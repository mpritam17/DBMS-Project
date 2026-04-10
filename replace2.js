const fs = require('fs');
const file = 'mern/backend/server.js';
let content = fs.readFileSync(file, 'utf8');

const regex = /\/\/ Image search endpoint - proxies to Python image search API[\s\S]*?app.get\("\/api\/images\/:id",/m;

const replacement = `// Image search endpoint - strictly runs the actual C++ R-tree benchmark
app.post("/api/image-search", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ ok: false, error: "No image file provided" });
    }

    const k = Number(req.body.k || 10);
    if (!Number.isInteger(k) || k < 1) {
      return res.status(400).json({ ok: false, error: "k must be >= 1" });
    }

    const { dbPath } = req.body;
    const resolvedDbPath = resolveDbPath(dbPath);

    // Forward the request to the Python extract API
    const formData = new FormData();
    formData.append("image", req.file.buffer, {
      filename: req.file.originalname || "image.jpg",
      contentType: req.file.mimetype || "image/jpeg",
      knownLength: req.file.buffer.length,
    });

    let extractResponse;
    try {
      extractResponse = await new Promise((resolve, reject) => {
        formData.submit(\`\${IMAGE_SEARCH_API}/extract\`, (err, response) => {
          if (err) reject(err);
          else resolve(response);
        });
      });
    } catch (fetchError) {
      console.error("Fetch error:", fetchError);
      return res.status(503).json({
        ok: false,
        error: "Image search service offline. Start the Python API first.",
      });
    }

    let extractDataRaw = "";
    for await (const chunk of extractResponse) {
      extractDataRaw += chunk;
    }
    let extractData;
    try {
      extractData = JSON.parse(extractDataRaw);
    } catch {
      return res.status(500).json({ ok: false, error: "Invalid vector response from python" });
    }
    
    if (!extractData.ok) {
      return res.status(500).json({ ok: false, error: extractData.error || "Failed to extract vector" });
    }

    let dbDims = 128;
    try {
      const dummyRun = await execFileAsync(BENCHMARK_BIN, [resolvedDbPath, "all:1", "1"]);
      const match = dummyRun.stdout.match(/dims:\\s+(\\d+)/);
      if (match) dbDims = Number(match[1]);
    } catch (e) {}

    let chosenVec = extractData.vector;
    if (extractData.pca_dims && extractData.pca_dims === dbDims) {
      chosenVec = extractData.vector_pca;
    }

    const vectorStr = chosenVec.join(",");
    const querySelector = \`vec:\${vectorStr}\`;

    const startMs = Date.now();
    const parsed = await runBenchmark({ dbPath: resolvedDbPath, querySelector, k });
    const endMs = Date.now();

    return res.json({
      ok: true,
      k,
      results: parsed.neighbors.map((n) => ({
        id: n.id,
        distance: n.distance,
        imageUrl: \`/api/images/\${n.id}\`,
      })),
      timing: {
        imageLoad_ms: 0,
        embedding_ms: 0,
        pca_ms: 0,
        knnSearch_ms: (parsed.metrics.rtreeUs / 1000.0) || (endMs - startMs),
        total_ms: endMs - startMs,
      },
      dims: dbDims,
      pcaEnabled: extractData.pca_dims === dbDims,
    });
  } catch (error) {
    console.error("Image search error:", error);
    return res.status(500).json({ ok: false, error: error.message || "Internal server error" });
  }
});

// Serve images by ID
app.get("/api/images/:id",`;

content = content.replace(regex, replacement);
fs.writeFileSync(file, content);
