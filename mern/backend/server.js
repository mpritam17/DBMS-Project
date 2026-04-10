import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import FormData from "form-data";
import mongoose from "mongoose";
import multer from "multer";
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

dotenv.config();

const execFileAsync = promisify(execFile);
const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

app.use(cors());
app.use(express.json());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");

const PORT = Number(process.env.PORT || 5000);
const DEFAULT_DB_PATH = process.env.DEFAULT_DB_PATH || path.join(repoRoot, "sample.db");
const BENCHMARK_BIN = process.env.BENCHMARK_BIN || path.join(repoRoot, "build", "week4_query_benchmark");
const IMAGE_SEARCH_API = process.env.IMAGE_SEARCH_API || "http://127.0.0.1:5001";
const IMAGE_DIR = process.env.IMAGE_DIR || path.join(repoRoot, "data", "sample_images");

const queryLogSchema = new mongoose.Schema(
  {
    queryId: { type: Number, required: true },
    k: { type: Number, required: true },
    dbPath: { type: String, required: true },
    metrics: { type: Object, required: true },
    results: { type: Array, required: true },
  },
  { timestamps: true }
);

const QueryLog = mongoose.model("QueryLog", queryLogSchema);

function parseBenchmarkOutput(output) {
  const metrics = {};
  const results = [];

  const pull = (key, regex, cast = Number) => {
    const match = output.match(regex);
    if (match) {
      metrics[key] = cast(match[1]);
    }
  };

  pull("vectorsRaw", /vectors\(raw\):\s+(\d+)/);
  pull("vectorsUnique", /vectors\(unique\):\s+(\d+)/);
  pull("dims", /dims:\s+(\d+)/);
  pull("k", /k:\s+(\d+)/);
  pull("rtreeUs", /Average R-tree KNN latency:\s+([0-9.]+)\s+us/);
  pull("bruteUs", /Average brute-force latency:\s+([0-9.]+)\s+us/);
  pull("recall", /Average recall@k:\s+([0-9.]+)/);
  pull("diskReads", /reads=(\d+)/);
  pull("diskWrites", /writes=(\d+)/);

  for (const line of output.split("\n")) {
    const row = line.trim().match(/^\d+\.\s+(\d+),\s+([0-9.]+),\s+([0-9.]+),\s+([0-9.]+)/);
    if (row) {
      results.push({
        queryId: Number(row[1]),
        rtreeUs: Number(row[2]),
        bruteUs: Number(row[3]),
        recall: Number(row[4]),
      });
    }
  }

  return { metrics, results, raw: output };
}

async function runBenchmark({ dbPath, querySelector, k }) {
  const args = [dbPath, String(querySelector), String(k)];
  const timeoutMs = 300000; // Increased timeout for large databases
  const { stdout, stderr } = await execFileAsync(BENCHMARK_BIN, args, { timeout: timeoutMs });
  if (stderr && stderr.trim().length > 0) {
    console.error("Benchmark stderr:", stderr);
  }
  return parseBenchmarkOutput(stdout);
}

function resolveDbPath(inputPath) {
  const candidate = String(inputPath || DEFAULT_DB_PATH).trim();
  if (candidate.length === 0) {
    return DEFAULT_DB_PATH;
  }

  if (path.isAbsolute(candidate)) {
    return path.normalize(candidate);
  }

  // Treat relative paths as repository-root relative so "sample.db" is stable.
  return path.resolve(repoRoot, candidate);
}

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, service: "week4-mern-backend" });
});

app.post("/api/query", async (req, res) => {
  try {
    const rawQueryId = req.body.queryId;
    const k = Number(req.body.k || 10);
    const dbPath = resolveDbPath(req.body.dbPath);
    const isAll = String(rawQueryId).toLowerCase() === "all";
    const queryId = isAll ? null : Number(rawQueryId);
    const querySelector = isAll ? "all" : queryId;

    if (!isAll && (!Number.isInteger(queryId) || queryId < 0)) {
      return res.status(400).json({ ok: false, error: "queryId must be a non-negative integer or 'all'" });
    }
    if (!Number.isInteger(k) || k < 1) {
      return res.status(400).json({ ok: false, error: "k must be >= 1" });
    }
    if (!fs.existsSync(dbPath)) {
      return res.status(400).json({ ok: false, error: `Database not found: ${dbPath}` });
    }

    const parsed = await runBenchmark({ dbPath, querySelector, k });

    if (mongoose.connection.readyState === 1) {
      await QueryLog.create({
        queryId: queryId ?? -1,
        k,
        dbPath,
        metrics: parsed.metrics,
        results: parsed.results,
      });
    }

    return res.json({ ok: true, query: { queryId: querySelector, k, dbPath }, data: parsed });
  } catch (error) {
    if (error && (error.killed || error.code === "ETIMEDOUT" || error.signal === "SIGTERM")) {
      return res.status(408).json({
        ok: false,
        error:
          "Query timed out. This benchmark rebuilds an index per request; use a smaller DB (e.g. large_3000.db) for interactive UI, or run CLI benchmarks offline.",
      });
    }
    return res.status(500).json({ ok: false, error: error.message });
  }
});

app.post("/api/query-image", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ ok: false, error: "No image file provided" });
    }
    const k = Number(req.body.k || 10);
    const dbPath = resolveDbPath(req.body.dbPath);

    if (!Number.isInteger(k) || k < 1) {
      return res.status(400).json({ ok: false, error: "k must be >= 1" });
    }
    if (!fs.existsSync(dbPath)) {
      return res.status(400).json({ ok: false, error: `Database not found: ${dbPath}` });
    }

    // Extract embedding
    const formData = new FormData();
    formData.append("image", req.file.buffer, {
      filename: req.file.originalname || "image.jpg",
      contentType: req.file.mimetype || "image/jpeg",
      knownLength: req.file.buffer.length,
    });

    let extractResponse;
    try {
      extractResponse = await new Promise((resolve, reject) => {
        formData.submit(`${IMAGE_SEARCH_API}/extract`, (err, res) => {
          if (err) reject(err);
          else resolve(res);
        });
      });
    } catch (fetchError) {
      return res.status(503).json({ ok: false, error: "Image extraction service unavailable." });
    }

    let extractDataRaw = "";
    for await (const chunk of extractResponse) {
      extractDataRaw += chunk;
    }
    const extractData = JSON.parse(extractDataRaw);

    if (!extractData.ok) {
      return res.status(500).json({ ok: false, error: extractData.error || "Failed to extract vector" });
    }

    // Determine DB dimensions using a dummy query
    let dbDims = 128;
    try {
      const dummyRun = await execFileAsync(BENCHMARK_BIN, [dbPath, "all:1", "1"]);
      const match = dummyRun.stdout.match(/dims:\s+(\d+)/);
      if (match) dbDims = Number(match[1]);
    } catch (e) {
      // Ignored if query fails, we guess 128
    }

    let chosenVec = extractData.vector;
    if (extractData.pca_dims && extractData.pca_dims === dbDims) {
      chosenVec = extractData.vector_pca;
    }

    const vectorStr = chosenVec.join(",");
    const querySelector = `vec:${vectorStr}`;

    const parsed = await runBenchmark({ dbPath, querySelector, k });

    if (mongoose.connection.readyState === 1) {
      await QueryLog.create({
        queryId: -2,
        k,
        dbPath,
        metrics: parsed.metrics,
        results: parsed.results,
      });
    }

    return res.json({ ok: true, query: { queryId: "image", k, dbPath }, data: parsed });
  } catch (error) {
    if (error && (error.killed || error.code === "ETIMEDOUT" || error.signal === "SIGTERM")) {
      return res.status(408).json({
        ok: false,
        error: "Query timed out.",
      });
    }
    return res.status(500).json({ ok: false, error: error.message });
  }
});

app.get("/api/query-logs", async (_req, res) => {
  try {
    if (mongoose.connection.readyState !== 1) {
      return res.json({ ok: true, connectedMongo: false, items: [] });
    }
    const logs = await QueryLog.find().sort({ createdAt: -1 }).limit(20).lean();
    return res.json({ ok: true, connectedMongo: true, items: logs });
  } catch (error) {
    return res.status(500).json({ ok: false, error: error.message });
  }
});

// Image search endpoint - proxies to Python image search API
app.post("/api/image-search", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ ok: false, error: "No image file provided" });
    }

    const k = Number(req.body.k || 10);
    if (!Number.isInteger(k) || k < 1) {
      return res.status(400).json({ ok: false, error: "k must be >= 1" });
    }

    // Forward the request to the Python image search API using form-data package
    const formData = new FormData();
    formData.append("image", req.file.buffer, {
      filename: req.file.originalname || "image.jpg",
      contentType: req.file.mimetype || "image/jpeg",
      knownLength: req.file.buffer.length,
    });
    formData.append("k", String(k));

    let response;
    try {
      // Use form-data's submit method for proper multipart encoding
      response = await new Promise((resolve, reject) => {
        formData.submit(`${IMAGE_SEARCH_API}/search`, (err, res) => {
          if (err) reject(err);
          else resolve(res);
        });
      });
    } catch (fetchError) {
      console.error("Fetch error:", fetchError);
      if (fetchError.code === "ECONNREFUSED") {
        return res.status(503).json({
          ok: false,
          error: "Image search service unavailable. Start the Python API with: python scripts/image_search_api.py --vec-file data/sample_vecs.bin --image-dir data/sample_images --pca-model data/pca_model.npz",
        });
      }
      throw fetchError;
    }

    // Read response body
    let responseText = "";
    for await (const chunk of response) {
      responseText += chunk.toString();
    }

    if (response.statusCode !== 200) {
      let errorData = {};
      try {
        errorData = JSON.parse(responseText);
      } catch {
        errorData = { error: responseText || `HTTP ${response.statusCode}` };
      }
      console.error("Image search API error:", response.statusCode, errorData);
      return res.status(response.statusCode).json({
        ok: false,
        error: errorData.error || `Image search API returned ${response.statusCode}`,
      });
    }

    let data;
    try {
      data = JSON.parse(responseText);
    } catch {
      console.error("Failed to parse response:", responseText);
      return res.status(500).json({ ok: false, error: "Invalid response from image search API" });
    }
    
    // Rewrite image URLs to go through this server
    if (data.results) {
      data.results = data.results.map((r) => ({
        ...r,
        imageUrl: `/api/images/${r.id}`,
      }));
    }

    return res.json(data);
  } catch (error) {
    console.error("Image search error:", error);
    return res.status(500).json({ ok: false, error: error.message || "Internal server error" });
  }
});

// Serve images by ID
app.get("/api/images/:id", (req, res) => {
  const imageId = parseInt(req.params.id, 10);
  if (isNaN(imageId) || imageId < 0) {
    return res.status(400).json({ ok: false, error: "Invalid image ID" });
  }

  // Try common naming patterns
  const patterns = [
    `img_${String(imageId).padStart(4, "0")}.png`,
    `img_${String(imageId).padStart(4, "0")}.jpg`,
    `img_${imageId}.png`,
    `img_${imageId}.jpg`,
    `${imageId}.png`,
    `${imageId}.jpg`,
  ];

  for (const pattern of patterns) {
    const imagePath = path.join(IMAGE_DIR, pattern);
    if (fs.existsSync(imagePath)) {
      return res.sendFile(imagePath);
    }
  }

  return res.status(404).json({ ok: false, error: `Image ${imageId} not found` });
});

// Check if image search service is available
app.get("/api/image-search/status", async (_req, res) => {
  try {
    const response = await fetch(`${IMAGE_SEARCH_API}/health`, { timeout: 2000 });
    if (response.ok) {
      const data = await response.json();
      return res.json({ ok: true, service: "available", ...data });
    }
    return res.json({ ok: false, service: "unavailable" });
  } catch {
    return res.json({ ok: false, service: "unavailable" });
  }
});

async function main() {
  if (process.env.MONGODB_URI) {
    try {
      await mongoose.connect(process.env.MONGODB_URI);
      console.log("MongoDB connected");
    } catch (error) {
      console.warn("MongoDB connection failed, continuing without persistence:", error.message);
    }
  } else {
    console.log("MONGODB_URI not set, running without persistence");
  }

  app.listen(PORT, () => {
    console.log(`Week4 MERN backend listening on http://localhost:${PORT}`);
    console.log(`Using benchmark binary: ${BENCHMARK_BIN}`);
    console.log(`Default DB path: ${DEFAULT_DB_PATH}`);
    console.log(`Image search API: ${IMAGE_SEARCH_API}`);
    console.log(`Image directory: ${IMAGE_DIR}`);
  });
}

main();
