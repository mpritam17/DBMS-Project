import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import FormData from "form-data";
import mongoose from "mongoose";
import multer from "multer";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
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
const INCREMENTAL_INSERT_BIN = process.env.INCREMENTAL_INSERT_BIN || path.join(repoRoot, "build", "incremental_insert");
const IMAGE_SEARCH_API = process.env.IMAGE_SEARCH_API || "http://127.0.0.1:5001";
const IMAGE_DIR = process.env.IMAGE_DIR || path.join(repoRoot, "data", "sample_images");
const VEC_FILE = process.env.VEC_FILE || path.join(repoRoot, "data", "sample_vecs.bin");
const BENCHMARK_USE_FAIR = process.env.BENCHMARK_USE_FAIR !== "0";
const BENCHMARK_BPM_PAGES = process.env.BENCHMARK_BPM_PAGES || "auto";

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
let imageHashToId = null;

function parseBenchmarkOutput(output) {
  const metrics = {};
  const results = [];
  const neighbors = [];
  const pointMatches = [];

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
  pull("kdBuildUs", /kd_build_us:\s+([0-9.]+)/);
  pull("rtreeUs", /Average R-tree KNN latency:\s+([0-9.]+)\s+us/);
  pull("kdUs", /Average KD-tree latency:\s+([0-9.]+)\s+us/);
  pull("bruteUs", /Average brute-force latency:\s+([0-9.]+)\s+us/);
  pull("pointSearchUs", /Average R-tree point-search latency:\s+([0-9.]+)\s+us/);
  pull("pointHitRate", /Point-search hit-rate:\s+([0-9.]+)/);
  pull("pointMatchesPerQuery", /Point-search matches\/query:\s+([0-9.]+)/);
  pull("pointNodesVisited", /Point-search avg nodes visited:\s+([0-9.]+)/);
  pull("pointEntriesExamined", /Point-search avg entries examined:\s+([0-9.]+)/);
  pull("kdRecall", /Average KD recall@k:\s+([0-9.]+)/);
  pull("recall", /Average recall@k:\s+([0-9.]+)/);
  pull("bpmFetchRequests", /Buffer-pool fetch requests:\s+(\d+)/);
  pull("bpmFetchHits", /Buffer-pool hits:\s+(\d+)/);
  pull("bpmFetchMisses", /Buffer-pool misses:\s+(\d+)/);
  pull("bpmHitRate", /Buffer-pool hit-rate:\s+([0-9.]+)/);
  pull("diskReads", /reads=(\d+)/);
  pull("diskWrites", /writes=(\d+)/);

  for (const line of output.split("\n")) {
    const trimmed = line.trim();
    const rowMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    if (rowMatch) {
      const parts = rowMatch[1].split(",").map((part) => part.trim());
      if (parts.length >= 8) {
        results.push({
          queryId: Number(parts[0]),
          rtreeUs: Number(parts[1]),
          kdUs: Number(parts[2]),
          bruteUs: Number(parts[3]),
          pointUs: Number(parts[4]),
          pointMatches: Number(parts[5]),
          kdRecall: Number(parts[6]),
          recall: Number(parts[7]),
        });
      } else if (parts.length >= 4) {
        results.push({
          queryId: Number(parts[0]),
          rtreeUs: Number(parts[1]),
          bruteUs: Number(parts[2]),
          recall: Number(parts[3]),
        });
      }
    }

    const neighborMatch = trimmed.match(/^\{\s*"id":\s*(\d+),\s*"distance":\s*([^}]+)\s*\}$/);
    if (neighborMatch) {
      neighbors.push({
        id: Number(neighborMatch[1]),
        distance: Number(neighborMatch[2].trim())
      });
    }

    const pointMatch = trimmed.match(/^\{\s*"point_id":\s*(\d+)\s*\}$/);
    if (pointMatch) {
      pointMatches.push(Number(pointMatch[1]));
    }
  }

  return { metrics, results, neighbors, pointMatches, raw: output };
}

function parseImageIdFromFilename(filename) {
  const name = String(filename || "").trim();
  if (!name) return null;

  // img_0001.png, img_12345.jpg
  let match = name.match(/^img_(\d+)\.(?:png|jpg|jpeg)$/i);
  if (match) {
    return Number(match[1]);
  }

  // 42.png, 42.jpg
  match = name.match(/^(\d+)\.(?:png|jpg|jpeg)$/i);
  if (match) {
    return Number(match[1]);
  }

  return null;
}

function hashBuffer(buffer) {
  return createHash("sha1").update(buffer).digest("hex");
}

function buildImageHashIndex() {
  const index = new Map();
  if (!fs.existsSync(IMAGE_DIR)) {
    return index;
  }

  let entries = [];
  try {
    entries = fs.readdirSync(IMAGE_DIR, { withFileTypes: true });
  } catch (error) {
    console.warn("Failed to read image directory for hash index:", error.message);
    return index;
  }

  for (const entry of entries) {
    if (!entry.isFile()) {
      continue;
    }

    const imageId = parseImageIdFromFilename(entry.name);
    if (!Number.isInteger(imageId) || imageId < 0) {
      continue;
    }

    const imagePath = path.join(IMAGE_DIR, entry.name);
    try {
      const digest = hashBuffer(fs.readFileSync(imagePath));
      index.set(digest, imageId);
    } catch {
      // Skip unreadable files and continue indexing.
    }
  }

  console.log(`Image hash index built: ${index.size} entries`);
  return index;
}

function getImageHashIndex() {
  if (imageHashToId === null) {
    imageHashToId = buildImageHashIndex();
  }
  return imageHashToId;
}

function resolveUploadedImageId(file) {
  const byName = parseImageIdFromFilename(file?.originalname);
  if (Number.isInteger(byName) && byName >= 0) {
    return { id: byName, matchType: "filename" };
  }

  if (!file?.buffer || !Buffer.isBuffer(file.buffer)) {
    return null;
  }

  const byHash = getImageHashIndex().get(hashBuffer(file.buffer));
  if (Number.isInteger(byHash) && byHash >= 0) {
    return { id: byHash, matchType: "content-hash" };
  }

  return null;
}

function readVecByIdFromVec1(vecPath, targetId) {
  if (!fs.existsSync(vecPath)) {
    return null;
  }

  const raw = fs.readFileSync(vecPath);
  if (raw.length < 24 || raw.toString("ascii", 0, 4) !== "VEC1") {
    return null;
  }

  const count = Number(raw.readBigUInt64LE(8));
  const dims = raw.readUInt32LE(16);
  const entryBytes = 8 + (dims * 4);
  let offset = 24;

  for (let i = 0; i < count; i += 1) {
    if (offset + entryBytes > raw.length) {
      break;
    }

    const id = Number(raw.readBigUInt64LE(offset));
    if (id === targetId) {
      const values = new Array(dims);
      const vecOffset = offset + 8;
      for (let d = 0; d < dims; d += 1) {
        values[d] = raw.readFloatLE(vecOffset + (d * 4));
      }
      return values;
    }

    offset += entryBytes;
  }

  return null;
}

function resolveNearDuplicateVector(extractData, resolveVectorById = null) {
  const nearDuplicate = extractData?.near_duplicate;
  if (!nearDuplicate || nearDuplicate.confident !== true) {
    return null;
  }

  const id = Number(nearDuplicate.id);
  if (!Number.isInteger(id) || id < 0) {
    return null;
  }

  let vec = null;
  if (typeof resolveVectorById === "function") {
    vec = resolveVectorById(id);
  }
  if (!vec) {
    vec = readVecByIdFromVec1(VEC_FILE, id);
  }
  if (!vec) {
    return null;
  }

  return {
    id,
    vec,
    hashDistance: Number(nearDuplicate.hashDistance),
    secondBestDistance: Number(nearDuplicate.secondBestDistance),
    gap: Number(nearDuplicate.gap),
  };
}

async function runBenchmark({
  dbPath,
  querySelector,
  k,
  fairMode = BENCHMARK_USE_FAIR,
  bpmPages = BENCHMARK_BPM_PAGES,
}) {
  const args = [dbPath, String(querySelector), String(k)];
  if (fairMode) {
    args.push("--fair");
  }
  if (bpmPages !== undefined && bpmPages !== null && String(bpmPages).trim().length > 0) {
    args.push("--bpm-pages", String(bpmPages).trim());
  }
  const timeoutMs = 600000; // 10 minutes timeout for large databases
  const { stdout, stderr } = await execFileAsync(BENCHMARK_BIN, args, { timeout: timeoutMs });
  if (stderr && stderr.trim().length > 0) {
    console.error("Benchmark stderr:", stderr);
  }
  return parseBenchmarkOutput(stdout);
}

async function runPointSearchBenchmark({
  dbPath,
  querySelector,
  bpmPages = BENCHMARK_BPM_PAGES,
}) {
  const args = [dbPath, String(querySelector), "1", "--point-only"];
  if (BENCHMARK_USE_FAIR) {
    args.push("--fair");
  }
  if (bpmPages !== undefined && bpmPages !== null && String(bpmPages).trim().length > 0) {
    args.push("--bpm-pages", String(bpmPages).trim());
  }

  const timeoutMs = 600000;
  const { stdout, stderr } = await execFileAsync(BENCHMARK_BIN, args, { timeout: timeoutMs });
  if (stderr && stderr.trim().length > 0) {
    console.error("Point-search benchmark stderr:", stderr);
  }
  return parseBenchmarkOutput(stdout);
}

function parseInsertIdToken(idInput) {
  if (idInput === undefined || idInput === null || String(idInput).trim() === "") {
    return "auto";
  }
  const normalized = String(idInput).trim().toLowerCase();
  if (normalized === "auto") {
    return "auto";
  }
  const parsed = Number(normalized);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error("id must be a non-negative integer or 'auto'");
  }
  return String(parsed);
}

async function extractImageVector(file) {
  const formData = new FormData();
  formData.append("image", file.buffer, {
    filename: file.originalname || "image.jpg",
    contentType: file.mimetype || "image/jpeg",
    knownLength: file.buffer.length,
  });

  let response;
  try {
    response = await new Promise((resolve, reject) => {
      formData.submit(`${IMAGE_SEARCH_API}/extract`, (err, res) => {
        if (err) {
          reject(err);
        } else {
          resolve(res);
        }
      });
    });
  } catch {
    const error = new Error("Image extraction service unavailable.");
    error.statusCode = 503;
    throw error;
  }

  let body = "";
  for await (const chunk of response) {
    body += chunk;
  }

  let parsed;
  try {
    parsed = JSON.parse(body);
  } catch {
    const error = new Error("Invalid response from image extraction service.");
    error.statusCode = 502;
    throw error;
  }

  if (!parsed.ok) {
    const error = new Error(parsed.error || "Image vector extraction failed.");
    error.statusCode = 502;
    throw error;
  }

  return parsed;
}

function toNumericVector(input, fieldName) {
  if (!Array.isArray(input) || input.length === 0) {
    throw new Error(`${fieldName} is missing or empty`);
  }
  return input.map((value) => {
    const n = Number(value);
    if (!Number.isFinite(n)) {
      throw new Error(`${fieldName} contains non-numeric values`);
    }
    return n;
  });
}

function chooseVectorForDims(extractData, expectedDims) {
  const rawVector = Array.isArray(extractData?.vector)
    ? toNumericVector(extractData.vector, "vector")
    : null;
  const pcaVector = Array.isArray(extractData?.vector_pca)
    ? toNumericVector(extractData.vector_pca, "vector_pca")
    : null;

  if (pcaVector && Number(extractData?.pca_dims) === expectedDims) {
    return { vector: pcaVector, source: "pca" };
  }
  if (rawVector && Number(extractData?.dims) === expectedDims) {
    return { vector: rawVector, source: "raw" };
  }
  if (pcaVector && pcaVector.length === expectedDims) {
    return { vector: pcaVector, source: "pca" };
  }
  if (rawVector && rawVector.length === expectedDims) {
    return { vector: rawVector, source: "raw" };
  }

  const rawDims = rawVector ? rawVector.length : 0;
  const pcaDims = pcaVector ? pcaVector.length : 0;
  throw new Error(
    `No extracted vector matches DB dims=${expectedDims} (raw=${rawDims}, pca=${pcaDims})`
  );
}

function saveInsertedImageFile(file, insertedId) {
  fs.mkdirSync(IMAGE_DIR, { recursive: true });

  const extFromName = path.extname(file?.originalname || "").toLowerCase();
  const allowed = new Set([".png", ".jpg", ".jpeg", ".bmp", ".webp"]);
  const ext = allowed.has(extFromName) ? extFromName : ".jpg";
  const normalizedExt = ext === ".jpeg" ? ".jpg" : ext;

  const fileName = `img_${insertedId}${normalizedExt}`;
  const imagePath = path.join(IMAGE_DIR, fileName);
  fs.writeFileSync(imagePath, file.buffer);

  if (imageHashToId !== null) {
    imageHashToId.set(hashBuffer(file.buffer), insertedId);
  }

  return imagePath;
}

function buildImageSearchOutputFromBenchmark({
  parsed,
  k,
  extractTiming,
  vectorSource,
  shortcutMatch,
  nearDuplicateShortcut,
}) {
  const pointMatches = Array.isArray(parsed?.pointMatches) ? parsed.pointMatches : [];
  const neighbors = Array.isArray(parsed?.neighbors) ? parsed.neighbors : [];
  const merged = [];
  const seen = new Set();

  for (const id of pointMatches) {
    if (seen.has(id)) {
      continue;
    }
    seen.add(id);
    merged.push({ id, distance: 0.0, source: "point" });
  }

  for (const neighbor of neighbors) {
    if (merged.length >= k) {
      break;
    }
    if (seen.has(neighbor.id)) {
      continue;
    }
    seen.add(neighbor.id);
    merged.push({ id: neighbor.id, distance: neighbor.distance, source: "knn" });
  }

  const pointSearchMs = Number(parsed?.metrics?.pointSearchUs || 0) / 1000.0;
  const knnSearchMs = Number(parsed?.metrics?.rtreeUs || 0) / 1000.0;
  const baseTiming = extractTiming || {};
  const totalMs = Number(baseTiming.total_ms || 0) + pointSearchMs + knnSearchMs;
  const method = pointMatches.length > 0
    ? (merged.length > pointMatches.length ? "point+knn" : "point-only")
    : "knn-only";

  return {
    ok: true,
    k,
    results: merged.slice(0, k).map((item) => ({
      ...item,
      imageUrl: `/api/images/${item.id}`,
    })),
    timing: {
      imageLoad_ms: Number(baseTiming.imageLoad_ms || 0),
      embedding_ms: Number(baseTiming.embedding_ms || 0),
      pca_ms: Number(baseTiming.pca_ms || 0),
      pointSearch_ms: pointSearchMs,
      knnSearch_ms: knnSearchMs,
      total_ms: totalMs,
    },
    searchDiagnostics: {
      method,
      pointMatchCount: pointMatches.length,
      pointMatchIds: pointMatches.slice(0, 10),
      pointNodesVisited: Number(parsed?.metrics?.pointNodesVisited || 0),
      pointEntriesExamined: Number(parsed?.metrics?.pointEntriesExamined || 0),
      pointHitRate: Number(parsed?.metrics?.pointHitRate || 0),
      rtreeUs: Number(parsed?.metrics?.rtreeUs || 0),
      kdUs: Number(parsed?.metrics?.kdUs || 0),
      bruteUs: Number(parsed?.metrics?.bruteUs || 0),
      vectorSource,
      shortcutMatch: shortcutMatch || null,
      nearDuplicateId: nearDuplicateShortcut?.id ?? null,
      nearDuplicateHashDistance: nearDuplicateShortcut?.hashDistance ?? null,
      nearDuplicateSecondBestDistance: nearDuplicateShortcut?.secondBestDistance ?? null,
      nearDuplicateGap: nearDuplicateShortcut?.gap ?? null,
    },
    dims: Number(parsed?.metrics?.dims || 0),
    pcaEnabled: vectorSource === "pca",
  };
}

function normalizeVectorSelector(vectorInput) {
  if (Array.isArray(vectorInput)) {
    if (vectorInput.length === 0) {
      throw new Error("vector array cannot be empty");
    }
    const values = vectorInput.map((value) => {
      const n = Number(value);
      if (!Number.isFinite(n)) {
        throw new Error("vector array contains non-numeric values");
      }
      return n;
    });
    return `vec:${values.join(",")}`;
  }

  const raw = String(vectorInput ?? "").trim();
  if (raw.length === 0) {
    throw new Error("vector is required");
  }

  const cleaned = raw.startsWith("vec:") ? raw.slice(4) : raw;
  const parts = cleaned.split(",").map((part) => part.trim());
  if (parts.length === 0 || parts.some((part) => part.length === 0)) {
    throw new Error("vector must be a comma-separated list of numbers");
  }
  for (const part of parts) {
    const n = Number(part);
    if (!Number.isFinite(n)) {
      throw new Error("vector must contain only numeric values");
    }
  }
  return `vec:${parts.join(",")}`;
}

function parseIncrementalInsertOutput(output) {
  const pull = (regex, cast = String) => {
    const match = output.match(regex);
    if (!match) {
      return null;
    }
    return cast(match[1]);
  };

  return {
    dbFile: pull(/db_file:\s+(.+)/),
    indexDbFile: pull(/index_db_file:\s+(.+)/),
    id: pull(/id:\s+(\d+)/, Number),
    dims: pull(/dims:\s+(\d+)/, Number),
    vectorsTotal: pull(/vectors_total:\s+(\d+)/, Number),
    rebuilt: /rebuild/i.test(output),
    raw: output,
  };
}

async function runIncrementalInsert({
  dbPath,
  idToken,
  vectorSelector,
  bpmPages,
}) {
  const args = [dbPath, String(idToken), vectorSelector];
  if (bpmPages !== undefined && bpmPages !== null) {
    args.push("--bpm-pages", String(bpmPages));
  }
  const timeoutMs = 600000;
  const { stdout, stderr } = await execFileAsync(INCREMENTAL_INSERT_BIN, args, { timeout: timeoutMs });
  if (stderr && stderr.trim().length > 0) {
    console.error("Incremental insert stderr:", stderr);
  }
  return parseIncrementalInsertOutput(stdout);
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

async function resolveDbDims(dbPath) {
  try {
    const dimRun = await execFileAsync(BENCHMARK_BIN, [dbPath, "dim", "1"]);
    const match = dimRun.stdout.match(/dims:\s+(\d+)/);
    if (match) {
      const dims = Number(match[1]);
      if (Number.isInteger(dims) && dims > 0) {
        return dims;
      }
    }
  } catch {
    // Fall through to error below.
  }

  throw new Error(`Failed to determine vector dimensions for DB: ${dbPath}`);
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
          "Query timed out. The benchmark process can still be expensive for large datasets; use a smaller DB (for example large_3000.db) for interactive UI, or run large sweeps offline.",
      });
    }
    return res.status(500).json({ ok: false, error: error.message });
  }
});

app.post("/api/incremental-insert", upload.single("image"), async (req, res) => {
  try {
    if (!fs.existsSync(INCREMENTAL_INSERT_BIN)) {
      return res.status(503).json({
        ok: false,
        error: `Incremental insert binary not found: ${INCREMENTAL_INSERT_BIN}`,
      });
    }

    if (!req.file) {
      return res.status(400).json({ ok: false, error: "No image file provided for incremental insert" });
    }

    const dbPath = resolveDbPath(req.body.dbPath);
    if (!fs.existsSync(dbPath)) {
      return res.status(400).json({ ok: false, error: `Database not found: ${dbPath}` });
    }

    const dbDims = await resolveDbDims(dbPath);
    const extractData = await extractImageVector(req.file);
    const selected = chooseVectorForDims(extractData, dbDims);
    const vectorSelector = normalizeVectorSelector(selected.vector);

    const idToken = parseInsertIdToken(req.body.id);

    let bpmPages;
    if (req.body.bpmPages !== undefined && req.body.bpmPages !== null && String(req.body.bpmPages).trim() !== "") {
      const parsed = Number(req.body.bpmPages);
      if (!Number.isInteger(parsed) || parsed < 1) {
        return res.status(400).json({ ok: false, error: "bpmPages must be an integer >= 1" });
      }
      bpmPages = parsed;
    }

    const data = await runIncrementalInsert({ dbPath, idToken, vectorSelector, bpmPages });
    const insertedId = Number(data?.id);

    let storedImagePath = null;
    if (Number.isInteger(insertedId)) {
      try {
        storedImagePath = saveInsertedImageFile(req.file, insertedId);
      } catch (error) {
        console.warn("Failed to persist inserted image file:", error.message);
      }
    }

    return res.json({
      ok: true,
      request: {
        dbPath,
        id: idToken,
        imageName: req.file.originalname || null,
        vectorDims: selected.vector.length,
        vectorSource: selected.source,
      },
      data,
      extractionTiming: extractData.timing || {},
      imageStoredAs: storedImagePath,
      cacheSync: {
        updated: false,
        reason: "not-applicable",
      },
    });
  } catch (error) {
    if (error?.statusCode) {
      return res.status(error.statusCode).json({ ok: false, error: error.message });
    }
    if (error && (error.killed || error.code === "ETIMEDOUT" || error.signal === "SIGTERM")) {
      return res.status(408).json({ ok: false, error: "Incremental insert timed out." });
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

    // Shortcut for dataset images: resolve by filename first, then content hash
    // so renamed uploads can still map to their exact stored vector.
    const shortcut = resolveUploadedImageId(req.file);
    if (shortcut) {
      const vec = readVecByIdFromVec1(VEC_FILE, shortcut.id);
      if (vec) {
        const querySelector = `vec:${vec.join(",")}`;
        const parsed = await runBenchmark({ dbPath, querySelector, k });
        return res.json({ ok: true, query: { queryId: "image", k, dbPath }, data: parsed });
      }
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

    const nearDuplicateShortcut = resolveNearDuplicateVector(extractData);
    if (nearDuplicateShortcut) {
      const querySelector = `vec:${nearDuplicateShortcut.vec.join(",")}`;
      const parsed = await runBenchmark({ dbPath, querySelector, k });
      return res.json({
        ok: true,
        query: {
          queryId: "image",
          k,
          dbPath,
          nearDuplicateId: nearDuplicateShortcut.id,
          nearDuplicateHashDistance: nearDuplicateShortcut.hashDistance,
          nearDuplicateGap: nearDuplicateShortcut.gap,
        },
        data: parsed,
      });
    }

    const dbDims = await resolveDbDims(dbPath);
    const selected = chooseVectorForDims(extractData, dbDims);

    const vectorStr = selected.vector.join(",");
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

// Image search endpoint: R-tree KNN + R-tree point-search feedback.
app.post("/api/image-search", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ ok: false, error: "No image file provided" });
    }

    const k = Number(req.body.k || 10);
    if (!Number.isInteger(k) || k < 1) {
      return res.status(400).json({ ok: false, error: "k must be >= 1" });
    }

    const resolvedDbPath = resolveDbPath(req.body.dbPath);
    if (!fs.existsSync(resolvedDbPath)) {
      return res.status(400).json({ ok: false, error: `Database not found: ${resolvedDbPath}` });
    }

    const dbDims = await resolveDbDims(resolvedDbPath);

    let queryVector = null;
    let vectorSource = "unknown";
    let shortcutMatch = null;
    let nearDuplicateShortcut = null;
    let extractTiming = { imageLoad_ms: 0, embedding_ms: 0, pca_ms: 0, total_ms: 0 };

    // Shortcut for dataset images: resolve by filename first, then content hash.
    const shortcut = resolveUploadedImageId(req.file);
    if (shortcut) {
      const vec = readVecByIdFromVec1(VEC_FILE, shortcut.id);
      if (Array.isArray(vec) && vec.length === dbDims) {
        queryVector = vec.slice();
        vectorSource = "shortcut";
        shortcutMatch = shortcut.matchType;
      }
    }

    if (!queryVector) {
      const extractData = await extractImageVector(req.file);
      extractTiming = extractData.timing || extractTiming;

      nearDuplicateShortcut = resolveNearDuplicateVector(extractData);

      if (
        nearDuplicateShortcut
        && Array.isArray(nearDuplicateShortcut.vec)
        && nearDuplicateShortcut.vec.length === dbDims
      ) {
        queryVector = nearDuplicateShortcut.vec.slice();
        vectorSource = "near-duplicate-shortcut";
        shortcutMatch = "near-duplicate-hash";
      } else {
        const selected = chooseVectorForDims(extractData, dbDims);
        queryVector = selected.vector;
        vectorSource = selected.source;
      }
    }

    if (!Array.isArray(queryVector) || queryVector.length === 0) {
      return res.status(500).json({
        ok: false,
        error: "Query vector could not be prepared for image search",
      });
    }

    if (queryVector.length !== dbDims) {
      return res.status(500).json({
        ok: false,
        error: `Query vector dimensions do not match DB dims=${dbDims}`,
      });
    }

    const querySelector = normalizeVectorSelector(queryVector);
    const parsed = await runBenchmark({
      dbPath: resolvedDbPath,
      querySelector,
      k,
      extractTiming,
      vectorSource,
      shortcutMatch,
      nearDuplicateShortcut,
    });

    const payload = buildImageSearchOutputFromBenchmark({
      parsed,
      k,
      extractTiming,
      vectorSource,
      shortcutMatch,
      nearDuplicateShortcut,
    });

    return res.json(payload);
  } catch (error) {
    console.error("Image search error:", error);
    if (error?.statusCode) {
      return res.status(error.statusCode).json({ ok: false, error: error.message });
    }
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
    console.log(`Using incremental insert binary: ${INCREMENTAL_INSERT_BIN}`);
    console.log(`Benchmark fair mode: ${BENCHMARK_USE_FAIR ? "on" : "off"}`);
    console.log(`Benchmark bpm pages: ${BENCHMARK_BPM_PAGES}`);
    console.log(`Default DB path: ${DEFAULT_DB_PATH}`);
    console.log(`Image search API: ${IMAGE_SEARCH_API}`);
    console.log(`Image directory: ${IMAGE_DIR}`);
  });
}

main();
