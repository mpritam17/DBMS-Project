# Week 4 Query Layer And Benchmarking

This milestone delivers an end-to-end query layer that connects:

- the page-backed embedding store (slotted pages), and
- the R-tree KNN index/search path.

## What Was Added

- New tool: `week4_query_benchmark`
- Source file: `cpp/tools/week4_query_benchmark.cpp`
- Build target added in `CMakeLists.txt`

The tool reads vectors from the on-disk embedding store (`sample.db`), reuses/builds a companion R-tree index through `BufferPoolManager`, executes KNN for a selected query vector, and compares the result with KD-tree and brute-force baselines.

## Inputs

Usage:

```bash
./build/week4_query_benchmark <db_file> <query_id|all|all:N|vec:x,y,...> <k> [csv_output_path] [--fair] [--point-only] [--bpm-pages N|auto]
```

Example:

```bash
./build/week4_query_benchmark sample.db 0 10
./build/week4_query_benchmark sample.db all 10 benchmark_week4.csv
./build/week4_query_benchmark sample.db all:200 10 benchmark_week4_sampled.csv
```

Arguments:

- `db_file`: slotted-page database produced by `bulk_load`
- `query_id|all|all:N`: specific vector id, all query vectors, or first `N` vectors for faster sampled runs
- `k`: number of nearest neighbours
- `csv_output_path` (optional): writes per-query rows for report plotting

## Reported Metrics

The tool prints:

- R-tree KNN latency (microseconds)
- KD-tree latency and build time (microseconds)
- Brute-force latency (microseconds)
- R-tree point-search latency and hit metrics
- Recall@k (overlap between R-tree result ids and brute-force ids)
- Disk I/O counters (`StorageManager::disk_reads`, `StorageManager::disk_writes`)
- R-tree metadata page id

In `all` mode, the tool reports average latency/recall across all query vectors and can export per-query rows to CSV.

In `--point-only` mode, the benchmark skips KD and brute paths and reports point-search feedback only (useful for image-query exact-match checks).

## SQLite Baseline Comparison

To compare normal SQLite scans against the custom R-tree path, use:

```bash
python scripts/benchmark_sqlite_vs_rtree.py \
  --db sample.db \
  --benchmark-bin ./build/week4_query_benchmark \
  --query-selector all:200 \
  --k 10 \
  --output-csv sqlite_vs_rtree_metrics.csv
```

The script runs `week4_query_benchmark`, builds a temporary SQLite table with the same vectors, measures SQLite KNN scan latency, and outputs merged per-query comparison rows in a CSV.

## Notes

- The embedding store is treated as source-of-truth for vectors; index pages are persisted separately in `<db>.rtree_tmp.db`.
- If the companion index is stale/corrupt for the current dimensions, it is rebuilt and then reused on later runs.
- Recall should stay high as data and tree balancing improve.

## Index Persistence Design

Current layout:

- `sample.db`: embedding store pages only
- `sample.db.rtree_tmp.db`: R-tree index pages only

Why this is preferred here:

- Keeps data and index page formats isolated and easier to debug.
- Allows deleting/rebuilding only the index file without touching vector data.
- Reduces risk while iterating on index internals.

An embedded-single-file layout can be better for strict one-file deployment and stronger transactional coupling, but it requires more complex metadata and recovery machinery.

## MERN Integration Layer

A MERN scaffold is now available under `mern/`:

- Backend: `mern/backend` (Express + optional MongoDB persistence)
- Frontend: `mern/frontend` (React + Vite)

Backend setup:

```bash
cd mern/backend
npm install
cp .env.example .env
npm run dev
```

Frontend setup:

```bash
cd mern/frontend
npm install
npm run dev
```

Backend endpoints:

- `GET /api/health`
- `POST /api/query`
- `GET /api/query-logs`

Sample request:

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"queryId":0,"k":10,"dbPath":"/home/quantumec/Documents/DBMS_term_project/sample.db"}'
```

The backend delegates benchmark endpoints to `./build/week4_query_benchmark` and returns parsed metrics/rows. For the image-search endpoint, backend now uses R-tree KNN neighbors from the benchmark output and still reports R-tree point-search diagnostics.

## Lightweight Python API (Alternative)

For a dependency-light fallback, you can still run:

```bash
python scripts/week4_query_api.py --db sample.db --host 127.0.0.1 --port 8080
```
