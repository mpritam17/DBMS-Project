# Week 4 Query Layer And Benchmarking

This milestone delivers an end-to-end query layer that connects:

- the page-backed embedding store (slotted pages), and
- the R-tree KNN index/search path.

## What Was Added

- New tool: `week4_query_benchmark`
- Source file: `cpp/tools/week4_query_benchmark.cpp`
- Build target added in `CMakeLists.txt`

The tool reads vectors from the on-disk embedding store (`sample.db`), builds an R-tree index through `BufferPoolManager`, executes KNN for a selected query vector, and compares the result with a brute-force scanner.

## Inputs

Usage:

```bash
./build/week4_query_benchmark <db_file> <query_id|all|all:N> <k> [csv_output_path]
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
- Brute-force latency (microseconds)
- Recall@k (overlap between R-tree result ids and brute-force ids)
- Disk I/O counters (`StorageManager::disk_reads`, `StorageManager::disk_writes`)
- R-tree metadata page id

In `all` mode, the tool reports average latency/recall across all query vectors and can export per-query rows to CSV.

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

- The current benchmark rebuilds the R-tree each run for deterministic and simple measurement.
- The embedding store is treated as source-of-truth for vectors, then the query layer indexes those vectors.
- Recall should stay high as data and tree balancing improve.

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

The backend delegates search execution to `./build/week4_query_benchmark` and returns parsed metrics/rows. When `MONGODB_URI` is set, query logs are persisted in MongoDB.

## Lightweight Python API (Alternative)

For a dependency-light fallback, you can still run:

```bash
python scripts/week4_query_api.py --db sample.db --host 127.0.0.1 --port 8080
```
