# DBMS Term Project

Implementation of a page-backed R-Tree for high-dimensional image indexing.

## Current Scope
- C++ storage manager for fixed-size pages (current page size: 16KB)
- Python embedding extraction and dataset generation pipeline
- Slotted-page embedding store with VEC1 parser and bulk loader
- Buffer pool manager with concurrent page latching and LRU-2 style replacement
- R-Tree insertion/split, persisted metadata, reopen support, and KNN search
- First-class exact-point query path in the R-Tree index
- Week 4 benchmark tool supporting both KNN and exact-point workloads
- In-memory KD-tree implementation for analysis benchmarks
- MERN backend/frontend integration, including image search and exact-point benchmark tab

## Repository Layout
- `cpp/`: C++ source, tools, and tests
- `scripts/`: Python utilities and fallback API
- `mern/`: backend/frontend web app
- `docs/`: milestone notes and implementation details

## Toolchain Requirements
- CMake 3.16+
- A C++17-capable compiler

Notes:
- The code uses C++17 headers such as `filesystem` and `optional`.
- Very old MinGW toolchains (for example GCC 6.x) are not sufficient.

## Python Setup
Create and use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Build And Run C++ Tests

```bash
cmake -S . -B build
cmake --build build

./build/storage_smoke
./build/packing_test
./build/bpm_test
./build/rtree_node_test
./build/rtree_insert_test
./build/rtree_knn_test
./build/rtree_point_test
./build/rtree_metadata_test
./build/kd_tree_test
```

## Embedding Export And Bulk Loading
1. Export vectors in VEC1 format:

```bash
source .venv/bin/activate
python scripts/extract_embeddings.py --dataset cifar10 --output data/cifar10_vecs.bin --dims 128 --limit 50
```

2. Bulk load into slotted-page DB:

```bash
./build/bulk_load data/cifar10_vecs.bin sample.db
```

## Week 4 Benchmark (KNN + Exact Point)

Build and run:

```bash
cmake -S . -B build
cmake --build build
```

KNN selectors:

```bash
./build/week4_query_benchmark sample.db 0 10
./build/week4_query_benchmark sample.db all 10 week4_knn_all.csv
./build/week4_query_benchmark sample.db all:200 10 week4_knn_sample.csv
./build/week4_query_benchmark sample.db vec:0.1,0.2,0.3,... 10
```

Exact-point selectors:

```bash
./build/week4_query_benchmark sample.db pointid:0 1
./build/week4_query_benchmark sample.db pointall 1 week4_point_all.csv
./build/week4_query_benchmark sample.db pointall:200 1 week4_point_sample.csv
./build/week4_query_benchmark sample.db pointvec:0.1,0.2,0.3,... 1
```

The benchmark reports:
- KNN latency/recall (KNN mode)
- Exact-point latency and exact-match rate (point mode)
- Storage reads/writes
- Buffer pool hit-rate stats

## KD-Tree Analysis Benchmark

Build and run:

```bash
cmake -S . -B build
cmake --build build
```

Run comparative analysis (R-tree vs KD-tree vs brute-force):

```bash
./build/kd_analysis_benchmark sample.db 0 10
./build/kd_analysis_benchmark sample.db all 10 kd_analysis_all.csv
./build/kd_analysis_benchmark sample.db all:200 10 kd_analysis_sample.csv
./build/kd_analysis_benchmark sample.db vec:0.1,0.2,0.3,... 10
```

The analysis benchmark reports:
- R-tree, KD-tree, and brute-force latency
- Recall@k for both R-tree and KD-tree against brute-force truth
- Storage I/O and buffer-pool hit-rate counters from the R-tree path

This benchmark is additive and does not change existing week4 query API behavior.

## SQLite Vs R-Tree Comparison

```bash
python scripts/benchmark_sqlite_vs_rtree.py \
  --db sample.db \
  --benchmark-bin ./build/week4_query_benchmark \
  --query-selector all:200 \
  --k 10 \
  --output-csv sqlite_vs_rtree_metrics.csv
```

## MERN App

Backend:

```bash
cd mern/backend
npm install
npm run dev
```

Frontend:

```bash
cd mern/frontend
npm install
npm run dev
```

Main backend endpoints:
- `POST /api/query` (KNN benchmark)
- `POST /api/point-query` (exact-point benchmark)
- `POST /api/query-image` and `POST /api/image-search` (image-driven flows)

Frontend includes three tabs:
- Image Search
- Query Benchmark
- Exact Point

## Optional Python Fallback API

```bash
python scripts/week4_query_api.py --db sample.db --host 127.0.0.1 --port 8080
```

## Dataset Regeneration (60k CIFAR-10, 64D)

```bash
source .venv/bin/activate
python scripts/populate_database.py --source cifar10 --count 60000 --pca-dims 64
./build/bulk_load data/sample_vecs.bin sample.db
```

## Cleanup Generated Files

```bash
bash scripts/cleanup_generated_files.sh --dry-run
bash scripts/cleanup_generated_files.sh --apply
bash scripts/cleanup_generated_files.sh --apply --include-node-modules
```
