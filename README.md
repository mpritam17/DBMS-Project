# DBMS Term Project

Implementation of a page-backed R-Tree for high-dimensional image indexing.

## Current Scope
- C++ storage manager for fixed-size 4KB pages
- Python embedding extraction pipeline for image datasets
- C++ slotted pages, `VEC1` binary file parser, and bulk loader
- Buffer pool manager with concurrent page latching and LRU-2 style replacement
- Initial R-Tree node page serialization and bounding-box primitives
- Recursive R-Tree insertion with node splitting and root growth
- Week 1 documentation for storage layout and vector export format
- Week 4 R-tree query layer and full integration tests
- Week 5 MERN integration scaling up to 60,000 images loaded at 64D with fixed ID map pipeline

## Repository Layout
- `cpp/`: C++ source, headers, and tests
- `scripts/`: Python utilities for embedding extraction
- `docs/`: planning and implementation notes

## Documentation
- `docs/WEEK1_START.md`: Week 1 goals and quickstart
- `docs/WEEK1_IMPLEMENTATION_DETAILS.md`: detailed record of completed implementation and validation
- `docs/WEEK4_QUERY_LAYER.md`: week 4 R-tree query layer
- `docs/WEEK5_IMAGE_SEARCH_AND_MERN.md`: Week 5 E2E image search API scaled to 60k high-accuracy (64D) images using MERN

## Python Setup
Create and use the workspace virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Build And Run Tests
Build the C++ executables with CMake:

```bash
cmake -S . -B build
cmake --build build

# Run the raw storage manager I/O test
./build/storage_smoke

# Run the slotted-page vector packing test
./build/packing_test

# Run the buffer-pool regression test
./build/bpm_test

# Run the R-tree node serialization test
./build/rtree_node_test

# Run the R-tree insertion and split test
./build/rtree_insert_test
```

## Embedding Export and Bulk Loading
1. Generate a binary vector file in the `VEC1` format using Python:

```bash
source .venv/bin/activate
python scripts/extract_embeddings.py --dataset cifar10 --output data/cifar10_vecs.bin --dims 128 --limit 50
```

2. Bulk load those vectors into the C++ slotted-page database:

```bash
./build/bulk_load data/cifar10_vecs.bin sample.db
```

## Next Milestones
- Persist index metadata so an R-tree can be reopened from disk by metadata page ID
- Week 4: end-to-end query layer connecting the KNN index to the embedding store

## Week 4 Query Benchmark

Build and run the Week 4 integration benchmark:

```bash
cmake -S . -B build
cmake --build build
./build/week4_query_benchmark sample.db 0 10
./build/week4_query_benchmark sample.db all 10 week4_metrics.csv
```

It reads vectors from the slotted-page embedding store, builds an R-tree query layer through the buffer pool, executes KNN, and compares R-tree latency/recall against brute-force search.

## End-to-End Image Search API (MERN)

A MERN scaffold is available in `mern/`.

Start backend:

```bash
cd mern/backend
npm install
cp .env.example .env
npm run dev
```

Start frontend:

```bash
cd mern/frontend
npm install
npm run dev
```

Backend query endpoint:

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"queryId":0,"k":10,"dbPath":"/home/quantumec/Documents/DBMS_term_project/sample.db"}'
```

Fallback lightweight Python API remains available:

```bash
python scripts/week4_query_api.py --db sample.db --host 127.0.0.1 --port 8080
```

## Generate New 64D Dataset
To regenerate the 60,000 CIFAR-10 embeddings mapping to 64D features over ResNet-18:
```bash
source .venv/bin/activate
# This safely parses and sets 60,000 photos, bypassing index sorting faults previously identified.
python scripts/populate_database.py --source cifar10 --count 60000 --pca-dims 64
./build/bulk_load data/sample_vecs.bin sample.db
```
