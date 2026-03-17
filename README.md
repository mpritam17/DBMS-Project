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

## Repository Layout
- `cpp/`: C++ source, headers, and tests
- `scripts/`: Python utilities for embedding extraction
- `docs/`: planning and implementation notes

## Documentation
- `docs/WEEK1_START.md`: Week 1 goals and quickstart
- `docs/WEEK1_IMPLEMENTATION_DETAILS.md`: detailed record of completed implementation and validation

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
```

It reads vectors from the slotted-page embedding store, builds an R-tree query layer through the buffer pool, executes KNN, and compares R-tree latency/recall against brute-force search.
