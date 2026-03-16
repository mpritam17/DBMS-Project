# DBMS Term Project

Implementation of a page-backed R-Tree for high-dimensional image indexing.

## Current Scope
- C++ storage manager for fixed-size 4KB pages
- Python embedding extraction pipeline for image datasets
- Week 1 documentation for storage layout and vector export format

## Repository Layout
- `cpp/`: C++ source, headers, and tests
- `scripts/`: Python utilities for embedding extraction
- `docs/`: planning and implementation notes

## Python Setup
Create and use the workspace virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Build And Run
Build the C++ smoke test with CMake:

```bash
cmake -S . -B build
cmake --build build
./build/storage_smoke
```

## Embedding Export
Generate a binary vector file in the `VEC1` format:

```bash
source .venv/bin/activate
python scripts/extract_embeddings.py --dataset cifar10 --output data/cifar10_vecs.bin --dims 128 --limit 5000
```

## Next Milestones
- Implement a `VEC1` reader and page bulk-loader in C++
- Add a buffer pool manager with LRU replacement
- Add R-Tree node serialization and search logic