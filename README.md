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

## Incremental Insert (CLI And MERN)

For a fast append-only path (without full `populate_database + bulk_load` rebuild), the low-level CLI stays vector-based:

```bash
./build/incremental_insert sample.db auto vec:0.1,0.2,0.3,...
```

In the MERN UI/API, incremental insert is image-based: upload an image and the backend extracts the embedding, chooses PCA/raw dimensions to match the DB, appends the vector, and updates index files.

Notes:

- `auto` assigns `max(id)+1` from the current store.
- You can also provide an explicit ID instead of `auto`.
- Index persistence is in a companion temp index DB file (`sample.db.rtree_tmp.db`), not embedded into the embedding store file (`sample.db`).
- If that index is stale/corrupt, the tool rebuilds it once and then continues incrementally.

### Companion Temp Index DB: Why This Design

We intentionally keep vectors and index pages in separate files:

- `sample.db`: source-of-truth embedding store (slotted pages with vector payloads)
- `sample.db.rtree_tmp.db`: R-tree index pages and metadata

Compared to embedding the R-tree into the same file as vectors, the companion-file approach is generally better for this project because:

- Simpler page-type isolation: embedding pages and index pages never collide, reducing parser/recovery complexity.
- Safer rebuild path: stale/corrupt index can be dropped/rebuilt without rewriting vector data.
- Faster iteration: benchmark and UI can reuse the existing temp index across runs while keeping the data file unchanged.
- Lower risk during development: storage format changes in index logic do not force migration of the embedding store.

When embedding index and vectors into one file can be better:

- You need strict single-file portability and transactional coupling between data and index pages.
- You are ready to implement stronger metadata/versioning and crash-recovery semantics in one physical file.

For this term-project stage, companion-file persistence is the better tradeoff: cleaner architecture and safer incremental development.

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
./build/week4_query_benchmark sample.db all:200 10 week4_metrics_sampled.csv
```

It reads vectors from the slotted-page embedding store, builds an R-tree query layer through the buffer pool, executes KNN, and compares R-tree latency/recall against brute-force search. The selector `all:N` runs only the first `N` query vectors, which is useful for fast benchmark iterations on larger datasets.

Operational notes:

- The R-tree pages are persisted in `sample.db.rtree_tmp.db` and reused when dimensions match.
- KD-tree build is process-local in `week4_query_benchmark` (measured via `kd_build_us`).
- The MERN image-search path uses R-tree KNN results from `week4_query_benchmark` and surfaces point-search diagnostics.

## SQLite Vs R-Tree Comparison

To compare baseline SQLite scan latency against your custom R-tree access path:

```bash
python scripts/benchmark_sqlite_vs_rtree.py \
  --db sample.db \
  --benchmark-bin ./build/week4_query_benchmark \
  --query-selector all:200 \
  --k 10 \
  --output-csv sqlite_vs_rtree_metrics.csv
```

This command:

- runs `week4_query_benchmark` for R-tree and brute-force timings,
- loads the same vectors into a temporary SQLite table,
- times SQLite KNN scans using SQL `ORDER BY` distance,
- writes merged per-query metrics to `sqlite_vs_rtree_metrics.csv`.

## Cleanup Generated Files

Preview cleanup targets:

```bash
bash scripts/cleanup_generated_files.sh --dry-run
```

Apply cleanup:

```bash
bash scripts/cleanup_generated_files.sh --apply
```

Optional: also delete frontend/backend `node_modules` and frontend `dist`:

```bash
bash scripts/cleanup_generated_files.sh --apply --include-node-modules
```

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

Image search API startup (PCA enabled by default via `data/pca_model.npz`):

```bash
source .venv/bin/activate
python scripts/image_search_api.py --vec-file data/sample_vecs.bin --image-dir data/sample_images
```

If needed, PCA can be disabled explicitly:

```bash
python scripts/image_search_api.py --vec-file data/sample_vecs.bin --image-dir data/sample_images --no-pca --embedding-dims 64
```

## Generate New 64D Dataset
To regenerate the 60,000 CIFAR-10 embeddings mapping to 64D features over ResNet-18:
```bash
source .venv/bin/activate
# This safely parses and sets 60,000 photos, bypassing index sorting faults previously identified.
python scripts/populate_database.py --source cifar10 --count 60000 --pca-dims 64
./build/bulk_load data/sample_vecs.bin sample.db
```
