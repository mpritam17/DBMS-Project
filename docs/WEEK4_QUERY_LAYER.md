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
./build/week4_query_benchmark <db_file> <query_id> <k>
```

Example:

```bash
./build/week4_query_benchmark sample.db 0 10
```

Arguments:

- `db_file`: slotted-page database produced by `bulk_load`
- `query_id`: vector id to use as the query point
- `k`: number of nearest neighbours

## Reported Metrics

The tool prints:

- R-tree KNN latency (microseconds)
- Brute-force latency (microseconds)
- Recall@k (overlap between R-tree result ids and brute-force ids)
- Disk I/O counters (`StorageManager::disk_reads`, `StorageManager::disk_writes`)
- R-tree metadata page id

## Notes

- The current benchmark rebuilds the R-tree each run for deterministic and simple measurement.
- The embedding store is treated as source-of-truth for vectors, then the query layer indexes those vectors.
- Recall should stay high as data and tree balancing improve.
