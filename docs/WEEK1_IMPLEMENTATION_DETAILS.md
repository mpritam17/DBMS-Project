# Week 1 Implementation Details

This document records exactly what has been implemented so far for the DBMS term project, how to run it, and what remains pending.

## 1. Completed Scope

### 1.1 Storage Manager (C++)
Implemented a page-based storage layer in:
- `cpp/include/storage_manager.h`
- `cpp/src/storage_manager.cpp`

Capabilities currently available:
- Fixed page size: `4096` bytes (`kPageSize`)
- Open/close database file
- Read page by page ID
- Write page by page ID
- Allocate new zero-initialized page
- Return current page count
- Flush pending writes

Safety checks implemented:
- Read rejects out-of-range page IDs
- Write rejects page buffers not equal to 4096 bytes
- Write rejects non-contiguous page creation (`page_id > num_pages_`)
- All public operations validate file-open state via `ensureOpen()`

### 1.2 Storage Smoke Test (C++)
Implemented test executable in:
- `cpp/tests/storage_manager_smoke.cpp`

Test flow:
1. Deletes prior `week1_smoke.db` if it exists.
2. Opens storage manager.
3. Allocates one page.
4. Writes sentinel bytes to the page.
5. Reads page back and verifies byte values.
6. Flushes and closes file.

Expected output:
- `StorageManager smoke test passed. Pages: 1`

### 1.3 Embedding Export Pipeline (Python)
Implemented script in:
- `scripts/extract_embeddings.py`

Supported dataset modes:
- `--dataset cifar10`: downloads and uses CIFAR-10 test split
- `--dataset folder`: recursively loads images from a local folder

Pipeline behavior:
1. Preprocess images to `224x224` with ImageNet normalization.
2. Extract features using pretrained `ResNet18` backbone (`fc` removed).
3. Project to user-defined dimensionality (`--dims`, default `128`).
4. L2-normalize vectors.
5. Write vectors to binary `VEC1` format.

### 1.4 Environment and Repo Setup
- Python virtual environment at `.venv/`
- Required Python packages installed:
  - `numpy`
  - `pillow`
  - `torch`
  - `torchvision`
- CMake-based project build configured in `CMakeLists.txt`
- Git repository initialized on branch `main`
- Ignore policy in `.gitignore` excludes virtualenv, build output, and generated runtime artifacts

### 1.5 VEC1 Inspection Utility
Implemented script in:
- `scripts/validate_vec1.py`

Utility behavior:
- Reads and validates `VEC1` header fields (`magic`, `version`, `count`, `dims`)
- Computes expected binary size using `entry_bytes = 8 + 4 * dims`
- Compares expected size against actual file size and reports consistency

## 2. Binary Formats

### 2.1 Storage Page (Current)
Current implementation stores 4096-byte pages and now includes explicit page layout constants in code.

Declared in `cpp/include/storage_manager.h`:
- `PageLayout::kPageSize = 4096`
- `PageLayout::kHeaderSize = 64`
- `PageLayout::kPayloadSize = 4032`
- `PageHeader` struct with `static_assert(sizeof(PageHeader) == 64)`

Status:
- Header constants and struct are now defined.
- Full slot-array packing logic is still pending.

### 2.2 `VEC1` Bulk-Load File
Produced by `scripts/extract_embeddings.py`.

Header layout (little-endian):
- Bytes `0..3`: magic `VEC1`
- Bytes `4..7`: version (`uint32`, value `1`)
- Bytes `8..15`: vector count (`uint64`)
- Bytes `16..19`: dimensions (`uint32`)
- Bytes `20..23`: reserved (`uint32`)

Record layout:
- `uint64 vector_id`
- `float32[dims] vector`

## 3. Build, Run, and Validation

### 3.1 C++ Build and Smoke Test
Commands:

```bash
cmake -S . -B build
cmake --build build -j
./build/storage_smoke
```

Latest verified result:
- Build succeeded (`storage_manager` + `storage_smoke` targets)
- Runtime output: `StorageManager smoke test passed. Pages: 1`

### 3.2 Python Script Validation
Command:

```bash
.venv/bin/python -m py_compile scripts/extract_embeddings.py scripts/validate_vec1.py
```

Latest verified result:
- Syntax compile passed with no errors for both scripts.

### 3.3 Dataset Artifact Validation
Generated and validated local folder-mode embeddings artifact:

```bash
.venv/bin/python scripts/extract_embeddings.py --dataset folder --data-root data/sample_images --output data/sample_vecs.bin --dims 128 --limit 500
.venv/bin/python scripts/validate_vec1.py --input data/sample_vecs.bin
```

Observed metadata:
- `file`: `data/sample_vecs.bin`
- `count`: `500`
- `dims`: `128`
- `entry_bytes`: `520`
- `expected_size`: `260024`
- `actual_size`: `260024`
- `size_match`: `True`

Note:
- CIFAR-10 download attempts were interrupted in this environment; folder-mode export was used to verify the end-to-end pipeline.

## 4. Current Constraints and Known Gaps

The following are intentionally pending and represent next implementation work:

1. No buffer pool manager yet.
2. Page header constants exist, but no slot-directory serialization/deserialization logic yet.
3. No `VEC1` reader in C++ yet.
4. No R-Tree node layout or insertion/split logic yet.
5. No KNN traversal/query execution yet.
6. No benchmark harness against SQLite and brute force yet.

## 5. Suggested Week 1 Closure Criteria

To declare Week 1 fully complete, finish these additions:

1. ~~Implement a C++ `VEC1` parser.~~ (Done: `cpp/src/vec1_reader.cpp`)
2. ~~Implement bulk-loader from `VEC1` into fixed-size pages.~~ (Done: `cpp/tools/bulk_load.cpp`)
3. ~~Implement page slot-directory serialization/deserialization logic using the declared header format.~~ (Done: `cpp/src/slotted_page.cpp`)
4. ~~Add at least one automated test that validates packing math (entry count/page) and reload integrity.~~ (Done: `cpp/tests/packing_test.cpp`)

Week 1 goals are now 100% complete!

## 6. File Inventory (Implemented Artifacts)

- `CMakeLists.txt`
- `cpp/include/storage_manager.h`
- `cpp/src/storage_manager.cpp`
- `cpp/tests/storage_manager_smoke.cpp`
- `scripts/extract_embeddings.py`
- `scripts/validate_vec1.py`
- `requirements.txt`
- `.gitignore`
- `docs/WEEK1_START.md`
