# Week 1 Start Guide

## Week 1 Goal
A working disk I/O interface and an embedding dataset exported as a binary bulk-load file.

## Suggested Page Layout (4KB)
- Page size: `4096` bytes
- Header bytes: `64`
- Payload bytes: `4032`
- Entry format (initial): `uint64 image_id + float32[dims]`

Entry size formula:
`entry_bytes = 8 + 4 * dims`

Quick capacity check:
- `dims=64` -> `entry=264` bytes -> `15` vectors/page
- `dims=128` -> `entry=520` bytes -> `7` vectors/page
- `dims=256` -> `entry=1032` bytes -> `3` vectors/page

Recommended starting point: `dims=128` for a balance between index quality and page utilization.

## Binary Bulk-Load File Format (`VEC1`)
- Bytes `0..3`: magic = `VEC1`
- Bytes `4..7`: version = `1` (`uint32`)
- Bytes `8..15`: number of vectors (`uint64`)
- Bytes `16..19`: dimensions (`uint32`)
- Bytes `20..23`: reserved (`uint32`)
- Then repeated records:
- `uint64 vector_id`
- `float32[dims]`

## What To Do Today
1. Build and run the C++ storage smoke test.
2. Install Python dependencies for embedding extraction.
3. Export CIFAR-10 embeddings into `data/cifar10_vecs.bin`.
4. Record file metadata (rows, dims, file size) for your report.

## Commands
```bash
cmake -S . -B build
cmake --build build
./build/storage_smoke

python3 -m pip install torch torchvision pillow numpy
python3 scripts/extract_embeddings.py --dataset cifar10 --output data/cifar10_vecs.bin --dims 128 --limit 5000
```
