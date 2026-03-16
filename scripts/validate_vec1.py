#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path


def read_header(path: Path):
    with path.open("rb") as f:
        raw = f.read(24)

    if len(raw) != 24:
        raise ValueError("File too small to contain VEC1 header")

    magic = raw[0:4]
    if magic != b"VEC1":
        raise ValueError(f"Invalid magic: {magic!r}")

    version = struct.unpack("<I", raw[4:8])[0]
    count = struct.unpack("<Q", raw[8:16])[0]
    dims = struct.unpack("<I", raw[16:20])[0]
    reserved = struct.unpack("<I", raw[20:24])[0]
    return version, count, dims, reserved


def main():
    parser = argparse.ArgumentParser(description="Validate and inspect VEC1 vector file header")
    parser.add_argument("--input", required=True, help="Path to VEC1 binary file")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    version, count, dims, reserved = read_header(path)
    entry_bytes = 8 + 4 * dims
    expected_size = 24 + count * entry_bytes
    actual_size = path.stat().st_size

    print(f"file={path}")
    print(f"version={version}")
    print(f"count={count}")
    print(f"dims={dims}")
    print(f"reserved={reserved}")
    print(f"entry_bytes={entry_bytes}")
    print(f"expected_size={expected_size}")
    print(f"actual_size={actual_size}")
    print(f"size_match={expected_size == actual_size}")


if __name__ == "__main__":
    main()