#!/usr/bin/env python3
"""Compare SQLite scan latency against the custom R-tree benchmark path.

The script reuses the existing `week4_query_benchmark` executable for R-tree and
brute-force metrics, then builds a temporary SQLite table from the same slotted
page vectors and times equivalent KNN SQL scans.
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import fmean


PAGE_SIZE = 16384
HEADER_SIZE = 64
PAGE_MAGIC = 0x50414745
SLOTTED_PAGE_TYPE = 0

HEADER_STRUCT = struct.Struct("<IHHIIHH44s")
SLOT_STRUCT = struct.Struct("<HH")


def read_vectors_from_slotted_db(db_path: Path) -> tuple[dict[int, tuple[float, ...]], int]:
    """Read unique vectors from the page-backed embedding store.

    Vector payload layout per slot item: uint64 id + float32[dims].
    """
    vectors_by_id: dict[int, tuple[float, ...]] = {}
    dims = 0

    with db_path.open("rb") as handle:
        page_index = 0
        while True:
            page = handle.read(PAGE_SIZE)
            if not page:
                break
            if len(page) != PAGE_SIZE:
                raise ValueError(
                    f"Corrupt page at index {page_index}: expected {PAGE_SIZE} bytes, got {len(page)}"
                )

            magic, page_type, _flags, _page_id, item_count, _free_offset, _free_bytes, _reserved = (
                HEADER_STRUCT.unpack_from(page, 0)
            )
            if magic != PAGE_MAGIC or page_type != SLOTTED_PAGE_TYPE:
                page_index += 1
                continue

            for item_index in range(item_count):
                slot_offset = HEADER_SIZE + item_index * SLOT_STRUCT.size
                if slot_offset + SLOT_STRUCT.size > PAGE_SIZE:
                    break

                item_offset, item_length = SLOT_STRUCT.unpack_from(page, slot_offset)
                if item_length < 8 or item_offset + item_length > PAGE_SIZE:
                    continue
                if (item_length - 8) % 4 != 0:
                    continue

                item_dims = (item_length - 8) // 4
                if item_dims <= 0:
                    continue

                if dims == 0:
                    dims = item_dims
                elif item_dims != dims:
                    continue

                vec_id = struct.unpack_from("<Q", page, item_offset)[0]
                values = struct.unpack_from(f"<{item_dims}f", page, item_offset + 8)
                vectors_by_id[int(vec_id)] = tuple(float(v) for v in values)

            page_index += 1

    if not vectors_by_id:
        raise ValueError("No vectors found in slotted-page embedding store")
    if dims <= 0:
        raise ValueError("Failed to infer vector dimensions from embedding store")

    return vectors_by_id, dims


def run_week4_benchmark(
    benchmark_bin: Path,
    db_path: Path,
    query_selector: str,
    k: int,
    csv_output: Path,
) -> str:
    """Run existing C++ week4 benchmark and return stdout text."""
    cmd = [str(benchmark_bin), str(db_path), query_selector, str(k), str(csv_output)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "week4_query_benchmark failed\n"
            f"  command: {' '.join(cmd)}\n"
            f"  return code: {completed.returncode}\n"
            f"  stdout:\n{completed.stdout}\n"
            f"  stderr:\n{completed.stderr}\n"
        )
    return completed.stdout


def read_week4_csv(csv_path: Path) -> dict[int, dict[str, float]]:
    """Read per-query rows emitted by week4_query_benchmark."""
    rows: dict[int, dict[str, float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            query_id = int(row["query_id"])
            rows[query_id] = {
                "rtree_us": float(row["rtree_us"]),
                "brute_us": float(row["brute_us"]),
                "recall": float(row["recall"]),
            }
    if not rows:
        raise ValueError(f"No benchmark rows found in CSV: {csv_path}")
    return rows


def create_sqlite_table(
    sqlite_path: Path,
    vectors_by_id: dict[int, tuple[float, ...]],
    dims: int,
) -> sqlite3.Connection:
    """Create and populate a SQLite baseline table for vector scans."""
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("PRAGMA journal_mode = OFF;")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -200000;")

    value_columns = [f"d{i}" for i in range(dims)]
    schema_cols = ", ".join(f"{col} REAL NOT NULL" for col in value_columns)
    conn.execute(f"CREATE TABLE embeddings (id INTEGER PRIMARY KEY, {schema_cols});")

    insert_columns = ", ".join(["id", *value_columns])
    placeholders = ", ".join("?" for _ in range(dims + 1))
    insert_sql = f"INSERT INTO embeddings ({insert_columns}) VALUES ({placeholders});"

    ordered_ids = sorted(vectors_by_id)
    with conn:
        conn.executemany(
            insert_sql,
            ((vec_id, *vectors_by_id[vec_id]) for vec_id in ordered_ids),
        )

    return conn


def build_sqlite_knn_query(dims: int) -> str:
    """Build SQL query that computes squared L2 distance and returns top-k IDs."""
    terms = [f"(d{i} - ?{i + 1}) * (d{i} - ?{i + 1})" for i in range(dims)]
    dist_expr = " + ".join(terms)
    return f"SELECT id FROM embeddings ORDER BY ({dist_expr}) ASC LIMIT ?{dims + 1};"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (p / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def choose_query_ids(
    week4_rows: dict[int, dict[str, float]],
    max_queries: int,
    sample_mode: str,
    seed: int,
) -> list[int]:
    ids = sorted(week4_rows)
    if max_queries <= 0 or max_queries >= len(ids):
        return ids

    if sample_mode == "head":
        return ids[:max_queries]

    rng = random.Random(seed)
    return sorted(rng.sample(ids, max_queries))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare custom R-tree query latency against baseline SQLite scans"
    )
    parser.add_argument("--db", default="sample.db", help="Path to slotted-page embedding DB")
    parser.add_argument(
        "--benchmark-bin",
        default="./build/week4_query_benchmark",
        help="Path to week4_query_benchmark executable",
    )
    parser.add_argument(
        "--query-selector",
        default="all:200",
        help="Forwarded selector for week4 benchmark (examples: 0, all, all:200)",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k neighbors")
    parser.add_argument(
        "--sqlite-repeats",
        type=int,
        default=3,
        help="Number of SQLite timings to average per query",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Optional cap on compared query rows (0 means no cap)",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["head", "random"],
        default="random",
        help="How to select rows when --max-queries is set",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--output-csv",
        default="sqlite_vs_rtree_metrics.csv",
        help="Output CSV path for merged comparison metrics",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Progress logging interval during SQLite timings",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary week4/sqlite files for inspection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_path = Path(args.db).expanduser().resolve()
    benchmark_bin = Path(args.benchmark_bin).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not db_path.exists():
        raise SystemExit(f"Embedding DB not found: {db_path}")
    if not benchmark_bin.exists():
        raise SystemExit(f"Benchmark executable not found: {benchmark_bin}")
    if args.k <= 0:
        raise SystemExit("k must be >= 1")
    if args.sqlite_repeats <= 0:
        raise SystemExit("sqlite-repeats must be >= 1")
    if args.progress_every <= 0:
        raise SystemExit("progress-every must be >= 1")

    tmp_ctx = tempfile.TemporaryDirectory(prefix="sqlite_rtree_compare_")
    temp_dir = Path(tmp_ctx.name)

    try:
        week4_csv = temp_dir / "week4_rows.csv"
        sqlite_db = temp_dir / "baseline.sqlite"

        print("[1/4] Running week4 benchmark...")
        run_week4_benchmark(
            benchmark_bin=benchmark_bin,
            db_path=db_path,
            query_selector=args.query_selector,
            k=args.k,
            csv_output=week4_csv,
        )

        week4_rows = read_week4_csv(week4_csv)
        query_ids = choose_query_ids(
            week4_rows=week4_rows,
            max_queries=args.max_queries,
            sample_mode=args.sample_mode,
            seed=args.seed,
        )

        print("[2/4] Reading vectors from slotted-page DB...")
        vectors_by_id, dims = read_vectors_from_slotted_db(db_path)

        missing_ids = [qid for qid in query_ids if qid not in vectors_by_id]
        if missing_ids:
            raise RuntimeError(
                f"Benchmark CSV has query IDs missing from embedding store: {missing_ids[:5]}"
            )

        print("[3/4] Building SQLite baseline table...")
        conn = create_sqlite_table(sqlite_db, vectors_by_id, dims)
        query_sql = build_sqlite_knn_query(dims)

        print(f"[4/4] Timing SQLite queries ({len(query_ids)} rows)...")
        merged_rows: list[dict[str, float | int]] = []

        for idx, qid in enumerate(query_ids, start=1):
            params = (*vectors_by_id[qid], args.k)
            trials_us = []
            for _ in range(args.sqlite_repeats):
                t0_ns = time.perf_counter_ns()
                conn.execute(query_sql, params).fetchall()
                elapsed_us = (time.perf_counter_ns() - t0_ns) / 1000.0
                trials_us.append(elapsed_us)

            sqlite_us = fmean(trials_us)
            week4_row = week4_rows[qid]
            rtree_us = week4_row["rtree_us"]
            ratio = sqlite_us / rtree_us if rtree_us > 0 else 0.0

            merged_rows.append(
                {
                    "query_id": qid,
                    "rtree_us": rtree_us,
                    "sqlite_us": sqlite_us,
                    "brute_us": week4_row["brute_us"],
                    "recall": week4_row["recall"],
                    "sqlite_over_rtree": ratio,
                }
            )

            if idx % args.progress_every == 0 or idx == len(query_ids):
                print(f"  - completed {idx}/{len(query_ids)}")

        conn.close()

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "query_id",
                    "rtree_us",
                    "sqlite_us",
                    "brute_us",
                    "recall",
                    "sqlite_over_rtree",
                ],
            )
            writer.writeheader()
            writer.writerows(merged_rows)

        rtree_us_values = [float(row["rtree_us"]) for row in merged_rows]
        sqlite_us_values = [float(row["sqlite_us"]) for row in merged_rows]
        brute_us_values = [float(row["brute_us"]) for row in merged_rows]
        recall_values = [float(row["recall"]) for row in merged_rows]

        avg_rtree = fmean(rtree_us_values)
        avg_sqlite = fmean(sqlite_us_values)
        avg_brute = fmean(brute_us_values)
        avg_recall = fmean(recall_values)
        speedup = avg_sqlite / avg_rtree if avg_rtree > 0 else 0.0

        print("\nComparison Summary")
        print(f"  vectors indexed: {len(vectors_by_id)}")
        print(f"  dims: {dims}")
        print(f"  query rows compared: {len(merged_rows)}")
        print(f"  average rtree_us: {avg_rtree:.3f}")
        print(f"  average sqlite_us: {avg_sqlite:.3f}")
        print(f"  average brute_us: {avg_brute:.3f}")
        print(f"  average recall: {avg_recall:.6f}")
        print(f"  rtree speedup vs sqlite (avg): {speedup:.3f}x")
        print(
            f"  sqlite p50/p95 us: {percentile(sqlite_us_values, 50):.3f} / {percentile(sqlite_us_values, 95):.3f}"
        )
        print(
            f"  rtree  p50/p95 us: {percentile(rtree_us_values, 50):.3f} / {percentile(rtree_us_values, 95):.3f}"
        )
        print(f"  merged CSV written: {output_csv}")

        if args.keep_temp:
            print(f"  temp files kept at: {temp_dir}")
    finally:
        if not args.keep_temp:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
