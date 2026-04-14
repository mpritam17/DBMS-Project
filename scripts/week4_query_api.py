#!/usr/bin/env python3
"""Minimal Week 4 HTTP query layer over the C++ benchmark/search executable.

Endpoints:
- GET /health
- GET /query?db=sample.db&id=0&k=10
"""

import argparse
import json
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def parse_benchmark_output(raw_output: str) -> dict:
    """Extract metrics and top-k rows from week4_query_benchmark output."""
    metrics = {}
    topk = []

    patterns = {
        "vectors_raw": r"vectors\(raw\):\s+(\d+)",
        "vectors_unique": r"vectors\(unique\):\s+(\d+)",
        "dims": r"dims:\s+(\d+)",
        "k": r"k:\s+(\d+)",
        "rtree_us": r"Average R-tree KNN latency:\s+([0-9.]+)\s+us",
        "brute_us": r"Average brute-force latency:\s+([0-9.]+)\s+us",
        "recall": r"Average recall@k:\s+([0-9.]+)",
        "disk_reads": r"Disk I/O counters \(StorageManager\): reads=(\d+)",
        "disk_writes": r"Disk I/O counters \(StorageManager\): reads=\d+, writes=(\d+)",
        "bpm_hit_rate_percent": r"Buffer Pool hit rate \(BPM\):\s+([0-9.]+)%",
        "bpm_hits": r"Buffer Pool hit rate \(BPM\):\s+[0-9.]+% \(hits=(\d+),",
        "bpm_fetches": r"Buffer Pool hit rate \(BPM\):\s+[0-9.]+% \(hits=\d+, fetches=(\d+),",
        "bpm_misses": r"Buffer Pool hit rate \(BPM\):\s+[0-9.]+% \(hits=\d+, fetches=\d+, misses=(\d+)\)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw_output)
        if match:
            value = match.group(1)
            metrics[key] = float(value) if "." in value else int(value)

    top_section = False
    row_section = False
    for line in raw_output.splitlines():
        if line.startswith("Top ") and "R-tree results" in line:
            top_section = True
            row_section = False
            continue
        if line.startswith("First ") and "per-query rows" in line:
            row_section = True
            top_section = False
            continue
        if top_section:
            stripped = line.strip()
            if not stripped:
                break
            top_match = re.match(r"\d+\.\s+([0-9.]+),\s+(\d+)", stripped)
            if top_match:
                topk.append(
                    {
                        "distance": float(top_match.group(1)),
                        "id": int(top_match.group(2)),
                    }
                )
        if row_section:
            stripped = line.strip()
            if not stripped:
                break
            row_match = re.match(r"\d+\.\s+(\d+),\s+([0-9.]+),\s+([0-9.]+),\s+([0-9.]+)", stripped)
            if row_match:
                topk.append(
                    {
                        "query_id": int(row_match.group(1)),
                        "rtree_us": float(row_match.group(2)),
                        "brute_us": float(row_match.group(3)),
                        "recall": float(row_match.group(4)),
                    }
                )

    return {"metrics": metrics, "results": topk, "raw": raw_output}


class Week4Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, benchmark_bin: Path, default_db: Path, **kwargs):
        self.benchmark_bin = benchmark_bin
        self.default_db = default_db
        super().__init__(*args, **kwargs)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"ok": True, "service": "week4-query-api"})
            return

        if parsed.path != "/query":
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return

        params = parse_qs(parsed.query)
        try:
            query_id = int(params.get("id", [None])[0])
            k = int(params.get("k", [10])[0])
            if query_id < 0:
                raise ValueError("id must be >= 0")
            if k <= 0:
                raise ValueError("k must be >= 1")
        except (TypeError, ValueError) as exc:
            self._send_json({"ok": False, "error": f"Invalid query params: {exc}"}, status=400)
            return

        db_param = params.get("db", [str(self.default_db)])[0]
        db_path = Path(db_param).expanduser().resolve()
        if not db_path.exists():
            self._send_json({"ok": False, "error": f"Database not found: {db_path}"}, status=400)
            return

        cmd = [str(self.benchmark_bin), str(db_path), str(query_id), str(k)]
        try:
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            self._send_json({"ok": False, "error": "Query timed out"}, status=504)
            return

        if completed.returncode != 0:
            self._send_json(
                {
                    "ok": False,
                    "error": "Benchmark command failed",
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                    "stdout": completed.stdout,
                },
                status=500,
            )
            return

        parsed_output = parse_benchmark_output(completed.stdout)
        self._send_json(
            {
                "ok": True,
                "query": {"id": query_id, "k": k, "db": str(db_path)},
                "data": parsed_output,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 4 query API over C++ benchmark executable")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--db", default="sample.db", help="Default embedding store database path")
    parser.add_argument("--benchmark-bin", default="./build/week4_query_benchmark")
    args = parser.parse_args()

    benchmark_bin = Path(args.benchmark_bin).expanduser().resolve()
    if not benchmark_bin.exists():
        raise SystemExit(f"benchmark executable not found: {benchmark_bin}")

    default_db = Path(args.db).expanduser().resolve()

    def handler(*h_args, **h_kwargs):
        return Week4Handler(*h_args, benchmark_bin=benchmark_bin, default_db=default_db, **h_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Week4 API listening on http://{args.host}:{args.port}")
    print("Endpoints: /health and /query?id=<id>&k=<k>&db=<path>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
