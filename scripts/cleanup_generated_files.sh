#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
apply=false
include_node_modules=false

usage() {
    cat <<'EOF'
Usage: bash scripts/cleanup_generated_files.sh [--dry-run] [--apply] [--include-node-modules]

Options:
  --dry-run               Show what would be removed (default)
  --apply                 Remove matching files/directories
  --include-node-modules  Also remove mern backend/frontend node_modules and frontend dist
  --help                  Show this help message
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            apply=false
            ;;
        --apply)
            apply=true
            ;;
        --include-node-modules)
            include_node_modules=true
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            usage >&2
            exit 1
            ;;
    esac
done

targets=(
    "api_debug.log"
    "d.txt"
    "err.log"
    "extract.log"
    "extract2.log"
    "hist.txt"
    "out.txt"
    "out2.log"
    "paths.log"
    "populate_60k.log"
    "nohup.out"
)

if [[ "$include_node_modules" == true ]]; then
    targets+=(
        "mern/backend/node_modules"
        "mern/frontend/node_modules"
        "mern/frontend/dist"
    )
fi

shopt -s nullglob
for temp_db in "$ROOT_DIR"/*.rtree_tmp.db "$ROOT_DIR"/*.sqlite_tmp.db; do
    rel_path="${temp_db#$ROOT_DIR/}"
    targets+=("$rel_path")
done
shopt -u nullglob

to_remove=()
for rel_path in "${targets[@]}"; do
    abs_path="$ROOT_DIR/$rel_path"
    if [[ -e "$abs_path" ]]; then
        to_remove+=("$rel_path")
    fi
done

if [[ ${#to_remove[@]} -eq 0 ]]; then
    echo "No generated artifacts found."
    exit 0
fi

if [[ "$apply" == false ]]; then
    echo "Dry run: the following paths would be removed:"
    for rel_path in "${to_remove[@]}"; do
        echo "  $rel_path"
    done
    exit 0
fi

echo "Removing generated artifacts:"
for rel_path in "${to_remove[@]}"; do
    abs_path="$ROOT_DIR/$rel_path"
    echo "  $rel_path"
    rm -rf "$abs_path"
done

echo "Cleanup complete."
