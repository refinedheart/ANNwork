#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_PATH="${ANN_DATA_PATH:-$ROOT_DIR/../../ANN/ann/anndata}"
OUT_DIR="$ROOT_DIR/results/report_pack"
OUT_CSV="$OUT_DIR/ann2_selected.csv"

mkdir -p "$OUT_DIR"

export ANN_DATA_PATH="$DATA_PATH"
export ANN_QUERY_LIMIT="${ANN_QUERY_LIMIT:-20}"
export ANN_IVF_NLIST="${ANN_IVF_NLIST:-128}"
export ANN_IVF_NPROBE="${ANN_IVF_NPROBE:-32}"
export ANN_HNSW_EF="${ANN_HNSW_EF:-64}"

cat > "$OUT_CSV" <<'EOF'
mode,threads,recall,latency_us,total_us
EOF

run_one() {
    local mode="$1"
    local threads="$2"
    local output
    output="$("$ROOT_DIR/main" "$mode" "$threads" 2>&1)"
    printf '%s\n' "$output"
    local recall latency total
    recall="$(printf '%s\n' "$output" | awk -F': ' '/average recall/ {print $2}' | tail -1)"
    latency="$(printf '%s\n' "$output" | awk -F': ' '/average latency/ {print $2}' | tail -1 | awk '{print $1}')"
    total="$(printf '%s\n' "$output" | awk -F': ' '/total time/ {print $2}' | tail -1 | awk '{print $1}')"
    printf '%s,%s,%s,%s,%s\n' "$mode" "$threads" "$recall" "$latency" "$total" >> "$OUT_CSV"
}

{
    run_one seq 1
    run_one pthread 8
    run_one openmp 8
    run_one pthread_part 8
    run_one openmp_part 8
    run_one ivf_neon 1
    run_one ivf_pthread 8
    run_one ivf_openmp 8
    run_one ivf_part_pthread 8
    run_one ivf_part_openmp 8
    run_one ivf_pq 1
    run_one ivf_pq_pthread 8
    run_one ivf_pq_openmp 8
    run_one hnsw 1
    run_one hnsw_pthread 8
    run_one hnsw_openmp 8
} | tee "$OUT_DIR/ann2_selected.log"

cat "$OUT_CSV"
