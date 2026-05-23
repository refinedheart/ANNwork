#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_PATH="${ANN_DATA_PATH:-$ROOT_DIR/../../ANN/ann/anndata}"
OUT_DIR="$ROOT_DIR/results/report_pack"
OUT_CSV="$OUT_DIR/batch_scale.csv"

mkdir -p "$OUT_DIR"

export ANN_DATA_PATH="$DATA_PATH"
export ANN_IVF_NLIST="${ANN_IVF_NLIST:-128}"
export ANN_IVF_NPROBE="${ANN_IVF_NPROBE:-32}"
export ANN_HNSW_EF="${ANN_HNSW_EF:-64}"

QUERY_LIMITS="${QUERY_LIMITS:-5 10 20 50 100}"
MODES="${MODES:-seq pthread openmp hnsw hnsw_pthread hnsw_openmp}"

cat > "$OUT_CSV" <<'EOF'
mode,threads,query_limit,recall,latency_us,total_us
EOF

run_one() {
    local mode="$1"
    local threads="$2"
    local query_limit="$3"
    local output
    output="$(ANN_QUERY_LIMIT="$query_limit" "$ROOT_DIR/main" "$mode" "$threads" 2>&1)"
    printf '%s\n' "$output"
    local recall latency total
    recall="$(printf '%s\n' "$output" | awk -F': ' '/average recall/ {print $2}' | tail -1)"
    latency="$(printf '%s\n' "$output" | awk -F': ' '/average latency/ {print $2}' | tail -1 | awk '{print $1}')"
    total="$(printf '%s\n' "$output" | awk -F': ' '/total time/ {print $2}' | tail -1 | awk '{print $1}')"
    printf '%s,%s,%s,%s,%s,%s\n' "$mode" "$threads" "$query_limit" "$recall" "$latency" "$total" >> "$OUT_CSV"
}

{
    for q in $QUERY_LIMITS; do
        for entry in $MODES; do
            case "$entry" in
                seq|hnsw)
                    run_one "$entry" 1 "$q"
                    ;;
                *)
                    run_one "$entry" 8 "$q"
                    ;;
            esac
        done
    done
} | tee "$OUT_DIR/batch_scale.log"

cat "$OUT_CSV"
