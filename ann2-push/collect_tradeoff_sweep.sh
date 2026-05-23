#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_PATH="${ANN_DATA_PATH:-$ROOT_DIR/../../ANN/ann/anndata}"
OUT_DIR="$ROOT_DIR/results/report_pack"
OUT_CSV="$OUT_DIR/tradeoff_sweep.csv"

mkdir -p "$OUT_DIR"

export ANN_DATA_PATH="$DATA_PATH"
export ANN_QUERY_LIMIT="${ANN_QUERY_LIMIT:-20}"
export ANN_IVF_NLIST="${ANN_IVF_NLIST:-128}"

IVF_NPROBES="${IVF_NPROBES:-8 16 32 64}"
HNSW_EFS="${HNSW_EFS:-16 32 64 128}"
PQ_RERANKS="${PQ_RERANKS:-100 200 550 1000}"

cat > "$OUT_CSV" <<'EOF'
family,mode,param_name,param_value,threads,recall,latency_us,total_us
EOF

run_one() {
    local family="$1"
    local mode="$2"
    local param_name="$3"
    local param_value="$4"
    local threads="$5"
    shift 5
    local output
    output="$("$@" 2>&1)"
    printf '%s\n' "$output"
    local recall latency total
    recall="$(printf '%s\n' "$output" | awk -F': ' '/average recall/ {print $2}' | tail -1)"
    latency="$(printf '%s\n' "$output" | awk -F': ' '/average latency/ {print $2}' | tail -1 | awk '{print $1}')"
    total="$(printf '%s\n' "$output" | awk -F': ' '/total time/ {print $2}' | tail -1 | awk '{print $1}')"
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$family" "$mode" "$param_name" "$param_value" "$threads" "$recall" "$latency" "$total" >> "$OUT_CSV"
}

{
    for p in $PQ_RERANKS; do
        run_one flat_pq seq ANN_PQ_RERANK "$p" 1 env ANN_PQ_RERANK="$p" "$ROOT_DIR/main" seq 1
    done

    for nprobe in $IVF_NPROBES; do
        run_one ivf ivf_neon ANN_IVF_NPROBE "$nprobe" 1 env ANN_IVF_NPROBE="$nprobe" "$ROOT_DIR/main" ivf_neon 1
        run_one ivf_pq ivf_pq ANN_IVF_NPROBE "$nprobe" 1 env ANN_IVF_NPROBE="$nprobe" "$ROOT_DIR/main" ivf_pq 1
    done

    for ef in $HNSW_EFS; do
        run_one hnsw hnsw ANN_HNSW_EF "$ef" 1 env ANN_HNSW_EF="$ef" "$ROOT_DIR/main" hnsw 1
    done
} | tee "$OUT_DIR/tradeoff_sweep.log"

cat "$OUT_CSV"
