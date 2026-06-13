#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$ROOT_DIR/results/gpu_report"
OUT_CSV="$OUT_DIR/gpu_ann.csv"

mkdir -p "$OUT_DIR"

bash "$ROOT_DIR/build_gpu.sh"

export ANN_DATA_PATH="${ANN_DATA_PATH:-/anndata/}"
export ANN_QUERY_LIMIT="${ANN_QUERY_LIMIT:-2000}"
export ANN_GPU_BATCH="${ANN_GPU_BATCH:-64}"
export ANN_IVF_NLIST="${ANN_IVF_NLIST:-128}"
export ANN_IVF_NPROBE="${ANN_IVF_NPROBE:-32}"

cat > "$OUT_CSV" <<'CSV'
mode,batch_size,queries,recall,latency_us,total_us,h2d_ms,flat_mm_ms,centroid_ms,refine_ms,topk_ms,d2h_ms,refine_pairs,skipped_pairs
CSV

run_one() {
    local mode="$1"
    local batch="$2"
    local output
    echo "=== $mode batch=$batch ===" >&2
    output="$("$ROOT_DIR/gpu_main" "$mode" "$batch" "$ANN_QUERY_LIMIT" 2>&1)"
    printf '%s\n' "$output" | tee "$OUT_DIR/${mode}_b${batch}.log"

    local queries recall latency total h2d flat centroid refine topk d2h pairs skipped
    queries="$(printf '%s\n' "$output" | awk -F': ' '/^queries:/ {print $2}' | tail -1)"
    recall="$(printf '%s\n' "$output" | awk -F': ' '/average recall/ {print $2}' | tail -1)"
    latency="$(printf '%s\n' "$output" | awk -F': ' '/average latency/ {print $2}' | tail -1)"
    total="$(printf '%s\n' "$output" | awk -F': ' '/total time/ {print $2}' | tail -1)"
    h2d="$(printf '%s\n' "$output" | awk -F': ' '/stage h2d query/ {print $2}' | tail -1)"
    flat="$(printf '%s\n' "$output" | awk -F': ' '/stage flat mm/ {print $2}' | tail -1)"
    centroid="$(printf '%s\n' "$output" | awk -F': ' '/stage centroid/ {print $2}' | tail -1)"
    refine="$(printf '%s\n' "$output" | awk -F': ' '/stage refine/ {print $2}' | tail -1)"
    topk="$(printf '%s\n' "$output" | awk -F': ' '/stage topk/ {print $2}' | tail -1)"
    d2h="$(printf '%s\n' "$output" | awk -F': ' '/stage d2h/ {print $2}' | tail -1)"
    pairs="$(printf '%s\n' "$output" | awk -F': ' '/ivf refine pairs/ {print $2}' | tail -1)"
    skipped="$(printf '%s\n' "$output" | awk -F': ' '/ivf skipped pairs/ {print $2}' | tail -1)"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$mode" "$batch" "$queries" "$recall" "$latency" "$total" \
        "$h2d" "$flat" "$centroid" "$refine" "$topk" "$d2h" \
        "$pairs" "$skipped" >> "$OUT_CSV"
}

for batch in 16 32 64; do
    run_one flat_gpu_naive "$batch"
    run_one flat_gpu_tiled "$batch"
done

for batch in 16 32 64; do
    run_one ivf_gpu "$batch"
    run_one ivf_gpu_grouped "$batch"
done

echo "saved $OUT_CSV"
