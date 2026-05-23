#!/bin/bash
# ANN 多线程完整性能测试脚本
# 用法: ./run_bench.sh

THREADS="1 2 4 8"
OUTPUT="bench_results.csv"

# 分组测试：每组有独立的索引构建
# Group A: PQ-SIMD 系列 (共用 PQ 索引)
PQ_MODES="seq pthread openmp pthread_lut openmp_lut pthread_part openmp_part"

# Group B: IVF 系列 (共用 IVF 索引)
IVF_MODES="ivf_neon ivf_pthread ivf_openmp ivf_part_pthread ivf_part_openmp"

# Group C: IVF-PQ 系列 (共用 IVF-PQ 索引)
IVFPQ_MODES="ivf_pq ivf_pq_pthread ivf_pq_openmp"

# Group D: HNSW 系列 (共用 HNSW 索引)
HNSW_MODES="hnsw hnsw_pthread hnsw_openmp"

echo "mode,threads,avg_recall,avg_latency_us,total_time_us" > $OUTPUT

run_test() {
    local mode=$1
    local nt=$2
    if { [ "$mode" = "seq" ] || [ "$mode" = "hnsw" ]; } && [ "$nt" -ne 1 ]; then
        return
    fi
    echo "=== [$mode] threads=$nt ===" >&2
    output=$(./main $mode $nt 2>&1)
    recall=$(echo "$output" | grep "average recall:" | awk '{print $3}')
    latency=$(echo "$output" | grep "average latency" | awk '{print $4}')
    total=$(echo "$output" | grep "total time" | awk '{print $4}')
    echo "$mode,$nt,$recall,$latency,$total" >> $OUTPUT
    echo "  recall=$recall latency=${latency}us total=${total}us" >&2
}

# Group A: PQ-SIMD
echo "=== Group A: PQ-SIMD ===" >&2
for mode in $PQ_MODES; do
    for nt in $THREADS; do
        run_test $mode $nt
    done
done

# Group B: IVF
echo "=== Group B: IVF ===" >&2
for mode in $IVF_MODES; do
    for nt in $THREADS; do
        run_test $mode $nt
    done
done

# Group C: IVF-PQ
echo "=== Group C: IVF-PQ ===" >&2
for mode in $IVFPQ_MODES; do
    for nt in $THREADS; do
        run_test $mode $nt
    done
done

# Group D: HNSW
echo "=== Group D: HNSW ===" >&2
for mode in $HNSW_MODES; do
    for nt in $THREADS; do
        run_test $mode $nt
    done
done

echo "=== All tests complete ===" >&2
echo "Results saved to $OUTPUT"
echo ""
echo "Summary:"
column -t -s',' $OUTPUT
