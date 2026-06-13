#!/bin/sh
#PBS -N ann_gpu
#PBS -e ann_gpu.e
#PBS -o ann_gpu.o

cd "$PBS_O_WORKDIR" || exit 1

bash ./build_gpu.sh || exit 1

MODE="${ANN_GPU_MODE:-ivf_gpu_grouped}"
BATCH="${ANN_GPU_BATCH:-64}"
QUERIES="${ANN_QUERY_LIMIT:-2000}"

./gpu_main "$MODE" "$BATCH" "$QUERIES"
