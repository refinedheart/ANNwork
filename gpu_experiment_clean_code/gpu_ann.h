#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <string>
#include <utility>
#include <vector>

constexpr size_t GPU_ANN_MAX_K = 32;

enum class GpuAnnFlatKernel {
    Naive,
    Tiled,
};

enum class GpuAnnIVFMode {
    WastefulBatch,
    Grouped,
};

struct GpuAnnConfig {
    size_t batch_size = 64;
    size_t k = 10;
    size_t nprobe = 32;
    size_t topk_threads = 256;
    GpuAnnFlatKernel flat_kernel = GpuAnnFlatKernel::Tiled;
    GpuAnnIVFMode ivf_mode = GpuAnnIVFMode::Grouped;
};

struct GpuAnnTiming {
    double h2d_query_ms = 0.0;
    double flat_mm_ms = 0.0;
    double centroid_ms = 0.0;
    double refine_ms = 0.0;
    double topk_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
    size_t batches = 0;
    size_t refine_groups = 0;
    size_t refine_pairs = 0;
    size_t skipped_pairs = 0;
};

struct GpuAnnTopK {
    size_t query_count = 0;
    size_t k = 0;
    std::vector<float> distances;
    std::vector<uint32_t> ids;
};

struct GpuAnnIndex {
    size_t base_count = 0;
    size_t dim = 0;
    size_t nlist = 0;
    size_t inverted_count = 0;
    float* d_base = nullptr;
    float* d_centroids = nullptr;
    uint32_t* d_offsets = nullptr;
    uint32_t* d_ids = nullptr;
    std::vector<uint32_t> h_offsets;
};

bool gpu_ann_has_device(std::string* message = nullptr);

bool gpu_ann_build_flat_index(GpuAnnIndex& index,
                              const float* base,
                              size_t base_count,
                              size_t dim,
                              std::string* error = nullptr);

bool gpu_ann_attach_ivf_index(GpuAnnIndex& index,
                              const float* centroids,
                              const uint32_t* offsets,
                              const uint32_t* ids,
                              size_t nlist,
                              size_t id_count,
                              std::string* error = nullptr);

void gpu_ann_free(GpuAnnIndex& index);

bool gpu_ann_flat_search_batch(const GpuAnnIndex& index,
                               const float* queries,
                               size_t query_count,
                               const GpuAnnConfig& config,
                               GpuAnnTopK& result,
                               GpuAnnTiming* timing = nullptr,
                               std::string* error = nullptr);

bool gpu_ann_ivf_search_batch(const GpuAnnIndex& index,
                              const float* queries,
                              size_t query_count,
                              const GpuAnnConfig& config,
                              GpuAnnTopK& result,
                              GpuAnnTiming* timing = nullptr,
                              std::string* error = nullptr);

std::priority_queue<std::pair<float, uint32_t>>
gpu_ann_queue_from_topk(const GpuAnnTopK& result, size_t query_id);
