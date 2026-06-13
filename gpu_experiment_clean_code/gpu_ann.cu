#include "gpu_ann.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace {

constexpr float GPU_ANN_INF_F = 3.4028234663852886e+38F;

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
    return false;
}

bool check_cuda(cudaError_t status, std::string* error, const char* context) {
    if (status == cudaSuccess) {
        return true;
    }
    std::ostringstream oss;
    oss << context << " failed: " << cudaGetErrorString(status);
    return set_error(error, oss.str());
}

bool validate_common(const GpuAnnIndex& index,
                     size_t query_count,
                     const GpuAnnConfig& config,
                     std::string* error) {
    if (!index.d_base || index.base_count == 0 || index.dim == 0) {
        return set_error(error, "GPU ANN index is empty");
    }
    if (query_count == 0) {
        return set_error(error, "query_count is zero");
    }
    if (config.k == 0 || config.k > GPU_ANN_MAX_K) {
        std::ostringstream oss;
        oss << "k must be in [1, " << GPU_ANN_MAX_K << "]";
        return set_error(error, oss.str());
    }
    if (config.batch_size == 0) {
        return set_error(error, "batch_size must be positive");
    }
    if (config.topk_threads == 0 || config.topk_threads > 1024) {
        return set_error(error, "topk_threads must be in [1, 1024]");
    }
    return true;
}

__device__ void insert_candidate(float* best_dist,
                                 uint32_t* best_ids,
                                 int k,
                                 float cand_dist,
                                 uint32_t cand_id) {
    if (!(cand_dist < GPU_ANN_INF_F) || cand_id == 0xffffffffu) {
        return;
    }

    int worst_pos = 0;
    float worst = best_dist[0];
    for (int i = 1; i < k; ++i) {
        if (best_dist[i] > worst) {
            worst = best_dist[i];
            worst_pos = i;
        }
    }

    if (cand_dist < worst ||
        (cand_dist == worst && cand_id < best_ids[worst_pos])) {
        best_dist[worst_pos] = cand_dist;
        best_ids[worst_pos] = cand_id;
    }
}

__global__ void flat_mm_naive_kernel(const float* __restrict__ base,
                                     const float* __restrict__ queries,
                                     float* __restrict__ scores,
                                     size_t base_count,
                                     size_t query_count,
                                     size_t dim) {
    const size_t base_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t query_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (base_id >= base_count || query_id >= query_count) {
        return;
    }

    float acc = 0.0f;
    const float* b = base + base_id * dim;
    const float* q = queries + query_id * dim;
    for (size_t d = 0; d < dim; ++d) {
        acc += b[d] * q[d];
    }
    scores[query_id * base_count + base_id] = 1.0f - acc;
}

__global__ void flat_mm_tiled_kernel(const float* __restrict__ base,
                                     const float* __restrict__ queries,
                                     float* __restrict__ scores,
                                     size_t base_count,
                                     size_t query_count,
                                     size_t dim) {
    constexpr int TILE = 16;
    __shared__ float q_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t base_id = blockIdx.x * TILE + tx;
    const size_t query_id = blockIdx.y * TILE + ty;

    float acc = 0.0f;
    for (size_t tile = 0; tile < dim; tile += TILE) {
        const size_t q_dim = tile + tx;
        const size_t b_dim = tile + tx;
        const size_t b_load_id = blockIdx.x * TILE + ty;

        q_tile[ty][tx] = (query_id < query_count && q_dim < dim)
                             ? queries[query_id * dim + q_dim]
                             : 0.0f;
        b_tile[ty][tx] = (b_load_id < base_count && b_dim < dim)
                             ? base[b_load_id * dim + b_dim]
                             : 0.0f;
        __syncthreads();

        for (int d = 0; d < TILE; ++d) {
            acc += q_tile[ty][d] * b_tile[tx][d];
        }
        __syncthreads();
    }

    if (base_id < base_count && query_id < query_count) {
        scores[query_id * base_count + base_id] = 1.0f - acc;
    }
}

__global__ void topk_from_scores_kernel(const float* __restrict__ scores,
                                        size_t row_stride,
                                        size_t item_count,
                                        int k,
                                        float* __restrict__ out_dist,
                                        uint32_t* __restrict__ out_ids,
                                        const uint32_t* __restrict__ id_map) {
    const size_t query_id = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ unsigned char shared_raw[];
    float* shared_dist = reinterpret_cast<float*>(shared_raw);
    uint32_t* shared_ids =
        reinterpret_cast<uint32_t*>(shared_dist + blockDim.x * k);

    float local_dist[GPU_ANN_MAX_K];
    uint32_t local_ids[GPU_ANN_MAX_K];
    for (int i = 0; i < k; ++i) {
        local_dist[i] = GPU_ANN_INF_F;
        local_ids[i] = 0xffffffffu;
    }

    const float* row = scores + query_id * row_stride;
    for (size_t i = tid; i < item_count; i += blockDim.x) {
        const float dist = row[i];
        const uint32_t id = id_map ? id_map[i] : static_cast<uint32_t>(i);
        insert_candidate(local_dist, local_ids, k, dist, id);
    }

    for (int i = 0; i < k; ++i) {
        shared_dist[tid * k + i] = local_dist[i];
        shared_ids[tid * k + i] = local_ids[i];
    }
    __syncthreads();

    if (tid == 0) {
        float merged_dist[GPU_ANN_MAX_K];
        uint32_t merged_ids[GPU_ANN_MAX_K];
        for (int i = 0; i < k; ++i) {
            merged_dist[i] = GPU_ANN_INF_F;
            merged_ids[i] = 0xffffffffu;
        }

        for (int t = 0; t < blockDim.x; ++t) {
            for (int i = 0; i < k; ++i) {
                insert_candidate(merged_dist,
                                 merged_ids,
                                 k,
                                 shared_dist[t * k + i],
                                 shared_ids[t * k + i]);
            }
        }

        for (int out = 0; out < k; ++out) {
            int best_pos = -1;
            float best = GPU_ANN_INF_F;
            uint32_t best_id = 0xffffffffu;
            for (int i = 0; i < k; ++i) {
                if (merged_dist[i] < best ||
                    (merged_dist[i] == best && merged_ids[i] < best_id)) {
                    best = merged_dist[i];
                    best_id = merged_ids[i];
                    best_pos = i;
                }
            }
            out_dist[query_id * k + out] = best;
            out_ids[query_id * k + out] = best_id;
            if (best_pos >= 0) {
                merged_dist[best_pos] = GPU_ANN_INF_F;
                merged_ids[best_pos] = 0xffffffffu;
            }
        }
    }
}

__device__ bool cluster_is_selected(const uint32_t* selected_clusters,
                                    size_t nprobe,
                                    size_t query_id,
                                    uint32_t cluster_id) {
    const uint32_t* row = selected_clusters + query_id * nprobe;
    for (size_t i = 0; i < nprobe; ++i) {
        if (row[i] == cluster_id) {
            return true;
        }
    }
    return false;
}

__global__ void ivf_refine_wasteful_kernel(const float* __restrict__ base,
                                           const float* __restrict__ queries,
                                           const uint32_t* __restrict__ list_ids,
                                           size_t list_len,
                                           size_t query_count,
                                           size_t dim,
                                           uint32_t cluster_id,
                                           const uint32_t* __restrict__ selected_clusters,
                                           size_t nprobe,
                                           float* __restrict__ scores) {
    const size_t list_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t query_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (list_pos >= list_len || query_id >= query_count) {
        return;
    }

    const uint32_t base_id = list_ids[list_pos];
    const float* b = base + static_cast<size_t>(base_id) * dim;
    const float* q = queries + query_id * dim;
    float acc = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        acc += b[d] * q[d];
    }

    float dist = 1.0f - acc;
    if (!cluster_is_selected(selected_clusters, nprobe, query_id, cluster_id)) {
            dist = GPU_ANN_INF_F;
    }
    scores[query_id * list_len + list_pos] = dist;
}

__global__ void ivf_refine_grouped_kernel(const float* __restrict__ base,
                                          const float* __restrict__ queries,
                                          const uint32_t* __restrict__ query_ids,
                                          const uint32_t* __restrict__ list_ids,
                                          size_t list_len,
                                          size_t group_query_count,
                                          size_t dim,
                                          float* __restrict__ scores) {
    const size_t list_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t group_query_pos = blockIdx.y * blockDim.y + threadIdx.y;
    if (list_pos >= list_len || group_query_pos >= group_query_count) {
        return;
    }

    const uint32_t query_id = query_ids[group_query_pos];
    const uint32_t base_id = list_ids[list_pos];
    const float* b = base + static_cast<size_t>(base_id) * dim;
    const float* q = queries + static_cast<size_t>(query_id) * dim;
    float acc = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        acc += b[d] * q[d];
    }
    scores[group_query_pos * list_len + list_pos] = 1.0f - acc;
}

void launch_flat_kernel(const GpuAnnIndex& index,
                        const float* d_queries,
                        size_t batch_count,
                        const GpuAnnConfig& config,
                        float* d_scores) {
    dim3 block(16, 16);
    dim3 grid((index.base_count + block.x - 1) / block.x,
              (batch_count + block.y - 1) / block.y);
    if (config.flat_kernel == GpuAnnFlatKernel::Naive) {
        flat_mm_naive_kernel<<<grid, block>>>(index.d_base,
                                              d_queries,
                                              d_scores,
                                              index.base_count,
                                              batch_count,
                                              index.dim);
    } else {
        flat_mm_tiled_kernel<<<grid, block>>>(index.d_base,
                                              d_queries,
                                              d_scores,
                                              index.base_count,
                                              batch_count,
                                              index.dim);
    }
}

void launch_centroid_kernel(const GpuAnnIndex& index,
                            const float* d_queries,
                            size_t batch_count,
                            const GpuAnnConfig& config,
                            float* d_scores) {
    dim3 block(16, 16);
    dim3 grid((index.nlist + block.x - 1) / block.x,
              (batch_count + block.y - 1) / block.y);
    if (config.flat_kernel == GpuAnnFlatKernel::Naive) {
        flat_mm_naive_kernel<<<grid, block>>>(index.d_centroids,
                                              d_queries,
                                              d_scores,
                                              index.nlist,
                                              batch_count,
                                              index.dim);
    } else {
        flat_mm_tiled_kernel<<<grid, block>>>(index.d_centroids,
                                              d_queries,
                                              d_scores,
                                              index.nlist,
                                              batch_count,
                                              index.dim);
    }
}

void launch_topk(const float* d_scores,
                 size_t row_stride,
                 size_t item_count,
                 size_t query_count,
                 const uint32_t* d_id_map,
                 const GpuAnnConfig& config,
                 float* d_out_dist,
                 uint32_t* d_out_ids) {
    const int threads = static_cast<int>(config.topk_threads);
    const int k = static_cast<int>(config.k);
    const size_t shared_bytes =
        static_cast<size_t>(threads) * static_cast<size_t>(k) *
        (sizeof(float) + sizeof(uint32_t));
    topk_from_scores_kernel<<<static_cast<unsigned int>(query_count),
                              threads,
                              shared_bytes>>>(d_scores,
                                              row_stride,
                                              item_count,
                                              k,
                                              d_out_dist,
                                              d_out_ids,
                                              d_id_map);
}

void push_host_candidate(std::priority_queue<std::pair<float, uint32_t>>& heap,
                         float dist,
                         uint32_t id,
                         size_t k) {
    if (!std::isfinite(dist) || id == std::numeric_limits<uint32_t>::max()) {
        return;
    }
    if (heap.size() < k) {
        heap.emplace(dist, id);
    } else if (dist < heap.top().first ||
               (dist == heap.top().first && id < heap.top().second)) {
        heap.pop();
        heap.emplace(dist, id);
    }
}

void finalize_heaps(std::vector<std::priority_queue<std::pair<float, uint32_t>>>& heaps,
                    size_t k,
                    GpuAnnTopK& result) {
    result.query_count = heaps.size();
    result.k = k;
    result.distances.assign(result.query_count * k,
                            std::numeric_limits<float>::infinity());
    result.ids.assign(result.query_count * k,
                      std::numeric_limits<uint32_t>::max());

    for (size_t q = 0; q < heaps.size(); ++q) {
        std::vector<std::pair<float, uint32_t>> ordered;
        ordered.reserve(heaps[q].size());
        while (!heaps[q].empty()) {
            ordered.push_back(heaps[q].top());
            heaps[q].pop();
        }
        std::sort(ordered.begin(),
                  ordered.end(),
                  [](const auto& lhs, const auto& rhs) {
                      if (lhs.first != rhs.first) {
                          return lhs.first < rhs.first;
                      }
                      return lhs.second < rhs.second;
                  });
        const size_t limit = std::min(k, ordered.size());
        for (size_t i = 0; i < limit; ++i) {
            result.distances[q * k + i] = ordered[i].first;
            result.ids[q * k + i] = ordered[i].second;
        }
    }
}

}

bool gpu_ann_has_device(std::string* message) {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        if (message) {
            *message = cudaGetErrorString(status);
        }
        return false;
    }
    if (count <= 0) {
        if (message) {
            *message = "no CUDA device found";
        }
        return false;
    }
    if (message) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        std::ostringstream oss;
        oss << "CUDA device 0: " << prop.name
            << ", SMs=" << prop.multiProcessorCount
            << ", global_mem_mb=" << (prop.totalGlobalMem / (1024 * 1024));
        *message = oss.str();
    }
    return true;
}

bool gpu_ann_build_flat_index(GpuAnnIndex& index,
                              const float* base,
                              size_t base_count,
                              size_t dim,
                              std::string* error) {
    gpu_ann_free(index);
    if (!base || base_count == 0 || dim == 0) {
        return set_error(error, "invalid base matrix");
    }

    index.base_count = base_count;
    index.dim = dim;
    const size_t bytes = base_count * dim * sizeof(float);
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&index.d_base), bytes),
                    error,
                    "cudaMalloc d_base")) {
        gpu_ann_free(index);
        return false;
    }
    if (!check_cuda(cudaMemcpy(index.d_base, base, bytes, cudaMemcpyHostToDevice),
                    error,
                    "cudaMemcpy base H2D")) {
        gpu_ann_free(index);
        return false;
    }
    return true;
}

bool gpu_ann_attach_ivf_index(GpuAnnIndex& index,
                              const float* centroids,
                              const uint32_t* offsets,
                              const uint32_t* ids,
                              size_t nlist,
                              size_t id_count,
                              std::string* error) {
    if (!index.d_base || index.dim == 0) {
        return set_error(error, "build the flat GPU index before attaching IVF");
    }
    if (!centroids || !offsets || !ids || nlist == 0) {
        return set_error(error, "invalid IVF index arrays");
    }

    if (index.d_centroids) {
        cudaFree(index.d_centroids);
        index.d_centroids = nullptr;
    }
    if (index.d_offsets) {
        cudaFree(index.d_offsets);
        index.d_offsets = nullptr;
    }
    if (index.d_ids) {
        cudaFree(index.d_ids);
        index.d_ids = nullptr;
    }

    index.nlist = nlist;
    index.inverted_count = id_count;
    index.h_offsets.assign(offsets, offsets + nlist + 1);

    const size_t centroid_bytes = nlist * index.dim * sizeof(float);
    const size_t offset_bytes = (nlist + 1) * sizeof(uint32_t);
    const size_t id_bytes = id_count * sizeof(uint32_t);

    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&index.d_centroids),
                               centroid_bytes),
                    error,
                    "cudaMalloc d_centroids") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&index.d_offsets),
                               offset_bytes),
                    error,
                    "cudaMalloc d_offsets") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&index.d_ids), id_bytes),
                    error,
                    "cudaMalloc d_ids")) {
        return false;
    }

    if (!check_cuda(cudaMemcpy(index.d_centroids,
                               centroids,
                               centroid_bytes,
                               cudaMemcpyHostToDevice),
                    error,
                    "cudaMemcpy centroids H2D") ||
        !check_cuda(cudaMemcpy(index.d_offsets,
                               offsets,
                               offset_bytes,
                               cudaMemcpyHostToDevice),
                    error,
                    "cudaMemcpy offsets H2D") ||
        !check_cuda(cudaMemcpy(index.d_ids,
                               ids,
                               id_bytes,
                               cudaMemcpyHostToDevice),
                    error,
                    "cudaMemcpy ids H2D")) {
        return false;
    }
    return true;
}

void gpu_ann_free(GpuAnnIndex& index) {
    if (index.d_base) {
        cudaFree(index.d_base);
    }
    if (index.d_centroids) {
        cudaFree(index.d_centroids);
    }
    if (index.d_offsets) {
        cudaFree(index.d_offsets);
    }
    if (index.d_ids) {
        cudaFree(index.d_ids);
    }
    index = GpuAnnIndex{};
}

bool gpu_ann_flat_search_batch(const GpuAnnIndex& index,
                               const float* queries,
                               size_t query_count,
                               const GpuAnnConfig& config,
                               GpuAnnTopK& result,
                               GpuAnnTiming* timing,
                               std::string* error) {
    if (!queries) {
        return set_error(error, "queries pointer is null");
    }
    if (!validate_common(index, query_count, config, error)) {
        return false;
    }

    GpuAnnTiming local_timing{};
    auto total_start = Clock::now();
    const size_t max_batch = std::min(config.batch_size, query_count);
    result.query_count = query_count;
    result.k = config.k;
    result.distances.assign(query_count * config.k,
                            std::numeric_limits<float>::infinity());
    result.ids.assign(query_count * config.k,
                      std::numeric_limits<uint32_t>::max());

    float* d_queries = nullptr;
    float* d_scores = nullptr;
    float* d_out_dist = nullptr;
    uint32_t* d_out_ids = nullptr;

    const size_t query_bytes = max_batch * index.dim * sizeof(float);
    const size_t score_bytes = max_batch * index.base_count * sizeof(float);
    const size_t out_dist_bytes = max_batch * config.k * sizeof(float);
    const size_t out_id_bytes = max_batch * config.k * sizeof(uint32_t);

    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_queries), query_bytes),
                    error,
                    "cudaMalloc d_queries") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_scores), score_bytes),
                    error,
                    "cudaMalloc d_scores") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out_dist), out_dist_bytes),
                    error,
                    "cudaMalloc d_out_dist") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_out_ids), out_id_bytes),
                    error,
                    "cudaMalloc d_out_ids")) {
        cudaFree(d_queries);
        cudaFree(d_scores);
        cudaFree(d_out_dist);
        cudaFree(d_out_ids);
        return false;
    }

    for (size_t offset = 0; offset < query_count; offset += config.batch_size) {
        const size_t batch_count = std::min(config.batch_size, query_count - offset);
        ++local_timing.batches;

        auto stage = Clock::now();
        if (!check_cuda(cudaMemcpy(d_queries,
                                   queries + offset * index.dim,
                                   batch_count * index.dim * sizeof(float),
                                   cudaMemcpyHostToDevice),
                        error,
                        "cudaMemcpy query H2D")) {
            cudaFree(d_queries);
            cudaFree(d_scores);
            cudaFree(d_out_dist);
            cudaFree(d_out_ids);
            return false;
        }
        local_timing.h2d_query_ms += elapsed_ms(stage);

        stage = Clock::now();
        launch_flat_kernel(index, d_queries, batch_count, config, d_scores);
        if (!check_cuda(cudaGetLastError(), error, "flat matrix kernel launch") ||
            !check_cuda(cudaDeviceSynchronize(), error, "flat matrix kernel sync")) {
            cudaFree(d_queries);
            cudaFree(d_scores);
            cudaFree(d_out_dist);
            cudaFree(d_out_ids);
            return false;
        }
        local_timing.flat_mm_ms += elapsed_ms(stage);

        stage = Clock::now();
        launch_topk(d_scores,
                    index.base_count,
                    index.base_count,
                    batch_count,
                    nullptr,
                    config,
                    d_out_dist,
                    d_out_ids);
        if (!check_cuda(cudaGetLastError(), error, "topk kernel launch") ||
            !check_cuda(cudaDeviceSynchronize(), error, "topk kernel sync")) {
            cudaFree(d_queries);
            cudaFree(d_scores);
            cudaFree(d_out_dist);
            cudaFree(d_out_ids);
            return false;
        }
        local_timing.topk_ms += elapsed_ms(stage);

        stage = Clock::now();
        if (!check_cuda(cudaMemcpy(result.distances.data() + offset * config.k,
                                   d_out_dist,
                                   batch_count * config.k * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                        error,
                        "cudaMemcpy flat distances D2H") ||
            !check_cuda(cudaMemcpy(result.ids.data() + offset * config.k,
                                   d_out_ids,
                                   batch_count * config.k * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost),
                        error,
                        "cudaMemcpy flat ids D2H")) {
            cudaFree(d_queries);
            cudaFree(d_scores);
            cudaFree(d_out_dist);
            cudaFree(d_out_ids);
            return false;
        }
        local_timing.d2h_ms += elapsed_ms(stage);
    }

    cudaFree(d_queries);
    cudaFree(d_scores);
    cudaFree(d_out_dist);
    cudaFree(d_out_ids);

    local_timing.total_ms = elapsed_ms(total_start);
    if (timing) {
        *timing = local_timing;
    }
    return true;
}

bool gpu_ann_ivf_search_batch(const GpuAnnIndex& index,
                              const float* queries,
                              size_t query_count,
                              const GpuAnnConfig& config,
                              GpuAnnTopK& result,
                              GpuAnnTiming* timing,
                              std::string* error) {
    if (!queries) {
        return set_error(error, "queries pointer is null");
    }
    if (!validate_common(index, query_count, config, error)) {
        return false;
    }
    if (!index.d_centroids || !index.d_offsets || !index.d_ids ||
        index.nlist == 0 || index.h_offsets.size() != index.nlist + 1) {
        return set_error(error, "IVF index is not attached");
    }
    if (config.nprobe == 0 || config.nprobe > index.nlist ||
        config.nprobe > GPU_ANN_MAX_K) {
        std::ostringstream oss;
        oss << "nprobe must be in [1, min(nlist, " << GPU_ANN_MAX_K << ")]";
        return set_error(error, oss.str());
    }

    GpuAnnTiming local_timing{};
    auto total_start = Clock::now();
    const size_t max_batch = std::min(config.batch_size, query_count);
    size_t max_list_len = 1;
    for (size_t c = 0; c < index.nlist; ++c) {
        max_list_len = std::max<size_t>(
            max_list_len,
            index.h_offsets[c + 1] - index.h_offsets[c]);
    }

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> heaps(query_count);
    float* d_queries = nullptr;
    float* d_centroid_scores = nullptr;
    float* d_selected_dist = nullptr;
    uint32_t* d_selected_ids = nullptr;
    float* d_refine_scores = nullptr;
    uint32_t* d_group_query_ids = nullptr;
    float* d_group_dist = nullptr;
    uint32_t* d_group_ids = nullptr;

    const size_t query_bytes = max_batch * index.dim * sizeof(float);
    const size_t centroid_score_bytes = max_batch * index.nlist * sizeof(float);
    const size_t selected_dist_bytes = max_batch * config.nprobe * sizeof(float);
    const size_t selected_id_bytes = max_batch * config.nprobe * sizeof(uint32_t);
    const size_t refine_score_bytes = max_batch * max_list_len * sizeof(float);
    const size_t group_query_bytes = max_batch * sizeof(uint32_t);
    const size_t group_dist_bytes = max_batch * config.k * sizeof(float);
    const size_t group_id_bytes = max_batch * config.k * sizeof(uint32_t);

    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_queries), query_bytes),
                    error,
                    "cudaMalloc ivf d_queries") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_centroid_scores),
                               centroid_score_bytes),
                    error,
                    "cudaMalloc d_centroid_scores") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_selected_dist),
                               selected_dist_bytes),
                    error,
                    "cudaMalloc d_selected_dist") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_selected_ids),
                               selected_id_bytes),
                    error,
                    "cudaMalloc d_selected_ids") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_refine_scores),
                               refine_score_bytes),
                    error,
                    "cudaMalloc d_refine_scores") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_group_query_ids),
                               group_query_bytes),
                    error,
                    "cudaMalloc d_group_query_ids") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_group_dist),
                               group_dist_bytes),
                    error,
                    "cudaMalloc d_group_dist") ||
        !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_group_ids),
                               group_id_bytes),
                    error,
                    "cudaMalloc d_group_ids")) {
        cudaFree(d_queries);
        cudaFree(d_centroid_scores);
        cudaFree(d_selected_dist);
        cudaFree(d_selected_ids);
        cudaFree(d_refine_scores);
        cudaFree(d_group_query_ids);
        cudaFree(d_group_dist);
        cudaFree(d_group_ids);
        return false;
    }

    std::vector<uint32_t> h_selected(max_batch * config.nprobe);
    std::vector<float> h_group_dist(max_batch * config.k);
    std::vector<uint32_t> h_group_ids(max_batch * config.k);

    for (size_t offset = 0; offset < query_count; offset += config.batch_size) {
        const size_t batch_count = std::min(config.batch_size, query_count - offset);
        ++local_timing.batches;

        auto stage = Clock::now();
        if (!check_cuda(cudaMemcpy(d_queries,
                                   queries + offset * index.dim,
                                   batch_count * index.dim * sizeof(float),
                                   cudaMemcpyHostToDevice),
                        error,
                        "cudaMemcpy ivf query H2D")) {
            return false;
        }
        local_timing.h2d_query_ms += elapsed_ms(stage);

        stage = Clock::now();
        launch_centroid_kernel(index, d_queries, batch_count, config, d_centroid_scores);
        if (!check_cuda(cudaGetLastError(), error, "centroid kernel launch") ||
            !check_cuda(cudaDeviceSynchronize(), error, "centroid kernel sync")) {
            return false;
        }
        local_timing.centroid_ms += elapsed_ms(stage);

        stage = Clock::now();
        GpuAnnConfig centroid_topk_config = config;
        centroid_topk_config.k = config.nprobe;
        launch_topk(d_centroid_scores,
                    index.nlist,
                    index.nlist,
                    batch_count,
                    nullptr,
                    centroid_topk_config,
                    d_selected_dist,
                    d_selected_ids);
        if (!check_cuda(cudaGetLastError(), error, "centroid topk launch") ||
            !check_cuda(cudaDeviceSynchronize(), error, "centroid topk sync")) {
            return false;
        }
        local_timing.topk_ms += elapsed_ms(stage);

        stage = Clock::now();
        if (!check_cuda(cudaMemcpy(h_selected.data(),
                                   d_selected_ids,
                                   batch_count * config.nprobe * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost),
                        error,
                        "cudaMemcpy selected clusters D2H")) {
            return false;
        }
        local_timing.d2h_ms += elapsed_ms(stage);

        std::vector<std::vector<uint32_t>> cluster_queries(index.nlist);
        std::vector<uint8_t> used(index.nlist, 0);
        std::vector<uint32_t> active_clusters;
        active_clusters.reserve(batch_count * config.nprobe);

        for (uint32_t q = 0; q < batch_count; ++q) {
            for (size_t p = 0; p < config.nprobe; ++p) {
                const uint32_t cid = h_selected[static_cast<size_t>(q) * config.nprobe + p];
                if (cid >= index.nlist) {
                    continue;
                }
                if (!used[cid]) {
                    used[cid] = 1;
                    active_clusters.push_back(cid);
                }
                cluster_queries[cid].push_back(q);
            }
        }

        for (uint32_t cid : active_clusters) {
            const size_t list_start = index.h_offsets[cid];
            const size_t list_end = index.h_offsets[static_cast<size_t>(cid) + 1];
            const size_t list_len = list_end - list_start;
            if (list_len == 0) {
                continue;
            }

            const uint32_t* d_list_ids = index.d_ids + list_start;
            size_t group_query_count = batch_count;
            if (config.ivf_mode == GpuAnnIVFMode::Grouped) {
                group_query_count = cluster_queries[cid].size();
                if (group_query_count == 0) {
                    continue;
                }

                stage = Clock::now();
                if (!check_cuda(cudaMemcpy(d_group_query_ids,
                                           cluster_queries[cid].data(),
                                           group_query_count * sizeof(uint32_t),
                                           cudaMemcpyHostToDevice),
                                error,
                                "cudaMemcpy group query ids H2D")) {
                    return false;
                }
                local_timing.h2d_query_ms += elapsed_ms(stage);

                dim3 block(16, 16);
                dim3 grid((list_len + block.x - 1) / block.x,
                          (group_query_count + block.y - 1) / block.y);

                stage = Clock::now();
                ivf_refine_grouped_kernel<<<grid, block>>>(index.d_base,
                                                           d_queries,
                                                           d_group_query_ids,
                                                           d_list_ids,
                                                           list_len,
                                                           group_query_count,
                                                           index.dim,
                                                           d_refine_scores);
                if (!check_cuda(cudaGetLastError(), error, "grouped refine launch") ||
                    !check_cuda(cudaDeviceSynchronize(), error, "grouped refine sync")) {
                    return false;
                }
                local_timing.refine_ms += elapsed_ms(stage);
                local_timing.refine_pairs += group_query_count * list_len;
            } else {
                dim3 block(16, 16);
                dim3 grid((list_len + block.x - 1) / block.x,
                          (batch_count + block.y - 1) / block.y);

                stage = Clock::now();
                ivf_refine_wasteful_kernel<<<grid, block>>>(index.d_base,
                                                            d_queries,
                                                            d_list_ids,
                                                            list_len,
                                                            batch_count,
                                                            index.dim,
                                                            cid,
                                                            d_selected_ids,
                                                            config.nprobe,
                                                            d_refine_scores);
                if (!check_cuda(cudaGetLastError(), error, "wasteful refine launch") ||
                    !check_cuda(cudaDeviceSynchronize(), error, "wasteful refine sync")) {
                    return false;
                }
                local_timing.refine_ms += elapsed_ms(stage);
                local_timing.refine_pairs += batch_count * list_len;
                local_timing.skipped_pairs +=
                    (batch_count - cluster_queries[cid].size()) * list_len;
            }

            ++local_timing.refine_groups;

            stage = Clock::now();
            launch_topk(d_refine_scores,
                        list_len,
                        list_len,
                        group_query_count,
                        d_list_ids,
                        config,
                        d_group_dist,
                        d_group_ids);
            if (!check_cuda(cudaGetLastError(), error, "refine topk launch") ||
                !check_cuda(cudaDeviceSynchronize(), error, "refine topk sync")) {
                return false;
            }
            local_timing.topk_ms += elapsed_ms(stage);

            stage = Clock::now();
            if (!check_cuda(cudaMemcpy(h_group_dist.data(),
                                       d_group_dist,
                                       group_query_count * config.k * sizeof(float),
                                       cudaMemcpyDeviceToHost),
                            error,
                            "cudaMemcpy group distances D2H") ||
                !check_cuda(cudaMemcpy(h_group_ids.data(),
                                       d_group_ids,
                                       group_query_count * config.k * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost),
                            error,
                            "cudaMemcpy group ids D2H")) {
                return false;
            }
            local_timing.d2h_ms += elapsed_ms(stage);

            for (size_t qpos = 0; qpos < group_query_count; ++qpos) {
                const size_t batch_q =
                    (config.ivf_mode == GpuAnnIVFMode::Grouped)
                        ? cluster_queries[cid][qpos]
                        : qpos;
                auto& heap = heaps[offset + batch_q];
                for (size_t i = 0; i < config.k; ++i) {
                    const size_t src = qpos * config.k + i;
                    push_host_candidate(heap,
                                        h_group_dist[src],
                                        h_group_ids[src],
                                        config.k);
                }
            }
        }
    }

    cudaFree(d_queries);
    cudaFree(d_centroid_scores);
    cudaFree(d_selected_dist);
    cudaFree(d_selected_ids);
    cudaFree(d_refine_scores);
    cudaFree(d_group_query_ids);
    cudaFree(d_group_dist);
    cudaFree(d_group_ids);

    finalize_heaps(heaps, config.k, result);
    local_timing.total_ms = elapsed_ms(total_start);
    if (timing) {
        *timing = local_timing;
    }
    return true;
}

std::priority_queue<std::pair<float, uint32_t>>
gpu_ann_queue_from_topk(const GpuAnnTopK& result, size_t query_id) {
    std::priority_queue<std::pair<float, uint32_t>> heap;
    if (query_id >= result.query_count) {
        return heap;
    }
    for (size_t i = 0; i < result.k; ++i) {
        const size_t idx = query_id * result.k + i;
        if (idx >= result.distances.size() || idx >= result.ids.size()) {
            break;
        }
        if (std::isfinite(result.distances[idx]) &&
            result.ids[idx] != std::numeric_limits<uint32_t>::max()) {
            heap.emplace(result.distances[idx], result.ids[idx]);
        }
    }
    return heap;
}
