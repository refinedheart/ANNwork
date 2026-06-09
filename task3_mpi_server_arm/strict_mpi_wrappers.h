#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include "hnsw_mt.h"
#include "ivf.h"
#include "ivf_hnsw.h"
#include "ivf_mt.h"
#include "ivf_neon.h"
#include "ivf_pq.h"

using StrictMPICandidate = std::pair<float, uint32_t>;

inline void strict_mpi_push_bounded_candidate(std::priority_queue<StrictMPICandidate>& heap,
                                              float dist,
                                              uint32_t id,
                                              size_t limit) {
    if (heap.size() < limit) {
        heap.emplace(dist, id);
    } else if (dist < heap.top().first) {
        heap.pop();
        heap.emplace(dist, id);
    }
}

inline void strict_mpi_serialize_topk_queue_to_buffers(
    std::priority_queue<StrictMPICandidate> result,
    size_t k,
    uint32_t global_offset,
    float* distances,
    uint32_t* ids) {
    std::fill_n(distances, k, std::numeric_limits<float>::infinity());
    std::fill_n(ids, k, std::numeric_limits<uint32_t>::max());

    std::vector<StrictMPICandidate> ordered;
    ordered.reserve(result.size());
    while (!result.empty()) {
        ordered.push_back(result.top());
        result.pop();
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const StrictMPICandidate& lhs, const StrictMPICandidate& rhs) {
                  if (lhs.first != rhs.first) {
                      return lhs.first < rhs.first;
                  }
                  return lhs.second < rhs.second;
              });

    const size_t limit = std::min(k, ordered.size());
    for (size_t i = 0; i < limit; ++i) {
        distances[i] = ordered[i].first;
        ids[i] = ordered[i].second + global_offset;
    }
}

inline std::priority_queue<StrictMPICandidate>
strict_mpi_merge_topk_arrays(const float* distances,
                             const uint32_t* ids,
                             size_t count,
                             size_t k) {
    std::priority_queue<StrictMPICandidate> merged;
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] == std::numeric_limits<uint32_t>::max() ||
            !std::isfinite(distances[i])) {
            continue;
        }
        strict_mpi_push_bounded_candidate(merged, distances[i], ids[i], k);
    }
    return merged;
}

inline std::vector<size_t> strict_mpi_make_balanced_counts(size_t total, int parts) {
    std::vector<size_t> counts(parts, total / static_cast<size_t>(parts));
    const size_t remainder = total % static_cast<size_t>(parts);
    for (size_t i = 0; i < remainder; ++i) {
        ++counts[i];
    }
    return counts;
}

inline std::vector<size_t> strict_mpi_make_displacements(const std::vector<size_t>& counts) {
    std::vector<size_t> displs(counts.size(), 0);
    for (size_t i = 1; i < counts.size(); ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
    return displs;
}

inline std::string strict_mpi_normalize_data_path(std::string data_path) {
    if (!data_path.empty() && data_path.back() != '/') {
        data_path.push_back('/');
    }
    return data_path;
}

enum class StrictMPIMode {
    IVF_DATA_SEQ,
    IVF_DATA_OPENMP,
    IVF_PQ_DATA_SEQ,
    HNSW_SHARD_SEQ,
    IVF_HNSW_DATA_SEQ,
    IVF_HNSW_DATA_OPENMP,
};

struct StrictMPIWrapperConfig {
    // Edit these values directly on the server when you need a new MPI test point.
    StrictMPIMode mode = StrictMPIMode::IVF_DATA_SEQ;
    int threads = 1;
    int requested_thread_level = MPI_THREAD_FUNNELED;
    size_t query_limit = 100;
    size_t base_limit = 100000;
    std::string data_path = "/anndata/";
    size_t ivf_nlist = 128;
    size_t nprobe = 32;
    size_t pq_rerank = 550;
    size_t ivf_local_p = 100;
    size_t hnsw_ef = 64;
    size_t ivf_hnsw_nlist = 16;
    size_t ivf_hnsw_kmeans_iters = 2;
    size_t ivf_hnsw_m = 12;
    size_t ivf_hnsw_ef_construction = 80;
    bool mute_non_root_stdout = true;
};

inline StrictMPIWrapperConfig& strict_mpi_config() {
    static StrictMPIWrapperConfig cfg{};
    return cfg;
}

struct StrictMPIWrapperState {
    bool mpi_ready = false;
    bool prepared = false;
    bool finalized_registered = false;
    bool non_root_stdout_muted = false;
    int rank = 0;
    int world_size = 1;
    size_t base_number = 0;
    size_t vecdim = 0;
    size_t local_n = 0;
    size_t global_offset = 0;
    float* base = nullptr;
    float* local_base = nullptr;
    std::vector<float> gathered_distances;
    std::vector<uint32_t> gathered_ids;
    std::vector<float> merged_distances;
    std::vector<uint32_t> merged_ids;
};

inline StrictMPIWrapperState& strict_mpi_state() {
    static StrictMPIWrapperState state{};
    return state;
}

inline const char* strict_mpi_mode_name(StrictMPIMode mode) {
    switch (mode) {
        case StrictMPIMode::IVF_DATA_SEQ:
            return "ivf_data_seq";
        case StrictMPIMode::IVF_DATA_OPENMP:
            return "ivf_data_openmp";
        case StrictMPIMode::IVF_PQ_DATA_SEQ:
            return "ivf_pq_data_seq";
        case StrictMPIMode::HNSW_SHARD_SEQ:
            return "hnsw_shard_seq";
        case StrictMPIMode::IVF_HNSW_DATA_SEQ:
            return "ivf_hnsw_data_seq";
        case StrictMPIMode::IVF_HNSW_DATA_OPENMP:
            return "ivf_hnsw_data_openmp";
        default:
            return "unknown";
    }
}

inline void strict_mpi_finalize_at_exit() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return;
    }
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

inline void strict_mpi_ensure_initialized() {
    auto& state = strict_mpi_state();
    if (state.mpi_ready) {
        return;
    }

    const auto& cfg = strict_mpi_config();
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = MPI_THREAD_SINGLE;
        MPI_Init_thread(nullptr, nullptr, cfg.requested_thread_level, &provided);
        (void)provided;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &state.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.world_size);
    state.mpi_ready = true;

    if (!state.finalized_registered) {
        std::atexit(strict_mpi_finalize_at_exit);
        state.finalized_registered = true;
    }

    if (cfg.mute_non_root_stdout && state.rank != 0 && !state.non_root_stdout_muted) {
        std::cout.setstate(std::ios_base::failbit);
        state.non_root_stdout_muted = true;
    }
}

inline void strict_mpi_reset_prepared_state() {
    auto& state = strict_mpi_state();
    state.prepared = false;
    state.base_number = 0;
    state.vecdim = 0;
    state.local_n = 0;
    state.global_offset = 0;
    state.base = nullptr;
    state.local_base = nullptr;
    state.gathered_distances.clear();
    state.gathered_ids.clear();
    state.merged_distances.clear();
    state.merged_ids.clear();
}

inline std::priority_queue<std::pair<float, uint32_t>>
strict_mpi_run_local_search(float* query, size_t k) {
    auto& state = strict_mpi_state();
    auto& cfg = strict_mpi_config();
    if (state.local_n == 0 || state.local_base == nullptr) {
        return {};
    }

    switch (cfg.mode) {
        case StrictMPIMode::IVF_DATA_SEQ:
            return ivf_search_neon(state.local_base, query, state.local_n, state.vecdim, k, cfg.nprobe);
        case StrictMPIMode::IVF_DATA_OPENMP:
            return ivf_search_centroid_partition_openmp(state.local_base, query, state.local_n,
                                                        state.vecdim, k, cfg.threads,
                                                        cfg.nprobe, cfg.ivf_local_p);
        case StrictMPIMode::IVF_PQ_DATA_SEQ:
            return ivf_pq_search(state.local_base, query, state.local_n, state.vecdim, k,
                                 cfg.nprobe, cfg.pq_rerank);
        case StrictMPIMode::HNSW_SHARD_SEQ:
            return hnsw_search_prebuilt(query, k);
        case StrictMPIMode::IVF_HNSW_DATA_SEQ:
            return ivf_hnsw_search(state.local_base, query, state.local_n, state.vecdim, k,
                                   cfg.nprobe, cfg.hnsw_ef);
        case StrictMPIMode::IVF_HNSW_DATA_OPENMP:
            return ivf_hnsw_search_openmp(state.local_base, query, state.local_n, state.vecdim,
                                          k, cfg.threads, cfg.nprobe, cfg.hnsw_ef);
        default:
            std::cerr << "unsupported strict mpi mode: " << strict_mpi_mode_name(cfg.mode) << "\n";
            std::exit(1);
    }
}

inline void strict_mpi_prepare(float* base, size_t base_number, size_t vecdim) {
    strict_mpi_ensure_initialized();

    auto& state = strict_mpi_state();
    auto& cfg = strict_mpi_config();
    if (state.prepared &&
        state.base == base &&
        state.base_number == base_number &&
        state.vecdim == vecdim) {
        return;
    }

    strict_mpi_reset_prepared_state();
    state.base = base;
    state.base_number = base_number;
    state.vecdim = vecdim;

    const auto counts = strict_mpi_make_balanced_counts(base_number, state.world_size);
    const auto displs = strict_mpi_make_displacements(counts);
    state.local_n = counts[state.rank];
    state.global_offset = displs[state.rank];
    state.local_base = base + state.global_offset * vecdim;

    if (state.rank == 0) {
        std::cerr << "[strict_mpi] mode=" << strict_mpi_mode_name(cfg.mode)
                  << " ranks=" << state.world_size
                  << " threads=" << cfg.threads << "\n";
    }

    if (state.local_n > 0) {
        switch (cfg.mode) {
            case StrictMPIMode::IVF_DATA_SEQ:
            case StrictMPIMode::IVF_DATA_OPENMP: {
                const size_t local_nlist = std::max<size_t>(1, std::min(cfg.ivf_nlist, state.local_n));
                build_ivf_index(state.local_base, state.local_n, vecdim, local_nlist);
                break;
            }
            case StrictMPIMode::IVF_PQ_DATA_SEQ: {
                const size_t local_nlist = std::max<size_t>(1, std::min(cfg.ivf_nlist, state.local_n));
                build_ivf_pq(state.local_base, state.local_n, vecdim, local_nlist);
                break;
            }
            case StrictMPIMode::HNSW_SHARD_SEQ:
                build_hnsw_index(state.local_base, state.local_n, vecdim, 1, 16, 150, cfg.hnsw_ef);
                break;
            case StrictMPIMode::IVF_HNSW_DATA_SEQ:
            case StrictMPIMode::IVF_HNSW_DATA_OPENMP: {
                const size_t local_nlist = std::max<size_t>(1, std::min(cfg.ivf_hnsw_nlist, state.local_n));
                build_ivf_hnsw_index(state.local_base, state.local_n, vecdim,
                                     local_nlist,
                                     cfg.ivf_hnsw_kmeans_iters,
                                     cfg.ivf_hnsw_m,
                                     cfg.ivf_hnsw_ef_construction,
                                     cfg.hnsw_ef);
                break;
            }
            default:
                break;
        }
    }

    state.prepared = true;
}

inline bool strict_mpi_is_root() {
    strict_mpi_ensure_initialized();
    return strict_mpi_state().rank == 0;
}

inline size_t strict_mpi_query_limit(size_t fallback) {
    return std::min(fallback, strict_mpi_config().query_limit);
}

inline size_t strict_mpi_base_limit(size_t fallback) {
    const size_t limit = strict_mpi_config().base_limit;
    return limit == 0 ? fallback : std::min(fallback, limit);
}

inline const std::string& strict_mpi_data_path() {
    static const std::string path = strict_mpi_normalize_data_path(strict_mpi_config().data_path);
    return path;
}

inline std::priority_queue<std::pair<float, uint32_t>>
strict_mpi_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    strict_mpi_prepare(base, base_number, vecdim);

    auto& state = strict_mpi_state();
    state.gathered_distances.resize(static_cast<size_t>(state.world_size) * k);
    state.gathered_ids.resize(static_cast<size_t>(state.world_size) * k);
    state.merged_distances.resize(k);
    state.merged_ids.resize(k);
    std::vector<float> local_distances(k);
    std::vector<uint32_t> local_ids(k);
    auto local_result = strict_mpi_run_local_search(query, k);
    strict_mpi_serialize_topk_queue_to_buffers(local_result,
                                               k,
                                               static_cast<uint32_t>(state.global_offset),
                                               local_distances.data(),
                                               local_ids.data());

    if (state.rank == 0) {
        state.gathered_distances.assign(static_cast<size_t>(state.world_size) * k,
                                        std::numeric_limits<float>::infinity());
        state.gathered_ids.assign(static_cast<size_t>(state.world_size) * k,
                                  std::numeric_limits<uint32_t>::max());
    }

    MPI_Gather(local_distances.data(),
               static_cast<int>(k),
               MPI_FLOAT,
               state.rank == 0 ? state.gathered_distances.data() : nullptr,
               static_cast<int>(k),
               MPI_FLOAT,
               0,
               MPI_COMM_WORLD);
    MPI_Gather(local_ids.data(),
               static_cast<int>(k),
               MPI_UINT32_T,
               state.rank == 0 ? state.gathered_ids.data() : nullptr,
               static_cast<int>(k),
               MPI_UINT32_T,
               0,
               MPI_COMM_WORLD);

    if (state.rank == 0) {
        auto merged = strict_mpi_merge_topk_arrays(state.gathered_distances.data(),
                                                   state.gathered_ids.data(),
                                                   static_cast<size_t>(state.world_size) * k,
                                                   k);
        strict_mpi_serialize_topk_queue_to_buffers(merged, k, 0,
                                                   state.merged_distances.data(),
                                                   state.merged_ids.data());
    }

    MPI_Bcast(state.merged_distances.data(),
              static_cast<int>(k),
              MPI_FLOAT,
              0,
              MPI_COMM_WORLD);
    MPI_Bcast(state.merged_ids.data(),
              static_cast<int>(k),
              MPI_UINT32_T,
              0,
              MPI_COMM_WORLD);

    return strict_mpi_merge_topk_arrays(state.merged_distances.data(),
                                        state.merged_ids.data(),
                                        k,
                                        k);
}
