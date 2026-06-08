#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include "ann_bench_utils.h"
#include "flat_scan_mt.h"
#include "hnsw_mt.h"
#include "hybrid_graph_mt.h"
#include "ivf_mt.h"
#include "ivf_neon.h"
#include "ivf_pq.h"

namespace {

struct MPIRunConfig {
    std::string mode = "ivf_data_seq";
    int threads = 1;
    size_t query_limit = 2000;
    size_t base_limit = 0;
    size_t mpi_batch = 32;
    size_t data_mpi_batch = 0;
    int requested_thread_level = MPI_THREAD_FUNNELED;
    int provided_thread_level = MPI_THREAD_SINGLE;
    size_t k = 10;
    size_t ivf_nlist = 128;
    size_t nprobe = 32;
    size_t pq_rerank = 550;
    size_t ivf_local_p = 100;
    size_t hnsw_ef = 64;
    size_t ivf_hnsw_cluster_k = 32;
    size_t hnsw_parts = 4;
    size_t hnsw_shard_probe = 2;
    std::string data_path = "/anndata/";
};

MPIRunConfig ParseConfig(int argc, char* argv[]) {
    MPIRunConfig cfg;
    if (argc > 1) {
        cfg.mode = argv[1];
    }
    if (argc > 2) {
        cfg.threads = std::max(1, std::atoi(argv[2]));
    } else {
        cfg.threads = std::max(1, GetEnvInt("ANN_THREADS", 1));
    }
    if (argc > 3) {
        cfg.query_limit = static_cast<size_t>(std::strtoull(argv[3], nullptr, 10));
    } else {
        cfg.query_limit = GetEnvSizeT("ANN_QUERY_LIMIT", cfg.query_limit);
    }

    cfg.k = GetEnvSizeT("ANN_TOPK", cfg.k);
    cfg.base_limit = GetEnvSizeT("ANN_BASE_LIMIT", cfg.base_limit);
    cfg.mpi_batch = GetEnvSizeT("ANN_MPI_BATCH", cfg.mpi_batch);
    cfg.data_mpi_batch = GetEnvSizeT("ANN_DATA_MPI_BATCH", cfg.data_mpi_batch);
    cfg.ivf_nlist = GetEnvSizeT("ANN_IVF_NLIST", cfg.ivf_nlist);
    cfg.nprobe = GetEnvSizeT("ANN_IVF_NPROBE", cfg.nprobe);
    cfg.pq_rerank = GetEnvSizeT("ANN_PQ_RERANK", cfg.pq_rerank);
    cfg.ivf_local_p = GetEnvSizeT("ANN_IVF_LOCAL_P", cfg.ivf_local_p);
    cfg.hnsw_ef = GetEnvSizeT("ANN_HNSW_EF", cfg.hnsw_ef);
    cfg.ivf_hnsw_cluster_k = GetEnvSizeT("ANN_IVF_HNSW_CLUSTER_K", cfg.ivf_hnsw_cluster_k);
    cfg.hnsw_parts = GetEnvSizeT("ANN_HNSW_PARTS", cfg.hnsw_parts);
    cfg.hnsw_shard_probe = GetEnvSizeT("ANN_HNSW_SHARD_PROBE", cfg.hnsw_shard_probe);
    cfg.data_path = NormalizeDataPath(GetEnvString("ANN_DATA_PATH", cfg.data_path));
    return cfg;
}

const char* ThreadLevelName(int level) {
    switch (level) {
        case MPI_THREAD_SINGLE:
            return "single";
        case MPI_THREAD_FUNNELED:
            return "funneled";
        case MPI_THREAD_SERIALIZED:
            return "serialized";
        case MPI_THREAD_MULTIPLE:
            return "multiple";
        default:
            return "unknown";
    }
}

int ParseThreadLevelFromEnv(const std::string& text) {
    if (text == "single") {
        return MPI_THREAD_SINGLE;
    }
    if (text == "funneled") {
        return MPI_THREAD_FUNNELED;
    }
    if (text == "serialized") {
        return MPI_THREAD_SERIALIZED;
    }
    if (text == "multiple") {
        return MPI_THREAD_MULTIPLE;
    }
    return MPI_THREAD_FUNNELED;
}

int DetermineRequestedThreadLevel(const MPIRunConfig& cfg) {
    const std::string env_level = GetEnvString("ANN_MPI_THREAD_LEVEL", "");
    if (!env_level.empty()) {
        return ParseThreadLevelFromEnv(env_level);
    }
    if (cfg.mode == "query_shard_rma_openmp_multiple") {
        return MPI_THREAD_MULTIPLE;
    }
    if (cfg.mode == "query_shard_rma_openmp_funneled") {
        return MPI_THREAD_FUNNELED;
    }
    return MPI_THREAD_FUNNELED;
}

bool IsQueryShardMode(const std::string& mode) {
    return mode == "query_shard_seq" || mode == "query_shard_openmp" ||
           mode == "query_shard_pthread" || mode == "query_shard_nb" ||
           mode == "query_shard_rma_openmp_funneled" ||
           mode == "query_shard_rma_openmp_multiple" ||
           mode == "query_shard_rma";
}

bool IsDataShardMode(const std::string& mode) {
    return mode == "ivf_data_seq" || mode == "ivf_data_pthread" ||
           mode == "ivf_data_openmp" || mode == "ivf_pq_data_seq" ||
           mode == "hnsw_shard_seq" || mode == "ivf_hnsw_data_seq" ||
           mode == "ivf_hnsw_data_openmp" || mode == "hnsw_on_hnsw_seq";
}

std::vector<int> ToIntVector(const std::vector<size_t>& values, size_t scale = 1) {
    std::vector<int> result(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = static_cast<int>(values[i] * scale);
    }
    return result;
}

void ScatterContiguousGroundTruth(int rank,
                                  int* test_gt,
                                  const std::vector<int>& sendcounts_gt,
                                  const std::vector<int>& displs_gt,
                                  size_t local_queries,
                                  size_t test_gt_d,
                                  std::vector<int>& local_gt) {
    local_gt.resize(local_queries * test_gt_d);
    MPI_Scatterv(test_gt,
                 sendcounts_gt.data(),
                 displs_gt.data(),
                 MPI_INT,
                 local_gt.empty() ? nullptr : local_gt.data(),
                 sendcounts_gt.empty() ? 0 : sendcounts_gt[rank],
                 MPI_INT,
                 0,
                 MPI_COMM_WORLD);
}

std::vector<std::vector<int>> ScatterBatchedGroundTruth(const MPIRunConfig& cfg,
                                                        int rank,
                                                        int world_size,
                                                        int* test_gt,
                                                        size_t test_number,
                                                        size_t test_gt_d) {
    const size_t data_batch = cfg.data_mpi_batch == 0 ? test_number : cfg.data_mpi_batch;
    const std::vector<size_t> batch_sizes = MakeBatchSizes(test_number, data_batch);
    std::vector<std::vector<int>> gt_batches(batch_sizes.size());

    size_t global_offset = 0;
    for (size_t b = 0; b < batch_sizes.size(); ++b) {
        const auto batch_counts = MakeBalancedCounts(batch_sizes[b], world_size);
        const auto batch_displs = MakeDisplacements(batch_counts);
        const auto sendcounts_gt = ToIntVector(batch_counts, test_gt_d);
        const auto displs_gt = ToIntVector(batch_displs, test_gt_d);
        gt_batches[b].resize(batch_counts[rank] * test_gt_d);

        MPI_Scatterv(rank == 0 ? test_gt + global_offset * test_gt_d : nullptr,
                     sendcounts_gt.data(),
                     displs_gt.data(),
                     MPI_INT,
                     gt_batches[b].empty() ? nullptr : gt_batches[b].data(),
                     sendcounts_gt[rank],
                     MPI_INT,
                     0,
                     MPI_COMM_WORLD);
        global_offset += batch_sizes[b];
    }

    return gt_batches;
}

void BuildLocalShardIndex(const MPIRunConfig& cfg,
                          float* local_base,
                          size_t local_n,
                          size_t vecdim) {
    if (local_n == 0) {
        return;
    }

    const size_t local_nlist = std::max<size_t>(1, std::min(cfg.ivf_nlist, local_n));
    if (cfg.mode == "ivf_data_seq" || cfg.mode == "ivf_data_pthread" ||
        cfg.mode == "ivf_data_openmp") {
        build_ivf_index(local_base, local_n, vecdim, local_nlist);
    } else if (cfg.mode == "ivf_pq_data_seq") {
        build_ivf_pq(local_base, local_n, vecdim, local_nlist);
    } else if (cfg.mode == "ivf_hnsw_data_seq" ||
               cfg.mode == "ivf_hnsw_data_openmp") {
        BuildIVFHNSWIndex(local_base, local_n, vecdim, local_nlist, cfg.hnsw_ef);
    } else if (cfg.mode == "hnsw_shard_seq") {
        build_hnsw_index(local_base, local_n, vecdim, 1, 16, 150, cfg.hnsw_ef);
    } else if (cfg.mode == "hnsw_on_hnsw_seq") {
        BuildHNSWOnHNSWIndex(local_base, local_n, vecdim, cfg.hnsw_parts, cfg.hnsw_ef);
    }
}

std::priority_queue<AnnCandidate>
RunLocalShardSearch(const MPIRunConfig& cfg,
                    float* local_base,
                    float* query,
                    size_t local_n,
                    size_t vecdim) {
    if (local_n == 0) {
        return {};
    }

    if (cfg.mode == "ivf_data_seq") {
        return ivf_search_neon(local_base, query, local_n, vecdim, cfg.k, cfg.nprobe);
    }
    if (cfg.mode == "ivf_data_pthread") {
        return ivf_search_centroid_partition_pthread(local_base, query, local_n, vecdim,
                                                     cfg.k, cfg.threads, cfg.nprobe,
                                                     cfg.ivf_local_p);
    }
    if (cfg.mode == "ivf_data_openmp") {
        return ivf_search_centroid_partition_openmp(local_base, query, local_n, vecdim,
                                                    cfg.k, cfg.threads, cfg.nprobe,
                                                    cfg.ivf_local_p);
    }
    if (cfg.mode == "ivf_pq_data_seq") {
        return ivf_pq_search(local_base, query, local_n, vecdim, cfg.k,
                             cfg.nprobe, cfg.pq_rerank);
    }
    if (cfg.mode == "ivf_hnsw_data_seq") {
        const size_t local_nlist = std::max<size_t>(1, std::min(cfg.ivf_nlist, local_n));
        return ivf_hnsw_search(local_base, query, local_n, vecdim, cfg.k,
                               local_nlist, cfg.nprobe,
                               cfg.ivf_hnsw_cluster_k, cfg.hnsw_ef);
    }
    if (cfg.mode == "ivf_hnsw_data_openmp") {
        const size_t local_nlist = std::max<size_t>(1, std::min(cfg.ivf_nlist, local_n));
        return ivf_hnsw_search_openmp(local_base, query, local_n, vecdim, cfg.k,
                                      cfg.threads, local_nlist, cfg.nprobe,
                                      cfg.ivf_hnsw_cluster_k, cfg.hnsw_ef);
    }
    if (cfg.mode == "hnsw_shard_seq") {
        return hnsw_search_prebuilt(query, cfg.k);
    }
    if (cfg.mode == "hnsw_on_hnsw_seq") {
        return hnsw_on_hnsw_search(local_base, query, local_n, vecdim, cfg.k,
                                   cfg.hnsw_parts, cfg.hnsw_shard_probe,
                                   cfg.ivf_local_p, cfg.hnsw_ef);
    }

    std::cerr << "unsupported data-shard mode: " << cfg.mode << "\n";
    std::exit(1);
}

void PrintSummary(const MPIRunConfig& cfg,
                  int world_size,
                  size_t base_vectors,
                  size_t queries,
                  double avg_recall,
                  double avg_latency_us,
                  double total_us) {
    std::cout << "mode: " << cfg.mode << "\n";
    std::cout << "mpi ranks: " << world_size << "\n";
    std::cout << "threads per rank: " << cfg.threads << "\n";
    std::cout << "mpi thread level requested: " << ThreadLevelName(cfg.requested_thread_level) << "\n";
    std::cout << "mpi thread level provided: " << ThreadLevelName(cfg.provided_thread_level) << "\n";
    std::cout << "base vectors: " << base_vectors << "\n";
    std::cout << "queries: " << queries << "\n";
    std::cout << "average recall: " << avg_recall << "\n";
    std::cout << "average latency (us): " << avg_latency_us << "\n";
    std::cout << "total time (us): " << total_us << "\n";
}

void RunQueryShardNonblocking(const MPIRunConfig& cfg,
                              int rank,
                              int world_size,
                              float* base,
                              float* test_query,
                              size_t test_number,
                              size_t base_number,
                              size_t vecdim,
                              size_t test_gt_d,
                              const std::vector<std::vector<int>>& gt_batches,
                              double& local_recall_sum,
                              double& local_latency_sum) {
    const std::vector<size_t> batch_sizes = MakeBatchSizes(test_number, cfg.mpi_batch);
    const size_t num_batches = batch_sizes.size();

    std::vector<size_t> local_counts(num_batches, 0);
    std::vector<std::vector<size_t>> batch_counts(num_batches);
    std::vector<std::vector<size_t>> batch_displs(num_batches);

    for (size_t b = 0; b < num_batches; ++b) {
        batch_counts[b] = MakeBalancedCounts(batch_sizes[b], world_size);
        batch_displs[b] = MakeDisplacements(batch_counts[b]);
        local_counts[b] = batch_counts[b][rank];
    }

    std::vector<std::vector<float>> query_batches(num_batches);
    std::vector<MPI_Request> recv_query_reqs(num_batches, MPI_REQUEST_NULL);
    std::vector<MPI_Request> send_reqs;

    for (size_t b = 0; b < num_batches; ++b) {
        query_batches[b].resize(local_counts[b] * vecdim);
        const int query_tag = 11000 + static_cast<int>(b);

        if (rank != 0 && local_counts[b] > 0) {
            MPI_Irecv(query_batches[b].data(),
                      static_cast<int>(local_counts[b] * vecdim),
                      MPI_FLOAT, 0, query_tag, MPI_COMM_WORLD,
                      &recv_query_reqs[b]);
        }
    }

    if (rank == 0) {
        size_t global_offset = 0;
        for (size_t b = 0; b < num_batches; ++b) {
            const int query_tag = 11000 + static_cast<int>(b);
            for (int r = 0; r < world_size; ++r) {
                const size_t count = batch_counts[b][r];
                if (count == 0) {
                    continue;
                }
                const size_t batch_offset = global_offset + batch_displs[b][r];
                if (r == 0) {
                    std::copy(test_query + batch_offset * vecdim,
                              test_query + (batch_offset + count) * vecdim,
                              query_batches[b].begin());
                } else {
                    MPI_Request query_req = MPI_REQUEST_NULL;
                    MPI_Isend(test_query + batch_offset * vecdim,
                              static_cast<int>(count * vecdim),
                              MPI_FLOAT, r, query_tag, MPI_COMM_WORLD, &query_req);
                    send_reqs.push_back(query_req);
                }
            }
            global_offset += batch_sizes[b];
        }
    }

    local_recall_sum = 0.0;
    local_latency_sum = 0.0;
    const unsigned long converter = 1000 * 1000;

    for (size_t b = 0; b < num_batches; ++b) {
        if (local_counts[b] == 0) {
            continue;
        }
        if (rank != 0) {
            MPI_Wait(&recv_query_reqs[b], MPI_STATUS_IGNORE);
        }

        for (size_t i = 0; i < local_counts[b]; ++i) {
            struct timeval val;
            gettimeofday(&val, nullptr);
            auto res = ivf_search_neon(base,
                                       query_batches[b].data() + i * vecdim,
                                       base_number, vecdim, cfg.k, cfg.nprobe);
            struct timeval new_val;
            gettimeofday(&new_val, nullptr);
            const int64_t diff = (new_val.tv_sec * converter + new_val.tv_usec) -
                                 (val.tv_sec * converter + val.tv_usec);
            local_recall_sum += ComputeRecallFromQueue(
                res, gt_batches[b].data() + i * test_gt_d, cfg.k);
            local_latency_sum += static_cast<double>(diff);
        }
    }

    if (rank == 0 && !send_reqs.empty()) {
        MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(), MPI_STATUSES_IGNORE);
    }

    if (rank == 0) {
        std::cout << "nonblocking batch size: " << cfg.mpi_batch << "\n";
    }
}

void RunQueryShardRMA(const MPIRunConfig& cfg,
                      int rank,
                      int world_size,
                      float* base,
                      float* test_query,
                      size_t test_number,
                      size_t base_number,
                      size_t vecdim,
                      size_t test_gt_d,
                      std::vector<int>& local_gt,
                      double& local_recall_sum,
                      double& local_latency_sum) {
    const auto query_counts = MakeBalancedCounts(test_number, world_size);
    const auto query_displs = MakeDisplacements(query_counts);
    const size_t local_queries = query_counts[rank];
    const size_t global_query_offset = query_displs[rank];

    std::vector<float> local_query(local_queries * vecdim);

    MPI_Win query_win = MPI_WIN_NULL;
    float* query_base = (rank == 0) ? test_query : nullptr;

    MPI_Win_create(query_base,
                   static_cast<MPI_Aint>((rank == 0) ? test_number * vecdim * sizeof(float) : 0),
                   sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &query_win);

    if (local_queries > 0) {
        if (rank == 0) {
            std::copy(test_query + global_query_offset * vecdim,
                      test_query + (global_query_offset + local_queries) * vecdim,
                      local_query.begin());
        } else {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, query_win);
            MPI_Get(local_query.data(),
                    static_cast<int>(local_queries * vecdim),
                    MPI_FLOAT,
                    0,
                    static_cast<MPI_Aint>(global_query_offset * vecdim),
                    static_cast<int>(local_queries * vecdim),
                    MPI_FLOAT,
                    query_win);
            MPI_Win_unlock(0, query_win);
        }
    }

    local_recall_sum = 0.0;
    local_latency_sum = 0.0;
    const unsigned long converter = 1000 * 1000;

    for (size_t i = 0; i < local_queries; ++i) {
        struct timeval val;
        gettimeofday(&val, nullptr);
        auto res = ivf_search_neon(base,
                                   local_query.data() + i * vecdim,
                                   base_number, vecdim, cfg.k, cfg.nprobe);
        struct timeval new_val;
        gettimeofday(&new_val, nullptr);
        const int64_t diff = (new_val.tv_sec * converter + new_val.tv_usec) -
                             (val.tv_sec * converter + val.tv_usec);
        local_recall_sum += ComputeRecallFromQueue(
            res, local_gt.data() + i * test_gt_d, cfg.k);
        local_latency_sum += static_cast<double>(diff);
    }

    MPI_Win_free(&query_win);
}

void CreateQueryWindow(int rank,
                       float* test_query,
                       size_t test_number,
                       size_t vecdim,
                       MPI_Win* query_win) {
    float* query_base = (rank == 0) ? test_query : nullptr;
    MPI_Win_create(query_base,
                   static_cast<MPI_Aint>((rank == 0) ? test_number * vecdim * sizeof(float) : 0),
                   sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, query_win);
}

void FetchLocalQueriesRMAFunneled(int rank,
                                  float* test_query,
                                  size_t vecdim,
                                  size_t local_queries,
                                  size_t global_query_offset,
                                  MPI_Win query_win,
                                  std::vector<float>& local_query) {
    local_query.resize(local_queries * vecdim);
    if (local_queries == 0) {
        return;
    }
    if (rank == 0) {
        std::copy(test_query + global_query_offset * vecdim,
                  test_query + (global_query_offset + local_queries) * vecdim,
                  local_query.begin());
    } else {
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, query_win);
        MPI_Get(local_query.data(),
                static_cast<int>(local_queries * vecdim),
                MPI_FLOAT,
                0,
                static_cast<MPI_Aint>(global_query_offset * vecdim),
                static_cast<int>(local_queries * vecdim),
                MPI_FLOAT,
                query_win);
        MPI_Win_unlock(0, query_win);
    }
}

void RunQueryShardRMAOpenMPFunneled(const MPIRunConfig& cfg,
                                    int rank,
                                    int world_size,
                                    float* base,
                                    float* test_query,
                                    size_t test_number,
                                    size_t base_number,
                                    size_t vecdim,
                                    size_t test_gt_d,
                                    std::vector<int>& local_gt,
                                    double& local_recall_sum,
                                    double& local_latency_sum) {
    const auto query_counts = MakeBalancedCounts(test_number, world_size);
    const auto query_displs = MakeDisplacements(query_counts);
    const size_t local_queries = query_counts[rank];
    const size_t global_query_offset = query_displs[rank];

    MPI_Win query_win = MPI_WIN_NULL;
    CreateQueryWindow(rank, test_query, test_number, vecdim, &query_win);

    std::vector<float> local_query;
    FetchLocalQueriesRMAFunneled(rank, test_query, vecdim,
                                 local_queries, global_query_offset, query_win,
                                 local_query);
    MPI_Win_free(&query_win);

    std::vector<SearchResult> results;
    ivf_batch_openmp(base, local_query.data(), local_gt.data(), local_queries,
                     base_number, vecdim, test_gt_d, cfg.k, cfg.threads,
                     cfg.nprobe, results);

    local_recall_sum = 0.0;
    local_latency_sum = 0.0;
    for (const auto& result : results) {
        local_recall_sum += result.recall;
        local_latency_sum += static_cast<double>(result.latency);
    }
}

void RunQueryShardRMAOpenMPMultiple(const MPIRunConfig& cfg,
                                    int rank,
                                    int world_size,
                                    float* base,
                                    float* test_query,
                                    size_t test_number,
                                    size_t base_number,
                                    size_t vecdim,
                                    size_t test_gt_d,
                                    std::vector<int>& local_gt,
                                    double& local_recall_sum,
                                    double& local_latency_sum) {
    const auto query_counts = MakeBalancedCounts(test_number, world_size);
    const auto query_displs = MakeDisplacements(query_counts);
    const size_t local_queries = query_counts[rank];
    const size_t global_query_offset = query_displs[rank];

    MPI_Win query_win = MPI_WIN_NULL;
    CreateQueryWindow(rank, test_query, test_number, vecdim, &query_win);

    local_recall_sum = 0.0;
    local_latency_sum = 0.0;
    const unsigned long converter = 1000 * 1000;
    omp_set_num_threads(cfg.threads);
    if (rank != 0) {
        MPI_Win_lock_all(0, query_win);
    }

    #pragma omp parallel reduction(+:local_recall_sum, local_latency_sum)
    {
        std::vector<float> query_buf(vecdim);
        std::vector<int> gt_buf(test_gt_d);

        #pragma omp for schedule(static)
        for (long long local_idx = 0; local_idx < static_cast<long long>(local_queries); ++local_idx) {
            const size_t global_idx = global_query_offset + static_cast<size_t>(local_idx);
            if (rank == 0) {
                std::copy(test_query + global_idx * vecdim,
                          test_query + (global_idx + 1) * vecdim,
                          query_buf.begin());
            } else {
                MPI_Get(query_buf.data(),
                        static_cast<int>(vecdim),
                        MPI_FLOAT,
                        0,
                        static_cast<MPI_Aint>(global_idx * vecdim),
                        static_cast<int>(vecdim),
                        MPI_FLOAT,
                        query_win);
                MPI_Win_flush(0, query_win);
            }
            std::copy(local_gt.data() + static_cast<size_t>(local_idx) * test_gt_d,
                      local_gt.data() + (static_cast<size_t>(local_idx) + 1) * test_gt_d,
                      gt_buf.begin());

            struct timeval val;
            gettimeofday(&val, nullptr);
            auto res = ivf_search_neon(base, query_buf.data(), base_number, vecdim, cfg.k, cfg.nprobe);
            struct timeval new_val;
            gettimeofday(&new_val, nullptr);
            const int64_t diff = (new_val.tv_sec * converter + new_val.tv_usec) -
                                 (val.tv_sec * converter + val.tv_usec);
            local_recall_sum += ComputeRecallFromQueue(res, gt_buf.data(), cfg.k);
            local_latency_sum += static_cast<double>(diff);
        }
    }

    if (rank != 0) {
        MPI_Win_unlock_all(query_win);
    }
    MPI_Win_free(&query_win);
}

int RunQueryShard(const MPIRunConfig& cfg, int rank, int world_size) {
    size_t base_number = 0;
    size_t test_number = 0;
    size_t vecdim = 0;
    size_t test_gt_d = 0;

    float* base = LoadData<float>(cfg.data_path + "DEEP100K.base.100k.fbin",
                                  base_number, vecdim, rank == 0);

    float* test_query = nullptr;
    int* test_gt = nullptr;
    int* original_test_gt = nullptr;
    if (rank == 0) {
        test_query = LoadData<float>(cfg.data_path + "DEEP100K.query.fbin",
                                     test_number, vecdim);
        test_gt = LoadData<int>(cfg.data_path + "DEEP100K.gt.query.100k.top100.bin",
                                test_number, test_gt_d);
        original_test_gt = test_gt;
        test_number = std::min(test_number, cfg.query_limit);
        if (test_number == 0) {
            std::cerr << "ANN_QUERY_LIMIT results in zero queries\n";
        }
    }

    uint64_t test_number_u64 = static_cast<uint64_t>(test_number);
    uint64_t vecdim_u64 = static_cast<uint64_t>(vecdim);
    uint64_t test_gt_d_u64 = static_cast<uint64_t>(test_gt_d);
    MPI_Bcast(&test_number_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vecdim_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_gt_d_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    test_number = static_cast<size_t>(test_number_u64);
    vecdim = static_cast<size_t>(vecdim_u64);
    test_gt_d = static_cast<size_t>(test_gt_d_u64);
    if (cfg.base_limit > 0) {
        base_number = std::min(base_number, cfg.base_limit);
    }
    if (test_number == 0) {
        delete[] base;
        delete[] test_query;
        delete[] original_test_gt;
        return 1;
    }

    std::vector<int> recomputed_gt;
    if (rank == 0 && cfg.base_limit > 0) {
        recomputed_gt = BuildExactGroundTruthForSubset(
            base, test_query, test_number, base_number, vecdim, cfg.k);
        test_gt = recomputed_gt.data();
        test_gt_d = cfg.k;
    }
    uint64_t test_gt_d_sync = static_cast<uint64_t>(test_gt_d);
    MPI_Bcast(&test_gt_d_sync, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    test_gt_d = static_cast<size_t>(test_gt_d_sync);

    const auto query_counts = MakeBalancedCounts(test_number, world_size);
    const auto query_displs = MakeDisplacements(query_counts);
    const std::vector<int> sendcounts_query = ToIntVector(query_counts, vecdim);
    const std::vector<int> displs_query = ToIntVector(query_displs, vecdim);
    const std::vector<int> sendcounts_gt = ToIntVector(query_counts, test_gt_d);
    const std::vector<int> displs_gt = ToIntVector(query_displs, test_gt_d);

    const size_t local_queries = query_counts[rank];
    const bool use_scatterv = (cfg.mode == "query_shard_seq" ||
                               cfg.mode == "query_shard_pthread" ||
                               cfg.mode == "query_shard_openmp");
    std::vector<float> local_query;
    std::vector<int> local_gt;
    std::vector<std::vector<int>> gt_batches;
    if (cfg.mode == "query_shard_nb") {
        gt_batches = ScatterBatchedGroundTruth(cfg, rank, world_size, test_gt,
                                               test_number, test_gt_d);
    } else {
        ScatterContiguousGroundTruth(rank, test_gt, sendcounts_gt, displs_gt,
                                     local_queries, test_gt_d, local_gt);
        if (use_scatterv) {
            local_query.resize(local_queries * vecdim);
        }
    }

    build_ivf_index(base, base_number, vecdim, cfg.ivf_nlist);

    std::vector<SearchResult> results;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    double local_recall_sum = 0.0;
    double local_latency_sum = 0.0;

    if (use_scatterv) {
        MPI_Scatterv(test_query, sendcounts_query.data(), displs_query.data(), MPI_FLOAT,
                     local_query.empty() ? nullptr : local_query.data(),
                     sendcounts_query[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    if (cfg.mode == "query_shard_nb") {
        RunQueryShardNonblocking(cfg, rank, world_size, base, test_query,
                                 test_number, base_number, vecdim, test_gt_d,
                                 gt_batches,
                                 local_recall_sum, local_latency_sum);
    } else if (cfg.mode == "query_shard_rma") {
        RunQueryShardRMA(cfg, rank, world_size, base, test_query,
                         test_number, base_number, vecdim, test_gt_d, local_gt,
                         local_recall_sum, local_latency_sum);
    } else if (cfg.mode == "query_shard_rma_openmp_funneled") {
        RunQueryShardRMAOpenMPFunneled(cfg, rank, world_size, base, test_query,
                                       test_number, base_number, vecdim, test_gt_d, local_gt,
                                       local_recall_sum, local_latency_sum);
    } else if (cfg.mode == "query_shard_rma_openmp_multiple") {
        RunQueryShardRMAOpenMPMultiple(cfg, rank, world_size, base, test_query,
                                       test_number, base_number, vecdim, test_gt_d, local_gt,
                                       local_recall_sum, local_latency_sum);
    } else if (cfg.mode == "query_shard_seq") {
        results.resize(local_queries);
        const unsigned long converter = 1000 * 1000;
        for (size_t i = 0; i < local_queries; ++i) {
            struct timeval val;
            gettimeofday(&val, nullptr);
            auto res = ivf_search_neon(base, local_query.data() + i * vecdim,
                                       base_number, vecdim, cfg.k, cfg.nprobe);
            struct timeval new_val;
            gettimeofday(&new_val, nullptr);
            const int64_t diff = (new_val.tv_sec * converter + new_val.tv_usec) -
                                 (val.tv_sec * converter + val.tv_usec);
            const float recall = ComputeRecallFromQueue(
                res, local_gt.data() + i * test_gt_d, cfg.k);
            results[i] = {recall, diff};
        }
    } else if (cfg.mode == "query_shard_pthread") {
        ivf_batch_pthread(base, local_query.data(), local_gt.data(), local_queries,
                          base_number, vecdim, test_gt_d, cfg.k, cfg.threads,
                          cfg.nprobe, results);
    } else {
        ivf_batch_openmp(base, local_query.data(), local_gt.data(), local_queries,
                         base_number, vecdim, test_gt_d, cfg.k, cfg.threads,
                         cfg.nprobe, results);
    }

    if (cfg.mode != "query_shard_nb" &&
        cfg.mode != "query_shard_rma" &&
        cfg.mode != "query_shard_rma_openmp_funneled" &&
        cfg.mode != "query_shard_rma_openmp_multiple") {
        for (const auto& result : results) {
            local_recall_sum += result.recall;
            local_latency_sum += static_cast<double>(result.latency);
        }
    }

    double global_recall_sum = 0.0;
    double global_latency_sum = 0.0;
    MPI_Reduce(&local_recall_sum, &global_recall_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_latency_sum, &global_latency_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    if (rank == 0) {
        const double total_us = (end - start) * 1e6;
        PrintSummary(cfg, world_size, base_number, test_number,
                     global_recall_sum / static_cast<double>(test_number),
                     total_us / static_cast<double>(test_number),
                     total_us);
        std::cout << "average local compute latency (us): "
                  << global_latency_sum / static_cast<double>(test_number) << "\n";
    }

    delete[] base;
    delete[] test_query;
    delete[] original_test_gt;
    return 0;
}

int RunDataShard(const MPIRunConfig& cfg, int rank, int world_size) {
    size_t test_number = 0;
    size_t base_number = 0;
    size_t vecdim = 0;
    size_t test_gt_d = 0;

    float* base = nullptr;
    float* test_query = nullptr;
    int* test_gt = nullptr;
    int* original_test_gt = nullptr;

    if (rank == 0) {
        test_query = LoadData<float>(cfg.data_path + "DEEP100K.query.fbin",
                                     test_number, vecdim);
        test_gt = LoadData<int>(cfg.data_path + "DEEP100K.gt.query.100k.top100.bin",
                                test_number, test_gt_d);
        original_test_gt = test_gt;
        base = LoadData<float>(cfg.data_path + "DEEP100K.base.100k.fbin",
                               base_number, vecdim);
        test_number = std::min(test_number, cfg.query_limit);
        if (cfg.base_limit > 0) {
            base_number = std::min(base_number, cfg.base_limit);
        }
        if (test_number == 0) {
            std::cerr << "ANN_QUERY_LIMIT results in zero queries\n";
        }
    }

    uint64_t test_number_u64 = static_cast<uint64_t>(test_number);
    uint64_t base_number_u64 = static_cast<uint64_t>(base_number);
    uint64_t vecdim_u64 = static_cast<uint64_t>(vecdim);
    uint64_t test_gt_d_u64 = static_cast<uint64_t>(test_gt_d);
    MPI_Bcast(&test_number_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&base_number_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vecdim_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_gt_d_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    test_number = static_cast<size_t>(test_number_u64);
    base_number = static_cast<size_t>(base_number_u64);
    vecdim = static_cast<size_t>(vecdim_u64);
    test_gt_d = static_cast<size_t>(test_gt_d_u64);
    if (cfg.base_limit > 0) {
        base_number = std::min(base_number, cfg.base_limit);
    }
    if (test_number == 0) {
        delete[] base;
        delete[] test_query;
        delete[] original_test_gt;
        return 1;
    }

    std::vector<int> recomputed_gt;
    if (rank == 0 && cfg.base_limit > 0) {
        recomputed_gt = BuildExactGroundTruthForSubset(
            base, test_query, test_number, base_number, vecdim, cfg.k);
        test_gt = recomputed_gt.data();
        test_gt_d = cfg.k;
    }

    const auto base_counts = MakeBalancedCounts(base_number, world_size);
    const auto base_displs = MakeDisplacements(base_counts);
    const std::vector<int> sendcounts_base = ToIntVector(base_counts, vecdim);
    const std::vector<int> displs_base = ToIntVector(base_displs, vecdim);

    const size_t local_n = base_counts[rank];
    std::vector<float> local_base(local_n * vecdim);
    MPI_Scatterv(base, sendcounts_base.data(), displs_base.data(), MPI_FLOAT,
                 local_base.empty() ? nullptr : local_base.data(),
                 sendcounts_base[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    delete[] base;
    base = nullptr;

    BuildLocalShardIndex(cfg, local_base.data(), local_n, vecdim);

    double recall_sum = 0.0;
    double local_bcast_us = 0.0;
    double local_search_us = 0.0;
    double local_gather_us = 0.0;
    double root_merge_us = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    const double total_start = MPI_Wtime();
    const std::vector<size_t> batch_sizes = MakeBatchSizes(test_number, cfg.mpi_batch);
    const size_t max_batch = batch_sizes.empty() ? 0 : *std::max_element(batch_sizes.begin(), batch_sizes.end());
    std::vector<float> query_batch(max_batch * vecdim);
    std::vector<float> local_distances(max_batch * cfg.k);
    std::vector<uint32_t> local_ids(max_batch * cfg.k);
    std::vector<float> gathered_distances;
    std::vector<uint32_t> gathered_ids;
    std::vector<float> merge_distances(static_cast<size_t>(world_size) * cfg.k);
    std::vector<uint32_t> merge_ids(static_cast<size_t>(world_size) * cfg.k);
    if (rank == 0) {
        gathered_distances.resize(static_cast<size_t>(world_size) * max_batch * cfg.k);
        gathered_ids.resize(static_cast<size_t>(world_size) * max_batch * cfg.k);
    }

    size_t query_offset = 0;
    for (size_t batch_size : batch_sizes) {
        if (rank == 0) {
            std::copy(test_query + query_offset * vecdim,
                      test_query + (query_offset + batch_size) * vecdim,
                      query_batch.begin());
        }

        const double bcast_start = MPI_Wtime();
        MPI_Bcast(query_batch.data(),
                  static_cast<int>(batch_size * vecdim),
                  MPI_FLOAT,
                  0,
                  MPI_COMM_WORLD);
        local_bcast_us += (MPI_Wtime() - bcast_start) * 1e6;

        const double search_start = MPI_Wtime();
        for (size_t qi = 0; qi < batch_size; ++qi) {
            auto local_result = RunLocalShardSearch(cfg,
                                                    local_base.data(),
                                                    query_batch.data() + qi * vecdim,
                                                    local_n,
                                                    vecdim);
            SerializeTopKQueueToBuffers(local_result,
                                        cfg.k,
                                        static_cast<uint32_t>(base_displs[rank]),
                                        local_distances.data() + qi * cfg.k,
                                        local_ids.data() + qi * cfg.k);
        }
        local_search_us += (MPI_Wtime() - search_start) * 1e6;

        const double gather_start = MPI_Wtime();
        MPI_Gather(local_distances.data(),
                   static_cast<int>(batch_size * cfg.k),
                   MPI_FLOAT,
                   rank == 0 ? gathered_distances.data() : nullptr,
                   static_cast<int>(batch_size * cfg.k),
                   MPI_FLOAT,
                   0,
                   MPI_COMM_WORLD);
        MPI_Gather(local_ids.data(),
                   static_cast<int>(batch_size * cfg.k),
                   MPI_UINT32_T,
                   rank == 0 ? gathered_ids.data() : nullptr,
                   static_cast<int>(batch_size * cfg.k),
                   MPI_UINT32_T,
                   0,
                   MPI_COMM_WORLD);
        local_gather_us += (MPI_Wtime() - gather_start) * 1e6;

        if (rank == 0) {
            const double merge_start = MPI_Wtime();
            const size_t rank_block = batch_size * cfg.k;
            for (size_t qi = 0; qi < batch_size; ++qi) {
                for (int r = 0; r < world_size; ++r) {
                    const size_t src = static_cast<size_t>(r) * rank_block + qi * cfg.k;
                    const size_t dst = static_cast<size_t>(r) * cfg.k;
                    std::copy_n(gathered_distances.data() + src,
                                cfg.k,
                                merge_distances.data() + dst);
                    std::copy_n(gathered_ids.data() + src,
                                cfg.k,
                                merge_ids.data() + dst);
                }
                auto merged = MergeTopKArrays(merge_distances.data(),
                                              merge_ids.data(),
                                              static_cast<size_t>(world_size) * cfg.k,
                                              cfg.k);
                recall_sum += ComputeRecallFromQueue(merged,
                                                     test_gt + (query_offset + qi) * test_gt_d,
                                                     cfg.k);
            }
            root_merge_us += (MPI_Wtime() - merge_start) * 1e6;
        }

        query_offset += batch_size;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double total_end = MPI_Wtime();
    double max_bcast_us = 0.0;
    double max_search_us = 0.0;
    double max_gather_us = 0.0;
    MPI_Reduce(&local_bcast_us, &max_bcast_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_search_us, &max_search_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_gather_us, &max_gather_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double total_us = (total_end - total_start) * 1e6;
        PrintSummary(cfg, world_size, base_number, test_number,
                     recall_sum / static_cast<double>(test_number),
                     total_us / static_cast<double>(test_number),
                     total_us);
        std::cout << "stage bcast total (us): " << max_bcast_us << "\n";
        std::cout << "stage search total (us): " << max_search_us << "\n";
        std::cout << "stage gather total (us): " << max_gather_us << "\n";
        std::cout << "stage merge total (us): " << root_merge_us << "\n";
    }

    delete[] test_query;
    delete[] original_test_gt;
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    MPIRunConfig cfg = ParseConfig(argc, argv);
    cfg.requested_thread_level = DetermineRequestedThreadLevel(cfg);

    int provided = 0;
    MPI_Init_thread(&argc, &argv, cfg.requested_thread_level, &provided);
    cfg.provided_thread_level = provided;

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (cfg.mode == "query_shard_rma_openmp_multiple" &&
        cfg.provided_thread_level < MPI_THREAD_MULTIPLE) {
        if (rank == 0) {
            std::cerr << "query_shard_rma_openmp_multiple requires MPI_THREAD_MULTIPLE, but provided "
                      << ThreadLevelName(cfg.provided_thread_level) << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    int exit_code = 0;
    if (IsQueryShardMode(cfg.mode)) {
        exit_code = RunQueryShard(cfg, rank, world_size);
    } else if (IsDataShardMode(cfg.mode)) {
        exit_code = RunDataShard(cfg, rank, world_size);
    } else {
        if (rank == 0) {
            std::cerr << "unknown mpi mode: " << cfg.mode << "\n";
            std::cerr << "supported modes: query_shard_seq, query_shard_pthread, query_shard_openmp, query_shard_nb, query_shard_rma, "
                      << "query_shard_rma_openmp_funneled, query_shard_rma_openmp_multiple, "
                      << "ivf_data_seq, ivf_data_pthread, ivf_data_openmp, "
                      << "ivf_pq_data_seq, ivf_hnsw_data_seq, ivf_hnsw_data_openmp, "
                      << "hnsw_shard_seq, hnsw_on_hnsw_seq\n";
        }
        exit_code = 1;
    }

    MPI_Finalize();
    return exit_code;
}
