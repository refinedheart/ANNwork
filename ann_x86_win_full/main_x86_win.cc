#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "common.h"
#include "distance_x86.h"
#include "flat_x86.h"
#include "hnsw_x86.h"
#include "ivf_x86.h"

static void PrintUsage() {
    std::cerr
        << "Usage: ann_x86_win <mode> [num_threads]\n"
        << "Modes:\n"
        << "  serial, sse4, sse8, sse16, avx8, avx16, auto\n"
        << "  thread_flat, openmp_flat, thread_flat_part, openmp_flat_part\n"
        << "  ivf, ivf_pq, thread_ivf_pq, openmp_ivf_pq\n"
        << "  hnsw, thread_hnsw, openmp_hnsw\n";
}

static KernelKind ParseKernel(const std::string& mode) {
    if (mode == "serial") return KernelKind::Serial;
    if (mode == "sse4") return KernelKind::SSE4P;
    if (mode == "sse8") return KernelKind::SSE8P;
    if (mode == "sse16") return KernelKind::SSE16P;
    if (mode == "avx8") return KernelKind::AVX8P;
    if (mode == "avx16") return KernelKind::AVX16P;
    if (mode == "auto") return KernelKind::AutoVec;
    return KernelKind::AVX16P;
}

static void PrintSummary(const std::string& mode, int threads, const std::vector<SearchResult>& results, long long total_us) {
    float avg_recall = 0.0f;
    double avg_latency = 0.0;
    for (const auto& result : results) {
        avg_recall += result.recall;
        avg_latency += static_cast<double>(result.latency_us);
    }
    avg_recall /= static_cast<float>(results.size());
    avg_latency /= static_cast<double>(results.size());

    std::cout << "mode: " << mode << ", threads: " << threads << "\n";
    std::cout << "average recall: " << avg_recall << "\n";
    std::cout << "average latency (us): " << avg_latency << "\n";
    std::cout << "total time (us): " << total_us << "\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            PrintUsage();
            return 1;
        }

        std::string mode = argv[1];
        int num_threads = 8;
        if (argc >= 3) {
            num_threads = std::stoi(argv[2]);
        }

        const char* env_path = std::getenv("ANN_DATA_PATH");
        std::string data_path = env_path ? env_path : "./anndata";
        if (!data_path.empty() && data_path.back() != '/' && data_path.back() != '\\') {
            data_path.push_back('/');
        }

        size_t query_number = 0, base_number = 0;
        size_t gt_dim = 0, dim = 0;
        auto test_query_holder = std::unique_ptr<float[]>(LoadData<float>(data_path + "DEEP100K.query.fbin", query_number, dim));
        auto test_gt_holder = std::unique_ptr<int[]>(LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", query_number, gt_dim));
        auto base_holder = std::unique_ptr<float[]>(LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, dim));

        size_t query_limit = ReadEnvSizeT("ANN_QUERY_LIMIT", 2000);
        size_t rerank_p = ReadEnvSizeT("ANN_PQ_RERANK", 550);
        size_t ivf_nlist = ReadEnvSizeT("ANN_IVF_NLIST", 128);
        size_t ivf_nprobe = ReadEnvSizeT("ANN_IVF_NPROBE", 32);
        size_t hnsw_ef = ReadEnvSizeT("ANN_HNSW_EF", 64);
        query_number = std::min(query_number, query_limit);
        if (query_number == 0) {
            throw std::runtime_error("no queries selected");
        }

        float* base = base_holder.get();
        float* test_query = test_query_holder.get();
        int* test_gt = test_gt_holder.get();
        const size_t k = 10;

        if (mode == "ivf" || mode == "ivf_pq" || mode == "thread_ivf_pq" || mode == "openmp_ivf_pq") {
            build_ivf_x86(base, base_number, dim, ivf_nlist);
        }
        if (mode == "ivf_pq" || mode == "thread_ivf_pq" || mode == "openmp_ivf_pq") {
            build_pq_x86(base, base_number, dim);
        }
        if (mode == "hnsw" || mode == "thread_hnsw" || mode == "openmp_hnsw") {
            build_hnsw_x86(base, base_number, dim, hnsw_ef);
        }

        std::vector<SearchResult> results;
        results.reserve(query_number);
        auto total_start = std::chrono::steady_clock::now();

        if (mode == "serial" || mode == "sse4" || mode == "sse8" || mode == "sse16" || mode == "avx8" || mode == "avx16" || mode == "auto") {
            KernelKind kernel = ParseKernel(mode);
            results.resize(query_number);
            for (size_t i = 0; i < query_number; ++i) {
                results[i] = RunSingleFlat(base, test_query + i * dim, test_gt + i * gt_dim, base_number, dim, k, kernel);
            }
        } else if (mode == "thread_flat") {
            results = flat_batch_threads(base, test_query, test_gt, query_number, base_number, dim, gt_dim, k, num_threads, KernelKind::AVX16P);
        } else if (mode == "openmp_flat") {
            results = flat_batch_openmp(base, test_query, test_gt, query_number, base_number, dim, gt_dim, k, num_threads, KernelKind::AVX16P);
        } else if (mode == "thread_flat_part") {
            results.resize(query_number);
            for (size_t i = 0; i < query_number; ++i) {
                results[i] = RunSingleFlatPartitionThreads(base, test_query + i * dim, test_gt + i * gt_dim,
                                                           base_number, dim, k, num_threads, KernelKind::AVX16P, rerank_p);
            }
        } else if (mode == "openmp_flat_part") {
            results.resize(query_number);
            for (size_t i = 0; i < query_number; ++i) {
                results[i] = RunSingleFlatPartitionOpenMP(base, test_query + i * dim, test_gt + i * gt_dim,
                                                          base_number, dim, k, num_threads, KernelKind::AVX16P, rerank_p);
            }
        } else if (mode == "ivf") {
            results.resize(query_number);
            for (size_t i = 0; i < query_number; ++i) {
                auto start = std::chrono::steady_clock::now();
                auto res = ivf_search_x86(base, test_query + i * dim, base_number, dim, k, ivf_nprobe, KernelKind::AVX16P);
                auto end = std::chrono::steady_clock::now();
                results[i] = {ComputeRecallTopK(res, test_gt + i * gt_dim, k), ElapsedUs(start, end)};
            }
        } else if (mode == "ivf_pq") {
            results.resize(query_number);
            for (size_t i = 0; i < query_number; ++i) {
                results[i] = RunSingleIVFPQX86(base, test_query + i * dim, test_gt + i * gt_dim,
                                               base_number, dim, k, ivf_nprobe, rerank_p, KernelKind::AVX16P);
            }
        } else if (mode == "thread_ivf_pq") {
            results = ivf_pq_batch_threads(base, test_query, test_gt, query_number, base_number, dim, gt_dim,
                                           k, num_threads, ivf_nprobe, rerank_p, KernelKind::AVX16P);
        } else if (mode == "openmp_ivf_pq") {
            results = ivf_pq_batch_openmp(base, test_query, test_gt, query_number, base_number, dim, gt_dim,
                                          k, num_threads, ivf_nprobe, rerank_p, KernelKind::AVX16P);
        } else if (mode == "hnsw") {
            results.resize(query_number);
            for (size_t i = 0; i < query_number; ++i) {
                results[i] = RunSingleHNSWX86(test_query + i * dim, test_gt + i * gt_dim, k);
            }
        } else if (mode == "thread_hnsw") {
            results = hnsw_batch_threads_x86(base, test_query, test_gt, query_number, base_number, dim, gt_dim, k, num_threads, hnsw_ef);
        } else if (mode == "openmp_hnsw") {
            results = hnsw_batch_openmp_x86(base, test_query, test_gt, query_number, base_number, dim, gt_dim, k, num_threads, hnsw_ef);
        } else {
            PrintUsage();
            return 1;
        }

        auto total_end = std::chrono::steady_clock::now();
        PrintSummary(mode, num_threads, results, ElapsedUs(total_start, total_end));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << "\n";
        return 1;
    }
}
