#pragma once

#include <omp.h>
#include <thread>
#include <vector>
#include "vendor/hnswlib/hnswlib/hnswlib.h"
#include "common.h"

struct HNSWStateX86 {
    hnswlib::InnerProductSpace* space = nullptr;
    hnswlib::HierarchicalNSW<float>* index = nullptr;
    size_t n = 0;
    size_t dim = 0;
    size_t ef_search = 0;
    bool built = false;
};

static HNSWStateX86 g_hnsw_x86;

inline void reset_hnsw_x86() {
    delete g_hnsw_x86.index;
    delete g_hnsw_x86.space;
    g_hnsw_x86 = HNSWStateX86{};
}

inline void build_hnsw_x86(float* base, size_t n, size_t dim, size_t ef_search,
                           size_t m = 16, size_t ef_construction = 150) {
    size_t effective_ef = std::max<size_t>(ef_search, 32);
    if (!g_hnsw_x86.built || g_hnsw_x86.n != n || g_hnsw_x86.dim != dim) {
        reset_hnsw_x86();
        g_hnsw_x86.space = new hnswlib::InnerProductSpace(dim);
        g_hnsw_x86.index = new hnswlib::HierarchicalNSW<float>(g_hnsw_x86.space, n, m, ef_construction);
        for (size_t i = 0; i < n; ++i) {
            g_hnsw_x86.index->addPoint(base + i * dim, static_cast<hnswlib::labeltype>(i));
        }
        g_hnsw_x86.n = n;
        g_hnsw_x86.dim = dim;
        g_hnsw_x86.built = true;
    }
    if (g_hnsw_x86.ef_search != effective_ef) {
        g_hnsw_x86.index->setEf(effective_ef);
        g_hnsw_x86.ef_search = effective_ef;
    }
}

inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_x86(const float* query, size_t k) {
    auto raw = g_hnsw_x86.index->searchKnn(query, k);
    std::priority_queue<std::pair<float, uint32_t>> res;
    while (!raw.empty()) {
        res.emplace(raw.top().first, static_cast<uint32_t>(raw.top().second));
        raw.pop();
    }
    return res;
}

inline SearchResult RunSingleHNSWX86(const float* query, const int* gt_row, size_t k) {
    auto start = std::chrono::steady_clock::now();
    auto res = hnsw_search_x86(query, k);
    auto end = std::chrono::steady_clock::now();
    return {ComputeRecallTopK(res, gt_row, k), ElapsedUs(start, end)};
}

inline std::vector<SearchResult>
hnsw_batch_threads_x86(float* base, const float* test_query, const int* test_gt,
                       size_t test_number, size_t base_number, size_t dim, size_t gt_dim, size_t k,
                       int num_threads, size_t ef_search) {
    build_hnsw_x86(base, base_number, dim, ef_search);
    std::vector<SearchResult> results(test_number);
    std::vector<std::thread> workers;
    size_t chunk = (test_number + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            size_t begin = static_cast<size_t>(t) * chunk;
            size_t end = std::min(begin + chunk, test_number);
            for (size_t i = begin; i < end; ++i) {
                results[i] = RunSingleHNSWX86(test_query + i * dim, test_gt + i * gt_dim, k);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return results;
}

inline std::vector<SearchResult>
hnsw_batch_openmp_x86(float* base, const float* test_query, const int* test_gt,
                      size_t test_number, size_t base_number, size_t dim, size_t gt_dim, size_t k,
                      int num_threads, size_t ef_search) {
    build_hnsw_x86(base, base_number, dim, ef_search);
    std::vector<SearchResult> results(test_number);
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(test_number); ++i) {
        size_t idx = static_cast<size_t>(i);
        results[idx] = RunSingleHNSWX86(test_query + idx * dim, test_gt + idx * gt_dim, k);
    }
    return results;
}
