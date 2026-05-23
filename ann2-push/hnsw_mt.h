#pragma once

#include <algorithm>
#include <cstdint>
#include <pthread.h>
#include <queue>
#include <set>
#include <sys/time.h>
#include <vector>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan_mt.h"

struct HNSWIndexState {
    hnswlib::InnerProductSpace* space;
    hnswlib::HierarchicalNSW<float>* index;
    size_t n;
    size_t dim;
    size_t ef_search;
    bool built;

    HNSWIndexState()
        : space(nullptr), index(nullptr), n(0), dim(0), ef_search(0), built(false) {}
};

static HNSWIndexState g_hnsw_index;

inline void reset_hnsw_index() {
    delete g_hnsw_index.index;
    delete g_hnsw_index.space;
    g_hnsw_index.index = nullptr;
    g_hnsw_index.space = nullptr;
    g_hnsw_index.n = 0;
    g_hnsw_index.dim = 0;
    g_hnsw_index.ef_search = 0;
    g_hnsw_index.built = false;
}

inline void build_hnsw_index(float* base, size_t n, size_t dim,
                             int build_threads = 1,
                             size_t m = 16,
                             size_t ef_construction = 150,
                             size_t ef_search = 100) {
    const size_t effective_ef = std::max(ef_search, size_t(32));

    if (!g_hnsw_index.built || g_hnsw_index.n != n || g_hnsw_index.dim != dim) {
        reset_hnsw_index();

        g_hnsw_index.space = new hnswlib::InnerProductSpace(dim);
        g_hnsw_index.index = new hnswlib::HierarchicalNSW<float>(
            g_hnsw_index.space, n, m, ef_construction);

        omp_set_num_threads(build_threads);
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            g_hnsw_index.index->addPoint(base + static_cast<size_t>(i) * dim,
                                         static_cast<hnswlib::labeltype>(i));
        }

        g_hnsw_index.n = n;
        g_hnsw_index.dim = dim;
        g_hnsw_index.built = true;
    }

    if (g_hnsw_index.ef_search != effective_ef) {
        g_hnsw_index.index->setEf(effective_ef);
        g_hnsw_index.ef_search = effective_ef;
    }
}

inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_prebuilt(float* query, size_t k) {
    auto raw = g_hnsw_index.index->searchKnn(query, k);
    std::priority_queue<std::pair<float, uint32_t>> res;
    while (!raw.empty()) {
        res.emplace(raw.top().first, static_cast<uint32_t>(raw.top().second));
        raw.pop();
    }
    return res;
}

inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search(float* base, float* query, size_t n, size_t dim, size_t k,
            size_t ef_search = 100, int build_threads = 1) {
    build_hnsw_index(base, n, dim, build_threads, 16, 150, ef_search);
    return hnsw_search_prebuilt(query, k);
}

inline SearchResult hnsw_search_with_stats(float* query, int* gt_row, size_t k) {
    const unsigned long converter = 1000 * 1000;
    struct timeval val;
    gettimeofday(&val, NULL);

    auto res = hnsw_search_prebuilt(query, k);

    struct timeval new_val;
    gettimeofday(&new_val, NULL);
    int64_t diff = (new_val.tv_sec * converter + new_val.tv_usec) -
                   (val.tv_sec * converter + val.tv_usec);

    std::set<uint32_t> gtset;
    for (size_t j = 0; j < k; ++j) {
        gtset.insert(static_cast<uint32_t>(gt_row[j]));
    }

    size_t acc = 0;
    while (!res.empty()) {
        if (gtset.find(res.top().second) != gtset.end()) {
            ++acc;
        }
        res.pop();
    }

    return {(float)acc / k, diff};
}

struct HNSWQueryArgs {
    float* test_query;
    int* test_gt;
    size_t start_idx;
    size_t end_idx;
    size_t vecdim;
    size_t test_gt_d;
    size_t k;
    SearchResult* results;
};

static void* hnsw_query_worker(void* arg) {
    HNSWQueryArgs* args = (HNSWQueryArgs*)arg;
    for (size_t i = args->start_idx; i < args->end_idx; ++i) {
        args->results[i] = hnsw_search_with_stats(
            args->test_query + i * args->vecdim,
            args->test_gt + i * args->test_gt_d,
            args->k);
    }
    return NULL;
}

inline void hnsw_batch_pthread(float* base, float* test_query, int* test_gt,
                               size_t test_number, size_t base_number,
                               size_t vecdim, size_t test_gt_d,
                               size_t k, int num_threads,
                               std::vector<SearchResult>& results,
                               size_t ef_search = 100) {
    build_hnsw_index(base, base_number, vecdim, 1, 16, 150, ef_search);

    results.resize(test_number);
    pthread_t* threads = new pthread_t[num_threads];
    HNSWQueryArgs* args = new HNSWQueryArgs[num_threads];
    size_t chunk = (test_number + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].test_query = test_query;
        args[t].test_gt = test_gt;
        args[t].start_idx = t * chunk;
        args[t].end_idx = std::min((t + 1) * chunk, test_number);
        args[t].vecdim = vecdim;
        args[t].test_gt_d = test_gt_d;
        args[t].k = k;
        args[t].results = results.data();
        pthread_create(&threads[t], NULL, hnsw_query_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    delete[] threads;
    delete[] args;
}

inline void hnsw_batch_openmp(float* base, float* test_query, int* test_gt,
                              size_t test_number, size_t base_number,
                              size_t vecdim, size_t test_gt_d,
                              size_t k, int num_threads,
                              std::vector<SearchResult>& results,
                              size_t ef_search = 100) {
    build_hnsw_index(base, base_number, vecdim, 1, 16, 150, ef_search);

    results.resize(test_number);
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(test_number); ++i) {
        size_t idx = static_cast<size_t>(i);
        results[idx] = hnsw_search_with_stats(
            test_query + idx * vecdim,
            test_gt + idx * test_gt_d,
            k);
    }
}
