#ifndef IVF_MT_H
#define IVF_MT_H

#include <pthread.h>
#include <omp.h>
#include "ivf_neon.h"

// ============================================================
// IVF Query 级并行 (不同线程处理不同 query)
// ============================================================

// ---- Pthread query-level parallelism ----

struct IVFQueryArgs {
    int thread_id;
    float* base;
    float* test_query;
    int* test_gt;
    size_t start_idx;
    size_t end_idx;
    size_t base_number;
    size_t vecdim;
    size_t test_gt_d;
    size_t k;
    size_t nprobe;
    SearchResult* results;
};

static void* ivf_query_worker(void* arg) {
    IVFQueryArgs* args = (IVFQueryArgs*)arg;
    const unsigned long Converter = 1000 * 1000;

    for (size_t i = args->start_idx; i < args->end_idx; ++i) {
        struct timeval val;
        gettimeofday(&val, NULL);

        auto res = ivf_search_neon(args->base,
                                    args->test_query + i * args->vecdim,
                                    args->base_number,
                                    args->vecdim,
                                    args->k,
                                    args->nprobe);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for (size_t j = 0; j < args->k; ++j) {
            gtset.insert(args->test_gt[j + i * args->test_gt_d]);
        }
        size_t acc = 0;
        while (res.size()) {
            if (gtset.find(res.top().second) != gtset.end()) ++acc;
            res.pop();
        }
        args->results[i] = {(float)acc / args->k, diff};
    }
    return NULL;
}

inline void ivf_batch_pthread(float* base, float* test_query, int* test_gt,
                               size_t test_number, size_t base_number,
                               size_t vecdim, size_t test_gt_d,
                               size_t k, int num_threads, size_t nprobe,
                               std::vector<SearchResult>& results) {
    results.resize(test_number);
    pthread_t* threads = new pthread_t[num_threads];
    IVFQueryArgs* args = new IVFQueryArgs[num_threads];
    size_t chunk = (test_number + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].thread_id = t;
        args[t].base = base;
        args[t].test_query = test_query;
        args[t].test_gt = test_gt;
        args[t].start_idx = t * chunk;
        args[t].end_idx = (t + 1) * chunk < test_number ?
                          (t + 1) * chunk : test_number;
        args[t].base_number = base_number;
        args[t].vecdim = vecdim;
        args[t].test_gt_d = test_gt_d;
        args[t].k = k;
        args[t].nprobe = nprobe;
        args[t].results = results.data();

        pthread_create(&threads[t], NULL, ivf_query_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
    delete[] threads;
    delete[] args;
}

// ---- OpenMP query-level parallelism ----

inline void ivf_batch_openmp(float* base, float* test_query, int* test_gt,
                              size_t test_number, size_t base_number,
                              size_t vecdim, size_t test_gt_d,
                              size_t k, int num_threads, size_t nprobe,
                              std::vector<SearchResult>& results) {
    results.resize(test_number);
    const unsigned long Converter = 1000 * 1000;
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < test_number; ++i) {
        struct timeval val;
        gettimeofday(&val, NULL);

        auto res = ivf_search_neon(base, test_query + i * vecdim,
                                    base_number, vecdim, k, nprobe);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for (size_t j = 0; j < k; ++j) {
            gtset.insert(test_gt[j + i * test_gt_d]);
        }
        size_t acc = 0;
        while (res.size()) {
            if (gtset.find(res.top().second) != gtset.end()) ++acc;
            res.pop();
        }
        results[i] = {(float)acc / k, diff};
    }
}

// ============================================================
// IVF 簇划分并行 (per-query: 簇粗排 + 精排并行)
// ============================================================

// ---- Pthread: 精排阶段簇划分 ----

struct IVFClusterArgs {
    float* base;
    float* query;
    const std::vector<uint32_t>* selected_clusters;  // 共享：选中的簇 ID 列表
    size_t start_c;   // 起始簇索引 (在 selected_clusters 中的索引)
    size_t end_c;     // 结束簇索引
    size_t dim;
    size_t k;
    size_t local_p;
    std::priority_queue<std::pair<float, uint32_t>> local_result;
};

static void* ivf_cluster_worker(void* arg) {
    IVFClusterArgs* args = (IVFClusterArgs*)arg;
    auto& q = args->local_result;

    for (size_t ci = args->start_c; ci < args->end_c; ++ci) {
        uint32_t cid = (*args->selected_clusters)[ci];
        for (auto vid : g_ivf.inverted_lists[cid]) {
            const float* v = &args->base[vid * args->dim];
            float* qry = args->query;

            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            float32x4_t sum4 = vdupq_n_f32(0.0f);

            size_t d = 0;
            for (; d + 16 <= args->dim; d += 16) {
                sum1 = vmlaq_f32(sum1, vld1q_f32(&v[d]),      vld1q_f32(&qry[d]));
                sum2 = vmlaq_f32(sum2, vld1q_f32(&v[d + 4]),  vld1q_f32(&qry[d + 4]));
                sum3 = vmlaq_f32(sum3, vld1q_f32(&v[d + 8]),  vld1q_f32(&qry[d + 8]));
                sum4 = vmlaq_f32(sum4, vld1q_f32(&v[d + 12]), vld1q_f32(&qry[d + 12]));
            }
            float32x4_t total = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));
            float32x2_t lo = vget_low_f32(total);
            lo = vpadd_f32(lo, vget_high_f32(total));
            lo = vpadd_f32(lo, lo);
            float ip = vget_lane_f32(lo, 0);
            for (; d < args->dim; ++d) { ip += v[d] * qry[d]; }

            float dist = 1.0f - ip;
            if (q.size() < args->local_p) {
                q.emplace(dist, vid);
            } else if (dist < q.top().first) {
                q.pop();
                q.emplace(dist, vid);
            }
        }
    }
    return NULL;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_cluster_partition_pthread(float* base, float* query, size_t n,
                                      size_t dim, size_t k, int num_threads,
                                      size_t nprobe = 64, size_t local_p = 100) {
    if (!g_ivf.built) { build_ivf_index(base, n, dim); }

    size_t nlist = g_ivf.nlist;

    // Step 1: 粗排 (单线程 — 计算量小)
    std::priority_queue<std::pair<float, uint32_t>> coarse_clusters;
    for (size_t c = 0; c < nlist; ++c) {
        float ip = neon_ip_f32(query, &g_ivf.centroids[c * dim], dim);
        float dist = 1.0f - ip;
        if (coarse_clusters.size() < nprobe) {
            coarse_clusters.emplace(dist, (uint32_t)c);
        } else if (dist < coarse_clusters.top().first) {
            coarse_clusters.pop();
            coarse_clusters.emplace(dist, (uint32_t)c);
        }
    }

    std::vector<uint32_t> selected_clusters;
    selected_clusters.reserve(nprobe);
    while (!coarse_clusters.empty()) {
        selected_clusters.push_back(coarse_clusters.top().second);
        coarse_clusters.pop();
    }

    // Step 2: 精排 (多线程簇划分)
    pthread_t* threads = new pthread_t[num_threads];
    IVFClusterArgs* args = new IVFClusterArgs[num_threads];
    size_t chunk = (selected_clusters.size() + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].base = base;
        args[t].query = query;
        args[t].selected_clusters = &selected_clusters;
        args[t].start_c = t * chunk;
        args[t].end_c = (t + 1) * chunk < selected_clusters.size() ?
                        (t + 1) * chunk : selected_clusters.size();
        args[t].dim = dim;
        args[t].k = k;
        args[t].local_p = local_p;
        args[t].local_result = std::priority_queue<std::pair<float, uint32_t>>();

        pthread_create(&threads[t], NULL, ivf_cluster_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    // merge
    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (int t = 0; t < num_threads; ++t) {
        while (!args[t].local_result.empty()) {
            auto item = args[t].local_result.top();
            args[t].local_result.pop();
            if (merged.size() < k) {
                merged.push(item);
            } else if (item.first < merged.top().first) {
                merged.push(item);
                merged.pop();
            }
        }
    }

    delete[] threads;
    delete[] args;
    return merged;
}

// ---- OpenMP: 精排阶段簇划分 (dynamic schedule) ----

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_cluster_partition_openmp(float* base, float* query, size_t n,
                                     size_t dim, size_t k, int num_threads,
                                     size_t nprobe = 64, size_t local_p = 100) {
    if (!g_ivf.built) { build_ivf_index(base, n, dim); }

    size_t nlist = g_ivf.nlist;

    // Step 1: 粗排
    std::priority_queue<std::pair<float, uint32_t>> coarse_clusters;
    for (size_t c = 0; c < nlist; ++c) {
        float ip = neon_ip_f32(query, &g_ivf.centroids[c * dim], dim);
        float dist = 1.0f - ip;
        if (coarse_clusters.size() < nprobe) {
            coarse_clusters.emplace(dist, (uint32_t)c);
        } else if (dist < coarse_clusters.top().first) {
            coarse_clusters.pop();
            coarse_clusters.emplace(dist, (uint32_t)c);
        }
    }

    std::vector<uint32_t> selected_clusters;
    selected_clusters.reserve(nprobe);
    while (!coarse_clusters.empty()) {
        selected_clusters.push_back(coarse_clusters.top().second);
        coarse_clusters.pop();
    }

    // Step 2: 精排 (OpenMP dynamic — 应对负载不均)
    omp_set_num_threads(num_threads);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(num_threads);

    size_t n_selected = selected_clusters.size();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& q = local_results[tid];

        #pragma omp for schedule(dynamic, 1)
        for (size_t ci = 0; ci < n_selected; ++ci) {
            uint32_t cid = selected_clusters[ci];
            for (auto vid : g_ivf.inverted_lists[cid]) {
                const float* v = &base[vid * dim];

                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);
                float32x4_t sum4 = vdupq_n_f32(0.0f);

                size_t d = 0;
                for (; d + 16 <= dim; d += 16) {
                    sum1 = vmlaq_f32(sum1, vld1q_f32(&v[d]),      vld1q_f32(&query[d]));
                    sum2 = vmlaq_f32(sum2, vld1q_f32(&v[d + 4]),  vld1q_f32(&query[d + 4]));
                    sum3 = vmlaq_f32(sum3, vld1q_f32(&v[d + 8]),  vld1q_f32(&query[d + 8]));
                    sum4 = vmlaq_f32(sum4, vld1q_f32(&v[d + 12]), vld1q_f32(&query[d + 12]));
                }
                float32x4_t total = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));
                float32x2_t lo = vget_low_f32(total);
                lo = vpadd_f32(lo, vget_high_f32(total));
                lo = vpadd_f32(lo, lo);
                float ip = vget_lane_f32(lo, 0);
                for (; d < dim; ++d) { ip += v[d] * query[d]; }

                float dist = 1.0f - ip;
                if (q.size() < local_p) {
                    q.emplace(dist, vid);
                } else if (dist < q.top().first) {
                    q.pop();
                    q.emplace(dist, vid);
                }
            }
        }
    }

    // merge
    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (int t = 0; t < num_threads; ++t) {
        while (!local_results[t].empty()) {
            auto item = local_results[t].top();
            local_results[t].pop();
            if (merged.size() < k) {
                merged.push(item);
            } else if (item.first < merged.top().first) {
                merged.push(item);
                merged.pop();
            }
        }
    }

    return merged;
}

// ============================================================
// IVF centroid 粗排并行
// ============================================================

inline void merge_bounded_heap(std::priority_queue<std::pair<float, uint32_t>>& dst,
                               std::priority_queue<std::pair<float, uint32_t>>& src,
                               size_t limit) {
    while (!src.empty()) {
        auto item = src.top();
        src.pop();
        if (dst.size() < limit) {
            dst.push(item);
        } else if (item.first < dst.top().first) {
            dst.pop();
            dst.push(item);
        }
    }
}

struct IVFCentroidArgs {
    float* query;
    size_t start_c;
    size_t end_c;
    size_t dim;
    size_t nprobe;
    std::priority_queue<std::pair<float, uint32_t>> local_clusters;
};

static void* ivf_centroid_worker(void* arg) {
    IVFCentroidArgs* args = (IVFCentroidArgs*)arg;
    auto& q = args->local_clusters;

    for (size_t c = args->start_c; c < args->end_c; ++c) {
        float ip = neon_ip_f32(args->query, &g_ivf.centroids[c * args->dim], args->dim);
        float dist = 1.0f - ip;
        if (q.size() < args->nprobe) {
            q.emplace(dist, (uint32_t)c);
        } else if (dist < q.top().first) {
            q.pop();
            q.emplace(dist, (uint32_t)c);
        }
    }
    return NULL;
}

inline std::vector<uint32_t>
ivf_select_centroids_pthread(float* query, size_t dim, int num_threads,
                             size_t nprobe) {
    size_t nlist = g_ivf.nlist;
    nprobe = std::min(nprobe, nlist);

    pthread_t* threads = new pthread_t[num_threads];
    IVFCentroidArgs* args = new IVFCentroidArgs[num_threads];
    size_t chunk = (nlist + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].query = query;
        args[t].start_c = t * chunk;
        args[t].end_c = std::min((size_t)(t + 1) * chunk, nlist);
        args[t].dim = dim;
        args[t].nprobe = nprobe;
        args[t].local_clusters = std::priority_queue<std::pair<float, uint32_t>>();
        pthread_create(&threads[t], NULL, ivf_centroid_worker, &args[t]);
    }

    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
        merge_bounded_heap(merged, args[t].local_clusters, nprobe);
    }

    std::vector<uint32_t> selected_clusters;
    selected_clusters.reserve(merged.size());
    while (!merged.empty()) {
        selected_clusters.push_back(merged.top().second);
        merged.pop();
    }

    delete[] threads;
    delete[] args;
    return selected_clusters;
}

inline std::vector<uint32_t>
ivf_select_centroids_openmp(float* query, size_t dim, int num_threads,
                            size_t nprobe) {
    size_t nlist = g_ivf.nlist;
    nprobe = std::min(nprobe, nlist);
    omp_set_num_threads(num_threads);

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_clusters(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& q = local_clusters[tid];

        #pragma omp for schedule(static)
        for (size_t c = 0; c < nlist; ++c) {
            float ip = neon_ip_f32(query, &g_ivf.centroids[c * dim], dim);
            float dist = 1.0f - ip;
            if (q.size() < nprobe) {
                q.emplace(dist, (uint32_t)c);
            } else if (dist < q.top().first) {
                q.pop();
                q.emplace(dist, (uint32_t)c);
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (int t = 0; t < num_threads; ++t) {
        merge_bounded_heap(merged, local_clusters[t], nprobe);
    }

    std::vector<uint32_t> selected_clusters;
    selected_clusters.reserve(merged.size());
    while (!merged.empty()) {
        selected_clusters.push_back(merged.top().second);
        merged.pop();
    }
    return selected_clusters;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_centroid_partition_pthread(float* base, float* query, size_t n,
                                      size_t dim, size_t k, int num_threads,
                                      size_t nprobe = 64, size_t local_p = 100) {
    if (!g_ivf.built) { build_ivf_index(base, n, dim); }

    std::vector<uint32_t> selected_clusters =
        ivf_select_centroids_pthread(query, dim, num_threads, nprobe);

    pthread_t* threads = new pthread_t[num_threads];
    IVFClusterArgs* args = new IVFClusterArgs[num_threads];
    size_t chunk = (selected_clusters.size() + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].base = base;
        args[t].query = query;
        args[t].selected_clusters = &selected_clusters;
        args[t].start_c = t * chunk;
        args[t].end_c = std::min((size_t)(t + 1) * chunk, selected_clusters.size());
        args[t].dim = dim;
        args[t].k = k;
        args[t].local_p = local_p;
        args[t].local_result = std::priority_queue<std::pair<float, uint32_t>>();

        pthread_create(&threads[t], NULL, ivf_cluster_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (int t = 0; t < num_threads; ++t) {
        merge_bounded_heap(merged, args[t].local_result, k);
    }

    delete[] threads;
    delete[] args;
    return merged;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_centroid_partition_openmp(float* base, float* query, size_t n,
                                     size_t dim, size_t k, int num_threads,
                                     size_t nprobe = 64, size_t local_p = 100) {
    if (!g_ivf.built) { build_ivf_index(base, n, dim); }

    std::vector<uint32_t> selected_clusters =
        ivf_select_centroids_openmp(query, dim, num_threads, nprobe);

    omp_set_num_threads(num_threads);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(num_threads);
    size_t n_selected = selected_clusters.size();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& q = local_results[tid];

        #pragma omp for schedule(dynamic, 1)
        for (size_t ci = 0; ci < n_selected; ++ci) {
            uint32_t cid = selected_clusters[ci];
            for (auto vid : g_ivf.inverted_lists[cid]) {
                const float* v = &base[vid * dim];
                float ip = neon_ip_f32(v, query, dim);
                float dist = 1.0f - ip;

                if (q.size() < local_p) {
                    q.emplace(dist, vid);
                } else if (dist < q.top().first) {
                    q.pop();
                    q.emplace(dist, vid);
                }
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (int t = 0; t < num_threads; ++t) {
        merge_bounded_heap(merged, local_results[t], k);
    }

    return merged;
}

// ============================================================
// IVF-PQ 多线程优化
// ============================================================

#include "ivf_pq.h"

// ---- IVF-PQ query 级并行 (Pthread) ----

struct IVFPQQueryArgs {
    float* base;
    float* test_query;
    int* test_gt;
    size_t start_idx;
    size_t end_idx;
    size_t base_number;
    size_t vecdim;
    size_t test_gt_d;
    size_t k;
    size_t nprobe;
    SearchResult* results;
};

static void* ivf_pq_query_worker(void* arg) {
    IVFPQQueryArgs* args = (IVFPQQueryArgs*)arg;
    const unsigned long Converter = 1000 * 1000;

    for (size_t i = args->start_idx; i < args->end_idx; ++i) {
        struct timeval val;
        gettimeofday(&val, NULL);

        auto res = ivf_pq_search(args->base,
                                  args->test_query + i * args->vecdim,
                                  args->base_number,
                                  args->vecdim,
                                  args->k,
                                  args->nprobe,
                                  550);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for (size_t j = 0; j < args->k; ++j) {
            gtset.insert(args->test_gt[j + i * args->test_gt_d]);
        }
        size_t acc = 0;
        while (res.size()) {
            if (gtset.find(res.top().second) != gtset.end()) ++acc;
            res.pop();
        }
        args->results[i] = {(float)acc / args->k, diff};
    }
    return NULL;
}

inline void ivf_pq_batch_pthread(float* base, float* test_query, int* test_gt,
                                  size_t test_number, size_t base_number,
                                  size_t vecdim, size_t test_gt_d,
                                  size_t k, int num_threads, size_t nprobe,
                                  std::vector<SearchResult>& results) {
    results.resize(test_number);
    pthread_t* threads = new pthread_t[num_threads];
    IVFPQQueryArgs* args = new IVFPQQueryArgs[num_threads];
    size_t chunk = (test_number + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].base = base;
        args[t].test_query = test_query;
        args[t].test_gt = test_gt;
        args[t].start_idx = t * chunk;
        args[t].end_idx = (t + 1) * chunk < test_number ?
                          (t + 1) * chunk : test_number;
        args[t].base_number = base_number;
        args[t].vecdim = vecdim;
        args[t].test_gt_d = test_gt_d;
        args[t].k = k;
        args[t].nprobe = nprobe;
        args[t].results = results.data();

        pthread_create(&threads[t], NULL, ivf_pq_query_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
    delete[] threads;
    delete[] args;
}

// ---- IVF-PQ query 级并行 (OpenMP) ----

inline void ivf_pq_batch_openmp(float* base, float* test_query, int* test_gt,
                                 size_t test_number, size_t base_number,
                                 size_t vecdim, size_t test_gt_d,
                                 size_t k, int num_threads, size_t nprobe,
                                 std::vector<SearchResult>& results) {
    results.resize(test_number);
    const unsigned long Converter = 1000 * 1000;
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < test_number; ++i) {
        struct timeval val;
        gettimeofday(&val, NULL);

        auto res = ivf_pq_search(base, test_query + i * vecdim,
                                  base_number, vecdim, k, nprobe, 550);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for (size_t j = 0; j < k; ++j) {
            gtset.insert(test_gt[j + i * test_gt_d]);
        }
        size_t acc = 0;
        while (res.size()) {
            if (gtset.find(res.top().second) != gtset.end()) ++acc;
            res.pop();
        }
        results[i] = {(float)acc / k, diff};
    }
}

#endif // IVF_MT_H
