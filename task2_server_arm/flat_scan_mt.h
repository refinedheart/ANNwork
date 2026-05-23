#pragma once

#include <cstdint>
#include <pthread.h>
#include <omp.h>
#include <arm_neon.h>
#include <vector>
#include <queue>
#include <set>
#include <sys/time.h>
#include "flat_scan_pq_adc_neon.h"

struct SearchResult {
    float recall;
    int64_t latency;
};

// ============================================================
// Phase 2: PQ LUT 构建阶段并行化
// ============================================================

// Pthread: parallel LUT computation (subspace-level parallelism)
struct LutThreadArgs {
    float* query;
    PQParams* pq;
    float* lut;
    size_t m_start;
    size_t m_end;
};

static void* lut_compute_worker(void* arg) {
    LutThreadArgs* args = (LutThreadArgs*)arg;
    for (size_t m = args->m_start; m < args->m_end; ++m) {
        const float* qs = &args->query[m * PQ_DSUB];
        float* lm = &args->lut[m * PQ_KS];
        const float* cents = &args->pq->cent[m * PQ_KS * PQ_DSUB];

        size_t k = 0;
        for (; k + 4 <= PQ_KS; k += 4) {
            lm[k+0] = neon_ip_f32(qs, &cents[(k+0) * PQ_DSUB], PQ_DSUB);
            lm[k+1] = neon_ip_f32(qs, &cents[(k+1) * PQ_DSUB], PQ_DSUB);
            lm[k+2] = neon_ip_f32(qs, &cents[(k+2) * PQ_DSUB], PQ_DSUB);
            lm[k+3] = neon_ip_f32(qs, &cents[(k+3) * PQ_DSUB], PQ_DSUB);
        }
        for (; k < PQ_KS; ++k) {
            lm[k] = neon_ip_f32(qs, &cents[k * PQ_DSUB], PQ_DSUB);
        }
    }
    return NULL;
}

inline void pq_compute_lut_pthread(float* query, PQParams& pq, float* lut,
                                    int num_threads) {
    pthread_t* threads = new pthread_t[num_threads];
    LutThreadArgs* args = new LutThreadArgs[num_threads];
    size_t chunk = (PQ_M + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].query = query;
        args[t].pq = &pq;
        args[t].lut = lut;
        args[t].m_start = t * chunk;
        args[t].m_end = (t + 1) * chunk < PQ_M ? (t + 1) * chunk : PQ_M;
        pthread_create(&threads[t], NULL, lut_compute_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
    delete[] threads;
    delete[] args;
}

// OpenMP: parallel LUT computation
inline void pq_compute_lut_openmp(float* query, PQParams& pq, float* lut,
                                   int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < PQ_M; ++m) {
        const float* qs = &query[m * PQ_DSUB];
        float* lm = &lut[m * PQ_KS];
        const float* cents = &pq.cent[m * PQ_KS * PQ_DSUB];

        size_t k = 0;
        for (; k + 4 <= PQ_KS; k += 4) {
            lm[k+0] = neon_ip_f32(qs, &cents[(k+0) * PQ_DSUB], PQ_DSUB);
            lm[k+1] = neon_ip_f32(qs, &cents[(k+1) * PQ_DSUB], PQ_DSUB);
            lm[k+2] = neon_ip_f32(qs, &cents[(k+2) * PQ_DSUB], PQ_DSUB);
            lm[k+3] = neon_ip_f32(qs, &cents[(k+3) * PQ_DSUB], PQ_DSUB);
        }
        for (; k < PQ_KS; ++k) {
            lm[k] = neon_ip_f32(qs, &cents[k * PQ_DSUB], PQ_DSUB);
        }
    }
}

// Search variants with parallel LUT + single-thread scan
inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_pq_lut_pthread(float* base, float* query, size_t n, size_t dim,
                            size_t k, int num_threads, size_t p = 550) {
    if (!g_pq_kmeans_built) { build_pq_kmeans_neon(base, n, dim); }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_pthread(query, g_pq_kmeans, lut, num_threads);

    // single-thread coarse scan (same as original)
    std::priority_queue<std::pair<float, uint32_t>> coarse;
    for (size_t i = 0; i < n; ++i) {
        uint8_t* code = &g_pq_kmeans.codes[i * PQ_M];
        float acc = 0.0f;
        for (size_t m = 0; m < PQ_M; ++m) {
            acc += lut[m * PQ_KS + code[m]];
        }
        float dist = 1.0f - acc;
        if (coarse.size() < p) { coarse.emplace(dist, (uint32_t)i); }
        else if (dist < coarse.top().first) { coarse.pop(); coarse.emplace(dist, (uint32_t)i); }
    }
    delete[] lut;

    // rerank
    std::vector<std::pair<float, uint32_t>> cands;
    cands.reserve(coarse.size());
    while (!coarse.empty()) { cands.push_back(coarse.top()); coarse.pop(); }

    std::priority_queue<std::pair<float, uint32_t>> res;
    for (auto& cand : cands) {
        size_t idx = cand.second;
        float ip = ip_exact_adc_neon(&base[idx * dim], query, dim);
        float dist = 1.0f - ip;
        if (res.size() < k) { res.emplace(dist, (uint32_t)idx); }
        else if (dist < res.top().first) { res.pop(); res.emplace(dist, (uint32_t)idx); }
    }
    return res;
}

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_pq_lut_openmp(float* base, float* query, size_t n, size_t dim,
                           size_t k, int num_threads, size_t p = 550) {
    if (!g_pq_kmeans_built) { build_pq_kmeans_neon(base, n, dim); }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_openmp(query, g_pq_kmeans, lut, num_threads);

    std::priority_queue<std::pair<float, uint32_t>> coarse;
    for (size_t i = 0; i < n; ++i) {
        uint8_t* code = &g_pq_kmeans.codes[i * PQ_M];
        float acc = 0.0f;
        for (size_t m = 0; m < PQ_M; ++m) {
            acc += lut[m * PQ_KS + code[m]];
        }
        float dist = 1.0f - acc;
        if (coarse.size() < p) { coarse.emplace(dist, (uint32_t)i); }
        else if (dist < coarse.top().first) { coarse.pop(); coarse.emplace(dist, (uint32_t)i); }
    }
    delete[] lut;

    std::vector<std::pair<float, uint32_t>> cands;
    cands.reserve(coarse.size());
    while (!coarse.empty()) { cands.push_back(coarse.top()); coarse.pop(); }

    std::priority_queue<std::pair<float, uint32_t>> res;
    for (auto& cand : cands) {
        size_t idx = cand.second;
        float ip = ip_exact_adc_neon(&base[idx * dim], query, dim);
        float dist = 1.0f - ip;
        if (res.size() < k) { res.emplace(dist, (uint32_t)idx); }
        else if (dist < res.top().first) { res.pop(); res.emplace(dist, (uint32_t)idx); }
    }
    return res;
}

// ---- Pthread query-level parallelism ----

struct PthreadWorkerArgs {
    int thread_id;
    int num_threads;
    float* base;
    float* test_query;
    int* test_gt;
    size_t start_idx;
    size_t end_idx;
    size_t base_number;
    size_t vecdim;
    size_t test_gt_d;
    size_t k;
    SearchResult* results;
};

static void* pthread_query_worker(void* arg) {
    PthreadWorkerArgs* args = (PthreadWorkerArgs*)arg;

    const unsigned long Converter = 1000 * 1000;

    for (size_t i = args->start_idx; i < args->end_idx; ++i) {
        struct timeval val;
        gettimeofday(&val, NULL);

        auto res = flat_search_pq_adc_neon_twostage(
            args->base,
            args->test_query + i * args->vecdim,
            args->base_number,
            args->vecdim,
            args->k,
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
            int x = res.top().second;
            if (gtset.find(x) != gtset.end()) {
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc / args->k;

        args->results[i] = {recall, diff};
    }

    return NULL;
}

inline void batch_search_pthread(float* base, float* test_query, int* test_gt,
                                  size_t test_number, size_t base_number,
                                  size_t vecdim, size_t test_gt_d,
                                  size_t k, int num_threads,
                                  std::vector<SearchResult>& results) {
    results.resize(test_number);

    pthread_t* threads = new pthread_t[num_threads];
    PthreadWorkerArgs* args = new PthreadWorkerArgs[num_threads];

    size_t chunk_size = (test_number + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].thread_id = t;
        args[t].num_threads = num_threads;
        args[t].base = base;
        args[t].test_query = test_query;
        args[t].test_gt = test_gt;
        args[t].start_idx = t * chunk_size;
        args[t].end_idx = (t + 1) * chunk_size < test_number ?
                          (t + 1) * chunk_size : test_number;
        args[t].base_number = base_number;
        args[t].vecdim = vecdim;
        args[t].test_gt_d = test_gt_d;
        args[t].k = k;
        args[t].results = results.data();

        pthread_create(&threads[t], NULL, pthread_query_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    delete[] threads;
    delete[] args;
}

// ---- OpenMP query-level parallelism ----

inline void batch_search_openmp(float* base, float* test_query, int* test_gt,
                                 size_t test_number, size_t base_number,
                                 size_t vecdim, size_t test_gt_d,
                                 size_t k, int num_threads,
                                 std::vector<SearchResult>& results) {
    results.resize(test_number);

    const unsigned long Converter = 1000 * 1000;

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < test_number; ++i) {
        struct timeval val;
        gettimeofday(&val, NULL);

        auto res = flat_search_pq_adc_neon_twostage(
            base,
            test_query + i * vecdim,
            base_number,
            vecdim,
            k,
            550);

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
            int x = res.top().second;
            if (gtset.find(x) != gtset.end()) {
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc / k;

        results[i] = {recall, diff};
    }
}

// ---- Pthread: base vector partition (per-query parallel) ----

struct BasePartitionArgs {
    float* base;
    float* query;
    size_t start_i;
    size_t end_i;
    size_t base_number;
    size_t vecdim;
    size_t k;
    size_t local_p;
    std::priority_queue<std::pair<float, uint32_t>> local_result;
};

static void* base_partition_worker(void* arg) {
    BasePartitionArgs* args = (BasePartitionArgs*)arg;

    std::priority_queue<std::pair<float, uint32_t>>& q = args->local_result;
    size_t dim = args->vecdim;

    for (size_t i = args->start_i; i < args->end_i; ++i) {
        float* v = &args->base[i * dim];
        float* qry = args->query;

        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);

        size_t d = 0;
        for (; d + 16 <= dim; d += 16) {
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

        for (; d < dim; ++d) {
            ip += v[d] * qry[d];
        }

        float dis = 1.0f - ip;

        if (q.size() < args->local_p) {
            q.push({dis, (uint32_t)i});
        } else if (dis < q.top().first) {
            q.push({dis, (uint32_t)i});
            q.pop();
        }
    }

    return NULL;
}

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_pthread_partition(float* base, float* query, size_t base_number,
                               size_t vecdim, size_t k, int num_threads,
                               size_t local_p = 100) {
    pthread_t* threads = new pthread_t[num_threads];
    BasePartitionArgs* args = new BasePartitionArgs[num_threads];

    size_t chunk = (base_number + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].base = base;
        args[t].query = query;
        args[t].start_i = t * chunk;
        args[t].end_i = (t + 1) * chunk < base_number ?
                        (t + 1) * chunk : base_number;
        args[t].base_number = base_number;
        args[t].vecdim = vecdim;
        args[t].k = k;
        args[t].local_p = local_p;
        args[t].local_result = std::priority_queue<std::pair<float, uint32_t>>();

        pthread_create(&threads[t], NULL, base_partition_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    // merge local results
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

// ---- OpenMP: base vector partition ----

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_openmp_partition(float* base, float* query, size_t base_number,
                              size_t vecdim, size_t k, int num_threads,
                              size_t local_p = 100) {
    omp_set_num_threads(num_threads);

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& q = local_results[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < base_number; ++i) {
            float* v = &base[i * vecdim];

            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            float32x4_t sum4 = vdupq_n_f32(0.0f);

            size_t d = 0;
            for (; d + 16 <= vecdim; d += 16) {
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

            for (; d < vecdim; ++d) {
                ip += v[d] * query[d];
            }

            float dis = 1.0f - ip;

            if (q.size() < local_p) {
                q.push({dis, (uint32_t)i});
            } else if (dis < q.top().first) {
                q.push({dis, (uint32_t)i});
                q.pop();
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

// ---- Pthread: PQ base-vector partition (two-stage) ----

struct PQPartitionArgs {
    float* base;
    float* query;
    float* lut;
    size_t start_i;
    size_t end_i;
    size_t base_number;
    size_t vecdim;
    size_t k;
    size_t local_p;
    std::priority_queue<std::pair<float, uint32_t>> local_result;
};

static void* pq_partition_worker(void* arg) {
    PQPartitionArgs* args = (PQPartitionArgs*)arg;
    auto& q = args->local_result;

    for (size_t i = args->start_i; i < args->end_i; ++i) {
        uint8_t* code = &g_pq_kmeans.codes[i * PQ_M];
        float acc = 0.0f;
        for (size_t m = 0; m < PQ_M; ++m) {
            acc += args->lut[m * PQ_KS + code[m]];
        }
        float dist = 1.0f - acc;

        if (q.size() < args->local_p) {
            q.emplace(dist, (uint32_t)i);
        } else if (dist < q.top().first) {
            q.pop();
            q.emplace(dist, (uint32_t)i);
        }
    }
    return NULL;
}

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_pq_partition_pthread(float* base, float* query, size_t base_number,
                                  size_t vecdim, size_t k, int num_threads,
                                  size_t local_p = 550) {
    // Step 1: compute LUT (single-thread, fast)
    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_pq_kmeans, lut);

    // Step 2: partition coarse PQ scan among threads
    pthread_t* threads = new pthread_t[num_threads];
    PQPartitionArgs* args = new PQPartitionArgs[num_threads];
    size_t chunk = (base_number + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].base = base;
        args[t].query = query;
        args[t].lut = lut;
        args[t].start_i = t * chunk;
        args[t].end_i = (t + 1) * chunk < base_number ?
                        (t + 1) * chunk : base_number;
        args[t].base_number = base_number;
        args[t].vecdim = vecdim;
        args[t].k = k;
        args[t].local_p = local_p;
        args[t].local_result = std::priority_queue<std::pair<float, uint32_t>>();

        pthread_create(&threads[t], NULL, pq_partition_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    // Step 3: merge local top-p into combined candidates
    std::vector<std::pair<float, uint32_t>> cands;
    for (int t = 0; t < num_threads; ++t) {
        while (!args[t].local_result.empty()) {
            cands.push_back(args[t].local_result.top());
            args[t].local_result.pop();
        }
    }

    delete[] lut;
    delete[] threads;
    delete[] args;

    // Step 4: rerank candidates with exact NEON IP
    std::priority_queue<std::pair<float, uint32_t>> res;
    for (auto& cand : cands) {
        size_t idx = cand.second;
        float* v = &base[idx * vecdim];

        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);

        size_t d = 0;
        for (; d + 16 <= vecdim; d += 16) {
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
        for (; d < vecdim; ++d) { ip += v[d] * query[d]; }

        float dist = 1.0f - ip;
        if (res.size() < k) {
            res.emplace(dist, (uint32_t)idx);
        } else if (dist < res.top().first) {
            res.pop();
            res.emplace(dist, (uint32_t)idx);
        }
    }

    return res;
}

// ---- OpenMP: PQ base-vector partition (two-stage) ----

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_pq_partition_openmp(float* base, float* query, size_t base_number,
                                 size_t vecdim, size_t k, int num_threads,
                                 size_t local_p = 550) {
    // Step 1: compute LUT
    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_pq_kmeans, lut);

    // Step 2: parallel PQ coarse scan
    omp_set_num_threads(num_threads);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& q = local_results[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < base_number; ++i) {
            uint8_t* code = &g_pq_kmeans.codes[i * PQ_M];
            float acc = 0.0f;
            for (size_t m = 0; m < PQ_M; ++m) {
                acc += lut[m * PQ_KS + code[m]];
            }
            float dist = 1.0f - acc;

            if (q.size() < local_p) {
                q.emplace(dist, (uint32_t)i);
            } else if (dist < q.top().first) {
                q.pop();
                q.emplace(dist, (uint32_t)i);
            }
        }
    }

    delete[] lut;

    // Step 3: merge
    std::vector<std::pair<float, uint32_t>> cands;
    for (int t = 0; t < num_threads; ++t) {
        while (!local_results[t].empty()) {
            cands.push_back(local_results[t].top());
            local_results[t].pop();
        }
    }

    // Step 4: rerank
    std::priority_queue<std::pair<float, uint32_t>> res;
    for (auto& cand : cands) {
        size_t idx = cand.second;
        float* v = &base[idx * vecdim];

        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);

        size_t d = 0;
        for (; d + 16 <= vecdim; d += 16) {
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
        for (; d < vecdim; ++d) { ip += v[d] * query[d]; }

        float dist = 1.0f - ip;
        if (res.size() < k) {
            res.emplace(dist, (uint32_t)idx);
        } else if (dist < res.top().first) {
            res.pop();
            res.emplace(dist, (uint32_t)idx);
        }
    }

    return res;
}
