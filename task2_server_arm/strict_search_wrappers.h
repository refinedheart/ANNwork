#pragma once

#include <algorithm>
#include <cstdint>
#include <pthread.h>
#include <queue>
#include <vector>
#include <omp.h>
#include <arm_neon.h>
#include "flat_scan_pq_adc_neon.h"

struct StrictPQPartitionArgs {
    float* base;
    float* query;
    float* lut;
    size_t start_i;
    size_t end_i;
    size_t vecdim;
    size_t local_p;
    std::priority_queue<std::pair<float, uint32_t>> local_result;
};

static void* strict_pq_partition_worker(void* arg) {
    StrictPQPartitionArgs* args = static_cast<StrictPQPartitionArgs*>(arg);
    auto& q = args->local_result;

    for (size_t i = args->start_i; i < args->end_i; ++i) {
        uint8_t* code = &g_pq_kmeans.codes[i * PQ_M];
        float acc = 0.0f;
        for (size_t m = 0; m < PQ_M; ++m) {
            acc += args->lut[m * PQ_KS + code[m]];
        }
        float dist = 1.0f - acc;
        if (q.size() < args->local_p) {
            q.emplace(dist, static_cast<uint32_t>(i));
        } else if (dist < q.top().first) {
            q.pop();
            q.emplace(dist, static_cast<uint32_t>(i));
        }
    }
    return NULL;
}

inline float strict_neon_ip(float* a, float* b, size_t dim) {
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);

    size_t d = 0;
    for (; d + 16 <= dim; d += 16) {
        sum1 = vmlaq_f32(sum1, vld1q_f32(&a[d]),      vld1q_f32(&b[d]));
        sum2 = vmlaq_f32(sum2, vld1q_f32(&a[d + 4]),  vld1q_f32(&b[d + 4]));
        sum3 = vmlaq_f32(sum3, vld1q_f32(&a[d + 8]),  vld1q_f32(&b[d + 8]));
        sum4 = vmlaq_f32(sum4, vld1q_f32(&a[d + 12]), vld1q_f32(&b[d + 12]));
    }

    float32x4_t total = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));
#if defined(__aarch64__)
    float ip = vaddvq_f32(total);
#else
    float32x2_t lo = vget_low_f32(total);
    lo = vpadd_f32(lo, vget_high_f32(total));
    lo = vpadd_f32(lo, lo);
    float ip = vget_lane_f32(lo, 0);
#endif

    for (; d < dim; ++d) {
        ip += a[d] * b[d];
    }
    return ip;
}

inline std::priority_queue<std::pair<float, uint32_t>>
strict_flat_search_pq_partition_pthread(float* base, float* query,
                                        size_t base_number, size_t vecdim,
                                        size_t k, int num_threads = 8,
                                        size_t local_p = 550) {
    if (!g_pq_kmeans_built) {
        build_pq_kmeans_neon(base, base_number, vecdim);
    }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_pq_kmeans, lut);

    pthread_t* threads = new pthread_t[num_threads];
    StrictPQPartitionArgs* args = new StrictPQPartitionArgs[num_threads];
    size_t chunk = (base_number + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].base = base;
        args[t].query = query;
        args[t].lut = lut;
        args[t].start_i = t * chunk;
        args[t].end_i = std::min((t + 1) * chunk, base_number);
        args[t].vecdim = vecdim;
        args[t].local_p = local_p;
        pthread_create(&threads[t], NULL, strict_pq_partition_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }

    std::vector<std::pair<float, uint32_t>> candidates;
    for (int t = 0; t < num_threads; ++t) {
        while (!args[t].local_result.empty()) {
            candidates.push_back(args[t].local_result.top());
            args[t].local_result.pop();
        }
    }

    delete[] lut;
    delete[] threads;
    delete[] args;

    std::priority_queue<std::pair<float, uint32_t>> result;
    for (auto& cand : candidates) {
        uint32_t idx = cand.second;
        float ip = strict_neon_ip(&base[static_cast<size_t>(idx) * vecdim], query, vecdim);
        float dist = 1.0f - ip;
        if (result.size() < k) {
            result.emplace(dist, idx);
        } else if (dist < result.top().first) {
            result.pop();
            result.emplace(dist, idx);
        }
    }
    return result;
}

inline std::priority_queue<std::pair<float, uint32_t>>
strict_flat_search_pq_partition_openmp(float* base, float* query,
                                       size_t base_number, size_t vecdim,
                                       size_t k, int num_threads = 8,
                                       size_t local_p = 550) {
    if (!g_pq_kmeans_built) {
        build_pq_kmeans_neon(base, base_number, vecdim);
    }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_pq_kmeans, lut);

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
                q.emplace(dist, static_cast<uint32_t>(i));
            } else if (dist < q.top().first) {
                q.pop();
                q.emplace(dist, static_cast<uint32_t>(i));
            }
        }
    }

    delete[] lut;

    std::priority_queue<std::pair<float, uint32_t>> result;
    for (int t = 0; t < num_threads; ++t) {
        while (!local_results[t].empty()) {
            uint32_t idx = local_results[t].top().second;
            local_results[t].pop();
            float ip = strict_neon_ip(&base[static_cast<size_t>(idx) * vecdim], query, vecdim);
            float dist = 1.0f - ip;
            if (result.size() < k) {
                result.emplace(dist, idx);
            } else if (dist < result.top().first) {
                result.pop();
                result.emplace(dist, idx);
            }
        }
    }
    return result;
}
