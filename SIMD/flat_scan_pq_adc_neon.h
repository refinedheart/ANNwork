#ifndef FLAT_SCAN_PQ_ADC_NEON_H
#define FLAT_SCAN_PQ_ADC_NEON_H

#include "flat_scan_pq_kmeans_neon.h"

inline float ip_exact_adc_neon(float* v, float* qry, size_t dim) {
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);

    for (size_t d = 0; d < dim; d += 16) {
        sum1 = vmlaq_f32(sum1, vld1q_f32(&v[d]),      vld1q_f32(&qry[d]));
        sum2 = vmlaq_f32(sum2, vld1q_f32(&v[d + 4]),  vld1q_f32(&qry[d + 4]));
        sum3 = vmlaq_f32(sum3, vld1q_f32(&v[d + 8]),  vld1q_f32(&qry[d + 8]));
        sum4 = vmlaq_f32(sum4, vld1q_f32(&v[d + 12]), vld1q_f32(&qry[d + 12]));
    }

    float32x4_t total = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));
    float32x2_t lo = vget_low_f32(total);
    lo = vpadd_f32(lo, vget_high_f32(total));
    lo = vpadd_f32(lo, lo);
    return vget_lane_f32(lo, 0);
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_adc_neon(float* base, float* query, size_t n, size_t dim, size_t k, size_t /*unused*/ = 0) {
    if (!g_pq_kmeans_built) { build_pq_kmeans_neon(base, n, dim); }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_pq_kmeans, lut);

    std::priority_queue<std::pair<float, uint32_t>> res;
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        uint8_t* code0 = &g_pq_kmeans.codes[(i+0) * PQ_M];
        uint8_t* code1 = &g_pq_kmeans.codes[(i+1) * PQ_M];
        uint8_t* code2 = &g_pq_kmeans.codes[(i+2) * PQ_M];
        uint8_t* code3 = &g_pq_kmeans.codes[(i+3) * PQ_M];

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        size_t m = 0;
        for (; m + 4 <= PQ_M; m += 4) {
            acc0 += lut[(m+0) * PQ_KS + code0[m+0]] + lut[(m+1) * PQ_KS + code0[m+1]] + lut[(m+2) * PQ_KS + code0[m+2]] + lut[(m+3) * PQ_KS + code0[m+3]];
            acc1 += lut[(m+0) * PQ_KS + code1[m+0]] + lut[(m+1) * PQ_KS + code1[m+1]] + lut[(m+2) * PQ_KS + code1[m+2]] + lut[(m+3) * PQ_KS + code1[m+3]];
            acc2 += lut[(m+0) * PQ_KS + code2[m+0]] + lut[(m+1) * PQ_KS + code2[m+1]] + lut[(m+2) * PQ_KS + code2[m+2]] + lut[(m+3) * PQ_KS + code2[m+3]];
            acc3 += lut[(m+0) * PQ_KS + code3[m+0]] + lut[(m+1) * PQ_KS + code3[m+1]] + lut[(m+2) * PQ_KS + code3[m+2]] + lut[(m+3) * PQ_KS + code3[m+3]];
        }
        for (; m < PQ_M; ++m) {
            acc0 += lut[m * PQ_KS + code0[m]];
            acc1 += lut[m * PQ_KS + code1[m]];
            acc2 += lut[m * PQ_KS + code2[m]];
            acc3 += lut[m * PQ_KS + code3[m]];
        }

        float d0 = 1.0f - acc0, d1 = 1.0f - acc1;
        float d2 = 1.0f - acc2, d3 = 1.0f - acc3;

        if (res.size() < k) { res.emplace(d0, (uint32_t)(i+0)); }
        else if (d0 < res.top().first) { res.pop(); res.emplace(d0, (uint32_t)(i+0)); }
        if (res.size() < k) { res.emplace(d1, (uint32_t)(i+1)); }
        else if (d1 < res.top().first) { res.pop(); res.emplace(d1, (uint32_t)(i+1)); }
        if (res.size() < k) { res.emplace(d2, (uint32_t)(i+2)); }
        else if (d2 < res.top().first) { res.pop(); res.emplace(d2, (uint32_t)(i+2)); }
        if (res.size() < k) { res.emplace(d3, (uint32_t)(i+3)); }
        else if (d3 < res.top().first) { res.pop(); res.emplace(d3, (uint32_t)(i+3)); }
    }

    for (; i < n; ++i) {
        uint8_t* code = &g_pq_kmeans.codes[i * PQ_M];
        float acc = 0.0f;
        for (size_t m = 0; m < PQ_M; ++m) {
            acc += lut[m * PQ_KS + code[m]];
        }
        float dist = 1.0f - acc;

        if (res.size() < k) {
            res.emplace(dist, (uint32_t)i);
        } else if (dist < res.top().first) {
            res.pop();
            res.emplace(dist, (uint32_t)i);
        }
    }

    delete[] lut;
    return res;
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_adc_neon_twostage(float* base, float* query, size_t n, size_t dim, size_t k, size_t p = 100) {
    if (!g_pq_kmeans_built) { build_pq_kmeans_neon(base, n, dim); }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_pq_kmeans, lut);

    std::priority_queue<std::pair<float, uint32_t>> coarse;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint8_t* c0 = &g_pq_kmeans.codes[(i+0) * PQ_M];
        uint8_t* c1 = &g_pq_kmeans.codes[(i+1) * PQ_M];
        uint8_t* c2 = &g_pq_kmeans.codes[(i+2) * PQ_M];
        uint8_t* c3 = &g_pq_kmeans.codes[(i+3) * PQ_M];

        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (size_t m = 0; m < PQ_M; ++m) {
            a0 += lut[m * PQ_KS + c0[m]];
            a1 += lut[m * PQ_KS + c1[m]];
            a2 += lut[m * PQ_KS + c2[m]];
            a3 += lut[m * PQ_KS + c3[m]];
        }

        float d0 = 1.0f - a0, d1 = 1.0f - a1;
        float d2 = 1.0f - a2, d3 = 1.0f - a3;

        if (coarse.size() < p) { coarse.emplace(d0, (uint32_t)(i+0)); }
        else if (d0 < coarse.top().first) { coarse.pop(); coarse.emplace(d0, (uint32_t)(i+0)); }
        if (coarse.size() < p) { coarse.emplace(d1, (uint32_t)(i+1)); }
        else if (d1 < coarse.top().first) { coarse.pop(); coarse.emplace(d1, (uint32_t)(i+1)); }
        if (coarse.size() < p) { coarse.emplace(d2, (uint32_t)(i+2)); }
        else if (d2 < coarse.top().first) { coarse.pop(); coarse.emplace(d2, (uint32_t)(i+2)); }
        if (coarse.size() < p) { coarse.emplace(d3, (uint32_t)(i+3)); }
        else if (d3 < coarse.top().first) { coarse.pop(); coarse.emplace(d3, (uint32_t)(i+3)); }
    }
    for (; i < n; ++i) {
        uint8_t* code = &g_pq_kmeans.codes[i * PQ_M];
        float acc = 0;
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
    while (!coarse.empty()) {
        cands.push_back(coarse.top());
        coarse.pop();
    }

    std::priority_queue<std::pair<float, uint32_t>> res;
    for (auto& cand : cands) {
        size_t idx = cand.second;
        float ip = ip_exact_adc_neon(&base[idx * dim], query, dim);
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

#endif
