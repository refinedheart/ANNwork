#ifndef FLAT_SCAN_PQ_BLOCKED_SOA_H
#define FLAT_SCAN_PQ_BLOCKED_SOA_H

#include "flat_scan_pq_kmeans_neon.h"

static PQParams g_pq_blocked;
static bool g_pq_blocked_built = false;

static const size_t KMEANS_BLOCK = 4096;

inline void kmeans_subspace_ip_blocked(float* data, size_t n, size_t dsub, float* cent, size_t ks, size_t iters) {
    bool* sel = new bool[n]();
    for (size_t k = 0; k < ks; ++k) {
        size_t idx;
        do { idx = rand() % n; } while (sel[idx]);
        sel[idx] = true;
        memcpy(&cent[k * dsub], &data[idx * dsub], dsub * sizeof(float));
    }
    delete[] sel;

    uint8_t* assign = new uint8_t[n];
    int* cnt = new int[ks];

    for (size_t iter = 0; iter < iters; ++iter) {
        for (size_t ib = 0; ib < n; ib += KMEANS_BLOCK) {
            size_t bend = (ib + KMEANS_BLOCK < n) ? ib + KMEANS_BLOCK : n;

            for (size_t i = ib; i < bend; ++i) {
                float best = -1e30f;
                uint8_t bk = 0;
                const float* pt = &data[i * dsub];

                size_t k = 0;
                for (; k + 4 <= ks; k += 4) {
                    float ip0 = neon_ip_f32(pt, &cent[(k+0) * dsub], dsub);
                    float ip1 = neon_ip_f32(pt, &cent[(k+1) * dsub], dsub);
                    float ip2 = neon_ip_f32(pt, &cent[(k+2) * dsub], dsub);
                    float ip3 = neon_ip_f32(pt, &cent[(k+3) * dsub], dsub);

                    if (ip0 > best) { best = ip0; bk = (uint8_t)(k+0); }
                    if (ip1 > best) { best = ip1; bk = (uint8_t)(k+1); }
                    if (ip2 > best) { best = ip2; bk = (uint8_t)(k+2); }
                    if (ip3 > best) { best = ip3; bk = (uint8_t)(k+3); }
                }
                assign[i] = bk;
            }
        }

        memset(cent, 0, ks * dsub * sizeof(float));
        memset(cnt, 0, ks * sizeof(int));
        for (size_t i = 0; i < n; ++i) {
            size_t k = assign[i];
            cnt[k]++;
            float* c = &cent[k * dsub];
            const float* pt = &data[i * dsub];
            size_t j = 0;
            for (; j + 4 <= dsub; j += 4) {
                float32x4_t pt_v = vld1q_f32(&pt[j]);
                float32x4_t cv   = vld1q_f32(&c[j]);
                vst1q_f32(&c[j], vaddq_f32(cv, pt_v));
            }
            for (; j < dsub; ++j) {
                c[j] += pt[j];
            }
        }
        for (size_t k = 0; k < ks; ++k) {
            if (cnt[k] > 0) {
                float inv = 1.0f / cnt[k];
                float32x4_t inv_v = vdupq_n_f32(inv);
                float* c = &cent[k * dsub];
                size_t j = 0;
                for (; j + 4 <= dsub; j += 4) {
                    float32x4_t cv = vld1q_f32(&c[j]);
                    vst1q_f32(&c[j], vmulq_f32(cv, inv_v));
                }
                for (; j < dsub; ++j) {
                    c[j] *= inv;
                }
            }
        }
    }

    delete[] assign;
    delete[] cnt;
}

inline void pq_train_blocked_soa(float* base, size_t n, size_t dim, PQParams& pq) {
    pq.n = n;
    pq.trained = true;
    pq.codes = new uint8_t[PQ_M * n];

    for (size_t m = 0; m < PQ_M; ++m) {
        float* sub = new float[n * PQ_DSUB];
        for (size_t i = 0; i < n; ++i) {
            memcpy(&sub[i * PQ_DSUB], &base[i * dim + m * PQ_DSUB], PQ_DSUB * sizeof(float));
        }

        float* c = &pq.cent[m * PQ_KS * PQ_DSUB];
        kmeans_subspace_ip_blocked(sub, n, PQ_DSUB, c, PQ_KS, PQ_ITERS);

        for (size_t i = 0; i < n; ++i) {
            float best = -1e30f;
            uint8_t bk = 0;
            const float* pt = &sub[i * PQ_DSUB];

            size_t k = 0;
            for (; k + 4 <= PQ_KS; k += 4) {
                float ip0 = neon_ip_f32(pt, &c[(k+0) * PQ_DSUB], PQ_DSUB);
                float ip1 = neon_ip_f32(pt, &c[(k+1) * PQ_DSUB], PQ_DSUB);
                float ip2 = neon_ip_f32(pt, &c[(k+2) * PQ_DSUB], PQ_DSUB);
                float ip3 = neon_ip_f32(pt, &c[(k+3) * PQ_DSUB], PQ_DSUB);

                if (ip0 > best) { best = ip0; bk = (uint8_t)(k+0); }
                if (ip1 > best) { best = ip1; bk = (uint8_t)(k+1); }
                if (ip2 > best) { best = ip2; bk = (uint8_t)(k+2); }
                if (ip3 > best) { best = ip3; bk = (uint8_t)(k+3); }
            }
            pq.codes[m * n + i] = bk;
        }

        delete[] sub;
    }
}

inline void build_pq_blocked_soa(float* base, size_t n, size_t dim) {
    srand(42);
    pq_train_blocked_soa(base, n, dim, g_pq_blocked);
    g_pq_blocked_built = true;
}

inline float adc_scan_soa_single(PQParams& pq, float* lut, size_t i) {
    float acc = 0.0f;
    for (size_t m = 0; m < PQ_M; ++m) {
        uint8_t c = pq.codes[m * pq.n + i];
        acc += lut[m * PQ_KS + c];
    }
    return 1.0f - acc;
}

inline void adc_scan_soa_batch4(PQParams& pq, float* lut, size_t i, float& d0, float& d1, float& d2, float& d3) {
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t nb = pq.n;

    size_t m = 0;
    for (; m + 4 <= PQ_M; m += 4) {
        uint32_t codes0 = *(uint32_t*)&pq.codes[(m+0) * nb + i];
        uint32_t codes1 = *(uint32_t*)&pq.codes[(m+1) * nb + i];
        uint32_t codes2 = *(uint32_t*)&pq.codes[(m+2) * nb + i];
        uint32_t codes3 = *(uint32_t*)&pq.codes[(m+3) * nb + i];

        a0 += lut[(m+0) * PQ_KS + (codes0 & 0xFF)] + lut[(m+1) * PQ_KS + (codes1 & 0xFF)] + lut[(m+2) * PQ_KS + (codes2 & 0xFF)] + lut[(m+3) * PQ_KS + (codes3 & 0xFF)];
        a1 += lut[(m+0) * PQ_KS + ((codes0 >> 8) & 0xFF)] + lut[(m+1) * PQ_KS + ((codes1 >> 8) & 0xFF)] + lut[(m+2) * PQ_KS + ((codes2 >> 8) & 0xFF)] + lut[(m+3) * PQ_KS + ((codes3 >> 8) & 0xFF)];
        a2 += lut[(m+0) * PQ_KS + ((codes0 >> 16) & 0xFF)] + lut[(m+1) * PQ_KS + ((codes1 >> 16) & 0xFF)] + lut[(m+2) * PQ_KS + ((codes2 >> 16) & 0xFF)] + lut[(m+3) * PQ_KS + ((codes3 >> 16) & 0xFF)];
        a3 += lut[(m+0) * PQ_KS + ((codes0 >> 24) & 0xFF)] + lut[(m+1) * PQ_KS + ((codes1 >> 24) & 0xFF)] + lut[(m+2) * PQ_KS + ((codes2 >> 24) & 0xFF)] + lut[(m+3) * PQ_KS + ((codes3 >> 24) & 0xFF)];
    }
    for (; m < PQ_M; ++m) {
        uint32_t codes = *(uint32_t*)&pq.codes[m * nb + i];
        a0 += lut[m * PQ_KS + (codes & 0xFF)];
        a1 += lut[m * PQ_KS + ((codes >> 8) & 0xFF)];
        a2 += lut[m * PQ_KS + ((codes >> 16) & 0xFF)];
        a3 += lut[m * PQ_KS + ((codes >> 24) & 0xFF)];
    }
    d0 = 1.0f - a0; d1 = 1.0f - a1;
    d2 = 1.0f - a2; d3 = 1.0f - a3;
}

inline float ip_exact_blocked(float* v, float* qry, size_t dim) {
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

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_blocked_soa(float* base, float* query, size_t n, size_t dim, size_t k, size_t /*unused*/ = 0) {
    if (!g_pq_blocked_built) { build_pq_blocked_soa(base, n, dim); }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_pq_blocked, lut);

    std::priority_queue<std::pair<float, uint32_t>> res;
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float d0, d1, d2, d3;
        adc_scan_soa_batch4(g_pq_blocked, lut, i, d0, d1, d2, d3);

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
        float dist = adc_scan_soa_single(g_pq_blocked, lut, i);
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

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_blocked_soa_twostage(float* base, float* query, size_t n, size_t dim, size_t k, size_t p = 100) {
    if (!g_pq_blocked_built) { build_pq_blocked_soa(base, n, dim); }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_pq_blocked, lut);

    std::priority_queue<std::pair<float, uint32_t>> coarse;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float d0, d1, d2, d3;
        adc_scan_soa_batch4(g_pq_blocked, lut, i, d0, d1, d2, d3);

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
        float dist = adc_scan_soa_single(g_pq_blocked, lut, i);
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
        float ip = ip_exact_blocked(&base[idx * dim], query, dim);
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
