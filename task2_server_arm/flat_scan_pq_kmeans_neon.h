#ifndef FLAT_SCAN_PQ_KMEANS_NEON_H
#define FLAT_SCAN_PQ_KMEANS_NEON_H

#include "flat_scan_pq.h"

static PQParams g_pq_kmeans;
static bool g_pq_kmeans_built = false;

inline float neon_hsum_f32x4(float32x4_t v) {
    float32x2_t lo = vget_low_f32(v);
    lo = vpadd_f32(lo, vget_high_f32(v));
    lo = vpadd_f32(lo, lo);
    return vget_lane_f32(lo, 0);
}

inline float neon_ip_f32(const float* a, const float* b, size_t d) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    size_t j = 0;
    for (; j + 8 <= d; j += 8) {
        sum0 = vmlaq_f32(sum0, vld1q_f32(&a[j]), vld1q_f32(&b[j]));
        sum1 = vmlaq_f32(sum1, vld1q_f32(&a[j + 4]), vld1q_f32(&b[j + 4]));
    }
    for (; j + 4 <= d; j += 4) {
        sum0 = vmlaq_f32(sum0, vld1q_f32(&a[j]), vld1q_f32(&b[j]));
    }
    float acc = neon_hsum_f32x4(vaddq_f32(sum0, sum1));
    for (; j < d; ++j) acc += a[j] * b[j];
    return acc;
}

inline void kmeans_subspace_ip_neon(float* data, size_t n, size_t dsub, float* cent, size_t ks, size_t iters) {
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
        for (size_t i = 0; i < n; ++i) {
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

inline void pq_train_kmeans_neon(float* base, size_t n, size_t dim, PQParams& pq) {
    pq.n = n;
    pq.trained = true;
    pq.codes = new uint8_t[n * PQ_M];

    for (size_t m = 0; m < PQ_M; ++m) {
        float* sub = new float[n * PQ_DSUB];
        for (size_t i = 0; i < n; ++i) {
            memcpy(&sub[i * PQ_DSUB], &base[i * dim + m * PQ_DSUB], PQ_DSUB * sizeof(float));
        }

        float* c = &pq.cent[m * PQ_KS * PQ_DSUB];
        kmeans_subspace_ip_neon(sub, n, PQ_DSUB, c, PQ_KS, PQ_ITERS);

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
            pq.codes[i * PQ_M + m] = bk;
        }

        delete[] sub;
    }
}

inline void build_pq_kmeans_neon(float* base, size_t n, size_t dim) {
    srand(42);
    pq_train_kmeans_neon(base, n, dim, g_pq_kmeans);
    g_pq_kmeans_built = true;
}

inline void pq_compute_lut_neon(float* query, PQParams& pq, float* lut) {
    for (size_t m = 0; m < PQ_M; ++m) {
        const float* qs = &query[m * PQ_DSUB];
        float* lm = &lut[m * PQ_KS];
        const float* cents = &pq.cent[m * PQ_KS * PQ_DSUB];

        for (size_t k = 0; k < PQ_KS; k += 4) {
            lm[k+0] = neon_ip_f32(qs, &cents[(k+0) * PQ_DSUB], PQ_DSUB);
            lm[k+1] = neon_ip_f32(qs, &cents[(k+1) * PQ_DSUB], PQ_DSUB);
            lm[k+2] = neon_ip_f32(qs, &cents[(k+2) * PQ_DSUB], PQ_DSUB);
            lm[k+3] = neon_ip_f32(qs, &cents[(k+3) * PQ_DSUB], PQ_DSUB);
        }
    }
}

inline float ip_exact_neon_kmeans(float* v, float* qry, size_t dim) {
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
    return neon_hsum_f32x4(total);
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_kmeans_neon(float* base, float* query, size_t n, size_t dim, size_t k, size_t /*unused*/ = 0) {
    if (!g_pq_kmeans_built) { build_pq_kmeans_neon(base, n, dim); }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_pq_kmeans, lut);

    std::priority_queue<std::pair<float, uint32_t>> res;
    for (size_t i = 0; i < n; ++i) {
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

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_kmeans_neon_twostage(float* base, float* query, size_t n, size_t dim, size_t k, size_t p = 100) {
    if (!g_pq_kmeans_built) { build_pq_kmeans_neon(base, n, dim); }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_pq_kmeans, lut);

    std::priority_queue<std::pair<float, uint32_t>> coarse;
    for (size_t i = 0; i < n; ++i) {
        uint8_t* code = &g_pq_kmeans.codes[i * PQ_M];
        float acc = 0.0f;
        for (size_t m = 0; m < PQ_M; ++m) {
            acc += lut[m * PQ_KS + code[m]];
        }
        float dist = 1.0f - acc;

        if (coarse.size() < p) {
            coarse.emplace(dist, (uint32_t)i);
        } else if (dist < coarse.top().first) {
            coarse.pop();
            coarse.emplace(dist, (uint32_t)i);
        }
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
        size_t i = cand.second;
        float ip = ip_exact_neon_kmeans(&base[i * dim], query, dim);
        float dist = 1.0f - ip;
        if (res.size() < k) {
            res.emplace(dist, (uint32_t)i);
        } else if (dist < res.top().first) {
            res.pop();
            res.emplace(dist, (uint32_t)i);
        }
    }
    return res;
}

#endif
