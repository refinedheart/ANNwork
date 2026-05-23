#ifndef FLAT_SCAN_PQ_H
#define FLAT_SCAN_PQ_H

#include <vector>
#include <queue>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <arm_neon.h>

#ifndef PQ_M
#define PQ_M 8
#endif
static const size_t PQ_M_VAL = PQ_M;
static const size_t PQ_DSUB = 96 / PQ_M;
static const size_t PQ_KS = 256;
static const size_t PQ_ITERS = 25;

struct PQParams {
    float cent[PQ_M * PQ_KS * PQ_DSUB];
    uint8_t* codes;
    size_t n;
    bool trained;
};

static PQParams g_pq;
static bool g_pq_built = false;

inline void kmeans_subspace_ip(float* data, size_t n, size_t dsub, float* cent, size_t ks, size_t iters) {
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
            for (size_t k = 0; k < ks; ++k) {
                const float* c = &cent[k * dsub];
                float ip = 0.0f;
                for (size_t j = 0; j < dsub; ++j) {
                    ip += pt[j] * c[j];
                }
                if (ip > best) { best = ip; bk = (uint8_t)k; }
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
            for (size_t j = 0; j < dsub; ++j) {
                c[j] += pt[j];
            }
        }
        for (size_t k = 0; k < ks; ++k) {
            if (cnt[k] > 0) {
                float inv = 1.0f / cnt[k];
                float* c = &cent[k * dsub];
                for (size_t j = 0; j < dsub; ++j) {
                    c[j] *= inv;
                }
            }
        }
    }

    delete[] assign;
    delete[] cnt;
}

inline void pq_train(float* base, size_t n, size_t dim, PQParams& pq) {
    pq.n = n;
    pq.trained = true;
    pq.codes = new uint8_t[n * PQ_M];

    for (size_t m = 0; m < PQ_M; ++m) {
        float* sub = new float[n * PQ_DSUB];
        for (size_t i = 0; i < n; ++i) {
            memcpy(&sub[i * PQ_DSUB], &base[i * dim + m * PQ_DSUB], PQ_DSUB * sizeof(float));
        }

        float* c = &pq.cent[m * PQ_KS * PQ_DSUB];
        kmeans_subspace_ip(sub, n, PQ_DSUB, c, PQ_KS, PQ_ITERS);

        for (size_t i = 0; i < n; ++i) {
            const float* pt = &sub[i * PQ_DSUB];
            float best = -1e30f;
            uint8_t bk = 0;
            for (size_t k = 0; k < PQ_KS; ++k) {
                const float* ck = &c[k * PQ_DSUB];
                float ip = 0.0f;
                for (size_t j = 0; j < PQ_DSUB; ++j) {
                    ip += pt[j] * ck[j];
                }
                if (ip > best) { best = ip; bk = (uint8_t)k; }
            }
            pq.codes[i * PQ_M + m] = bk;
        }

        delete[] sub;
    }
}

inline void build_pq(float* base, size_t n, size_t dim) {
    srand(42);
    pq_train(base, n, dim, g_pq);
    g_pq_built = true;
}

inline void pq_compute_lut(float* query, PQParams& pq, float* lut) {
    for (size_t m = 0; m < PQ_M; ++m) {
        const float* qs = &query[m * PQ_DSUB];
        float* lm = &lut[m * PQ_KS];
        const float* cents = &pq.cent[m * PQ_KS * PQ_DSUB];

        for (size_t k = 0; k < PQ_KS; ++k) {
            const float* c = &cents[k * PQ_DSUB];
            float acc = 0.0f;

            size_t d = 0;
            for (; d + 4 <= PQ_DSUB; d += 4) {
                float32x4_t qv = vld1q_f32(&qs[d]);
                float32x4_t cv = vld1q_f32(&c[d]);
                float32x4_t prod = vmulq_f32(qv, cv);
                float32x2_t lo = vget_low_f32(prod);
                float32x2_t hi = vget_high_f32(prod);
                lo = vpadd_f32(lo, hi);
                lo = vpadd_f32(lo, lo);
                acc += vget_lane_f32(lo, 0);
            }
            for (; d < PQ_DSUB; ++d) {
                acc += qs[d] * c[d];
            }

            lm[k] = acc;
        }
    }
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq(float* base, float* query, size_t n, size_t dim, size_t k, size_t /*unused*/ = 0) {
    if (!g_pq_built) { build_pq(base, n, dim); }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut(query, g_pq, lut);

    std::priority_queue<std::pair<float, uint32_t>> res;

    for (size_t i = 0; i < n; ++i) {
        uint8_t* code = &g_pq.codes[i * PQ_M];
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

inline float ip_exact_neon_pq(float* v, float* qry, size_t dim) {
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

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_pq_twostage(float* base, float* query, size_t n, size_t dim, size_t k, size_t p = 100) {
    if (!g_pq_built) { build_pq(base, n, dim); }

    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut(query, g_pq, lut);

    std::priority_queue<std::pair<float, uint32_t>> coarse;
    for (size_t i = 0; i < n; ++i) {
        uint8_t* code = &g_pq.codes[i * PQ_M];
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
        float ip = ip_exact_neon_pq(&base[i * dim], query, dim);
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
