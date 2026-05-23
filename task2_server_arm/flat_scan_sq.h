#ifndef FLAT_SCAN_SQ_H
#define FLAT_SCAN_SQ_H

#include <vector>
#include <queue>
#include <arm_neon.h>

struct SQParams {
    float* minv;
    float* scale;
    uint8_t* codes;
    size_t n;
    size_t dim;
};

static SQParams g_sq;
static bool g_sq_built = false;

inline void sq_train(float* base, size_t n, size_t dim, SQParams& sq) {
    sq.n = n;
    sq.dim = dim;
    sq.minv = new float[dim];
    sq.scale = new float[dim];
    sq.codes = new uint8_t[n * dim];

    for (size_t d = 0; d < dim; ++d) {
        float vmin = base[d], vmax = base[d];
        for (size_t i = 1; i < n; ++i) {
            float v = base[i * dim + d];
            if (v < vmin) { vmin = v; }
            if (v > vmax) { vmax = v; }
        }
        float range = vmax - vmin;
        if (range < 1e-8f) { range = 1.0f; }
        sq.minv[d] = vmin;
        sq.scale[d] = range / 255.0f;

        for (size_t i = 0; i < n; ++i) {
            float v = base[i * dim + d];
            sq.codes[i * dim + d] = (uint8_t)((v - vmin) / sq.scale[d] + 0.5f);
        }
    }
}

inline void build_sq(float* base, size_t n, size_t dim) {
    sq_train(base, n, dim, g_sq);
    g_sq_built = true;
}

inline float ip_sq_neon(uint8_t* code, float* query, float* qs, float qm, size_t dim) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    size_t d = 0;
    for (; d + 16 <= dim; d += 16) {
        uint8x16_t c = vld1q_u8(&code[d]);
        uint16x8_t c_lo = vmovl_u8(vget_low_u8(c));
        uint16x8_t c_hi = vmovl_u8(vget_high_u8(c));
        uint32x4_t c0 = vmovl_u16(vget_low_u16(c_lo));
        uint32x4_t c1 = vmovl_u16(vget_high_u16(c_lo));
        uint32x4_t c2 = vmovl_u16(vget_low_u16(c_hi));
        uint32x4_t c3 = vmovl_u16(vget_high_u16(c_hi));
        float32x4_t cf0 = vcvtq_f32_u32(c0);
        float32x4_t cf1 = vcvtq_f32_u32(c1);
        float32x4_t cf2 = vcvtq_f32_u32(c2);
        float32x4_t cf3 = vcvtq_f32_u32(c3);
        sum0 = vmlaq_f32(sum0, cf0, vld1q_f32(&qs[d]));
        sum1 = vmlaq_f32(sum1, cf1, vld1q_f32(&qs[d + 4]));
        sum2 = vmlaq_f32(sum2, cf2, vld1q_f32(&qs[d + 8]));
        sum3 = vmlaq_f32(sum3, cf3, vld1q_f32(&qs[d + 12]));
    }

    float32x4_t total = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    float32x2_t lo = vget_low_f32(total);
    lo = vpadd_f32(lo, vget_high_f32(total));
    lo = vpadd_f32(lo, lo);
    float acc = vget_lane_f32(lo, 0);

    for (; d < dim; ++d) {
        acc += (float)code[d] * qs[d];
    }

    return 1.0f - (acc + qm);
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_sq(float* base, float* query, size_t n, size_t dim, size_t k, size_t /*unused*/ = 0) {
    if (!g_sq_built) { build_sq(base, n, dim); }

    float* qs = new float[dim];
    float qm = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        qs[d] = query[d] * g_sq.scale[d];
        qm += query[d] * g_sq.minv[d];
    }

    std::priority_queue<std::pair<float, uint32_t>> res;
    for (size_t i = 0; i < n; ++i) {
        float dist = ip_sq_neon(g_sq.codes + i * dim, query, qs, qm, dim);
        if (res.size() < k) {
            res.emplace(dist, (uint32_t)i);
        } else if (dist < res.top().first) {
            res.pop();
            res.emplace(dist, (uint32_t)i);
        }
    }

    delete[] qs;
    return res;
}

inline float ip_exact_neon(float* v, float* qry, size_t dim) {
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

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_sq_twostage(float* base, float* query, size_t n, size_t dim, size_t k, size_t p = 100) {
    if (!g_sq_built) { build_sq(base, n, dim); }

    float* qs = new float[dim];
    float qm = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        qs[d] = query[d] * g_sq.scale[d];
        qm += query[d] * g_sq.minv[d];
    }

    std::priority_queue<std::pair<float, uint32_t>> coarse;
    for (size_t i = 0; i < n; ++i) {
        float dist = ip_sq_neon(g_sq.codes + i * dim, query, qs, qm, dim);
        if (coarse.size() < p) {
            coarse.emplace(dist, (uint32_t)i);
        } else if (dist < coarse.top().first) {
            coarse.pop();
            coarse.emplace(dist, (uint32_t)i);
        }
    }

    delete[] qs;

    std::vector<std::pair<float, uint32_t>> cands;
    cands.reserve(coarse.size());
    while (!coarse.empty()) {
        cands.push_back(coarse.top());
        coarse.pop();
    }

    std::priority_queue<std::pair<float, uint32_t>> res;
    for (auto& cand : cands) {
        size_t i = cand.second;
        float ip = ip_exact_neon(&base[i * dim], query, dim);
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
