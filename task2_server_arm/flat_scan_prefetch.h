#pragma once
#include <arm_neon.h>
#include <queue>

inline float ip_neon_16x_prefetch(float* v, float* qry, size_t dim) {
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
    return 1.0f - vget_lane_f32(lo, 0);
}

std::priority_queue<std::pair<float, uint32_t>> flat_search_neon16p_prefetch(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (int i = 0; i < n; ++i) {
        if (i + 4 < n) {
            __builtin_prefetch(&base[(i + 4) * dim], 0, 3);
        }

        float dis = ip_neon_16x_prefetch(&base[i * dim], query, dim);

        if (q.size() < k) {
            q.push({dis, i});
        } else {
            if (dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}
