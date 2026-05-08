#include <arm_neon.h>
#include <cstddef>
#include <queue>

inline float ip_neon_4x(float* v, float* qry, size_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    for (size_t d = 0; d < dim; d += 4) {
        sum = vmlaq_f32(sum, vld1q_f32(&v[d]), vld1q_f32(&qry[d]));
    }

    float32x2_t lo = vget_low_f32(sum);
    lo = vpadd_f32(lo, vget_high_f32(sum));
    lo = vpadd_f32(lo, lo);
    return 1.0f - vget_lane_f32(lo, 0);
}

inline float ip_neon_8x(float* v, float* qry, size_t dim) {
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);

    for (size_t d = 0; d < dim; d += 8) {
        sum1 = vmlaq_f32(sum1, vld1q_f32(&v[d]), vld1q_f32(&qry[d]));
        sum2 = vmlaq_f32(sum2, vld1q_f32(&v[d + 4]), vld1q_f32(&qry[d + 4]));
    }

    float32x4_t total = vaddq_f32(sum1, sum2);
    float32x2_t lo = vget_low_f32(total);
    lo = vpadd_f32(lo, vget_high_f32(total));
    lo = vpadd_f32(lo, lo);
    return 1.0f - vget_lane_f32(lo, 0);
}

inline float ip_neon_16x(float* v, float* qry, size_t dim) {
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

inline float ip_neon_16x_aligned(float* v, float* qry, size_t dim) {
    v = (float*)__builtin_assume_aligned(v, 16);
    qry = (float*)__builtin_assume_aligned(qry, 16);
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


std::priority_queue<std::pair<float, uint32_t>> flat_search_neon4p(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (int i = 0; i < n; ++i) {
        float dis = ip_neon_4x(&base[i * dim], query, dim);

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

std::priority_queue<std::pair<float, uint32_t>> flat_search_neon8p(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (int i = 0; i < n; ++i) {
        float dis = ip_neon_8x(&base[i * dim], query, dim);

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

std::priority_queue<std::pair<float, uint32_t>> flat_search_neon16p(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (int i = 0; i < n; ++i) {
        float dis = ip_neon_16x(&base[i * dim], query, dim);

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

std::priority_queue<std::pair<float, uint32_t>> flat_search_neon16p_aligned(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_neon_16x_aligned(&base[i * dim], query, dim);
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
