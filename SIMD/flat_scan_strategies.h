#pragma once
#include <arm_neon.h>
#include <queue>

inline float ip_neon_vaddv(float* v, float* qry, size_t dim) {
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
    return 1.0f - vaddvq_f32(total);
}

inline float ip_neon_vld4(float* v, float* qry, size_t dim) {
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);

    for (size_t d = 0; d < dim; d += 16) {
        float32x4x4_t vb = vld4q_f32(&v[d]);
        float32x4x4_t qb = vld4q_f32(&qry[d]);

        sum1 = vmlaq_f32(sum1, vb.val[0], qb.val[0]);
        sum2 = vmlaq_f32(sum2, vb.val[1], qb.val[1]);
        sum3 = vmlaq_f32(sum3, vb.val[2], qb.val[2]);
        sum4 = vmlaq_f32(sum4, vb.val[3], qb.val[3]);
    }

    float32x4_t total = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));
    float32x2_t lo = vget_low_f32(total);
    lo = vpadd_f32(lo, vget_high_f32(total));
    lo = vpadd_f32(lo, lo);
    return 1.0f - vget_lane_f32(lo, 0);
}

inline float ip_neon_combo(float* v, float* qry, size_t dim) {
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);

    for (size_t d = 0; d < dim; d += 16) {
        float32x4x4_t vb = vld4q_f32(&v[d]);
        float32x4x4_t qb = vld4q_f32(&qry[d]);

        sum1 = vmlaq_f32(sum1, vb.val[0], qb.val[0]);
        sum2 = vmlaq_f32(sum2, vb.val[1], qb.val[1]);
        sum3 = vmlaq_f32(sum3, vb.val[2], qb.val[2]);
        sum4 = vmlaq_f32(sum4, vb.val[3], qb.val[3]);
    }

    float32x4_t total = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));
    return 1.0f - vaddvq_f32(total);
}

inline float ip_neon_o3(float* v, float* qry, size_t dim) {
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

std::priority_queue<std::pair<float, uint32_t>> flat_search_neon16p_v2(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_neon_vaddv(&base[i * dim], query, dim);
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

std::priority_queue<std::pair<float, uint32_t>> flat_search_neon16p_vld4(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_neon_vld4(&base[i * dim], query, dim);
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

std::priority_queue<std::pair<float, uint32_t>> flat_search_neon16p_combo(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_neon_combo(&base[i * dim], query, dim);
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

std::priority_queue<std::pair<float, uint32_t>> flat_search_neon16p_o3(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_neon_o3(&base[i * dim], query, dim);
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
