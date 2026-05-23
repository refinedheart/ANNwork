#ifndef IVF_NEON_H
#define IVF_NEON_H

#include "ivf.h"
#include "flat_scan_pq_kmeans_neon.h"  // for neon_ip_f32

// NEON 优化的 IVF 查询
inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_neon(float* base, float* query, size_t n, size_t dim, size_t k,
                size_t nprobe = 64) {
    if (!g_ivf.built) { build_ivf_index(base, n, dim); }

    size_t nlist = g_ivf.nlist;

    // Step 1: 粗排 — NEON IP 计算 query 到所有簇中心的距离
    std::priority_queue<std::pair<float, uint32_t>> coarse_clusters;
    for (size_t c = 0; c < nlist; ++c) {
        float ip = neon_ip_f32(query, &g_ivf.centroids[c * dim], dim);
        float dist = 1.0f - ip;

        if (coarse_clusters.size() < nprobe) {
            coarse_clusters.emplace(dist, (uint32_t)c);
        } else if (dist < coarse_clusters.top().first) {
            coarse_clusters.pop();
            coarse_clusters.emplace(dist, (uint32_t)c);
        }
    }

    std::vector<uint32_t> selected_clusters;
    selected_clusters.reserve(nprobe);
    while (!coarse_clusters.empty()) {
        selected_clusters.push_back(coarse_clusters.top().second);
        coarse_clusters.pop();
    }

    // Step 2: 精排 — NEON IP (16-way unrolled)
    std::priority_queue<std::pair<float, uint32_t>> res;
    for (auto cid : selected_clusters) {
        for (auto vid : g_ivf.inverted_lists[cid]) {
            const float* v = &base[vid * dim];

            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            float32x4_t sum4 = vdupq_n_f32(0.0f);

            size_t d = 0;
            for (; d + 16 <= dim; d += 16) {
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

            for (; d < dim; ++d) { ip += v[d] * query[d]; }

            float dist = 1.0f - ip;

            if (res.size() < k) {
                res.emplace(dist, vid);
            } else if (dist < res.top().first) {
                res.pop();
                res.emplace(dist, vid);
            }
        }
    }

    return res;
}

#endif // IVF_NEON_H
