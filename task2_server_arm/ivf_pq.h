#ifndef IVF_PQ_H
#define IVF_PQ_H

#include "ivf.h"
#include "flat_scan_pq_kmeans_neon.h"

// IVF-PQ: IVF 聚类 + PQ 编码

struct IVFPQParams {
    PQParams pq;
    bool pq_trained;
};

static IVFPQParams g_ivf_pq;

// ---- IVF-PQ 索引构建 ----

inline void build_ivf_pq(float* base, size_t n, size_t dim,
                          size_t nlist = 1024) {
    srand(42);

    // Step 1: IVF 聚类
    if (!g_ivf.built) {
        build_ivf_index(base, n, dim, nlist);
    }

    // Step 2: PQ 编码 (对 base vectors 做 PQ)
    pq_train_kmeans_neon(base, n, dim, g_ivf_pq.pq);
    g_ivf_pq.pq_trained = true;

    fprintf(stderr, "IVF-PQ index built: nlist=%zu, PQ_M=%zu, PQ_KS=%zu\n",
            nlist, (size_t)PQ_M, (size_t)PQ_KS);
}

// ---- IVF-PQ 查询 (IVF 粗排 + PQ ADC scan + 精确 rerank) ----

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_pq_search(float* base, float* query, size_t n, size_t dim, size_t k,
              size_t nprobe = 64, size_t p = 550) {
    if (!g_ivf.built || !g_ivf_pq.pq_trained) { build_ivf_pq(base, n, dim); }

    size_t nlist = g_ivf.nlist;

    // Step 1: 粗排 — NEON IP 找到 nprobe 个最近的簇
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

    // Step 2: PQ 查表累加扫描
    float* lut = new float[PQ_M * PQ_KS];
    pq_compute_lut_neon(query, g_ivf_pq.pq, lut);

    std::priority_queue<std::pair<float, uint32_t>> pq_candidates;
    for (auto cid : selected_clusters) {
        for (auto vid : g_ivf.inverted_lists[cid]) {
            uint8_t* code = &g_ivf_pq.pq.codes[vid * PQ_M];
            float acc = 0.0f;
            for (size_t m = 0; m < PQ_M; ++m) {
                acc += lut[m * PQ_KS + code[m]];
            }
            float dist = 1.0f - acc;

            if (pq_candidates.size() < p) {
                pq_candidates.emplace(dist, vid);
            } else if (dist < pq_candidates.top().first) {
                pq_candidates.pop();
                pq_candidates.emplace(dist, vid);
            }
        }
    }
    delete[] lut;

    // Step 3: 精确 rerank
    std::vector<std::pair<float, uint32_t>> cands;
    cands.reserve(pq_candidates.size());
    while (!pq_candidates.empty()) {
        cands.push_back(pq_candidates.top());
        pq_candidates.pop();
    }

    std::priority_queue<std::pair<float, uint32_t>> res;
    for (auto& cand : cands) {
        size_t idx = cand.second;
        float* v = &base[idx * dim];

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
            res.emplace(dist, (uint32_t)idx);
        } else if (dist < res.top().first) {
            res.pop();
            res.emplace(dist, (uint32_t)idx);
        }
    }

    return res;
}

#endif // IVF_PQ_H
