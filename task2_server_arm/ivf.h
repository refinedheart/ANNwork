#ifndef IVF_H
#define IVF_H

#include <vector>
#include <queue>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// IVF 索引结构
struct IVFIndex {
    size_t nlist;         // 聚类中心数
    size_t dim;           // 向量维度
    size_t n;             // base 向量数量
    float* centroids;     // [nlist * dim] 聚类中心
    std::vector<std::vector<uint32_t>> inverted_lists;  // 倒排链
    bool built;

    IVFIndex() : nlist(0), dim(0), n(0), centroids(nullptr), built(false) {}

    ~IVFIndex() {
        if (centroids) { delete[] centroids; centroids = nullptr; }
    }
};

static IVFIndex g_ivf;

// ---- KMeans 聚类 (内积距离) ----

inline void kmeans_cluster_ip(float* data, size_t n, size_t dim,
                               float* centroids, size_t nlist, size_t iters = 25) {
    // 随机初始化聚类中心
    bool* sel = new bool[n]();
    for (size_t k = 0; k < nlist; ++k) {
        size_t idx;
        do { idx = rand() % n; } while (sel[idx]);
        sel[idx] = true;
        memcpy(&centroids[k * dim], &data[idx * dim], dim * sizeof(float));
    }
    delete[] sel;

    uint32_t* assign = new uint32_t[n];
    int* cnt = new int[nlist];

    for (size_t iter = 0; iter < iters; ++iter) {
        // E-step: 分配每个向量到最近的中心
        for (size_t i = 0; i < n; ++i) {
            float best = -1e30f;
            uint32_t bk = 0;
            const float* pt = &data[i * dim];

            // 朴素内积距离 (后续 NEON 优化)
            for (size_t k = 0; k < nlist; ++k) {
                const float* c = &centroids[k * dim];
                float ip = 0.0f;
                for (size_t j = 0; j < dim; ++j) {
                    ip += pt[j] * c[j];
                }
                if (ip > best) { best = ip; bk = (uint32_t)k; }
            }
            assign[i] = bk;
        }

        // M-step: 更新聚类中心
        memset(centroids, 0, nlist * dim * sizeof(float));
        memset(cnt, 0, nlist * sizeof(int));

        for (size_t i = 0; i < n; ++i) {
            size_t k = assign[i];
            cnt[k]++;
            float* c = &centroids[k * dim];
            const float* pt = &data[i * dim];
            for (size_t j = 0; j < dim; ++j) {
                c[j] += pt[j];
            }
        }
        for (size_t k = 0; k < nlist; ++k) {
            if (cnt[k] > 0) {
                float inv = 1.0f / cnt[k];
                float* c = &centroids[k * dim];
                for (size_t j = 0; j < dim; ++j) {
                    c[j] *= inv;
                }
            }
        }
    }

    delete[] assign;
    delete[] cnt;
}

// ---- IVF 索引构建 ----

inline void build_ivf_index(float* base, size_t n, size_t dim, size_t nlist = 1024) {
    srand(42);
    g_ivf.nlist = nlist;
    g_ivf.dim = dim;
    g_ivf.n = n;

    // Step 1: KMeans 聚类
    g_ivf.centroids = new float[nlist * dim];
    kmeans_cluster_ip(base, n, dim, g_ivf.centroids, nlist, 25);

    // Step 2: 建立倒排索引
    g_ivf.inverted_lists.resize(nlist);
    for (size_t i = 0; i < n; ++i) {
        const float* pt = &base[i * dim];
        float best = -1e30f;
        size_t bk = 0;

        for (size_t k = 0; k < nlist; ++k) {
            const float* c = &g_ivf.centroids[k * dim];
            float ip = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                ip += pt[j] * c[j];
            }
            if (ip > best) { best = ip; bk = k; }
        }
        g_ivf.inverted_lists[bk].push_back((uint32_t)i);
    }

    g_ivf.built = true;

    // 打印统计信息
    size_t max_len = 0, min_len = n, total = 0;
    for (size_t k = 0; k < nlist; ++k) {
        size_t len = g_ivf.inverted_lists[k].size();
        total += len;
        if (len > max_len) max_len = len;
        if (len < min_len) min_len = len;
    }
    fprintf(stderr, "IVF index built: nlist=%zu, avg_list_len=%.1f, max=%zu, min=%zu\n",
            nlist, (double)total / nlist, max_len, min_len);
}

// ---- IVF 查询 (朴素版本) ----

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search(float* base, float* query, size_t n, size_t dim, size_t k,
           size_t nprobe = 64) {
    if (!g_ivf.built) { build_ivf_index(base, n, dim); }

    size_t nlist = g_ivf.nlist;

    // Step 1: 粗排 — 找 nprobe 个最近的聚类中心
    std::priority_queue<std::pair<float, uint32_t>> coarse_clusters;
    for (size_t c = 0; c < nlist; ++c) {
        const float* cent = &g_ivf.centroids[c * dim];
        float ip = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            ip += query[j] * cent[j];
        }
        float dist = 1.0f - ip;

        if (coarse_clusters.size() < nprobe) {
            coarse_clusters.emplace(dist, (uint32_t)c);
        } else if (dist < coarse_clusters.top().first) {
            coarse_clusters.pop();
            coarse_clusters.emplace(dist, (uint32_t)c);
        }
    }

    // 收集选中的簇
    std::vector<uint32_t> selected_clusters;
    selected_clusters.reserve(nprobe);
    while (!coarse_clusters.empty()) {
        selected_clusters.push_back(coarse_clusters.top().second);
        coarse_clusters.pop();
    }

    // Step 2: 精排 — 扫描选中簇中的 vectors
    std::priority_queue<std::pair<float, uint32_t>> res;
    for (auto cid : selected_clusters) {
        for (auto vid : g_ivf.inverted_lists[cid]) {
            const float* v = &base[vid * dim];
            float ip = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                ip += v[j] * query[j];
            }
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

#endif // IVF_H
