#pragma once

#include <algorithm>
#include <cstring>
#include <omp.h>
#include <queue>
#include <thread>
#include <vector>
#include "common.h"
#include "distance_x86.h"
#include "pq_x86.h"

struct IVFIndexX86 {
    size_t nlist = 0;
    size_t dim = 0;
    size_t n = 0;
    std::vector<float> centroids;
    std::vector<std::vector<uint32_t>> inverted_lists;
    bool built = false;
};

static IVFIndexX86 g_ivf_x86;

inline void kmeans_cluster_ip_x86(float* data, size_t n, size_t dim, float* centroids,
                                  size_t nlist, size_t iters = 20) {
    std::vector<bool> sel(n, false);
    for (size_t k = 0; k < nlist; ++k) {
        size_t idx = 0;
        do { idx = static_cast<size_t>(rand()) % n; } while (sel[idx]);
        sel[idx] = true;
        std::memcpy(&centroids[k * dim], &data[idx * dim], dim * sizeof(float));
    }

    std::vector<uint32_t> assign(n, 0);
    std::vector<int> cnt(nlist, 0);

    for (size_t iter = 0; iter < iters; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            const float* pt = &data[i * dim];
            float best = -1e30f;
            uint32_t best_k = 0;
            for (size_t k = 0; k < nlist; ++k) {
                const float* c = &centroids[k * dim];
                float ip = 0.0f;
                for (size_t j = 0; j < dim; ++j) {
                    ip += pt[j] * c[j];
                }
                if (ip > best) {
                    best = ip;
                    best_k = static_cast<uint32_t>(k);
                }
            }
            assign[i] = best_k;
        }

        std::memset(centroids, 0, nlist * dim * sizeof(float));
        std::fill(cnt.begin(), cnt.end(), 0);
        for (size_t i = 0; i < n; ++i) {
            size_t k = assign[i];
            ++cnt[k];
            float* c = &centroids[k * dim];
            const float* pt = &data[i * dim];
            for (size_t j = 0; j < dim; ++j) {
                c[j] += pt[j];
            }
        }

        for (size_t k = 0; k < nlist; ++k) {
            if (cnt[k] == 0) {
                continue;
            }
            float inv = 1.0f / static_cast<float>(cnt[k]);
            float* c = &centroids[k * dim];
            for (size_t j = 0; j < dim; ++j) {
                c[j] *= inv;
            }
        }
    }
}

inline void build_ivf_x86(float* base, size_t n, size_t dim, size_t nlist) {
    srand(42);
    g_ivf_x86.nlist = nlist;
    g_ivf_x86.dim = dim;
    g_ivf_x86.n = n;
    g_ivf_x86.centroids.assign(nlist * dim, 0.0f);
    g_ivf_x86.inverted_lists.assign(nlist, {});

    kmeans_cluster_ip_x86(base, n, dim, g_ivf_x86.centroids.data(), nlist);

    for (size_t i = 0; i < n; ++i) {
        const float* pt = &base[i * dim];
        float best = -1e30f;
        size_t best_k = 0;
        for (size_t k = 0; k < nlist; ++k) {
            const float* c = &g_ivf_x86.centroids[k * dim];
            float ip = 0.0f;
            for (size_t j = 0; j < dim; ++j) {
                ip += pt[j] * c[j];
            }
            if (ip > best) {
                best = ip;
                best_k = k;
            }
        }
        g_ivf_x86.inverted_lists[best_k].push_back(static_cast<uint32_t>(i));
    }
    g_ivf_x86.built = true;
}

inline std::vector<uint32_t> select_ivf_clusters_x86(const float* query, size_t dim, size_t nprobe) {
    std::priority_queue<std::pair<float, uint32_t>> coarse;
    for (size_t c = 0; c < g_ivf_x86.nlist; ++c) {
        float dist = ComputeDistanceByKernel(&g_ivf_x86.centroids[c * dim], query, dim, KernelKind::Serial);
        if (coarse.size() < nprobe) {
            coarse.emplace(dist, static_cast<uint32_t>(c));
        } else if (dist < coarse.top().first) {
            coarse.pop();
            coarse.emplace(dist, static_cast<uint32_t>(c));
        }
    }
    std::vector<uint32_t> selected;
    selected.reserve(nprobe);
    while (!coarse.empty()) {
        selected.push_back(coarse.top().second);
        coarse.pop();
    }
    return selected;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_x86(const float* base, const float* query, size_t n, size_t dim, size_t k,
               size_t nprobe, KernelKind exact_kernel) {
    if (!g_ivf_x86.built) {
        build_ivf_x86(const_cast<float*>(base), n, dim, std::min<size_t>(128, n));
    }

    auto selected = select_ivf_clusters_x86(query, dim, nprobe);
    std::priority_queue<std::pair<float, uint32_t>> res;
    for (uint32_t cid : selected) {
        for (uint32_t vid : g_ivf_x86.inverted_lists[cid]) {
            float dist = ComputeDistanceByKernel(base + static_cast<size_t>(vid) * dim, query, dim, exact_kernel);
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

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_pq_search_x86(const float* base, const float* query, size_t n, size_t dim, size_t k,
                  size_t nprobe, size_t rerank_p, KernelKind exact_kernel) {
    if (!g_ivf_x86.built) {
        build_ivf_x86(const_cast<float*>(base), n, dim, std::min<size_t>(128, n));
    }
    if (!g_pq_x86_built) {
        build_pq_x86(const_cast<float*>(base), n, dim);
    }

    auto selected = select_ivf_clusters_x86(query, dim, nprobe);
    std::vector<float> lut(X86_PQ_M * X86_PQ_KS, 0.0f);
    pq_compute_lut_x86(query, g_pq_x86, lut.data());

    std::priority_queue<std::pair<float, uint32_t>> coarse;
    for (uint32_t cid : selected) {
        for (uint32_t vid : g_ivf_x86.inverted_lists[cid]) {
            const uint8_t* code = &g_pq_x86.codes[static_cast<size_t>(vid) * X86_PQ_M];
            float acc = 0.0f;
            for (size_t m = 0; m < X86_PQ_M; ++m) {
                acc += lut[m * X86_PQ_KS + code[m]];
            }
            float dist = 1.0f - acc;
            if (coarse.size() < rerank_p) {
                coarse.emplace(dist, vid);
            } else if (dist < coarse.top().first) {
                coarse.pop();
                coarse.emplace(dist, vid);
            }
        }
    }

    std::vector<std::pair<float, uint32_t>> cands;
    cands.reserve(coarse.size());
    while (!coarse.empty()) {
        cands.push_back(coarse.top());
        coarse.pop();
    }

    std::priority_queue<std::pair<float, uint32_t>> res;
    for (const auto& cand : cands) {
        size_t idx = cand.second;
        float dist = ComputeDistanceByKernel(base + idx * dim, query, dim, exact_kernel);
        if (res.size() < k) {
            res.emplace(dist, static_cast<uint32_t>(idx));
        } else if (dist < res.top().first) {
            res.pop();
            res.emplace(dist, static_cast<uint32_t>(idx));
        }
    }
    return res;
}

inline SearchResult RunSingleIVFPQX86(const float* base, const float* query, const int* gt_row,
                                      size_t n, size_t dim, size_t k, size_t nprobe,
                                      size_t rerank_p, KernelKind exact_kernel) {
    auto start = std::chrono::steady_clock::now();
    auto res = ivf_pq_search_x86(base, query, n, dim, k, nprobe, rerank_p, exact_kernel);
    auto end = std::chrono::steady_clock::now();
    return {ComputeRecallTopK(res, gt_row, k), ElapsedUs(start, end)};
}

inline std::vector<SearchResult>
ivf_pq_batch_threads(const float* base, const float* test_query, const int* test_gt, size_t test_number,
                     size_t base_number, size_t dim, size_t gt_dim, size_t k,
                     int num_threads, size_t nprobe, size_t rerank_p, KernelKind exact_kernel) {
    std::vector<SearchResult> results(test_number);
    std::vector<std::thread> workers;
    size_t chunk = (test_number + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            size_t begin = static_cast<size_t>(t) * chunk;
            size_t end = std::min(begin + chunk, test_number);
            for (size_t i = begin; i < end; ++i) {
                results[i] = RunSingleIVFPQX86(
                    base,
                    test_query + i * dim,
                    test_gt + i * gt_dim,
                    base_number,
                    dim,
                    k,
                    nprobe,
                    rerank_p,
                    exact_kernel);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return results;
}

inline std::vector<SearchResult>
ivf_pq_batch_openmp(const float* base, const float* test_query, const int* test_gt, size_t test_number,
                    size_t base_number, size_t dim, size_t gt_dim, size_t k,
                    int num_threads, size_t nprobe, size_t rerank_p, KernelKind exact_kernel) {
    std::vector<SearchResult> results(test_number);
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(test_number); ++i) {
        size_t idx = static_cast<size_t>(i);
        results[idx] = RunSingleIVFPQX86(
            base,
            test_query + idx * dim,
            test_gt + idx * gt_dim,
            base_number,
            dim,
            k,
            nprobe,
            rerank_p,
            exact_kernel);
    }
    return results;
}
