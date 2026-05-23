#pragma once

#include <cmath>
#include <cstring>
#include <omp.h>
#include <queue>
#include <thread>
#include <vector>
#include "common.h"
#include "distance_x86.h"

#ifndef PQ_M
#define PQ_M 8
#endif

static const size_t X86_PQ_M = PQ_M;
static const size_t X86_PQ_DSUB = 96 / X86_PQ_M;
static const size_t X86_PQ_KS = 256;
static const size_t X86_PQ_ITERS = 25;

struct PQParamsX86 {
    std::vector<float> cent;
    std::vector<uint8_t> codes;
    size_t n = 0;
    bool trained = false;

    PQParamsX86()
        : cent(X86_PQ_M * X86_PQ_KS * X86_PQ_DSUB, 0.0f) {}
};

static PQParamsX86 g_pq_x86;
static bool g_pq_x86_built = false;

inline void pq_kmeans_subspace_ip_scalar(float* data, size_t n, size_t dsub, float* cent, size_t ks, size_t iters) {
    std::vector<bool> selected(n, false);
    for (size_t k = 0; k < ks; ++k) {
        size_t idx = 0;
        do { idx = static_cast<size_t>(rand()) % n; } while (selected[idx]);
        selected[idx] = true;
        std::memcpy(&cent[k * dsub], &data[idx * dsub], dsub * sizeof(float));
    }

    std::vector<uint8_t> assign(n, 0);
    std::vector<int> cnt(ks, 0);

    for (size_t iter = 0; iter < iters; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            const float* pt = &data[i * dsub];
            float best = -1e30f;
            uint8_t best_k = 0;
            for (size_t k = 0; k < ks; ++k) {
                const float* c = &cent[k * dsub];
                float ip = 0.0f;
                for (size_t j = 0; j < dsub; ++j) {
                    ip += pt[j] * c[j];
                }
                if (ip > best) {
                    best = ip;
                    best_k = static_cast<uint8_t>(k);
                }
            }
            assign[i] = best_k;
        }

        std::memset(cent, 0, ks * dsub * sizeof(float));
        std::fill(cnt.begin(), cnt.end(), 0);
        for (size_t i = 0; i < n; ++i) {
            size_t k = assign[i];
            ++cnt[k];
            float* c = &cent[k * dsub];
            const float* pt = &data[i * dsub];
            for (size_t j = 0; j < dsub; ++j) {
                c[j] += pt[j];
            }
        }
        for (size_t k = 0; k < ks; ++k) {
            if (cnt[k] == 0) {
                continue;
            }
            float inv = 1.0f / static_cast<float>(cnt[k]);
            float* c = &cent[k * dsub];
            for (size_t j = 0; j < dsub; ++j) {
                c[j] *= inv;
            }
        }
    }
}

inline void build_pq_x86(float* base, size_t n, size_t dim) {
    srand(42);
    g_pq_x86.n = n;
    g_pq_x86.trained = true;
    g_pq_x86.codes.assign(n * X86_PQ_M, 0);

    for (size_t m = 0; m < X86_PQ_M; ++m) {
        std::vector<float> sub(n * X86_PQ_DSUB, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            std::memcpy(&sub[i * X86_PQ_DSUB], &base[i * dim + m * X86_PQ_DSUB], X86_PQ_DSUB * sizeof(float));
        }

        float* c = &g_pq_x86.cent[m * X86_PQ_KS * X86_PQ_DSUB];
        pq_kmeans_subspace_ip_scalar(sub.data(), n, X86_PQ_DSUB, c, X86_PQ_KS, X86_PQ_ITERS);

        for (size_t i = 0; i < n; ++i) {
            const float* pt = &sub[i * X86_PQ_DSUB];
            float best = -1e30f;
            uint8_t best_k = 0;
            for (size_t k = 0; k < X86_PQ_KS; ++k) {
                const float* ck = &c[k * X86_PQ_DSUB];
                float ip = 0.0f;
                for (size_t j = 0; j < X86_PQ_DSUB; ++j) {
                    ip += pt[j] * ck[j];
                }
                if (ip > best) {
                    best = ip;
                    best_k = static_cast<uint8_t>(k);
                }
            }
            g_pq_x86.codes[i * X86_PQ_M + m] = best_k;
        }
    }

    g_pq_x86_built = true;
}

inline void pq_compute_lut_x86(const float* query, const PQParamsX86& pq, float* lut) {
    for (size_t m = 0; m < X86_PQ_M; ++m) {
        const float* qs = &query[m * X86_PQ_DSUB];
        const float* cents = &pq.cent[m * X86_PQ_KS * X86_PQ_DSUB];
        float* lm = &lut[m * X86_PQ_KS];
        for (size_t k = 0; k < X86_PQ_KS; ++k) {
            const float* c = &cents[k * X86_PQ_DSUB];
            float acc = 0.0f;
            for (size_t d = 0; d < X86_PQ_DSUB; ++d) {
                acc += qs[d] * c[d];
            }
            lm[k] = acc;
        }
    }
}

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_pq_x86(const float* base, const float* query, size_t n, size_t dim, size_t k,
                   size_t rerank_p, KernelKind exact_kernel) {
    if (!g_pq_x86_built) {
        build_pq_x86(const_cast<float*>(base), n, dim);
    }

    std::vector<float> lut(X86_PQ_M * X86_PQ_KS, 0.0f);
    pq_compute_lut_x86(query, g_pq_x86, lut.data());

    std::priority_queue<std::pair<float, uint32_t>> coarse;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* code = &g_pq_x86.codes[i * X86_PQ_M];
        float acc = 0.0f;
        for (size_t m = 0; m < X86_PQ_M; ++m) {
            acc += lut[m * X86_PQ_KS + code[m]];
        }
        float dist = 1.0f - acc;
        if (coarse.size() < rerank_p) {
            coarse.emplace(dist, static_cast<uint32_t>(i));
        } else if (dist < coarse.top().first) {
            coarse.pop();
            coarse.emplace(dist, static_cast<uint32_t>(i));
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

inline SearchResult RunSinglePQX86(const float* base, const float* query, const int* gt_row,
                                   size_t n, size_t dim, size_t k, size_t rerank_p,
                                   KernelKind exact_kernel) {
    auto start = std::chrono::steady_clock::now();
    auto res = flat_search_pq_x86(base, query, n, dim, k, rerank_p, exact_kernel);
    auto end = std::chrono::steady_clock::now();
    return {ComputeRecallTopK(res, gt_row, k), ElapsedUs(start, end)};
}

inline std::vector<SearchResult>
pq_batch_threads(const float* base, const float* test_query, const int* test_gt, size_t test_number,
                 size_t base_number, size_t dim, size_t gt_dim, size_t k, int num_threads,
                 size_t rerank_p, KernelKind exact_kernel) {
    std::vector<SearchResult> results(test_number);
    std::vector<std::thread> workers;
    size_t chunk = (test_number + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            size_t begin = static_cast<size_t>(t) * chunk;
            size_t end = std::min(begin + chunk, test_number);
            for (size_t i = begin; i < end; ++i) {
                results[i] = RunSinglePQX86(
                    base,
                    test_query + i * dim,
                    test_gt + i * gt_dim,
                    base_number,
                    dim,
                    k,
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
pq_batch_openmp(const float* base, const float* test_query, const int* test_gt, size_t test_number,
                size_t base_number, size_t dim, size_t gt_dim, size_t k, int num_threads,
                size_t rerank_p, KernelKind exact_kernel) {
    std::vector<SearchResult> results(test_number);
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(test_number); ++i) {
        size_t idx = static_cast<size_t>(i);
        results[idx] = RunSinglePQX86(
            base,
            test_query + idx * dim,
            test_gt + idx * gt_dim,
            base_number,
            dim,
            k,
            rerank_p,
            exact_kernel);
    }
    return results;
}
