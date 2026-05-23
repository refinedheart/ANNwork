#pragma once

#include <omp.h>
#include <queue>
#include <thread>
#include <vector>
#include "common.h"
#include "distance_x86.h"

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_topk(const float* base, const float* query, size_t n, size_t dim, size_t k, KernelKind kernel) {
    std::priority_queue<std::pair<float, uint32_t>> res;
    for (size_t i = 0; i < n; ++i) {
        float dist = ComputeDistanceByKernel(base + i * dim, query, dim, kernel);
        if (res.size() < k) {
            res.emplace(dist, static_cast<uint32_t>(i));
        } else if (dist < res.top().first) {
            res.pop();
            res.emplace(dist, static_cast<uint32_t>(i));
        }
    }
    return res;
}

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_partition_threads(const float* base, const float* query, size_t n, size_t dim,
                              size_t k, int num_threads, KernelKind kernel, size_t local_p = 100) {
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(num_threads);
    std::vector<std::thread> workers;
    size_t chunk = (n + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            size_t begin = static_cast<size_t>(t) * chunk;
            size_t end = std::min(begin + chunk, n);
            auto& q = local_results[t];
            for (size_t i = begin; i < end; ++i) {
                float dist = ComputeDistanceByKernel(base + i * dim, query, dim, kernel);
                if (q.size() < local_p) {
                    q.emplace(dist, static_cast<uint32_t>(i));
                } else if (dist < q.top().first) {
                    q.pop();
                    q.emplace(dist, static_cast<uint32_t>(i));
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (auto& q : local_results) {
        while (!q.empty()) {
            auto item = q.top();
            q.pop();
            if (merged.size() < k) {
                merged.push(item);
            } else if (item.first < merged.top().first) {
                merged.pop();
                merged.push(item);
            }
        }
    }
    return merged;
}

inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_partition_openmp(const float* base, const float* query, size_t n, size_t dim,
                             size_t k, int num_threads, KernelKind kernel, size_t local_p = 100) {
    omp_set_num_threads(num_threads);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& q = local_results[tid];
        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            float dist = ComputeDistanceByKernel(base + static_cast<size_t>(i) * dim, query, dim, kernel);
            if (q.size() < local_p) {
                q.emplace(dist, static_cast<uint32_t>(i));
            } else if (dist < q.top().first) {
                q.pop();
                q.emplace(dist, static_cast<uint32_t>(i));
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (auto& q : local_results) {
        while (!q.empty()) {
            auto item = q.top();
            q.pop();
            if (merged.size() < k) {
                merged.push(item);
            } else if (item.first < merged.top().first) {
                merged.pop();
                merged.push(item);
            }
        }
    }
    return merged;
}

inline SearchResult RunSingleFlat(const float* base, const float* query, const int* gt_row,
                                  size_t n, size_t dim, size_t k, KernelKind kernel) {
    auto start = std::chrono::steady_clock::now();
    auto res = flat_search_topk(base, query, n, dim, k, kernel);
    auto end = std::chrono::steady_clock::now();
    return {ComputeRecallTopK(res, gt_row, k), ElapsedUs(start, end)};
}

inline SearchResult RunSingleFlatPartitionThreads(const float* base, const float* query, const int* gt_row,
                                                  size_t n, size_t dim, size_t k, int num_threads,
                                                  KernelKind kernel, size_t local_p = 100) {
    auto start = std::chrono::steady_clock::now();
    auto res = flat_search_partition_threads(base, query, n, dim, k, num_threads, kernel, local_p);
    auto end = std::chrono::steady_clock::now();
    return {ComputeRecallTopK(res, gt_row, k), ElapsedUs(start, end)};
}

inline SearchResult RunSingleFlatPartitionOpenMP(const float* base, const float* query, const int* gt_row,
                                                 size_t n, size_t dim, size_t k, int num_threads,
                                                 KernelKind kernel, size_t local_p = 100) {
    auto start = std::chrono::steady_clock::now();
    auto res = flat_search_partition_openmp(base, query, n, dim, k, num_threads, kernel, local_p);
    auto end = std::chrono::steady_clock::now();
    return {ComputeRecallTopK(res, gt_row, k), ElapsedUs(start, end)};
}

inline std::vector<SearchResult>
flat_batch_threads(const float* base, const float* test_query, const int* test_gt, size_t test_number,
                   size_t base_number, size_t dim, size_t gt_dim, size_t k,
                   int num_threads, KernelKind kernel) {
    std::vector<SearchResult> results(test_number);
    std::vector<std::thread> workers;
    size_t chunk = (test_number + static_cast<size_t>(num_threads) - 1) / static_cast<size_t>(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            size_t begin = static_cast<size_t>(t) * chunk;
            size_t end = std::min(begin + chunk, test_number);
            for (size_t i = begin; i < end; ++i) {
                results[i] = RunSingleFlat(
                    base,
                    test_query + i * dim,
                    test_gt + i * gt_dim,
                    base_number,
                    dim,
                    k,
                    kernel);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return results;
}

inline std::vector<SearchResult>
flat_batch_openmp(const float* base, const float* test_query, const int* test_gt, size_t test_number,
                  size_t base_number, size_t dim, size_t gt_dim, size_t k,
                  int num_threads, KernelKind kernel) {
    std::vector<SearchResult> results(test_number);
    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(test_number); ++i) {
        size_t idx = static_cast<size_t>(i);
        results[idx] = RunSingleFlat(
            base,
            test_query + idx * dim,
            test_gt + idx * gt_dim,
            base_number,
            dim,
            k,
            kernel);
    }
    return results;
}
