#ifndef IVF_HNSW_H
#define IVF_HNSW_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>
#include <vector>
#include <pthread.h>
#include <omp.h>
#include "ivf_neon.h"
#include "hnswlib/hnswlib/hnswlib.h"

struct IVFHNSWIndex {
    size_t nlist;
    size_t dim;
    size_t n;
    size_t ef_search;
    std::vector<float> centroids;
    std::vector<std::vector<uint32_t>> inverted_lists;
    std::vector<hnswlib::InnerProductSpace*> spaces;
    std::vector<hnswlib::HierarchicalNSW<float>*> graphs;
    bool built;

    IVFHNSWIndex() : nlist(0), dim(0), n(0), ef_search(0), built(false) {}
};

static IVFHNSWIndex g_ivf_hnsw;

inline void reset_ivf_hnsw_index() {
    for (auto* graph : g_ivf_hnsw.graphs) {
        delete graph;
    }
    for (auto* space : g_ivf_hnsw.spaces) {
        delete space;
    }
    g_ivf_hnsw = IVFHNSWIndex();
}

inline void ivf_hnsw_select_clusters(
    float* query,
    size_t nprobe,
    std::vector<uint32_t>& selected_clusters) {
    const size_t probe = std::min(nprobe, g_ivf_hnsw.nlist);
    std::priority_queue<std::pair<float, uint32_t>> coarse_clusters;

    for (size_t c = 0; c < g_ivf_hnsw.nlist; ++c) {
        float ip = neon_ip_f32(query, &g_ivf_hnsw.centroids[c * g_ivf_hnsw.dim],
                               g_ivf_hnsw.dim);
        float dist = 1.0f - ip;
        if (coarse_clusters.size() < probe) {
            coarse_clusters.emplace(dist, static_cast<uint32_t>(c));
        } else if (dist < coarse_clusters.top().first) {
            coarse_clusters.pop();
            coarse_clusters.emplace(dist, static_cast<uint32_t>(c));
        }
    }

    selected_clusters.clear();
    selected_clusters.reserve(probe);
    while (!coarse_clusters.empty()) {
        selected_clusters.push_back(coarse_clusters.top().second);
        coarse_clusters.pop();
    }
}

template <typename LabelT>
inline void ivf_hnsw_merge_result(
    std::priority_queue<std::pair<float, uint32_t>>& dst,
    std::priority_queue<std::pair<float, LabelT>>& src,
    size_t k) {
    while (!src.empty()) {
        auto raw_item = src.top();
        src.pop();
        std::pair<float, uint32_t> item(
            raw_item.first, static_cast<uint32_t>(raw_item.second));
        if (dst.size() < k) {
            dst.push(item);
        } else if (item.first < dst.top().first) {
            dst.pop();
            dst.push(item);
        }
    }
}

inline void ivf_hnsw_train_centroids(float* base,
                                     size_t n,
                                     size_t dim,
                                     size_t nlist,
                                     size_t kmeans_iters) {
    g_ivf_hnsw.centroids.assign(nlist * dim, 0.0f);

    for (size_t c = 0; c < nlist; ++c) {
        size_t idx = std::min((c * n) / nlist, n - 1);
        std::memcpy(&g_ivf_hnsw.centroids[c * dim],
                    &base[idx * dim],
                    dim * sizeof(float));
    }

    std::vector<uint32_t> assign(n);
    std::vector<float> next_centroids(nlist * dim);
    std::vector<uint32_t> counts(nlist);

    for (size_t iter = 0; iter < kmeans_iters; ++iter) {
        std::fill(next_centroids.begin(), next_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < n; ++i) {
            float* point = base + i * dim;
            float best = -1e30f;
            uint32_t best_c = 0;

            for (size_t c = 0; c < nlist; ++c) {
                float ip = neon_ip_f32(point, &g_ivf_hnsw.centroids[c * dim], dim);
                if (ip > best) {
                    best = ip;
                    best_c = static_cast<uint32_t>(c);
                }
            }

            assign[i] = best_c;
            counts[best_c]++;
            float* sum = &next_centroids[static_cast<size_t>(best_c) * dim];
            for (size_t d = 0; d < dim; ++d) {
                sum[d] += point[d];
            }
        }

        for (size_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) {
                continue;
            }
            float inv = 1.0f / static_cast<float>(counts[c]);
            float* centroid = &g_ivf_hnsw.centroids[c * dim];
            float* sum = &next_centroids[c * dim];
            for (size_t d = 0; d < dim; ++d) {
                centroid[d] = sum[d] * inv;
            }
        }
    }
}

inline void build_ivf_hnsw_index(float* base,
                                 size_t n,
                                 size_t dim,
                                 size_t nlist = 64,
                                 size_t kmeans_iters = 2,
                                 size_t m = 12,
                                 size_t ef_construction = 80,
                                 size_t ef_search = 64) {
    if (n == 0 || dim == 0) {
        return;
    }

    nlist = std::max<size_t>(1, std::min(nlist, n));
    reset_ivf_hnsw_index();

    g_ivf_hnsw.nlist = nlist;
    g_ivf_hnsw.dim = dim;
    g_ivf_hnsw.n = n;
    g_ivf_hnsw.ef_search = ef_search;
    g_ivf_hnsw.inverted_lists.assign(nlist, std::vector<uint32_t>());
    g_ivf_hnsw.spaces.assign(nlist, nullptr);
    g_ivf_hnsw.graphs.assign(nlist, nullptr);

    ivf_hnsw_train_centroids(base, n, dim, nlist, kmeans_iters);

    for (size_t i = 0; i < n; ++i) {
        float* point = base + i * dim;
        float best = -1e30f;
        uint32_t best_c = 0;
        for (size_t c = 0; c < nlist; ++c) {
            float ip = neon_ip_f32(point, &g_ivf_hnsw.centroids[c * dim], dim);
            if (ip > best) {
                best = ip;
                best_c = static_cast<uint32_t>(c);
            }
        }
        g_ivf_hnsw.inverted_lists[best_c].push_back(static_cast<uint32_t>(i));
    }

    size_t non_empty = 0;
    size_t total = 0;
    size_t max_len = 0;
    size_t min_len = n;
    for (size_t c = 0; c < nlist; ++c) {
        const auto& ids = g_ivf_hnsw.inverted_lists[c];
        total += ids.size();
        if (!ids.empty()) {
            non_empty++;
            max_len = std::max(max_len, ids.size());
            min_len = std::min(min_len, ids.size());

            g_ivf_hnsw.spaces[c] = new hnswlib::InnerProductSpace(dim);
            g_ivf_hnsw.graphs[c] = new hnswlib::HierarchicalNSW<float>(
                g_ivf_hnsw.spaces[c], ids.size(), m, ef_construction);
            for (uint32_t vid : ids) {
                g_ivf_hnsw.graphs[c]->addPoint(base + static_cast<size_t>(vid) * dim,
                                               static_cast<hnswlib::labeltype>(vid));
            }
            g_ivf_hnsw.graphs[c]->setEf(ef_search);
        }
    }

    if (non_empty == 0) {
        min_len = 0;
    }
    g_ivf_hnsw.built = true;

    std::fprintf(stderr,
                 "IVF-HNSW index built: nlist=%zu, non_empty=%zu, avg_list_len=%.1f, max=%zu, min=%zu, kmeans_iters=%zu\n",
                 nlist, non_empty, static_cast<double>(total) / nlist,
                 max_len, min_len, kmeans_iters);
}

inline void ivf_hnsw_set_ef(size_t ef_search) {
    if (!g_ivf_hnsw.built || g_ivf_hnsw.ef_search == ef_search) {
        return;
    }
    for (auto* graph : g_ivf_hnsw.graphs) {
        if (graph != nullptr) {
            graph->setEf(ef_search);
        }
    }
    g_ivf_hnsw.ef_search = ef_search;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_hnsw_search(float* base,
                float* query,
                size_t n,
                size_t dim,
                size_t k,
                size_t nprobe = 8,
                size_t ef_search = 64) {
    if (!g_ivf_hnsw.built) {
        build_ivf_hnsw_index(base, n, dim);
    }
    ivf_hnsw_set_ef(ef_search);

    std::vector<uint32_t> selected_clusters;
    ivf_hnsw_select_clusters(query, nprobe, selected_clusters);

    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (uint32_t cid : selected_clusters) {
        auto* graph = g_ivf_hnsw.graphs[cid];
        if (graph == nullptr) {
            continue;
        }
        size_t local_k = std::min(k, g_ivf_hnsw.inverted_lists[cid].size());
        if (local_k == 0) {
            continue;
        }
        auto raw = graph->searchKnn(query, local_k);
        ivf_hnsw_merge_result(merged, raw, k);
    }

    return merged;
}

struct IVFHNSWClusterArgs {
    float* query;
    const std::vector<uint32_t>* selected_clusters;
    size_t start_c;
    size_t end_c;
    size_t k;
    std::priority_queue<std::pair<float, uint32_t>> local_result;
};

static void* ivf_hnsw_cluster_worker(void* arg) {
    IVFHNSWClusterArgs* args = static_cast<IVFHNSWClusterArgs*>(arg);
    for (size_t pos = args->start_c; pos < args->end_c; ++pos) {
        uint32_t cid = (*args->selected_clusters)[pos];
        auto* graph = g_ivf_hnsw.graphs[cid];
        if (graph == nullptr) {
            continue;
        }
        size_t local_k = std::min(args->k, g_ivf_hnsw.inverted_lists[cid].size());
        if (local_k == 0) {
            continue;
        }
        auto raw = graph->searchKnn(args->query, local_k);
        ivf_hnsw_merge_result(args->local_result, raw, args->k);
    }
    return nullptr;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_hnsw_search_pthread(float* base,
                        float* query,
                        size_t n,
                        size_t dim,
                        size_t k,
                        int num_threads,
                        size_t nprobe = 8,
                        size_t ef_search = 64) {
    if (!g_ivf_hnsw.built) {
        build_ivf_hnsw_index(base, n, dim);
    }
    ivf_hnsw_set_ef(ef_search);

    std::vector<uint32_t> selected_clusters;
    ivf_hnsw_select_clusters(query, nprobe, selected_clusters);

    int threads_to_use = std::max(1, std::min<int>(num_threads,
        static_cast<int>(selected_clusters.size())));
    pthread_t* threads = new pthread_t[threads_to_use];
    IVFHNSWClusterArgs* args = new IVFHNSWClusterArgs[threads_to_use];
    size_t chunk = (selected_clusters.size() + threads_to_use - 1) / threads_to_use;

    for (int t = 0; t < threads_to_use; ++t) {
        args[t].query = query;
        args[t].selected_clusters = &selected_clusters;
        args[t].start_c = static_cast<size_t>(t) * chunk;
        args[t].end_c = std::min(static_cast<size_t>(t + 1) * chunk,
                                 selected_clusters.size());
        args[t].k = k;
        args[t].local_result = std::priority_queue<std::pair<float, uint32_t>>();
        pthread_create(&threads[t], nullptr, ivf_hnsw_cluster_worker, &args[t]);
    }

    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (int t = 0; t < threads_to_use; ++t) {
        pthread_join(threads[t], nullptr);
        ivf_hnsw_merge_result(merged, args[t].local_result, k);
    }

    delete[] threads;
    delete[] args;
    return merged;
}

inline std::priority_queue<std::pair<float, uint32_t>>
ivf_hnsw_search_openmp(float* base,
                       float* query,
                       size_t n,
                       size_t dim,
                       size_t k,
                       int num_threads,
                       size_t nprobe = 8,
                       size_t ef_search = 64) {
    if (!g_ivf_hnsw.built) {
        build_ivf_hnsw_index(base, n, dim);
    }
    ivf_hnsw_set_ef(ef_search);

    std::vector<uint32_t> selected_clusters;
    ivf_hnsw_select_clusters(query, nprobe, selected_clusters);

    int threads_to_use = std::max(1, std::min<int>(num_threads,
        static_cast<int>(selected_clusters.size())));
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(
        threads_to_use);

    omp_set_num_threads(threads_to_use);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local = local_results[tid];
        #pragma omp for schedule(dynamic, 1)
        for (size_t pos = 0; pos < selected_clusters.size(); ++pos) {
            uint32_t cid = selected_clusters[pos];
            auto* graph = g_ivf_hnsw.graphs[cid];
            if (graph == nullptr) {
                continue;
            }
            size_t local_k = std::min(k, g_ivf_hnsw.inverted_lists[cid].size());
            if (local_k == 0) {
                continue;
            }
            auto raw = graph->searchKnn(query, local_k);
            ivf_hnsw_merge_result(local, raw, k);
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> merged;
    for (auto& local : local_results) {
        ivf_hnsw_merge_result(merged, local, k);
    }
    return merged;
}

#endif // IVF_HNSW_H
