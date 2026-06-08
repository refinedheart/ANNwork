#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include <omp.h>

#include "ann_bench_utils.h"
#include "hnswlib/hnswlib/hnswlib.h"
#include "ivf.h"
#include "ivf_neon.h"

struct LocalHNSWShard {
    hnswlib::InnerProductSpace* space;
    hnswlib::HierarchicalNSW<float>* index;
    size_t size;

    LocalHNSWShard() : space(nullptr), index(nullptr), size(0) {}
};

inline void ResetLocalHNSWShard(LocalHNSWShard& shard) {
    delete shard.index;
    delete shard.space;
    shard.index = nullptr;
    shard.space = nullptr;
    shard.size = 0;
}

struct IVFHNSWIndexState {
    size_t n;
    size_t dim;
    size_t nlist;
    size_t ef_search;
    bool built;
    std::vector<LocalHNSWShard> cluster_indexes;

    IVFHNSWIndexState()
        : n(0), dim(0), nlist(0), ef_search(0), built(false) {}
};

static IVFHNSWIndexState g_ivf_hnsw_index;

inline void ResetIVFHNSWIndex() {
    for (auto& shard : g_ivf_hnsw_index.cluster_indexes) {
        ResetLocalHNSWShard(shard);
    }
    g_ivf_hnsw_index.cluster_indexes.clear();
    g_ivf_hnsw_index.n = 0;
    g_ivf_hnsw_index.dim = 0;
    g_ivf_hnsw_index.nlist = 0;
    g_ivf_hnsw_index.ef_search = 0;
    g_ivf_hnsw_index.built = false;
}

inline std::priority_queue<AnnCandidate>
SelectTopCentroidsForQuery(float* query, size_t dim, size_t nprobe) {
    std::priority_queue<AnnCandidate> coarse_clusters;
    const size_t effective_nprobe = std::min(nprobe, g_ivf.nlist);
    for (size_t c = 0; c < g_ivf.nlist; ++c) {
        const float ip = neon_ip_f32(query, &g_ivf.centroids[c * dim], dim);
        const float dist = 1.0f - ip;
        PushBoundedCandidate(coarse_clusters, dist, static_cast<uint32_t>(c), effective_nprobe);
    }
    return coarse_clusters;
}

inline std::vector<uint32_t>
DrainSelectedCentroids(std::priority_queue<AnnCandidate> coarse_clusters) {
    std::vector<uint32_t> selected;
    selected.reserve(coarse_clusters.size());
    while (!coarse_clusters.empty()) {
        selected.push_back(coarse_clusters.top().second);
        coarse_clusters.pop();
    }
    return selected;
}

inline void BuildIVFHNSWIndex(float* base,
                              size_t n,
                              size_t dim,
                              size_t nlist,
                              size_t ef_search) {
    build_ivf_index(base, n, dim, nlist);

    ResetIVFHNSWIndex();
    g_ivf_hnsw_index.cluster_indexes.resize(g_ivf.nlist);
    g_ivf_hnsw_index.n = n;
    g_ivf_hnsw_index.dim = dim;
    g_ivf_hnsw_index.nlist = g_ivf.nlist;
    g_ivf_hnsw_index.ef_search = std::max<size_t>(ef_search, 32);

    for (size_t cid = 0; cid < g_ivf.nlist; ++cid) {
        const auto& ids = g_ivf.inverted_lists[cid];
        if (ids.empty()) {
            continue;
        }

        auto& shard = g_ivf_hnsw_index.cluster_indexes[cid];
        shard.space = new hnswlib::InnerProductSpace(dim);
        shard.index = new hnswlib::HierarchicalNSW<float>(
            shard.space, ids.size(), 16, 150);
        for (size_t local_idx = 0; local_idx < ids.size(); ++local_idx) {
            const uint32_t vid = ids[local_idx];
            shard.index->addPoint(base + static_cast<size_t>(vid) * dim,
                                  static_cast<hnswlib::labeltype>(local_idx));
        }
        shard.index->setEf(g_ivf_hnsw_index.ef_search);
        shard.size = ids.size();
    }

    g_ivf_hnsw_index.built = true;
}

inline std::priority_queue<AnnCandidate>
ivf_hnsw_search(float* base,
                float* query,
                size_t n,
                size_t dim,
                size_t k,
                size_t nlist,
                size_t nprobe,
                size_t per_cluster_k,
                size_t ef_search) {
    if (!g_ivf_hnsw_index.built ||
        g_ivf_hnsw_index.n != n ||
        g_ivf_hnsw_index.dim != dim ||
        g_ivf_hnsw_index.nlist != nlist ||
        g_ivf_hnsw_index.ef_search != std::max<size_t>(ef_search, 32)) {
        BuildIVFHNSWIndex(base, n, dim, nlist, ef_search);
    }

    const auto selected_clusters = DrainSelectedCentroids(
        SelectTopCentroidsForQuery(query, dim, nprobe));
    const size_t cluster_k = std::max(k, per_cluster_k);

    std::priority_queue<AnnCandidate> merged;
    for (uint32_t cid : selected_clusters) {
        const auto& ids = g_ivf.inverted_lists[cid];
        auto& shard = g_ivf_hnsw_index.cluster_indexes[cid];
        if (ids.empty() || shard.index == nullptr) {
            continue;
        }

        auto raw = shard.index->searchKnn(query, std::min(cluster_k, shard.size));
        while (!raw.empty()) {
            const auto item = raw.top();
            raw.pop();
            const uint32_t local_label = static_cast<uint32_t>(item.second);
            PushBoundedCandidate(merged, item.first, ids[local_label], k);
        }
    }
    return merged;
}

inline std::priority_queue<AnnCandidate>
ivf_hnsw_search_openmp(float* base,
                       float* query,
                       size_t n,
                       size_t dim,
                       size_t k,
                       int num_threads,
                       size_t nlist,
                       size_t nprobe,
                       size_t per_cluster_k,
                       size_t ef_search) {
    if (!g_ivf_hnsw_index.built ||
        g_ivf_hnsw_index.n != n ||
        g_ivf_hnsw_index.dim != dim ||
        g_ivf_hnsw_index.nlist != nlist ||
        g_ivf_hnsw_index.ef_search != std::max<size_t>(ef_search, 32)) {
        BuildIVFHNSWIndex(base, n, dim, nlist, ef_search);
    }

    const auto selected_clusters = DrainSelectedCentroids(
        SelectTopCentroidsForQuery(query, dim, nprobe));
    const size_t cluster_k = std::max(k, per_cluster_k);

    omp_set_num_threads(num_threads);
    std::vector<std::vector<AnnCandidate>> local_candidates(
        static_cast<size_t>(std::max(1, num_threads)));

    #pragma omp parallel for schedule(static)
    for (long long cluster_idx = 0;
         cluster_idx < static_cast<long long>(selected_clusters.size());
         ++cluster_idx) {
        const int tid = omp_get_thread_num();
        const uint32_t cid = selected_clusters[static_cast<size_t>(cluster_idx)];
        const auto& ids = g_ivf.inverted_lists[cid];
        auto& shard = g_ivf_hnsw_index.cluster_indexes[cid];
        if (ids.empty() || shard.index == nullptr) {
            continue;
        }

        auto raw = shard.index->searchKnn(query, std::min(cluster_k, shard.size));
        while (!raw.empty()) {
            const auto item = raw.top();
            raw.pop();
            const uint32_t local_label = static_cast<uint32_t>(item.second);
            local_candidates[static_cast<size_t>(tid)].emplace_back(
                item.first, ids[local_label]);
        }
    }

    std::priority_queue<AnnCandidate> merged;
    for (const auto& candidates : local_candidates) {
        for (const auto& item : candidates) {
            PushBoundedCandidate(merged, item.first, item.second, k);
        }
    }
    return merged;
}

struct HNSWOnHNSWState {
    size_t n;
    size_t dim;
    size_t parts;
    size_t ef_search;
    bool built;
    LocalHNSWShard top_index;
    std::vector<std::vector<uint32_t>> shard_ids;
    std::vector<float> shard_centroids;
    std::vector<LocalHNSWShard> leaf_indexes;

    HNSWOnHNSWState()
        : n(0), dim(0), parts(0), ef_search(0), built(false) {}
};

static HNSWOnHNSWState g_hnsw_on_hnsw;

inline void ResetHNSWOnHNSWIndex() {
    ResetLocalHNSWShard(g_hnsw_on_hnsw.top_index);
    for (auto& shard : g_hnsw_on_hnsw.leaf_indexes) {
        ResetLocalHNSWShard(shard);
    }
    g_hnsw_on_hnsw.shard_ids.clear();
    g_hnsw_on_hnsw.shard_centroids.clear();
    g_hnsw_on_hnsw.leaf_indexes.clear();
    g_hnsw_on_hnsw.n = 0;
    g_hnsw_on_hnsw.dim = 0;
    g_hnsw_on_hnsw.parts = 0;
    g_hnsw_on_hnsw.ef_search = 0;
    g_hnsw_on_hnsw.built = false;
}

inline void BuildHNSWLeafShard(LocalHNSWShard& shard,
                               float* base,
                               const std::vector<uint32_t>& ids,
                               size_t dim,
                               size_t ef_search) {
    if (ids.empty()) {
        return;
    }

    shard.space = new hnswlib::InnerProductSpace(dim);
    shard.index = new hnswlib::HierarchicalNSW<float>(
        shard.space, ids.size(), 16, 150);
    for (size_t local_idx = 0; local_idx < ids.size(); ++local_idx) {
        const uint32_t vid = ids[local_idx];
        shard.index->addPoint(base + static_cast<size_t>(vid) * dim,
                              static_cast<hnswlib::labeltype>(local_idx));
    }
    shard.index->setEf(std::max<size_t>(ef_search, 32));
    shard.size = ids.size();
}

inline void BuildHNSWOnHNSWIndex(float* base,
                                 size_t n,
                                 size_t dim,
                                 size_t parts,
                                 size_t ef_search) {
    ResetHNSWOnHNSWIndex();

    const size_t effective_parts = std::max<size_t>(1, std::min(parts, n));
    const auto counts = MakeBalancedCounts(n, static_cast<int>(effective_parts));
    const auto displs = MakeDisplacements(counts);

    g_hnsw_on_hnsw.n = n;
    g_hnsw_on_hnsw.dim = dim;
    g_hnsw_on_hnsw.parts = effective_parts;
    g_hnsw_on_hnsw.ef_search = std::max<size_t>(ef_search, 32);
    g_hnsw_on_hnsw.shard_ids.resize(effective_parts);
    g_hnsw_on_hnsw.leaf_indexes.resize(effective_parts);
    g_hnsw_on_hnsw.shard_centroids.assign(effective_parts * dim, 0.0f);

    for (size_t part = 0; part < effective_parts; ++part) {
        auto& ids = g_hnsw_on_hnsw.shard_ids[part];
        ids.reserve(counts[part]);
        for (size_t offset = 0; offset < counts[part]; ++offset) {
            ids.push_back(static_cast<uint32_t>(displs[part] + offset));
        }

        float* centroid = g_hnsw_on_hnsw.shard_centroids.data() + part * dim;
        if (!ids.empty()) {
            for (uint32_t vid : ids) {
                const float* vec = base + static_cast<size_t>(vid) * dim;
                for (size_t d = 0; d < dim; ++d) {
                    centroid[d] += vec[d];
                }
            }
            const float inv = 1.0f / static_cast<float>(ids.size());
            for (size_t d = 0; d < dim; ++d) {
                centroid[d] *= inv;
            }
        }

        BuildHNSWLeafShard(g_hnsw_on_hnsw.leaf_indexes[part],
                           base, ids, dim, g_hnsw_on_hnsw.ef_search);
    }

    g_hnsw_on_hnsw.top_index.space = new hnswlib::InnerProductSpace(dim);
    g_hnsw_on_hnsw.top_index.index = new hnswlib::HierarchicalNSW<float>(
        g_hnsw_on_hnsw.top_index.space, effective_parts, 8, 100);
    for (size_t part = 0; part < effective_parts; ++part) {
        g_hnsw_on_hnsw.top_index.index->addPoint(
            g_hnsw_on_hnsw.shard_centroids.data() + part * dim,
            static_cast<hnswlib::labeltype>(part));
    }
    g_hnsw_on_hnsw.top_index.index->setEf(std::max<size_t>(8, effective_parts));
    g_hnsw_on_hnsw.top_index.size = effective_parts;
    g_hnsw_on_hnsw.built = true;
}

inline std::priority_queue<AnnCandidate>
hnsw_on_hnsw_search(float* base,
                    float* query,
                    size_t n,
                    size_t dim,
                    size_t k,
                    size_t parts,
                    size_t shard_probe,
                    size_t per_shard_k,
                    size_t ef_search) {
    if (!g_hnsw_on_hnsw.built ||
        g_hnsw_on_hnsw.n != n ||
        g_hnsw_on_hnsw.dim != dim ||
        g_hnsw_on_hnsw.parts != std::max<size_t>(1, std::min(parts, n)) ||
        g_hnsw_on_hnsw.ef_search != std::max<size_t>(ef_search, 32)) {
        BuildHNSWOnHNSWIndex(base, n, dim, parts, ef_search);
    }

    const size_t shard_k = std::max(k, per_shard_k);
    auto selected = g_hnsw_on_hnsw.top_index.index->searchKnn(
        query, std::min(shard_probe, g_hnsw_on_hnsw.top_index.size));

    std::priority_queue<AnnCandidate> merged;
    while (!selected.empty()) {
        const uint32_t shard_id = static_cast<uint32_t>(selected.top().second);
        selected.pop();

        const auto& ids = g_hnsw_on_hnsw.shard_ids[shard_id];
        auto& leaf = g_hnsw_on_hnsw.leaf_indexes[shard_id];
        if (ids.empty() || leaf.index == nullptr) {
            continue;
        }

        auto raw = leaf.index->searchKnn(query, std::min(shard_k, leaf.size));
        while (!raw.empty()) {
            const auto item = raw.top();
            raw.pop();
            const uint32_t local_label = static_cast<uint32_t>(item.second);
            PushBoundedCandidate(merged, item.first, ids[local_label], k);
        }
    }
    return merged;
}
