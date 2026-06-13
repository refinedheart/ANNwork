#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <set>
#include <string>
#include <utility>
#include <vector>

using AnnCandidate = std::pair<float, uint32_t>;

template <typename T>
T* LoadData(const std::string& data_path, size_t& n, size_t& d, bool verbose = true) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("failed to open data file: " + data_path);
    }

    fin.read(reinterpret_cast<char*>(&n), 4);
    if (!fin) {
        throw std::runtime_error("failed to read vector count from: " + data_path);
    }
    fin.read(reinterpret_cast<char*>(&d), 4);
    if (!fin) {
        throw std::runtime_error("failed to read vector dimension from: " + data_path);
    }

    T* data = new T[n * d];
    const int sz = sizeof(T);
    for (size_t i = 0; i < n; ++i) {
        fin.read(reinterpret_cast<char*>(data) + i * d * sz, d * sz);
        if (!fin) {
            delete[] data;
            throw std::runtime_error("unexpected EOF while reading data file: " + data_path);
        }
    }
    fin.close();

    if (verbose) {
        std::cerr << "load data " << data_path << "\n";
        std::cerr << "dimension: " << d << "  number:" << n
                  << "  size_per_element:" << sizeof(T) << "\n";
    }

    return data;
}

inline std::string GetEnvString(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : fallback;
}

inline size_t GetEnvSizeT(const char* name, size_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    return static_cast<size_t>(std::strtoull(value, nullptr, 10));
}

inline int GetEnvInt(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    return std::atoi(value);
}

inline std::string NormalizeDataPath(std::string data_path) {
    if (!data_path.empty() && data_path.back() != '/') {
        data_path.push_back('/');
    }
    return data_path;
}

inline void PushBoundedCandidate(std::priority_queue<AnnCandidate>& heap,
                                 float dist, uint32_t id, size_t limit) {
    if (heap.size() < limit) {
        heap.emplace(dist, id);
    } else if (dist < heap.top().first) {
        heap.pop();
        heap.emplace(dist, id);
    }
}

inline float ComputeRecallFromQueue(std::priority_queue<AnnCandidate> result,
                                    const int* gt_row,
                                    size_t k) {
    std::set<uint32_t> gtset;
    for (size_t j = 0; j < k; ++j) {
        gtset.insert(static_cast<uint32_t>(gt_row[j]));
    }

    size_t acc = 0;
    while (!result.empty()) {
        if (gtset.find(result.top().second) != gtset.end()) {
            ++acc;
        }
        result.pop();
    }
    return static_cast<float>(acc) / static_cast<float>(k);
}

inline std::vector<uint32_t> QueueToSortedIds(std::priority_queue<AnnCandidate> result) {
    std::vector<AnnCandidate> ordered;
    ordered.reserve(result.size());
    while (!result.empty()) {
        ordered.push_back(result.top());
        result.pop();
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const AnnCandidate& lhs, const AnnCandidate& rhs) {
                  if (lhs.first != rhs.first) {
                      return lhs.first < rhs.first;
                  }
                  return lhs.second < rhs.second;
              });

    std::vector<uint32_t> ids;
    ids.reserve(ordered.size());
    for (const auto& item : ordered) {
        ids.push_back(item.second);
    }
    return ids;
}

inline void SerializeTopKQueue(std::priority_queue<AnnCandidate> result,
                               size_t k,
                               uint32_t global_offset,
                               std::vector<float>& distances,
                               std::vector<uint32_t>& ids) {
    distances.assign(k, std::numeric_limits<float>::infinity());
    ids.assign(k, std::numeric_limits<uint32_t>::max());

    std::vector<AnnCandidate> ordered;
    ordered.reserve(result.size());
    while (!result.empty()) {
        ordered.push_back(result.top());
        result.pop();
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const AnnCandidate& lhs, const AnnCandidate& rhs) {
                  if (lhs.first != rhs.first) {
                      return lhs.first < rhs.first;
                  }
                  return lhs.second < rhs.second;
              });

    const size_t limit = std::min(k, ordered.size());
    for (size_t i = 0; i < limit; ++i) {
        distances[i] = ordered[i].first;
        ids[i] = ordered[i].second + global_offset;
    }
}

inline void SerializeTopKQueueToBuffers(std::priority_queue<AnnCandidate> result,
                                        size_t k,
                                        uint32_t global_offset,
                                        float* distances,
                                        uint32_t* ids) {
    std::fill_n(distances, k, std::numeric_limits<float>::infinity());
    std::fill_n(ids, k, std::numeric_limits<uint32_t>::max());

    std::vector<AnnCandidate> ordered;
    ordered.reserve(result.size());
    while (!result.empty()) {
        ordered.push_back(result.top());
        result.pop();
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const AnnCandidate& lhs, const AnnCandidate& rhs) {
                  if (lhs.first != rhs.first) {
                      return lhs.first < rhs.first;
                  }
                  return lhs.second < rhs.second;
              });

    const size_t limit = std::min(k, ordered.size());
    for (size_t i = 0; i < limit; ++i) {
        distances[i] = ordered[i].first;
        ids[i] = ordered[i].second + global_offset;
    }
}

inline std::priority_queue<AnnCandidate>
MergeTopKArrays(const float* distances,
                const uint32_t* ids,
                size_t count,
                size_t k) {
    std::priority_queue<AnnCandidate> merged;
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] == std::numeric_limits<uint32_t>::max() ||
            !std::isfinite(distances[i])) {
            continue;
        }
        PushBoundedCandidate(merged, distances[i], ids[i], k);
    }
    return merged;
}

inline std::vector<size_t> MakeBalancedCounts(size_t total, int parts) {
    std::vector<size_t> counts(parts, total / static_cast<size_t>(parts));
    const size_t remainder = total % static_cast<size_t>(parts);
    for (size_t i = 0; i < remainder; ++i) {
        ++counts[i];
    }
    return counts;
}

inline std::vector<size_t> MakeDisplacements(const std::vector<size_t>& counts) {
    std::vector<size_t> displs(counts.size(), 0);
    for (size_t i = 1; i < counts.size(); ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
    return displs;
}

inline std::vector<size_t> MakeBatchSizes(size_t total, size_t batch_size) {
    std::vector<size_t> batches;
    if (batch_size == 0) {
        batch_size = total;
    }
    for (size_t offset = 0; offset < total; offset += batch_size) {
        batches.push_back(std::min(batch_size, total - offset));
    }
    return batches;
}

inline std::vector<int> BuildExactGroundTruthForSubset(const float* base,
                                                       const float* queries,
                                                       size_t query_count,
                                                       size_t base_count,
                                                       size_t dim,
                                                       size_t k) {
    std::vector<int> gt(query_count * k, -1);

    for (size_t qi = 0; qi < query_count; ++qi) {
        const float* query = queries + qi * dim;
        std::priority_queue<AnnCandidate> heap;

        for (size_t bi = 0; bi < base_count; ++bi) {
            const float* vec = base + bi * dim;
            float ip = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                ip += vec[d] * query[d];
            }
            const float dist = 1.0f - ip;
            PushBoundedCandidate(heap, dist, static_cast<uint32_t>(bi), k);
        }

        std::vector<AnnCandidate> ordered;
        ordered.reserve(heap.size());
        while (!heap.empty()) {
            ordered.push_back(heap.top());
            heap.pop();
        }
        std::sort(ordered.begin(), ordered.end(),
                  [](const AnnCandidate& lhs, const AnnCandidate& rhs) {
                      if (lhs.first != rhs.first) {
                          return lhs.first < rhs.first;
                      }
                      return lhs.second < rhs.second;
                  });

        for (size_t i = 0; i < ordered.size() && i < k; ++i) {
            gt[qi * k + i] = static_cast<int>(ordered[i].second);
        }
    }

    return gt;
}
