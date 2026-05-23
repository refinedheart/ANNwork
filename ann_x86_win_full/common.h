#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct SearchResult {
    float recall;
    long long latency_us;
};

template <typename T>
T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("failed to open " + data_path +
                                 " (set ANN_DATA_PATH or copy anndata here)");
    }

    uint32_t n32 = 0;
    uint32_t d32 = 0;
    fin.read(reinterpret_cast<char*>(&n32), sizeof(n32));
    fin.read(reinterpret_cast<char*>(&d32), sizeof(d32));
    if (!fin) {
        throw std::runtime_error("failed to read header from " + data_path);
    }

    n = static_cast<size_t>(n32);
    d = static_cast<size_t>(d32);
    if (n == 0 || d == 0) {
        throw std::runtime_error("invalid shape in " + data_path);
    }

    T* data = new T[n * d];
    fin.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(n * d * sizeof(T)));
    if (!fin) {
        delete[] data;
        throw std::runtime_error("failed to read payload from " + data_path);
    }
    return data;
}

inline size_t ReadEnvSizeT(const char* name, size_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    return static_cast<size_t>(std::stoull(value));
}

template <typename QueueT>
inline float ComputeRecallTopK(QueueT res, const int* gt_row, size_t k) {
    std::set<uint32_t> gtset;
    for (size_t j = 0; j < k; ++j) {
        gtset.insert(static_cast<uint32_t>(gt_row[j]));
    }

    size_t acc = 0;
    while (!res.empty()) {
        if (gtset.find(res.top().second) != gtset.end()) {
            ++acc;
        }
        res.pop();
    }
    return static_cast<float>(acc) / static_cast<float>(k);
}

inline long long ElapsedUs(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}
