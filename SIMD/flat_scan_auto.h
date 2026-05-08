#pragma once
#include <queue>

inline float ip_plain(float* __restrict v, float* __restrict qry, size_t dim) {
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        sum += v[d] * qry[d];
    }
    return 1.0f - sum;
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_auto(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_plain(&base[i * dim], query, dim);
        if (q.size() < k) {
            q.push({dis, i});
        } else {
            if (dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}
