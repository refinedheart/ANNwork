#pragma once
#include <queue>

std::priority_queue<std::pair<float, uint32_t>> flat_search(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (int i = 0; i < n; ++i) {
        float dis = 0;
        for (int d = 0; d < dim; ++d) {
            dis += base[d + i * dim] * query[d];
        }
        dis = 1 - dis;

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
