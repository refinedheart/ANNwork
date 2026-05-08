#ifndef FLAT_SCAN_X86_H
#define FLAT_SCAN_X86_H

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <cstddef>
#include <queue>

inline float ip_sse4p(float* v, float* qry, size_t dim) {
    __m128 sum = _mm_setzero_ps();
    for (size_t d = 0; d < dim; d += 4) {
        sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(&v[d]), _mm_loadu_ps(&qry[d])));
    }
    __m128 hadd = _mm_hadd_ps(sum, sum);
    hadd = _mm_hadd_ps(hadd, hadd);
    return 1.0f - _mm_cvtss_f32(hadd);
}

inline float ip_sse8p(float* v, float* qry, size_t dim) {
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    for (size_t d = 0; d < dim; d += 8) {
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_loadu_ps(&v[d]), _mm_loadu_ps(&qry[d])));
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_loadu_ps(&v[d + 4]), _mm_loadu_ps(&qry[d + 4])));
    }
    __m128 total = _mm_add_ps(sum1, sum2);
    __m128 hadd = _mm_hadd_ps(total, total);
    hadd = _mm_hadd_ps(hadd, hadd);
    return 1.0f - _mm_cvtss_f32(hadd);
}

inline float ip_sse16p(float* v, float* qry, size_t dim) {
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    __m128 sum3 = _mm_setzero_ps();
    __m128 sum4 = _mm_setzero_ps();
    for (size_t d = 0; d < dim; d += 16) {
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_loadu_ps(&v[d]), _mm_loadu_ps(&qry[d])));
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_loadu_ps(&v[d + 4]), _mm_loadu_ps(&qry[d + 4])));
        sum3 = _mm_add_ps(sum3, _mm_mul_ps(_mm_loadu_ps(&v[d + 8]), _mm_loadu_ps(&qry[d + 8])));
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(_mm_loadu_ps(&v[d + 12]), _mm_loadu_ps(&qry[d + 12])));
    }
    __m128 total = _mm_add_ps(_mm_add_ps(sum1, sum2), _mm_add_ps(sum3, sum4));
    __m128 hadd = _mm_hadd_ps(total, total);
    hadd = _mm_hadd_ps(hadd, hadd);
    return 1.0f - _mm_cvtss_f32(hadd);
}

inline float ip_sse16p_aligned(float* v, float* qry, size_t dim) {
    v = (float*)__builtin_assume_aligned(v, 16);
    qry = (float*)__builtin_assume_aligned(qry, 16);
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    __m128 sum3 = _mm_setzero_ps();
    __m128 sum4 = _mm_setzero_ps();
    for (size_t d = 0; d < dim; d += 16) {
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_load_ps(&v[d]), _mm_load_ps(&qry[d])));
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_load_ps(&v[d + 4]), _mm_load_ps(&qry[d + 4])));
        sum3 = _mm_add_ps(sum3, _mm_mul_ps(_mm_load_ps(&v[d + 8]), _mm_load_ps(&qry[d + 8])));
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(_mm_load_ps(&v[d + 12]), _mm_load_ps(&qry[d + 12])));
    }
    __m128 total = _mm_add_ps(_mm_add_ps(sum1, sum2), _mm_add_ps(sum3, sum4));
    __m128 hadd = _mm_hadd_ps(total, total);
    hadd = _mm_hadd_ps(hadd, hadd);
    return 1.0f - _mm_cvtss_f32(hadd);
}

inline float ip_avx8p(float* v, float* qry, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    for (size_t d = 0; d < dim; d += 8) {
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(&v[d]), _mm256_loadu_ps(&qry[d])));
    }
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    __m128 hadd = _mm_hadd_ps(s128, s128);
    hadd = _mm_hadd_ps(hadd, hadd);
    return 1.0f - _mm_cvtss_f32(hadd);
}

inline float ip_avx16p(float* v, float* qry, size_t dim) {
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    for (size_t d = 0; d < dim; d += 16) {
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(_mm256_loadu_ps(&v[d]), _mm256_loadu_ps(&qry[d])));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(_mm256_loadu_ps(&v[d + 8]), _mm256_loadu_ps(&qry[d + 8])));
    }
    __m256 total = _mm256_add_ps(sum1, sum2);
    __m128 lo = _mm256_castps256_ps128(total);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    __m128 hadd = _mm_hadd_ps(s128, s128);
    hadd = _mm_hadd_ps(hadd, hadd);
    return 1.0f - _mm_cvtss_f32(hadd);
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_sse4p(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_sse4p(&base[i * dim], query, dim);
        if (q.size() < k) { q.push({dis, (uint32_t)i}); }
        else if (dis < q.top().first) { q.pop(); q.push({dis, (uint32_t)i}); }
    }
    return q;
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_sse8p(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_sse8p(&base[i * dim], query, dim);
        if (q.size() < k) { q.push({dis, (uint32_t)i}); }
        else if (dis < q.top().first) { q.pop(); q.push({dis, (uint32_t)i}); }
    }
    return q;
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_sse16p(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_sse16p(&base[i * dim], query, dim);
        if (q.size() < k) { q.push({dis, (uint32_t)i}); }
        else if (dis < q.top().first) { q.pop(); q.push({dis, (uint32_t)i}); }
    }
    return q;
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_sse16p_aligned(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_sse16p_aligned(&base[i * dim], query, dim);
        if (q.size() < k) { q.push({dis, (uint32_t)i}); }
        else if (dis < q.top().first) { q.pop(); q.push({dis, (uint32_t)i}); }
    }
    return q;
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_avx8p(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_avx8p(&base[i * dim], query, dim);
        if (q.size() < k) { q.push({dis, (uint32_t)i}); }
        else if (dis < q.top().first) { q.pop(); q.push({dis, (uint32_t)i}); }
    }
    return q;
}

inline std::priority_queue<std::pair<float, uint32_t>> flat_search_avx16p(float* base, float* query, size_t n, size_t dim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for (int i = 0; i < n; ++i) {
        float dis = ip_avx16p(&base[i * dim], query, dim);
        if (q.size() < k) { q.push({dis, (uint32_t)i}); }
        else if (dis < q.top().first) { q.pop(); q.push({dis, (uint32_t)i}); }
    }
    return q;
}

#endif
