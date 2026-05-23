#pragma once

#include <cstddef>
#include <immintrin.h>

enum class KernelKind {
    Serial,
    SSE4P,
    SSE8P,
    SSE16P,
    AVX8P,
    AVX16P,
    AutoVec,
};

inline float ip_distance_serial(const float* v, const float* qry, size_t dim) {
    float ip = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        ip += v[d] * qry[d];
    }
    return 1.0f - ip;
}

inline float ip_distance_sse4p(const float* v, const float* qry, size_t dim) {
    __m128 sum = _mm_setzero_ps();
    size_t d = 0;
    for (; d + 4 <= dim; d += 4) {
        sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(&v[d]), _mm_loadu_ps(&qry[d])));
    }
    __m128 hadd = _mm_hadd_ps(sum, sum);
    hadd = _mm_hadd_ps(hadd, hadd);
    float ip = _mm_cvtss_f32(hadd);
    for (; d < dim; ++d) {
        ip += v[d] * qry[d];
    }
    return 1.0f - ip;
}

inline float ip_distance_sse8p(const float* v, const float* qry, size_t dim) {
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    size_t d = 0;
    for (; d + 8 <= dim; d += 8) {
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_loadu_ps(&v[d]), _mm_loadu_ps(&qry[d])));
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_loadu_ps(&v[d + 4]), _mm_loadu_ps(&qry[d + 4])));
    }
    __m128 total = _mm_add_ps(sum1, sum2);
    __m128 hadd = _mm_hadd_ps(total, total);
    hadd = _mm_hadd_ps(hadd, hadd);
    float ip = _mm_cvtss_f32(hadd);
    for (; d < dim; ++d) {
        ip += v[d] * qry[d];
    }
    return 1.0f - ip;
}

inline float ip_distance_sse16p(const float* v, const float* qry, size_t dim) {
    __m128 sum1 = _mm_setzero_ps();
    __m128 sum2 = _mm_setzero_ps();
    __m128 sum3 = _mm_setzero_ps();
    __m128 sum4 = _mm_setzero_ps();
    size_t d = 0;
    for (; d + 16 <= dim; d += 16) {
        sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_loadu_ps(&v[d]), _mm_loadu_ps(&qry[d])));
        sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_loadu_ps(&v[d + 4]), _mm_loadu_ps(&qry[d + 4])));
        sum3 = _mm_add_ps(sum3, _mm_mul_ps(_mm_loadu_ps(&v[d + 8]), _mm_loadu_ps(&qry[d + 8])));
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(_mm_loadu_ps(&v[d + 12]), _mm_loadu_ps(&qry[d + 12])));
    }
    __m128 total = _mm_add_ps(_mm_add_ps(sum1, sum2), _mm_add_ps(sum3, sum4));
    __m128 hadd = _mm_hadd_ps(total, total);
    hadd = _mm_hadd_ps(hadd, hadd);
    float ip = _mm_cvtss_f32(hadd);
    for (; d < dim; ++d) {
        ip += v[d] * qry[d];
    }
    return 1.0f - ip;
}

inline float ip_distance_avx8p(const float* v, const float* qry, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    size_t d = 0;
    for (; d + 8 <= dim; d += 8) {
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(&v[d]), _mm256_loadu_ps(&qry[d])));
    }
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    __m128 hadd = _mm_hadd_ps(s128, s128);
    hadd = _mm_hadd_ps(hadd, hadd);
    float ip = _mm_cvtss_f32(hadd);
    for (; d < dim; ++d) {
        ip += v[d] * qry[d];
    }
    return 1.0f - ip;
}

inline float ip_distance_avx16p(const float* v, const float* qry, size_t dim) {
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    size_t d = 0;
    for (; d + 16 <= dim; d += 16) {
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(_mm256_loadu_ps(&v[d]), _mm256_loadu_ps(&qry[d])));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(_mm256_loadu_ps(&v[d + 8]), _mm256_loadu_ps(&qry[d + 8])));
    }
    __m256 total = _mm256_add_ps(sum1, sum2);
    __m128 lo = _mm256_castps256_ps128(total);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    __m128 hadd = _mm_hadd_ps(s128, s128);
    hadd = _mm_hadd_ps(hadd, hadd);
    float ip = _mm_cvtss_f32(hadd);
    for (; d < dim; ++d) {
        ip += v[d] * qry[d];
    }
    return 1.0f - ip;
}

inline float ip_distance_auto(const float* v, const float* qry, size_t dim) {
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        sum += v[d] * qry[d];
    }
    return 1.0f - sum;
}

inline float ComputeDistanceByKernel(const float* v, const float* qry, size_t dim, KernelKind kernel) {
    switch (kernel) {
        case KernelKind::Serial: return ip_distance_serial(v, qry, dim);
        case KernelKind::SSE4P: return ip_distance_sse4p(v, qry, dim);
        case KernelKind::SSE8P: return ip_distance_sse8p(v, qry, dim);
        case KernelKind::SSE16P: return ip_distance_sse16p(v, qry, dim);
        case KernelKind::AVX8P: return ip_distance_avx8p(v, qry, dim);
        case KernelKind::AVX16P: return ip_distance_avx16p(v, qry, dim);
        case KernelKind::AutoVec: return ip_distance_auto(v, qry, dim);
    }
    return ip_distance_serial(v, qry, dim);
}
