#include "gpu_ann.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "ann_bench_utils.h"
#include "ivf.h"

namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct RunConfig {
    std::string mode = "ivf_gpu_grouped";
    std::string data_path = "/anndata/";
    size_t query_limit = 2000;
    size_t base_limit = 0;
    size_t k = 10;
    size_t batch_size = 64;
    size_t ivf_nlist = 128;
    size_t nprobe = 32;
    size_t topk_threads = 256;
};

RunConfig ParseConfig(int argc, char** argv) {
    RunConfig cfg;
    cfg.mode = GetEnvString("ANN_GPU_MODE", cfg.mode);
    cfg.data_path = NormalizeDataPath(GetEnvString("ANN_DATA_PATH", cfg.data_path));
    cfg.query_limit = GetEnvSizeT("ANN_QUERY_LIMIT", cfg.query_limit);
    cfg.base_limit = GetEnvSizeT("ANN_BASE_LIMIT", cfg.base_limit);
    cfg.k = GetEnvSizeT("ANN_TOPK", cfg.k);
    cfg.batch_size = GetEnvSizeT("ANN_GPU_BATCH", cfg.batch_size);
    cfg.ivf_nlist = GetEnvSizeT("ANN_IVF_NLIST", cfg.ivf_nlist);
    cfg.nprobe = GetEnvSizeT("ANN_IVF_NPROBE", cfg.nprobe);
    cfg.topk_threads = GetEnvSizeT("ANN_GPU_TOPK_THREADS", cfg.topk_threads);

    if (argc > 1) {
        cfg.mode = argv[1];
    }
    if (argc > 2) {
        cfg.batch_size = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
    }
    if (argc > 3) {
        cfg.query_limit = static_cast<size_t>(std::strtoull(argv[3], nullptr, 10));
    }
    return cfg;
}

bool IsFlatMode(const std::string& mode) {
    return mode == "flat_gpu" || mode == "flat_gpu_tiled" ||
           mode == "flat_gpu_naive";
}

bool IsIVFMode(const std::string& mode) {
    return mode == "ivf_gpu" || mode == "ivf_gpu_grouped";
}

GpuAnnConfig MakeGpuConfig(const RunConfig& cfg) {
    GpuAnnConfig gpu;
    gpu.batch_size = cfg.batch_size;
    gpu.k = cfg.k;
    gpu.nprobe = cfg.nprobe;
    gpu.topk_threads = cfg.topk_threads;
    gpu.flat_kernel = cfg.mode == "flat_gpu_naive"
                          ? GpuAnnFlatKernel::Naive
                          : GpuAnnFlatKernel::Tiled;
    gpu.ivf_mode = cfg.mode == "ivf_gpu"
                       ? GpuAnnIVFMode::WastefulBatch
                       : GpuAnnIVFMode::Grouped;
    return gpu;
}

void FlattenIVFIndex(std::vector<uint32_t>& offsets,
                     std::vector<uint32_t>& ids) {
    offsets.assign(g_ivf.nlist + 1, 0);
    ids.clear();
    ids.reserve(g_ivf.n);

    for (size_t c = 0; c < g_ivf.nlist; ++c) {
        offsets[c] = static_cast<uint32_t>(ids.size());
        for (uint32_t id : g_ivf.inverted_lists[c]) {
            ids.push_back(id);
        }
    }
    offsets[g_ivf.nlist] = static_cast<uint32_t>(ids.size());
}

double ComputeRecall(const GpuAnnTopK& result,
                     const int* gt,
                     size_t gt_dim,
                     size_t k) {
    double recall_sum = 0.0;
    for (size_t q = 0; q < result.query_count; ++q) {
        std::set<uint32_t> gtset;
        for (size_t i = 0; i < k; ++i) {
            gtset.insert(static_cast<uint32_t>(gt[q * gt_dim + i]));
        }

        size_t hits = 0;
        for (size_t i = 0; i < k; ++i) {
            const uint32_t id = result.ids[q * result.k + i];
            if (gtset.find(id) != gtset.end()) {
                ++hits;
            }
        }
        recall_sum += static_cast<double>(hits) / static_cast<double>(k);
    }
    return recall_sum / static_cast<double>(result.query_count);
}

void PrintTiming(const GpuAnnTiming& timing) {
    std::cout << "stage h2d query total (ms): " << timing.h2d_query_ms << "\n";
    std::cout << "stage flat mm total (ms): " << timing.flat_mm_ms << "\n";
    std::cout << "stage centroid total (ms): " << timing.centroid_ms << "\n";
    std::cout << "stage refine total (ms): " << timing.refine_ms << "\n";
    std::cout << "stage topk total (ms): " << timing.topk_ms << "\n";
    std::cout << "stage d2h total (ms): " << timing.d2h_ms << "\n";
    std::cout << "gpu batches: " << timing.batches << "\n";
    std::cout << "ivf refine groups: " << timing.refine_groups << "\n";
    std::cout << "ivf refine pairs: " << timing.refine_pairs << "\n";
    std::cout << "ivf skipped pairs in wasteful baseline: "
              << timing.skipped_pairs << "\n";
}

void PrintUsage() {
    std::cerr
        << "Usage: ./gpu_main [mode] [batch_size] [query_limit]\n"
        << "Modes:\n"
        << "  flat_gpu_naive    exact flat batch matrix baseline\n"
        << "  flat_gpu_tiled    exact flat tiled matrix baseline\n"
        << "  flat_gpu          alias of flat_gpu_tiled\n"
        << "  ivf_gpu           IVF batch baseline, computes union clusters for all queries\n"
        << "  ivf_gpu_grouped   IVF grouped optimization, computes only queries that need each cluster\n"
        << "Env knobs: ANN_DATA_PATH, ANN_BASE_LIMIT, ANN_TOPK, ANN_GPU_BATCH,\n"
        << "           ANN_IVF_NLIST, ANN_IVF_NPROBE, ANN_GPU_TOPK_THREADS\n";
}

}

int main(int argc, char** argv) {
    RunConfig cfg = ParseConfig(argc, argv);
    if (!IsFlatMode(cfg.mode) && !IsIVFMode(cfg.mode)) {
        std::cerr << "unknown GPU ANN mode: " << cfg.mode << "\n";
        PrintUsage();
        return 1;
    }
    if (cfg.k == 0 || cfg.k > GPU_ANN_MAX_K ||
        (IsIVFMode(cfg.mode) && (cfg.nprobe == 0 || cfg.nprobe > GPU_ANN_MAX_K))) {
        std::cerr << "ANN_TOPK and ANN_IVF_NPROBE must be in [1, "
                  << GPU_ANN_MAX_K << "] for the GPU top-k kernel\n";
        return 1;
    }
    if (cfg.batch_size == 0) {
        std::cerr << "ANN_GPU_BATCH must be positive\n";
        return 1;
    }

    std::string device_message;
    if (!gpu_ann_has_device(&device_message)) {
        std::cerr << "CUDA device check failed: " << device_message << "\n";
        return 2;
    }
    std::cerr << device_message << "\n";

    size_t test_number = 0;
    size_t base_number = 0;
    size_t vecdim = 0;
    size_t test_gt_d = 0;

    float* test_query = nullptr;
    int* test_gt_storage = nullptr;
    float* base = nullptr;
    try {
        test_query = LoadData<float>(cfg.data_path + "DEEP100K.query.fbin",
                                     test_number,
                                     vecdim);
        test_gt_storage = LoadData<int>(
            cfg.data_path + "DEEP100K.gt.query.100k.top100.bin",
            test_number,
            test_gt_d);
        base = LoadData<float>(cfg.data_path + "DEEP100K.base.100k.fbin",
                               base_number,
                               vecdim);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        delete[] test_query;
        delete[] test_gt_storage;
        delete[] base;
        return 1;
    }

    test_number = std::min(test_number, cfg.query_limit);
    if (cfg.base_limit > 0) {
        base_number = std::min(base_number, cfg.base_limit);
    }
    if (test_number == 0 || base_number == 0) {
        std::cerr << "empty benchmark after applying limits\n";
        delete[] test_query;
        delete[] test_gt_storage;
        delete[] base;
        return 1;
    }

    std::vector<int> recomputed_gt;
    const int* test_gt = test_gt_storage;
    if (cfg.base_limit > 0) {
        recomputed_gt = BuildExactGroundTruthForSubset(base,
                                                       test_query,
                                                       test_number,
                                                       base_number,
                                                       vecdim,
                                                       cfg.k);
        test_gt = recomputed_gt.data();
        test_gt_d = cfg.k;
    }

    cfg.ivf_nlist = std::min(cfg.ivf_nlist, base_number);
    cfg.nprobe = std::min(cfg.nprobe, cfg.ivf_nlist);

    GpuAnnConfig gpu_cfg = MakeGpuConfig(cfg);
    GpuAnnIndex gpu_index;
    std::string error;

    auto build_start = Clock::now();
    if (!gpu_ann_build_flat_index(gpu_index,
                                  base,
                                  base_number,
                                  vecdim,
                                  &error)) {
        std::cerr << error << "\n";
        delete[] test_query;
        delete[] test_gt_storage;
        delete[] base;
        return 1;
    }
    const double flat_index_ms = elapsed_ms(build_start);

    double cpu_ivf_build_ms = 0.0;
    if (IsIVFMode(cfg.mode)) {
        auto ivf_start = Clock::now();
        build_ivf_index(base, base_number, vecdim, cfg.ivf_nlist);
        cpu_ivf_build_ms = elapsed_ms(ivf_start);

        std::vector<uint32_t> offsets;
        std::vector<uint32_t> ids;
        FlattenIVFIndex(offsets, ids);

        if (!gpu_ann_attach_ivf_index(gpu_index,
                                      g_ivf.centroids,
                                      offsets.data(),
                                      ids.data(),
                                      g_ivf.nlist,
                                      ids.size(),
                                      &error)) {
            std::cerr << error << "\n";
            gpu_ann_free(gpu_index);
            delete[] test_query;
            delete[] test_gt_storage;
            delete[] base;
            return 1;
        }
    }

    GpuAnnTopK result;
    GpuAnnTiming timing;
    bool ok = false;
    if (IsFlatMode(cfg.mode)) {
        ok = gpu_ann_flat_search_batch(gpu_index,
                                       test_query,
                                       test_number,
                                       gpu_cfg,
                                       result,
                                       &timing,
                                       &error);
    } else {
        ok = gpu_ann_ivf_search_batch(gpu_index,
                                      test_query,
                                      test_number,
                                      gpu_cfg,
                                      result,
                                      &timing,
                                      &error);
    }

    if (!ok) {
        std::cerr << error << "\n";
        gpu_ann_free(gpu_index);
        delete[] test_query;
        delete[] test_gt_storage;
        delete[] base;
        return 1;
    }

    const double avg_recall = ComputeRecall(result, test_gt, test_gt_d, cfg.k);
    const double avg_latency_us =
        timing.total_ms * 1000.0 / static_cast<double>(test_number);

    std::cout << "mode: " << cfg.mode << "\n";
    std::cout << "base vectors: " << base_number << "\n";
    std::cout << "queries: " << test_number << "\n";
    std::cout << "dimension: " << vecdim << "\n";
    std::cout << "topk: " << cfg.k << "\n";
    std::cout << "gpu batch size: " << cfg.batch_size << "\n";
    if (IsIVFMode(cfg.mode)) {
        std::cout << "ivf nlist: " << cfg.ivf_nlist << "\n";
        std::cout << "ivf nprobe: " << cfg.nprobe << "\n";
    }
    std::cout << "index base copy time (ms): " << flat_index_ms << "\n";
    std::cout << "cpu ivf build time (ms): " << cpu_ivf_build_ms << "\n";
    std::cout << "average recall: " << avg_recall << "\n";
    std::cout << "average latency (us): " << avg_latency_us << "\n";
    std::cout << "total time (us): " << timing.total_ms * 1000.0 << "\n";
    PrintTiming(timing);

    gpu_ann_free(gpu_index);
    delete[] test_query;
    delete[] test_gt_storage;
    delete[] base;
    return 0;
}
