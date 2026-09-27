/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/pipelined_image_loader.hpp"

#include <stdexcept>

namespace lfs::io {
    namespace {
        [[noreturn]] void loader_unavailable() {
            throw std::runtime_error("Pipelined image loading is not part of this build");
        }
    } // namespace

    PipelinedImageLoader::PipelinedImageLoader(PipelinedLoaderConfig) {}
    PipelinedImageLoader::~PipelinedImageLoader() = default;

    void PipelinedImageLoader::prefetch(const std::vector<ImageRequest>&) {}
    void PipelinedImageLoader::prefetch(size_t, const std::filesystem::path&, const LoadParams&) {}
    void PipelinedImageLoader::canonicalize(const std::vector<ImageRequest>&) {}

    ReadyImage PipelinedImageLoader::get() { loader_unavailable(); }
    std::optional<ReadyImage> PipelinedImageLoader::try_get() { return std::nullopt; }
    std::optional<ReadyImage> PipelinedImageLoader::try_get_for(std::chrono::milliseconds) {
        return std::nullopt;
    }

    lfs::Result<LoaderCompletion> PipelinedImageLoader::get_completion() { loader_unavailable(); }
    std::optional<LoaderCompletion> PipelinedImageLoader::try_get_completion() { return std::nullopt; }
    std::optional<LoaderCompletion> PipelinedImageLoader::try_get_completion_for(std::chrono::milliseconds) {
        return std::nullopt;
    }

    lfs::core::Tensor PipelinedImageLoader::load_image_immediate(const std::filesystem::path&, const LoadParams&) {
        loader_unavailable();
    }

    size_t PipelinedImageLoader::ready_count() const { return 0; }
    size_t PipelinedImageLoader::in_flight_count() const { return 0; }
    void PipelinedImageLoader::observe_training_iteration(double, double, std::size_t) {}
    void PipelinedImageLoader::record_decode_latency(double) {}
    size_t PipelinedImageLoader::adaptive_prefetch_target() const { return 0; }
    void PipelinedImageLoader::clear() {}
    void PipelinedImageLoader::reclaim_idle_decoded_frames() {}
    size_t PipelinedImageLoader::release_host_cache(size_t) { return 0; }
    void PipelinedImageLoader::shutdown() {}
    PipelinedImageLoader::CacheStats PipelinedImageLoader::get_stats() const { return {}; }
    PipelinedImageLoader::GpuMemoryStats PipelinedImageLoader::get_gpu_memory_stats() const { return {}; }
    std::filesystem::path PipelinedImageLoader::run_spill_directory() const { return {}; }

} // namespace lfs::io
