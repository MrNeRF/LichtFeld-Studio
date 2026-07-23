/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/failure_report.hpp"
#include "core/logger.hpp"
#include "io/pipelined_image_loader.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <set>
#include <thread>
#include <vector>

namespace {

    using lfs::io::ImageRequest;
    using lfs::io::LoaderCompletion;
    using lfs::io::PipelinedImageLoader;
    using lfs::io::PipelinedLoaderConfig;
    using lfs::io::SidecarCount;
    using lfs::io::SidecarKind;
    using lfs::io::SidecarPolicy;

    class LogHandlerGuard {
    public:
        explicit LogHandlerGuard(lfs::core::LogHandler handler)
            : token_(lfs::core::Logger::get().add_log_handler(std::move(handler))) {}
        ~LogHandlerGuard() { lfs::core::Logger::get().remove_log_handler(token_); }

    private:
        lfs::core::LogHandlerToken token_;
    };

    std::filesystem::path image_path() {
        return std::filesystem::path(TEST_DATA_DIR) / "bicycle/images_4/_DSC8744.JPG";
    }

    std::filesystem::path mask_path() {
        return std::filesystem::path(TEST_DATA_DIR) / "bicycle/masks/_DSC8744.png";
    }

    PipelinedLoaderConfig config() {
        PipelinedLoaderConfig result;
        result.jpeg_batch_size = 4;
        result.prefetch_count = 16;
        result.output_queue_size = 16;
        result.decoder_pool_size = 4;
        result.io_threads = 2;
        result.cold_process_threads = 2;
        result.use_filesystem_cache = false;
        result.max_cache_bytes = 128ULL * 1024ULL * 1024ULL;
        return result;
    }

    ImageRequest request(const std::uint64_t sequence) {
        ImageRequest result;
        result.sequence_id = sequence;
        result.path = image_path();
        result.params.max_width = 32;
        return result;
    }

    void set_sidecar(ImageRequest& request, const SidecarKind kind,
                     const std::filesystem::path& path) {
        switch (kind) {
        case SidecarKind::Mask: request.mask_path = path; break;
        case SidecarKind::Depth:
            request.depth_path = path;
            request.aux_target_width = 32;
            request.aux_target_height = 32;
            break;
        case SidecarKind::Normal:
            request.normal_path = path;
            request.aux_target_width = 32;
            request.aux_target_height = 32;
            break;
        }
    }

    std::filesystem::path valid_sidecar(const SidecarKind kind) {
        return kind == SidecarKind::Normal ? image_path() : mask_path();
    }

    const SidecarCount& count_for(const lfs::io::SidecarTally& tally,
                                  const SidecarKind kind) {
        switch (kind) {
        case SidecarKind::Mask: return tally.mask;
        case SidecarKind::Depth: return tally.depth;
        case SidecarKind::Normal: return tally.normal;
        }
        std::abort();
    }

    void expect_count(const SidecarCount& count, const std::uint64_t requested,
                      const std::uint64_t delivered, const std::uint64_t failed) {
        EXPECT_EQ(count.requested, requested);
        EXPECT_EQ(count.delivered, delivered);
        EXPECT_EQ(count.failed, failed);
    }

    void retire_events(LoaderCompletion& completion) {
        if (!completion.outcome) {
            return;
        }
        if (completion.outcome->depth_ready_event) {
            EXPECT_EQ(cudaEventDestroy(completion.outcome->depth_ready_event), cudaSuccess);
            completion.outcome->depth_ready_event = nullptr;
        }
        if (completion.outcome->normal_ready_event) {
            EXPECT_EQ(cudaEventDestroy(completion.outcome->normal_ready_event), cudaSuccess);
            completion.outcome->normal_ready_event = nullptr;
        }
    }

    LoaderCompletion get_completion(PipelinedImageLoader& loader) {
        auto result = loader.get_completion();
        EXPECT_TRUE(result.has_value());
        if (!result) {
            throw std::runtime_error(std::string(result.error().detail()));
        }
        return std::move(*result);
    }

    void expect_reconciled(const PipelinedImageLoader& loader) {
        const auto stats = loader.get_stats();
        EXPECT_EQ(stats.prefetch_queue_size, 0u);
        EXPECT_EQ(stats.hot_queue_size, 0u);
        EXPECT_EQ(stats.cold_queue_size, 0u);
        EXPECT_EQ(stats.output_queue_size, 0u);
        EXPECT_EQ(stats.pending_pairs_count, 0u);
        EXPECT_EQ(loader.in_flight_count(), 0u);
        EXPECT_EQ(stats.accepted_sequences,
                  stats.succeeded_sequences + stats.failed_sequences +
                      stats.cancelled_sequences);
    }

} // namespace

// FailureReport dedup fingerprints on {domain, code, detection site, operation}; two tests
// tripping the same report site in one binary run collide unless each resets the window
// (same requirement as PlyErrorTaxonomyTest).
class PipelinedLoaderLedger : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::core::reset_failure_report_dedup_for_testing();
    }
};

TEST_F(PipelinedLoaderLedger, WarnAndContinueDeliversDegradedCamera) {
    ASSERT_TRUE(std::filesystem::is_regular_file(image_path()));
    std::atomic<std::uint64_t> degradation_reports{0};
    LogHandlerGuard log_guard(
        [&](lfs::core::LogLevel, const lfs::core::SourceSite&, const std::string_view message) {
            if (message.find("optional sidecar degradation") != std::string_view::npos) {
                degradation_reports.fetch_add(1, std::memory_order_relaxed);
            }
        });
    auto cfg = config();
    PipelinedImageLoader loader(cfg);
    auto item = request(1);
    item.mask_path = image_path().parent_path() / "missing-mask.png";
    loader.prefetch({item});

    auto completion = get_completion(loader);
    ASSERT_TRUE(completion.outcome.has_value());
    EXPECT_FALSE(completion.outcome->mask.has_value());
    expect_count(completion.sidecars.mask, 1, 0, 1);
    retire_events(completion);
    EXPECT_EQ(loader.get_stats().total_images_loaded, 1u);
    EXPECT_EQ(degradation_reports.load(std::memory_order_relaxed), 1u);
}

TEST_F(PipelinedLoaderLedger, RequiredSidecarFailsCamera) {
    auto cfg = config();
    cfg.mask_policy = SidecarPolicy::Required;
    PipelinedImageLoader loader(cfg);
    auto item = request(2);
    item.mask_path = image_path().parent_path() / "missing-required-mask.png";
    loader.prefetch({item});

    auto completion = get_completion(loader);
    ASSERT_FALSE(completion.outcome.has_value());
    EXPECT_EQ(completion.outcome.error().code(), lfs::ErrorCode::DataLoss);
    EXPECT_NE(std::string(completion.outcome.error().user_message()).find(image_path().filename().string()),
              std::string::npos);
    expect_count(completion.sidecars.mask, 1, 0, 1);
}

TEST_F(PipelinedLoaderLedger, ExactSidecarTalliesForOneHundredFailuresPerKind) {
    constexpr std::uint64_t kFailures = 100;
    constexpr std::uint64_t kDelivered = 100;

    for (const auto kind : {SidecarKind::Mask, SidecarKind::Depth, SidecarKind::Normal}) {
        auto cfg = config();
        PipelinedImageLoader loader(cfg);
        std::vector<ImageRequest> requests;
        requests.reserve(kFailures + kDelivered);
        for (std::uint64_t i = 0; i < kFailures + kDelivered; ++i) {
            auto item = request(i);
            set_sidecar(item, kind,
                        i < kFailures ? image_path().parent_path() /
                                            ("missing-sidecar-" + std::to_string(i))
                                      : valid_sidecar(kind));
            requests.push_back(std::move(item));
        }
        loader.prefetch(requests);

        for (std::uint64_t i = 0; i < requests.size(); ++i) {
            auto completion = get_completion(loader);
            ASSERT_TRUE(completion.outcome.has_value());
            retire_events(completion);
        }

        const auto stats = loader.get_stats();
        expect_count(count_for(stats.aggregate_sidecar_tally, kind), 200, 100, 100);
        for (const auto other : {SidecarKind::Mask, SidecarKind::Depth, SidecarKind::Normal}) {
            if (other != kind) {
                expect_count(count_for(stats.aggregate_sidecar_tally, other), 0, 0, 0);
            }
        }
        EXPECT_EQ(loader.in_flight_count(), 0u);
    }
}

TEST_F(PipelinedLoaderLedger, EveryAcceptedSequenceCompletesExactlyOnce) {
    constexpr std::uint64_t kRequests = 500;
    auto cfg = config();
    cfg.prefetch_count = 8;
    cfg.output_queue_size = 4;
    PipelinedImageLoader loader(cfg);

    std::vector<ImageRequest> requests;
    requests.reserve(kRequests);
    for (std::uint64_t i = 0; i < kRequests; ++i) {
        auto item = request(i);
        if (i % 5 == 0) {
            item.path = image_path().parent_path() / ("missing-primary-" + std::to_string(i));
        }
        requests.push_back(std::move(item));
    }
    loader.prefetch(requests);

    std::set<std::uint64_t> sequences;
    for (std::uint64_t i = 0; i < kRequests; ++i) {
        auto completion = get_completion(loader);
        EXPECT_TRUE(sequences.insert(completion.sequence).second);
        retire_events(completion);
    }
    EXPECT_EQ(sequences.size(), kRequests);
    EXPECT_EQ(loader.in_flight_count(), 0u);
    const auto stats = loader.get_stats();
    EXPECT_EQ(stats.accepted_sequences, kRequests);
    EXPECT_EQ(stats.accepted_sequences,
              stats.succeeded_sequences + stats.failed_sequences + stats.cancelled_sequences);
}

TEST_F(PipelinedLoaderLedger, FullOutputQueueDoesNotLoseFollowingFailure) {
    auto cfg = config();
    cfg.output_queue_size = 1;
    cfg.prefetch_count = 2;
    PipelinedImageLoader loader(cfg);

    auto success = request(10);
    auto failure = request(11);
    failure.path = image_path().parent_path() / "missing-after-full-output.png";
    loader.prefetch({success});

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (loader.ready_count() != 1 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    ASSERT_EQ(loader.ready_count(), 1u);
    loader.prefetch({failure});

    auto first = get_completion(loader);
    auto second = get_completion(loader);
    EXPECT_EQ(first.sequence, 10u);
    EXPECT_TRUE(first.outcome.has_value());
    EXPECT_EQ(second.sequence, 11u);
    EXPECT_FALSE(second.outcome.has_value());
    retire_events(first);
    EXPECT_EQ(loader.in_flight_count(), 0u);
}

TEST_F(PipelinedLoaderLedger, ClearCancelsOldGenerationBeforeSequenceReuse) {
    auto cfg = config();
    cfg.output_queue_size = 1;
    cfg.prefetch_count = 1;
    PipelinedImageLoader loader(cfg);

    loader.prefetch({request(0)});
    loader.clear();

    auto replacement = request(0);
    replacement.path = image_path().parent_path() / "missing-after-reset.png";
    loader.prefetch({replacement});
    auto completion = get_completion(loader);
    EXPECT_EQ(completion.sequence, 0u);
    EXPECT_FALSE(completion.outcome.has_value());

    const auto stats = loader.get_stats();
    EXPECT_EQ(stats.accepted_sequences, 2u);
    EXPECT_EQ(stats.failed_sequences, 1u);
    EXPECT_EQ(stats.cancelled_sequences, 1u);
    EXPECT_EQ(stats.accepted_sequences,
              stats.succeeded_sequences + stats.failed_sequences + stats.cancelled_sequences);
}

TEST_F(PipelinedLoaderLedger, ShutdownRacesReconcileEveryQueueStage) {
    for (const auto delay : std::array{
             std::chrono::milliseconds(0), std::chrono::milliseconds(1),
             std::chrono::milliseconds(5), std::chrono::milliseconds(20)}) {
        auto cfg = config();
        cfg.output_queue_size = 1;
        cfg.prefetch_count = 2;
        cfg.batch_collect_timeout = std::chrono::milliseconds(20);
        PipelinedImageLoader loader(cfg);
        std::vector<ImageRequest> requests;
        for (std::uint64_t i = 0; i < 32; ++i) {
            requests.push_back(request(i));
        }
        loader.prefetch(requests);
        if (delay.count() != 0) {
            std::this_thread::sleep_for(delay);
        }
        loader.shutdown();
        expect_reconciled(loader);

        const auto accepted_before = loader.get_stats().accepted_sequences;
        loader.prefetch(999, image_path(), lfs::io::LoadParams{});
        EXPECT_EQ(loader.get_stats().accepted_sequences, accepted_before);

        void* probe = nullptr;
        ASSERT_EQ(cudaMallocAsync(&probe, 4096, nullptr), cudaSuccess);
        ASSERT_EQ(cudaFreeAsync(probe, nullptr), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
}
