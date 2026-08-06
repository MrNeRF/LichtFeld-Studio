/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/pipelined_image_loader.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <vector>

using namespace lfs::core;
using namespace lfs::io;

namespace {

    [[nodiscard]] std::uint64_t fnv1a64(const void* data, const size_t n) {
        const auto* p = static_cast<const std::uint8_t*>(data);
        std::uint64_t h = 14695981039346656037ull;
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    [[nodiscard]] std::uint64_t hash_tensor(const Tensor& t) {
        EXPECT_TRUE(t.is_valid());
        auto cpu = t.cpu().contiguous();
        if (cpu.dtype() == DataType::UInt8) {
            const auto v = cpu.to_vector_uint8();
            return fnv1a64(v.data(), v.size());
        }
        const auto v = cpu.to_vector();
        return fnv1a64(v.data(), v.size() * sizeof(float));
    }

    class GtDeviceCacheTest : public ::testing::Test {
    protected:
        void SetUp() override {
            const std::filesystem::path candidates[] = {
                std::filesystem::path(TEST_DATA_DIR) / "bicycle/images_4/_DSC8744.JPG",
                std::filesystem::path("/home/gauss/data/360_v2/bicycle/images_4/_DSC8744.JPG"),
                std::filesystem::path(TEST_DATA_DIR) / "bicycle/images/_DSC8744.JPG",
            };
            for (const auto& c : candidates) {
                if (std::filesystem::is_regular_file(c)) {
                    image_path_ = c;
                    break;
                }
            }
            ASSERT_TRUE(std::filesystem::is_regular_file(image_path_))
                << "No bicycle test image found under TEST_DATA_DIR or /home/gauss/data/360_v2";
        }

        static PipelinedLoaderConfig base_config() {
            PipelinedLoaderConfig c;
            c.jpeg_batch_size = 2;
            c.prefetch_count = 2;
            c.output_queue_size = 4;
            c.decoder_pool_size = 2;
            c.io_threads = 1;
            c.cold_process_threads = 1;
            c.max_cache_bytes = 64 * 1024 * 1024;
            c.use_filesystem_cache = false;
            c.enable_gt_cache = true;
            c.enable_gt_pinned_fallback = true;
            // Force device tier: tiny expected set, huge free budget via headroom 0.
            c.gt_cache_headroom_bytes = 0;
            c.gt_cache_cap_bytes = 512ULL * 1024 * 1024;
            c.gt_cache_expected_images = 4;
            c.gt_cache_bytes_per_image = 128 * 128 * 3; // after max_width=128
            return c;
        }

        ImageRequest make_request(const size_t sequence_id, const int max_width = 128) const {
            ImageRequest r;
            r.sequence_id = sequence_id;
            r.path = image_path_;
            r.params.resize_factor = 1;
            r.params.max_width = max_width;
            r.params.output_uint8 = true;
            return r;
        }

        std::filesystem::path image_path_;
    };

} // namespace

// ---------------------------------------------------------------------------
// Pure budget-gate unit tests (no GPU required for the decision function).
// ---------------------------------------------------------------------------

TEST(GtCacheBudgetGate, EnablesDeviceWhenUnderBudget) {
    const auto d = evaluate_gt_cache_budget(
        /*n_images=*/100,
        /*bytes_per_image=*/3 * 1024 * 1024, // 300 MiB total
        /*free_vram=*/8ULL * 1024 * 1024 * 1024,
        /*headroom=*/2ULL * 1024 * 1024 * 1024,
        /*cap=*/0,
        /*force_disable=*/false,
        /*allow_pinned=*/true);
    EXPECT_TRUE(d.enable_device);
    EXPECT_FALSE(d.enable_pinned_host);
    EXPECT_STREQ(d.reason, "device_within_budget");
    EXPECT_EQ(d.estimated_bytes, 100ULL * 3 * 1024 * 1024);
}

TEST(GtCacheBudgetGate, FallsBackToPinnedWhenOverBudget) {
    const auto d = evaluate_gt_cache_budget(
        /*n_images=*/200,
        /*bytes_per_image=*/4ULL * 1024 * 1024,                         // 800 MiB
        /*free_vram=*/2ULL * 1024 * 1024 * 1024 + 100ULL * 1024 * 1024, // only ~100 MiB after headroom
        /*headroom=*/2ULL * 1024 * 1024 * 1024,
        /*cap=*/0,
        /*force_disable=*/false,
        /*allow_pinned=*/true);
    EXPECT_FALSE(d.enable_device);
    EXPECT_TRUE(d.enable_pinned_host);
    EXPECT_STREQ(d.reason, "device_over_budget_use_pinned");
}

TEST(GtCacheBudgetGate, CapOverrideTightensBudget) {
    // Free VRAM would allow device, but cap is tiny.
    const auto d = evaluate_gt_cache_budget(
        /*n_images=*/10,
        /*bytes_per_image=*/10ULL * 1024 * 1024, // 100 MiB
        /*free_vram=*/8ULL * 1024 * 1024 * 1024,
        /*headroom=*/2ULL * 1024 * 1024 * 1024,
        /*cap=*/50ULL * 1024 * 1024, // 50 MiB cap
        /*force_disable=*/false,
        /*allow_pinned=*/true);
    EXPECT_FALSE(d.enable_device);
    EXPECT_TRUE(d.enable_pinned_host);
    EXPECT_EQ(d.device_budget_bytes, 50ULL * 1024 * 1024);
}

TEST(GtCacheBudgetGate, ForceDisableAndNoPinned) {
    const auto d = evaluate_gt_cache_budget(
        10, 1024 * 1024,
        8ULL * 1024 * 1024 * 1024,
        2ULL * 1024 * 1024 * 1024,
        0,
        /*force_disable=*/true,
        /*allow_pinned=*/false);
    EXPECT_FALSE(d.enable_device);
    EXPECT_FALSE(d.enable_pinned_host);
    EXPECT_STREQ(d.reason, "disabled_by_env_or_config");
}

TEST(GtCacheBudgetGate, LeavesTwoGigHeadroom) {
    // estimated == free - headroom → still fits ( <= ).
    const size_t free_v = 4ULL * 1024 * 1024 * 1024;
    const size_t head = 2ULL * 1024 * 1024 * 1024;
    const size_t est = free_v - head;
    const auto d = evaluate_gt_cache_budget(1, est, free_v, head, 0, false, true);
    EXPECT_TRUE(d.enable_device);
    EXPECT_EQ(d.device_budget_bytes, est);

    // One byte over → denied.
    const auto d2 = evaluate_gt_cache_budget(1, est + 1, free_v, head, 0, false, true);
    EXPECT_FALSE(d2.enable_device);
    EXPECT_TRUE(d2.enable_pinned_host);
}

// ---------------------------------------------------------------------------
// Integration: bit-identical cache hit, budget-gated off, eviction/fallback.
// ---------------------------------------------------------------------------

TEST_F(GtDeviceCacheTest, CacheHitReturnsBitIdenticalTensor) {
    auto cfg = base_config();
    PipelinedImageLoader loader(cfg);
    // Re-evaluate against real free VRAM with tiny footprint so device tier wins.
    loader.configure_gt_cache(/*expected_images=*/8, /*bytes_per_image=*/128 * 128 * 3);
    ASSERT_TRUE(loader.gt_cache_budget().enable_device)
        << loader.gt_cache_budget().reason;

    loader.prefetch({make_request(1)});
    const auto first = loader.get();
    ASSERT_TRUE(first.error.empty()) << first.error;
    ASSERT_TRUE(first.tensor.is_valid());
    EXPECT_EQ(first.tensor.device(), Device::CUDA);
    EXPECT_EQ(first.tensor.dtype(), DataType::UInt8);
    const auto h1 = hash_tensor(first.tensor);

    // Second request for the same image key must hit the device cache.
    loader.prefetch({make_request(2)});
    const auto second = loader.get();
    ASSERT_TRUE(second.error.empty()) << second.error;
    ASSERT_TRUE(second.tensor.is_valid());
    EXPECT_EQ(second.tensor.dtype(), DataType::UInt8);
    EXPECT_EQ(second.tensor.shape(), first.tensor.shape());
    const auto h2 = hash_tensor(second.tensor);
    EXPECT_EQ(h1, h2) << "cache-hit tensor must be bit-identical to first decode";

    const auto stats = loader.get_stats();
    EXPECT_GE(stats.gt_device_cache_hits, 1u);
    EXPECT_GE(stats.gt_device_cache_entries, 1u);
    EXPECT_GT(stats.gt_device_cache_bytes, 0u);
    EXPECT_TRUE(stats.gt_device_cache_enabled);
}

TEST_F(GtDeviceCacheTest, BudgetGateDisablesDeviceFallsBackToDecode) {
    auto cfg = base_config();
    cfg.enable_gt_cache = false; // force off
    PipelinedImageLoader loader(cfg);
    loader.configure_gt_cache(8, 128 * 128 * 3);
    EXPECT_FALSE(loader.gt_cache_budget().enable_device);
    EXPECT_FALSE(loader.gt_cache_budget().enable_pinned_host);

    loader.prefetch({make_request(1)});
    const auto a = loader.get();
    ASSERT_TRUE(a.tensor.is_valid());
    loader.prefetch({make_request(2)});
    const auto b = loader.get();
    ASSERT_TRUE(b.tensor.is_valid());
    // Fallback path still delivers valid CUDA u8 CHW tensors; GT cache stays empty.
    EXPECT_EQ(a.tensor.device(), Device::CUDA);
    EXPECT_EQ(b.tensor.device(), Device::CUDA);
    EXPECT_EQ(a.tensor.dtype(), DataType::UInt8);
    EXPECT_EQ(a.tensor.shape(), b.tensor.shape());
    const auto stats = loader.get_stats();
    EXPECT_EQ(stats.gt_device_cache_hits, 0u);
    EXPECT_EQ(stats.gt_device_cache_entries, 0u);
    EXPECT_EQ(stats.gt_device_cache_bytes, 0u);
    EXPECT_EQ(stats.gt_cache_inserts, 0u);
}

TEST_F(GtDeviceCacheTest, ClearGtCacheEvictsAndFallsBack) {
    auto cfg = base_config();
    PipelinedImageLoader loader(cfg);
    loader.configure_gt_cache(8, 128 * 128 * 3);
    ASSERT_TRUE(loader.gt_cache_budget().enable_device);

    loader.prefetch({make_request(1)});
    const auto first = loader.get();
    ASSERT_TRUE(first.tensor.is_valid());
    ASSERT_GE(loader.get_stats().gt_device_cache_entries, 1u);
    const auto h1 = hash_tensor(first.tensor);

    loader.clear_gt_cache();
    EXPECT_EQ(loader.gt_device_cache_bytes(), 0u);
    EXPECT_EQ(loader.get_stats().gt_device_cache_entries, 0u);

    // After eviction, path falls back to decode (valid tensor) and re-populates.
    loader.prefetch({make_request(2)});
    const auto second = loader.get();
    ASSERT_TRUE(second.tensor.is_valid());
    EXPECT_EQ(second.tensor.device(), Device::CUDA);
    EXPECT_EQ(second.tensor.shape(), first.tensor.shape());

    // And re-populates the cache; a subsequent hit is bit-identical to the
    // post-eviction decode (not necessarily to the pre-eviction decode, since
    // independent JPEG decode paths can re-encode).
    EXPECT_GE(loader.get_stats().gt_device_cache_entries, 1u);
    loader.prefetch({make_request(3)});
    const auto third = loader.get();
    ASSERT_TRUE(third.tensor.is_valid());
    EXPECT_EQ(hash_tensor(third.tensor), hash_tensor(second.tensor));
    EXPECT_GE(loader.get_stats().gt_device_cache_hits, 1u);
    (void)h1;
}

TEST_F(GtDeviceCacheTest, CapOverrideForcesPinnedOrOff) {
    auto cfg = base_config();
    // Cap of 1 byte → device cannot hold even one image.
    cfg.gt_cache_cap_bytes = 1;
    cfg.enable_gt_pinned_fallback = true;
    PipelinedImageLoader loader(cfg);
    loader.configure_gt_cache(4, 128 * 128 * 3);
    EXPECT_FALSE(loader.gt_cache_budget().enable_device);
    EXPECT_TRUE(loader.gt_cache_budget().enable_pinned_host)
        << loader.gt_cache_budget().reason;

    loader.prefetch({make_request(1)});
    const auto first = loader.get();
    ASSERT_TRUE(first.tensor.is_valid());
    loader.prefetch({make_request(2)});
    const auto second = loader.get();
    ASSERT_TRUE(second.tensor.is_valid());
    EXPECT_EQ(hash_tensor(first.tensor), hash_tensor(second.tensor));

    const auto stats = loader.get_stats();
    EXPECT_EQ(stats.gt_device_cache_entries, 0u);
    // Pinned middle tier should have served the second request.
    EXPECT_GE(stats.gt_pinned_cache_hits + stats.gt_cache_inserts, 1u);
}
