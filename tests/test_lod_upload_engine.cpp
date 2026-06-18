/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/lod_pool_quant.hpp"
#include "rendering/lod_upload_engine.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <thread>
#include <vector>

namespace {

    using lfs::vis::LodPageCache;
    using lfs::vis::LodUploadEngine;

    constexpr std::size_t kPageSplats = LodPageCache::kChunkSplats;

    struct DevicePool {
        void* base = nullptr;
        void* meta_base = nullptr;
        std::array<std::size_t, 7> region_offset{};
        std::size_t splat_capacity = 0;
        std::size_t meta_links_offset = 0;

        explicit DevicePool(const std::size_t pages, const std::uint32_t dst_rest) {
            splat_capacity = pages * kPageSplats;
            std::array<std::size_t, 7> region_bytes{};
            region_bytes[0] = splat_capacity * lfs::vis::lodq::kXyzBytes;
            region_bytes[1] = splat_capacity * lfs::vis::lodq::kSh0Bytes;
            region_bytes[2] = dst_rest == 0u
                                  ? 4u * sizeof(float)
                                  : lfs::core::sh_swizzled_byte_count(splat_capacity, dst_rest) / 4u;
            region_bytes[3] = splat_capacity * lfs::vis::lodq::kRotationBytes;
            region_bytes[4] = splat_capacity * lfs::vis::lodq::kScalingBytes;
            region_bytes[5] = splat_capacity * lfs::vis::lodq::kOpacityBytes;
            region_bytes[6] = pages * lfs::vis::lodq::kPageFrameBytes;
            std::size_t offset = 0;
            for (std::size_t i = 0; i < region_bytes.size(); ++i) {
                region_offset[i] = offset;
                offset += (region_bytes[i] + 255u) & ~std::size_t{255u};
            }
            EXPECT_EQ(cudaMalloc(&base, offset), cudaSuccess);
            EXPECT_EQ(cudaMemset(base, 0, offset), cudaSuccess);

            meta_links_offset = splat_capacity * lfs::vis::lodq::kMetaBoundsBytes;
            const std::size_t meta_bytes =
                meta_links_offset + splat_capacity * lfs::vis::lodq::kMetaLinksBytes;
            EXPECT_EQ(cudaMalloc(&meta_base, meta_bytes), cudaSuccess);
            EXPECT_EQ(cudaMemset(meta_base, 0, meta_bytes), cudaSuccess);
        }
        ~DevicePool() {
            if (base != nullptr) {
                (void)cudaFree(base);
            }
            if (meta_base != nullptr) {
                (void)cudaFree(meta_base);
            }
        }

        [[nodiscard]] LodUploadEngine::DeviceLayout layout(const std::uint32_t dst_rest) const {
            return {
                .device_base = base,
                .region_offset = region_offset,
                .splat_capacity = splat_capacity,
                .dst_rest = dst_rest,
                .dst_slots = lfs::core::sh_float4_slots_for_rest(dst_rest),
                .meta_base = meta_base,
                .meta_bounds_offset = 0,
                .meta_links_offset = meta_links_offset,
                .meta_capacity_nodes = splat_capacity,
            };
        }
    };

    // Builds a packed slot the way decode_rad_chunk_packed lays one out
    // (f32 means + f32 alpha planes, dimension-major, plus sidecar links)
    // and submits it through the dequant kernel.
    void packAndSubmit(LodUploadEngine& engine,
                       const std::uint32_t page,
                       const std::uint32_t chunk,
                       const std::size_t count) {
        auto* const slot = engine.acquireStagingSlot();
        ASSERT_NE(slot, nullptr);

        lfs::io::RadPagePackedDesc desc{};
        desc.count = static_cast<std::uint32_t>(count);
        desc.lod_opacity = 1;
        desc.chunk = chunk;

        std::size_t cursor = 0;
        const auto alloc_plane = [&](const std::size_t bytes) {
            const std::size_t at = (cursor + 15u) & ~std::size_t{15u};
            cursor = at + bytes;
            return at;
        };

        const std::size_t means_offset = alloc_plane(count * 3u * sizeof(float));
        auto* const means = reinterpret_cast<float*>(slot->data + means_offset);
        for (std::size_t d = 0; d < 3u; ++d) {
            for (std::size_t i = 0; i < count; ++i) {
                means[d * count + i] =
                    static_cast<float>(chunk) * 1000.0f + static_cast<float>(i * 3u + d);
            }
        }
        desc.props[desc.property_count++] = {
            .kind = static_cast<std::uint32_t>(lfs::io::RadPackedKind::Means),
            .encoding = static_cast<std::uint32_t>(lfs::io::RadPackedEncoding::F32),
            .plane_offset = static_cast<std::uint32_t>(means_offset),
            .plane_bytes = static_cast<std::uint32_t>(count * 3u * sizeof(float)),
        };

        const std::size_t alpha_offset = alloc_plane(count * sizeof(float));
        auto* const alpha = reinterpret_cast<float*>(slot->data + alpha_offset);
        std::iota(alpha, alpha + count, static_cast<float>(chunk));
        desc.props[desc.property_count++] = {
            .kind = static_cast<std::uint32_t>(lfs::io::RadPackedKind::Alpha),
            .encoding = static_cast<std::uint32_t>(lfs::io::RadPackedEncoding::F32),
            .plane_offset = static_cast<std::uint32_t>(alpha_offset),
            .plane_bytes = static_cast<std::uint32_t>(count * sizeof(float)),
        };

        const std::size_t bounds_offset =
            alloc_plane(kPageSplats * sizeof(lfs::core::RadMetaBoundsQ));
        std::memset(slot->data + bounds_offset, 0,
                    kPageSplats * sizeof(lfs::core::RadMetaBoundsQ));
        const std::size_t links_offset =
            alloc_plane(kPageSplats * sizeof(lfs::core::RadMetaLinksQ));
        auto* const links =
            reinterpret_cast<lfs::core::RadMetaLinksQ*>(slot->data + links_offset);
        for (std::size_t i = 0; i < kPageSplats; ++i) {
            links[i] = {.child_start = chunk, .packed = 0, .parent = 0};
        }
        desc.meta_bounds_offset = static_cast<std::uint32_t>(bounds_offset);
        desc.meta_links_offset = static_cast<std::uint32_t>(links_offset);
        desc.meta_node_count = static_cast<std::uint32_t>(kPageSplats);
        desc.used_bytes = static_cast<std::uint32_t>(cursor);

        engine.submitPackedPage(slot, desc, page, 1);
    }

    std::vector<LodPageCache::PendingUpload> collectAll(LodUploadEngine& engine,
                                                        const std::size_t expected) {
        std::vector<LodPageCache::PendingUpload> published;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (published.size() < expected && std::chrono::steady_clock::now() < deadline) {
            auto batch = engine.collectPublished();
            for (auto& upload : batch) {
                published.push_back(std::move(upload));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return published;
    }

    TEST(LodUploadEngine, RoundTripUploadsPayloadAndMetadata) {
        DevicePool pool(4, 0);
        LodUploadEngine engine;
        (void)engine.configure(pool.layout(0), nullptr);
        ASSERT_TRUE(engine.configured());

        constexpr std::size_t kCount = 1000;
        packAndSubmit(engine, 2, 7, kCount);
        const auto published = collectAll(engine, 1);
        ASSERT_EQ(published.size(), 1u);
        EXPECT_TRUE(published.front().error.empty()) << published.front().error;
        EXPECT_EQ(published.front().chunk, 7u);
        EXPECT_EQ(engine.lastPublishedSignalValue(), 1u);

        const std::size_t dst_start = 2u * kPageSplats;
        std::vector<float> means(kCount * 3u);
        ASSERT_EQ(cudaMemcpy(means.data(),
                             static_cast<std::uint8_t*>(pool.base) + pool.region_offset[0] +
                                 dst_start * 3u * sizeof(float),
                             means.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_FLOAT_EQ(means[0], 7000.0f);
        EXPECT_FLOAT_EQ(means.back(), 7000.0f + static_cast<float>(kCount * 3u - 1));

        std::vector<lfs::core::RadMetaLinksQ> links(4);
        ASSERT_EQ(cudaMemcpy(links.data(),
                             static_cast<std::uint8_t*>(pool.meta_base) + pool.meta_links_offset +
                                 dst_start * lfs::vis::lodq::kMetaLinksBytes,
                             links.size() * sizeof(lfs::core::RadMetaLinksQ),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (std::size_t i = 0; i < links.size(); ++i) {
            // Records pass through untouched; logical is selector-derived.
            EXPECT_EQ(links[i].child_start, 7u);
            EXPECT_EQ(links[i].parent, 0u);
        }
    }

    TEST(LodUploadEngine, DrainAndSyncCompletesInFlightWork) {
        DevicePool pool(8, 0);
        LodUploadEngine engine;
        (void)engine.configure(pool.layout(0), nullptr);

        for (std::uint32_t i = 0; i < 8; ++i) {
            packAndSubmit(engine, i, 100 + i, kPageSplats);
        }
        const auto drained = engine.drainAndSync();
        EXPECT_EQ(drained.size(), 8u);
        for (const auto& upload : drained) {
            EXPECT_TRUE(upload.error.empty()) << upload.error;
        }
        EXPECT_TRUE(engine.idle());
    }

    TEST(LodUploadEngine, ReconfigureDrainsPriorLayout) {
        DevicePool pool_a(2, 0);
        DevicePool pool_b(2, 0);
        LodUploadEngine engine;
        (void)engine.configure(pool_a.layout(0), nullptr);

        packAndSubmit(engine, 1, 5, 256);
        const auto drained = engine.configure(pool_b.layout(0), nullptr);
        EXPECT_EQ(drained.size(), 1u);
        EXPECT_TRUE(engine.idle());

        packAndSubmit(engine, 0, 6, 256);
        const auto published = collectAll(engine, 1);
        ASSERT_EQ(published.size(), 1u);
        EXPECT_TRUE(published.front().error.empty()) << published.front().error;
    }

    TEST(LodUploadEngine, MultiWorkerSignalsStayMonotone) {
        DevicePool pool(8, 0);
        LodUploadEngine engine;
        (void)engine.configure(pool.layout(0), nullptr);

        constexpr std::size_t kWorkers = 4;
        constexpr std::size_t kPerWorker = 16;
        std::atomic<std::uint32_t> next_chunk{0};
        std::vector<std::thread> workers;
        workers.reserve(kWorkers);
        for (std::size_t w = 0; w < kWorkers; ++w) {
            workers.emplace_back([&] {
                for (std::size_t i = 0; i < kPerWorker; ++i) {
                    const std::uint32_t chunk = next_chunk.fetch_add(1);
                    packAndSubmit(engine, chunk % 8u, chunk, 1024);
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        const auto drained = engine.drainAndSync();
        EXPECT_EQ(drained.size(), kWorkers * kPerWorker);
        for (const auto& upload : drained) {
            EXPECT_TRUE(upload.error.empty()) << upload.error;
        }
        EXPECT_EQ(engine.lastPublishedSignalValue(), kWorkers * kPerWorker);
    }

    TEST(LodUploadEngine, UnconfiguredAcquireReturnsNull) {
        LodUploadEngine engine;
        EXPECT_EQ(engine.acquireStagingSlot(), nullptr);
        EXPECT_TRUE(engine.idle());
        EXPECT_TRUE(engine.collectPublished().empty());
    }

} // namespace
