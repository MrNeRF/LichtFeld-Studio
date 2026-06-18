/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/lod_page_cache.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace {

    using lfs::vis::LodPageCache;

    constexpr std::uint32_t kInvalid = LodPageCache::kInvalidPage;

    std::vector<LodPageCache::PendingUpload> drainAndComplete(LodPageCache& cache) {
        auto uploads = cache.drainPendingUploads();
        cache.completeUploads(uploads);
        return uploads;
    }

    TEST(LodPageCache, ConfigureBoundedPinsRootAndStaysPartial) {
        LodPageCache cache;
        cache.configure(100, 10, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        const auto& snapshot = cache.snapshot();
        EXPECT_EQ(snapshot.logical_chunks, 100u);
        EXPECT_EQ(snapshot.physical_pages, 10u);
        EXPECT_TRUE(cache.configured());
        EXPECT_FALSE(cache.fullyResident());
        EXPECT_EQ(snapshot.chunk_to_page[0], 0u);
        EXPECT_EQ(snapshot.page_to_chunk[0], 0u);
        for (std::uint32_t chunk = 1; chunk < 100; ++chunk) {
            EXPECT_EQ(snapshot.chunk_to_page[chunk], kInvalid);
        }
    }

    TEST(LodPageCache, ConfigureFullCapacityIsFullyResident) {
        LodPageCache cache;
        cache.configure(16, 0, 1);

        EXPECT_EQ(cache.snapshot().physical_pages, 16u);
        EXPECT_TRUE(cache.fullyResident());
        for (std::uint32_t chunk = 0; chunk < 16; ++chunk) {
            EXPECT_EQ(cache.snapshot().chunk_to_page[chunk], chunk);
        }
    }

    TEST(LodPageCache, SubmitTraversalPriorityRespectsRequestBudget) {
        LodPageCache cache;
        cache.configure(100, 10, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache); // root bootstrap upload

        std::vector<std::uint32_t> wanted(40);
        std::iota(wanted.begin(), wanted.end(), 10u);
        cache.submitTraversalPriority(wanted);

        // Budget for 10 pages: clamp(10/4, min(8,10), min(64,10)) = 8.
        const auto uploads = cache.drainPendingUploads();
        EXPECT_EQ(uploads.size(), 8u);
        for (std::size_t i = 0; i < uploads.size(); ++i) {
            EXPECT_EQ(uploads[i].chunk, wanted[i]) << "caller priority order must be preserved";
        }
    }

    TEST(LodPageCache, CompleteUploadsPublishesResidency) {
        LodPageCache cache;
        cache.configure(100, 10, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        const std::uint64_t generation_before = cache.snapshot().generation;
        const std::vector<std::uint32_t> wanted{42, 43};
        cache.submitTraversalPriority(wanted);
        EXPECT_TRUE(cache.hasOutstandingWork());
        EXPECT_EQ(cache.admittedRequestCount(), 2u);
        (void)drainAndComplete(cache);

        EXPECT_FALSE(cache.hasOutstandingWork());
        EXPECT_NE(cache.snapshot().chunk_to_page[42], kInvalid);
        EXPECT_NE(cache.snapshot().chunk_to_page[43], kInvalid);
        EXPECT_GT(cache.snapshot().generation, generation_before);
    }

    TEST(LodPageCache, CompleteUploadsReleasesReservationOnError) {
        LodPageCache cache;
        cache.configure(100, 3, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        const std::vector<std::uint32_t> wanted{42, 43};
        cache.submitTraversalPriority(wanted);
        auto uploads = cache.drainPendingUploads();
        ASSERT_EQ(uploads.size(), 2u);

        // Async-stage failure after a successful decode (e.g. CUDA copy
        // error): the reservation must release or the slot leaks forever.
        uploads[0].error = "simulated upload failure";
        cache.completeUploads(uploads);
        EXPECT_EQ(cache.snapshot().chunk_to_page[42], kInvalid);
        EXPECT_NE(cache.snapshot().chunk_to_page[43], kInvalid);
        EXPECT_FALSE(cache.hasOutstandingWork());

        // The released slot is reusable: the same chunk can be re-requested
        // and published normally.
        cache.submitTraversalPriority(std::vector<std::uint32_t>{42});
        (void)drainAndComplete(cache);
        EXPECT_NE(cache.snapshot().chunk_to_page[42], kInvalid);
    }

    TEST(LodPageCache, ProtectedChunksSurviveEvictionPressure) {
        LodPageCache cache;
        cache.configure(100, 4, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        std::vector<std::uint32_t> first{10, 11, 12};
        cache.submitTraversalPriority(first);
        (void)drainAndComplete(cache);
        ASSERT_NE(cache.snapshot().chunk_to_page[10], kInvalid);
        ASSERT_NE(cache.snapshot().chunk_to_page[11], kInvalid);
        ASSERT_NE(cache.snapshot().chunk_to_page[12], kInvalid);

        // The pool is now full (root + 3). Requesting new chunks while
        // protecting 10 and 11 may only evict chunk 12.
        std::vector<std::uint32_t> next{20};
        std::vector<std::uint32_t> protected_chunks{10, 11};
        cache.submitTraversalPriority(next, protected_chunks);
        (void)drainAndComplete(cache);

        EXPECT_NE(cache.snapshot().chunk_to_page[10], kInvalid);
        EXPECT_NE(cache.snapshot().chunk_to_page[11], kInvalid);
        EXPECT_EQ(cache.snapshot().chunk_to_page[12], kInvalid);
        EXPECT_NE(cache.snapshot().chunk_to_page[20], kInvalid);
    }

    TEST(LodPageCache, PinnedRootIsNeverEvicted) {
        LodPageCache cache;
        cache.configure(100, 2, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        for (std::uint32_t chunk = 10; chunk < 30; ++chunk) {
            std::vector<std::uint32_t> wanted{chunk};
            cache.submitTraversalPriority(wanted);
            (void)drainAndComplete(cache);
        }

        EXPECT_EQ(cache.snapshot().chunk_to_page[0], 0u) << "pinned root must stay resident";
        EXPECT_NE(cache.snapshot().chunk_to_page[29], kInvalid);
    }

    TEST(LodPageCache, LruEvictsLeastRecentlyTouchedChunk) {
        LodPageCache cache;
        cache.configure(100, 3, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        std::vector<std::uint32_t> first{10, 11};
        cache.submitTraversalPriority(first);
        (void)drainAndComplete(cache);

        // Refresh chunk 10's LRU clock, then force one eviction.
        std::vector<std::uint32_t> refresh{10, 20};
        cache.submitTraversalPriority(refresh);
        (void)drainAndComplete(cache);

        EXPECT_NE(cache.snapshot().chunk_to_page[10], kInvalid);
        EXPECT_EQ(cache.snapshot().chunk_to_page[11], kInvalid) << "stale chunk must be evicted first";
        EXPECT_NE(cache.snapshot().chunk_to_page[20], kInvalid);
    }

    std::vector<LodPageCache::ChunkRequest> makeRequests(std::initializer_list<std::uint32_t> chunks,
                                                         const std::uint32_t priority = 100) {
        std::vector<LodPageCache::ChunkRequest> requests;
        for (const std::uint32_t chunk : chunks) {
            requests.push_back({.chunk = chunk, .priority = priority});
        }
        return requests;
    }

    TEST(LodPageCache, ProtectionWindowBlocksEvictionAcrossFrames) {
        LodPageCache cache;
        cache.configure(100, 4, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        cache.beginFrame();
        const auto fill = makeRequests({10, 11, 12});
        const std::vector<std::uint32_t> protect{10, 11, 12};
        cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(fill), protect);
        (void)drainAndComplete(cache);
        ASSERT_NE(cache.snapshot().chunk_to_page[12], kInvalid);

        // Within the protection window the protected chunks survive pressure
        // even when later frames stop listing them.
        for (int frame = 0; frame < 2; ++frame) {
            cache.beginFrame();
            const auto want = makeRequests({20u + static_cast<std::uint32_t>(frame)});
            cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(want),
                                          std::span<const std::uint32_t>{});
            (void)drainAndComplete(cache);
            EXPECT_NE(cache.snapshot().chunk_to_page[10], kInvalid);
            EXPECT_NE(cache.snapshot().chunk_to_page[11], kInvalid);
            EXPECT_NE(cache.snapshot().chunk_to_page[12], kInvalid);
        }

        // Past the window they become evictable again — by a strictly
        // higher-priority request (eviction admission).
        for (int frame = 0; frame < 4; ++frame) {
            cache.beginFrame();
        }
        // Admission needs a full hysteresis margin over the victims.
        const auto want = makeRequests({30}, 0x01000000);
        cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(want),
                                      std::span<const std::uint32_t>{});
        (void)drainAndComplete(cache);
        EXPECT_NE(cache.snapshot().chunk_to_page[30], kInvalid);
    }

    TEST(LodPageCache, NoEvictableSlotDefersRequestWithoutEviction) {
        LodPageCache cache;
        cache.configure(100, 3, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        cache.beginFrame();
        const auto fill = makeRequests({10, 11});
        const std::vector<std::uint32_t> protect{10, 11};
        cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(fill), protect);
        (void)drainAndComplete(cache);

        const std::uint64_t generation_before = cache.snapshot().generation;
        cache.beginFrame();
        const auto want = makeRequests({20, 21});
        cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(want), protect);
        const auto uploads = cache.drainPendingUploads();

        EXPECT_TRUE(uploads.empty()) << "deferred requests must not enqueue work";
        EXPECT_EQ(cache.snapshot().generation, generation_before);
        EXPECT_NE(cache.snapshot().chunk_to_page[10], kInvalid);
        EXPECT_NE(cache.snapshot().chunk_to_page[11], kInvalid);
        EXPECT_EQ(cache.deferredRequestCount(), 2u);
        // Deferral must be visible without creating work: the renderer keeps
        // the frame chain alive on deferredRequestCount() so decay and the
        // threshold controller can resolve the want on later frames.
        // deferred>0 with admitted==0 and no outstanding work is the
        // admission-freeze signal that halts the threshold descent.
        EXPECT_FALSE(cache.hasOutstandingWork());
        EXPECT_EQ(cache.admittedRequestCount(), 0u);
    }

    TEST(LodPageCache, SteadyStateProducesNoWork) {
        LodPageCache cache;
        cache.configure(100, 8, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        cache.beginFrame();
        const auto cut = makeRequests({10, 11, 12, 13});
        const std::vector<std::uint32_t> protect{10, 11, 12, 13};
        cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(cut), protect);
        (void)drainAndComplete(cache);
        const std::uint64_t generation = cache.snapshot().generation;

        for (int frame = 0; frame < 10; ++frame) {
            cache.beginFrame();
            cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(cut), protect);
            const auto uploads = cache.drainPendingUploads();
            EXPECT_TRUE(uploads.empty()) << "converged cut must not produce uploads";
            EXPECT_EQ(cache.snapshot().generation, generation) << "no residency churn at rest";
            EXPECT_FALSE(cache.hasOutstandingWork());
        }
    }

    TEST(LodPageCache, EvictionAdmissionRequiresHigherPriority) {
        LodPageCache cache;
        cache.configure(100, 3, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        // Priorities are float bits of pixel scale, like the GPU selector's.
        constexpr std::uint32_t kPriA = 0x40000000; // 2.0f
        constexpr std::uint32_t kPriB = 0x3F800000; // 1.0f
        constexpr std::uint32_t kPriC = 0x3F000000; // 0.5f

        cache.beginFrame();
        const std::vector<LodPageCache::ChunkRequest> fill{
            {.chunk = 10, .priority = kPriA},
            {.chunk = 11, .priority = kPriB},
        };
        cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(fill), {});
        (void)drainAndComplete(cache);
        ASSERT_NE(cache.snapshot().chunk_to_page[10], kInvalid);
        ASSERT_NE(cache.snapshot().chunk_to_page[11], kInvalid);
        const std::uint64_t generation = cache.snapshot().generation;

        // Pool full of fresher, higher-priority content: the lower-priority
        // want must defer instead of rotating pages (capacity-thrash guard).
        const std::vector<LodPageCache::ChunkRequest> contested{
            {.chunk = 10, .priority = kPriA},
            {.chunk = 11, .priority = kPriB},
            {.chunk = 12, .priority = kPriC},
        };
        for (int frame = 0; frame < 3; ++frame) {
            cache.beginFrame();
            cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(contested), {});
            EXPECT_TRUE(cache.drainPendingUploads().empty());
            EXPECT_GE(cache.deferredRequestCount(), 1u);
        }
        EXPECT_EQ(cache.snapshot().generation, generation);

        // Once interest in chunk 11 lapses, the idle-freshness window expires,
        // its resistance drops to zero, and the slot is reclaimed.
        const std::vector<LodPageCache::ChunkRequest> shifted{
            {.chunk = 10, .priority = kPriA},
            {.chunk = 12, .priority = kPriC},
        };
        bool reclaimed = false;
        for (int frame = 0; frame < 140 && !reclaimed; ++frame) {
            cache.beginFrame();
            cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(shifted), {});
            reclaimed = !drainAndComplete(cache).empty();
        }
        ASSERT_TRUE(reclaimed);
        EXPECT_NE(cache.snapshot().chunk_to_page[12], kInvalid);
        EXPECT_EQ(cache.snapshot().chunk_to_page[11], kInvalid);

        // Equal-priority counter-request must not flip the slot back: strict
        // admission breaks rotation cycles.
        const std::uint64_t settled = cache.snapshot().generation;
        const std::vector<LodPageCache::ChunkRequest> counter{
            {.chunk = 10, .priority = kPriA},
            {.chunk = 11, .priority = kPriC},
        };
        cache.beginFrame();
        cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(counter), {});
        EXPECT_TRUE(cache.drainPendingUploads().empty());
        EXPECT_EQ(cache.snapshot().generation, settled);
    }

    TEST(LodPageCache, RequestsHonorPriorityOrderViaLegacyOverload) {
        LodPageCache cache;
        cache.configure(100, 10, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        // Legacy span overload synthesizes descending priority from order.
        std::vector<std::uint32_t> wanted{50, 40, 30};
        cache.submitTraversalPriority(wanted);
        const auto uploads = cache.drainPendingUploads();
        ASSERT_EQ(uploads.size(), 3u);
        EXPECT_EQ(uploads[0].chunk, 50u);
        EXPECT_EQ(uploads[1].chunk, 40u);
        EXPECT_EQ(uploads[2].chunk, 30u);
    }

    TEST(LodPageCache, DiskBackedFullCapacityStillStreams) {
        // Out-of-core RAD whose chunk count fits the pool entirely: only a
        // preview prefix is loaded, so nothing may be pre-published and the
        // roots must bootstrap through the normal streaming path.
        LodPageCache cache;
        cache.configure(64, 64, 2, 1024, /*disk_backed=*/true);
        EXPECT_EQ(cache.snapshot().resident_chunks, 0u);

        cache.ensureRootResidency();
        const auto uploads = cache.drainPendingUploads();
        ASSERT_EQ(uploads.size(), 2u);
        EXPECT_EQ(uploads[0].chunk, 0u);
        EXPECT_EQ(uploads[1].chunk, 1u);
        cache.completeUploads(uploads);
        EXPECT_EQ(cache.snapshot().resident_chunks, 2u);
    }

    TEST(LodPageCache, UploadDrainHonorsByteBudget) {
        LodPageCache cache;
        constexpr std::size_t kPageBytes = 1024;
        cache.configure(100, 80, 1, kPageBytes);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        cache.beginFrame();
        const auto want = makeRequests({10, 11, 12, 13, 14});
        cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(want),
                                      std::span<const std::uint32_t>{});

        auto first = cache.drainPendingUploads(3 * kPageBytes);
        EXPECT_EQ(first.size(), 3u);
        EXPECT_TRUE(cache.hasOutstandingWork()) << "remainder stays pending";
        cache.completeUploads(first);

        auto second = cache.drainPendingUploads(0);
        EXPECT_EQ(second.size(), 2u);
        cache.completeUploads(second);
        EXPECT_FALSE(cache.hasOutstandingWork());
    }

    TEST(LodPageCache, ResidentSinceFrameStampsPublishOrder) {
        LodPageCache cache;
        cache.configure(100, 10, 1);
        cache.ensureRootResidency();
        (void)drainAndComplete(cache);

        for (int frame = 0; frame < 5; ++frame) {
            cache.beginFrame();
        }
        const auto want = makeRequests({42});
        cache.submitTraversalPriority(std::span<const LodPageCache::ChunkRequest>(want),
                                      std::span<const std::uint32_t>{});
        auto uploads = cache.drainPendingUploads();
        ASSERT_EQ(uploads.size(), 1u);
        const std::uint32_t page = uploads[0].page;
        const std::uint32_t stamp_before = cache.snapshot().page_resident_frame[page];

        cache.beginFrame();
        cache.completeUploads(uploads);
        EXPECT_EQ(cache.snapshot().page_resident_frame[page], 6u)
            << "stamp must be set at publish, not at reserve";
        EXPECT_NE(cache.snapshot().page_resident_frame[page], stamp_before);
    }

} // namespace
