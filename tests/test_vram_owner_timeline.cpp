/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "diagnostics/vram_owner_model.hpp"
#include "diagnostics/vram_timeline.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <utility>

namespace {
    constexpr std::size_t MiB = 1024ull * 1024ull;

    TEST(VramOwnerModel, PierFiveMillionIteration8500LedgerRows) {
        using namespace lfs::diagnostics;
        const auto bytes = [](double mib) {
            return static_cast<std::size_t>(std::llround(mib * MiB));
        };
        for (const bool gui : {false, true}) {
            VramProfilerSnapshot s;
            s.process.cuda_pool_valid = true;
            s.process.cuda_pool_reserved = (gui ? 1120 : 864) * MiB;
            s.process.cuda_pool_used = (gui ? 974 : 767) * MiB;
            s.process.cuda_slab_reserved_bytes = 27 * MiB;
            s.process.exportable_splat_bytes = gui ? 704 * MiB : 0;
            s.process.shared_scratch_bytes = gui ? 586 * MiB : 0;
            s.accounted_arena_live_bytes = gui ? 0 : 590 * MiB;
            s.gauges.push_back({"vram.audit.tensor.cuda_direct_live_bytes",
                                static_cast<double>(bytes(gui ? 724.8 : 1463.9))});
            const auto sampled = [&](std::string scope, std::string label, double mib,
                                     VramAllocationMethod method) {
                s.rows.push_back({std::move(scope), std::move(label), bytes(mib), bytes(mib),
                                  0, 0, 0, 0, method, VramRowKind::Sampled});
            };
            for (const auto& [label, mib] : {
                     std::pair{"shN.exp_avg", 457.8},
                     {"rotation.exp_avg", 76.3},
                     {"means.exp_avg", 57.2},
                     {"scaling.exp_avg", 57.2},
                     {"sh0.exp_avg", 57.2},
                     {"opacity.exp_avg", 19.1}})
                sampled("optimizer.adam", label, mib, VramAllocationMethod::Direct);
            for (const auto& [label, mib] : {
                     std::pair{"shN", 429.2},
                     {"rotation", 76.3},
                     {"means", 57.2},
                     {"scaling", 57.2},
                     {"sh0", 57.2},
                     {"opacity", 19.1},
                     {"densification_info", 38.1},
                     {"deleted", 4.8}})
                sampled("model.gaussians", label, mib, VramAllocationMethod::External);
            if (gui) {
                sampled("shared.scratch", "per_splat", 329.9, VramAllocationMethod::External);
                sampled("shared.scratch", "sort_buffers", 76.3, VramAllocationMethod::External);
                sampled("shared.scratch", "reserve_unbound", 137.8, VramAllocationMethod::External);
                sampled("shared.scratch", "cuda_vulkan_arena#2", 586.0, VramAllocationMethod::External);
            } else {
                s.rows.push_back({"train.step/train.rasterize_forward/rasterizer.arena.vmm", "", 590 * MiB,
                                  590 * MiB, 0, 0, 0, 0, VramAllocationMethod::Arena,
                                  VramRowKind::Hooked});
            }
            const auto result = buildVramOwnerBreakdown(s, gui);
            const auto get = [&](VramOwner owner) { return result.bytes[static_cast<std::size_t>(owner)]; };
            EXPECT_NEAR(static_cast<double>(get(VramOwner::Optimizer)) / MiB, 724.8, .2);
            EXPECT_NEAR(static_cast<double>(get(VramOwner::Model)) / MiB, 696.2, .2);
            EXPECT_NEAR(static_cast<double>(get(VramOwner::Rasterizer)) / MiB, gui ? 586.0 : 590.0, .2);
            EXPECT_NEAR(static_cast<double>(get(VramOwner::Densification)) / MiB, 42.9, .2);
        }
    }

    TEST(VramOwnerModel, ArenaCapacityPersistsBetweenTrainingScopes) {
        lfs::diagnostics::VramProfilerSnapshot s;
        s.rows.push_back({"rasterizer.fastgs", "arena.capacity", 590 * MiB, 590 * MiB,
                          0, 0, 0, 0, lfs::diagnostics::VramAllocationMethod::External,
                          lfs::diagnostics::VramRowKind::Sampled});
        const auto result = lfs::diagnostics::buildVramOwnerBreakdown(s, true);
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(lfs::diagnostics::VramOwner::Rasterizer)],
                  590 * static_cast<std::int64_t>(MiB));
    }

    TEST(VramOwnerModel, LiveStrategyGaugesUseDirectTensorBacking) {
        using namespace lfs::diagnostics;
        VramProfilerSnapshot s;
        s.process.process_memory_valid = true;
        s.process.process_used = 100 * MiB;
        s.gauges = {
            {"vram.audit.tensor.cuda_direct_live_bytes", static_cast<double>(100 * MiB)},
            {"vram.audit.mrnf.edge.score_sum.allocated_bytes", static_cast<double>(45 * MiB)},
            {"vram.audit.mrnf.free_mask.allocated_bytes", static_cast<double>(30 * MiB)},
            {"vram.audit.mrnf.densify_child.allocated_bytes", static_cast<double>(200 * MiB)},
        };
        s.rows.push_back({"optimizer.adam", "means.exp_avg", 25 * MiB, 25 * MiB,
                          0, 0, 0, 0, VramAllocationMethod::Direct, VramRowKind::Sampled});
        const auto result = buildVramOwnerBreakdown(s);
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(VramOwner::Optimizer)],
                  25 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(VramOwner::Densification)],
                  75 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(VramOwner::Unattributed)], 0);
    }

    TEST(VramOwnerModel, NestedTrainingScopesAndMeasuredProcessBalance) {
        using namespace lfs::diagnostics;
        VramProfilerSnapshot s;
        s.process.process_memory_valid = true;
        s.process.process_used = 800 * MiB;
        s.process.cuda_pool_valid = true;
        s.process.cuda_pool_reserved = 200 * MiB;
        s.process.cuda_context_baseline = 80 * MiB;
        s.rows = {
            {"Training execution/train.step/train.rasterize_forward", "async", 61 * MiB, 61 * MiB,
             0, 0, 0, 0, VramAllocationMethod::Async, VramRowKind::Hooked},
            {"Training execution/train.step", "async", 53 * MiB, 53 * MiB,
             0, 0, 0, 0, VramAllocationMethod::Async, VramRowKind::Hooked},
            {"Training execution/train.step", "bucketed", 30 * MiB, 30 * MiB,
             0, 0, 0, 0, VramAllocationMethod::Bucketed, VramRowKind::Hooked},
        };
        const auto result = buildVramOwnerBreakdown(s);
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(VramOwner::Rasterizer)],
                  61 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(VramOwner::LossStep)],
                  83 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(VramOwner::Context)],
                  600 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.context_inferred_bytes, 520 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(VramOwner::Unattributed)], 0);
        EXPECT_EQ(result.sum(), 800 * static_cast<std::int64_t>(MiB));
    }

    TEST(VramOwnerModel, ClosedCategoriesAndAliases) {
        lfs::diagnostics::VramProfilerSnapshot s;
        s.enabled = true;
        s.process.process_memory_valid = true;
        s.process.cuda_pool_valid = true;
        s.process.cuda_pool_reserved = 320 * MiB;
        s.process.cuda_pool_used = 280 * MiB;
        s.process.cuda_pool_bucket_cache_bytes = 20 * MiB;
        s.process.cuda_pool_bucket_live_waste_bytes = 10 * MiB;
        s.process.exportable_splat_bytes = 100 * MiB;
        s.process.vulkan_vma_block_bytes = 60 * MiB;
        s.process.cuda_context_baseline = 80 * MiB;
        s.process.process_used = 560 * MiB;
        s.rows = {
            {"model.gaussians", "means", 100 * MiB, 100 * MiB, 0, 0, 0, 0,
             lfs::diagnostics::VramAllocationMethod::External, lfs::diagnostics::VramRowKind::Sampled},
            {"optimizer.adam", "means.grad", 70 * MiB, 70 * MiB, 0, 0, 0, 0,
             lfs::diagnostics::VramAllocationMethod::Async, lfs::diagnostics::VramRowKind::Hooked},
            {"io.nvimagecodec", "default_pool", 50 * MiB, 50 * MiB, 0, 0, 0, 0,
             lfs::diagnostics::VramAllocationMethod::Async, lfs::diagnostics::VramRowKind::Static},
            {"vulkan.ui_texture.image", "poster", 40 * MiB, 40 * MiB, 0, 0, 0, 0,
             lfs::diagnostics::VramAllocationMethod::External, lfs::diagnostics::VramRowKind::Sampled},
            {"vulkan.vma", "allocator_free_in_blocks", 20 * MiB, 20 * MiB, 0, 0, 0, 0,
             lfs::diagnostics::VramAllocationMethod::External, lfs::diagnostics::VramRowKind::Sampled},
            {"vulkan.external.imported_block", "alias", 100 * MiB, 100 * MiB, 0, 0, 0, 0,
             lfs::diagnostics::VramAllocationMethod::External, lfs::diagnostics::VramRowKind::Sampled},
        };
        const auto result = lfs::diagnostics::buildVramOwnerBreakdown(s);
        EXPECT_EQ(result.sum(), static_cast<std::int64_t>(s.process.process_used));
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(lfs::diagnostics::VramOwner::Optimizer)],
                  70 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(lfs::diagnostics::VramOwner::Model)],
                  100 * static_cast<std::int64_t>(MiB));
        EXPECT_FALSE(result.over_attributed);
    }

    TEST(VramOwnerModel, SignedOverAttribution) {
        lfs::diagnostics::VramProfilerSnapshot s;
        s.process.process_memory_valid = true;
        s.process.process_used = 100 * MiB;
        s.process.cuda_pool_valid = true;
        s.process.cuda_pool_reserved = 120 * MiB;
        s.process.cuda_pool_used = 120 * MiB;
        const auto result = lfs::diagnostics::buildVramOwnerBreakdown(s);
        EXPECT_TRUE(result.over_attributed);
        EXPECT_EQ(result.sum(), 100 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.signed_residual_bytes, -20 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(lfs::diagnostics::VramOwner::Unattributed)],
                  -20 * static_cast<std::int64_t>(MiB));
    }

    TEST(VramOwnerModel, SampledTensorFillsOnlyUnclaimedPoolBytes) {
        lfs::diagnostics::VramProfilerSnapshot s;
        s.process.process_memory_valid = true;
        s.process.process_used = 200 * MiB;
        s.process.cuda_pool_valid = true;
        s.process.cuda_pool_reserved = 200 * MiB;
        s.process.cuda_pool_used = 200 * MiB;
        s.rows = {
            {"optimizer.adam", "means.grad", 80 * MiB, 80 * MiB, 0, 0, 0, 0,
             lfs::diagnostics::VramAllocationMethod::Async, lfs::diagnostics::VramRowKind::Hooked},
            {"optimizer.adam", "means.exp_avg", 100 * MiB, 100 * MiB, 0, 0, 0, 0,
             lfs::diagnostics::VramAllocationMethod::Unknown, lfs::diagnostics::VramRowKind::Sampled},
            {"train.inputs", "gt_tile", 60 * MiB, 60 * MiB, 0, 0, 0, 0,
             lfs::diagnostics::VramAllocationMethod::Unknown, lfs::diagnostics::VramRowKind::Sampled},
        };
        const auto result = lfs::diagnostics::buildVramOwnerBreakdown(s);
        EXPECT_EQ(result.sum(), 200 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(lfs::diagnostics::VramOwner::Optimizer)],
                  180 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.bytes[static_cast<std::size_t>(lfs::diagnostics::VramOwner::LossStep)],
                  20 * static_cast<std::int64_t>(MiB));
        EXPECT_EQ(result.signed_residual_bytes, 0);
    }

    TEST(VramTimeline, CompactionSpikesAndCsv) {
        lfs::diagnostics::VramTimeline series;
        lfs::diagnostics::VramProfilerSnapshot s;
        s.enabled = true;
        s.process.process_memory_valid = true;
        s.process.total = 4096 * MiB;
        for (int i = 0; i < 40'000; ++i) {
            s.iteration = i;
            s.process.process_used = (i < 2 ? 100 : 200) * MiB;
            series.push(s, static_cast<std::int64_t>(i) * 1000);
        }
        EXPECT_LE(series.points().size(), 1800u + 4u * 4320u);
        EXPECT_EQ(series.points().front().epoch_ms, 0);
        EXPECT_TRUE(lfs::diagnostics::isVramProcessSpike(100 * MiB, 200 * MiB));
        EXPECT_FALSE(lfs::diagnostics::isVramProcessSpike(100 * MiB, 150 * MiB));
        EXPECT_NE(series.csv().find("resolution_seconds"), std::string::npos);
        ASSERT_FALSE(series.markers().empty());
        EXPECT_EQ(series.markers().front().kind, "spike");
        const auto points = series.points();
        EXPECT_EQ(lfs::diagnostics::vramTimelineAxisMax(points), 256 * MiB);
    }

    TEST(VramTimeline, FitScaleIgnoresDeviceAndCapacity) {
        lfs::diagnostics::VramTimelinePoint point;
        point.process_bytes = 200 * MiB;
        point.device_bytes = 20'000 * MiB;
        point.capacity_bytes = 24'000 * MiB;
        const std::array points{point};
        EXPECT_EQ(lfs::diagnostics::vramTimelineAxisMax(points), 256 * MiB);
        EXPECT_EQ(lfs::diagnostics::vramTimelineAxisMax(points, true), 24'576 * MiB);
    }

    TEST(VramTimeline, LargeRowAndEscapedEvent) {
        lfs::diagnostics::VramTimeline series;
        lfs::diagnostics::VramProfilerSnapshot s;
        s.enabled = true;
        s.process.process_memory_valid = true;
        s.process.process_used = 100 * MiB;
        series.push(s, 1000);
        s.markers.push_back({1, 2000, 2, "save", "a,\"b\"", 0, 0.0});
        s.rows.push_back({"io.decode", "thread-buffer", 70 * MiB, 70 * MiB, 0, 0, 0, 0,
                          lfs::diagnostics::VramAllocationMethod::External,
                          lfs::diagnostics::VramRowKind::Sampled});
        series.push(s, 2000);
        bool saw_allocation = false;
        for (const auto& marker : series.markers())
            saw_allocation |= marker.kind == "allocation" && marker.text == "io.decode/thread-buffer";
        EXPECT_TRUE(saw_allocation);
        EXPECT_NE(series.csv().find("\"a,\"\"b\"\"\""), std::string::npos);
    }
} // namespace
