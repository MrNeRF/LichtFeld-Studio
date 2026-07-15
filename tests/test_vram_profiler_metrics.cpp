/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_label.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <gtest/gtest.h>
#include <thread>

using lfs::diagnostics::VramAllocationMethod;
using lfs::diagnostics::VramProfiler;

namespace {

    class VramProfilerMetricsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            auto& p = VramProfiler::instance();
            p.setEnabled(false);
            p.setEnabled(true);
        }

        void TearDown() override {
            VramProfiler::instance().setEnabled(false);
        }
    };

    TEST_F(VramProfilerMetricsTest, GaugeRoundTripsThroughSnapshot) {
        auto& p = VramProfiler::instance();
        p.setGauge("model.gaussians", 12345.0);
        p.setGauge("model.gaussians", 23456.0);
        p.setGauge("model.capacity", 100000.0);

        const auto snap = p.snapshot();
        ASSERT_EQ(snap.gauges.size(), 2u);
        EXPECT_EQ(snap.gauges[0].key, "model.capacity");
        EXPECT_DOUBLE_EQ(snap.gauges[0].value, 100000.0);
        EXPECT_EQ(snap.gauges[1].key, "model.gaussians");
        EXPECT_DOUBLE_EQ(snap.gauges[1].value, 23456.0);
    }

    TEST_F(VramProfilerMetricsTest, PinnedHostMemoryRoundTripsThroughSnapshot) {
        auto& p = VramProfiler::instance();
        p.setPinnedHostMemory(11, 22, 44);

        const auto snap = p.snapshot();
        EXPECT_EQ(snap.process.pinned_host_used, 11u);
        EXPECT_EQ(snap.process.pinned_host_cached, 22u);
        EXPECT_EQ(snap.process.pinned_host_peak, 44u);
    }

    TEST_F(VramProfilerMetricsTest, IterCounterResetsOnBeginIteration) {
        auto& p = VramProfiler::instance();
        p.beginIteration(0);
        p.addCounter("strategy.clones", 7, /*per_iteration=*/true);
        p.addCounter("strategy.clones", 3, /*per_iteration=*/true);

        auto snap = p.snapshot();
        ASSERT_EQ(snap.iter_counters.size(), 1u);
        EXPECT_EQ(snap.iter_counters[0].key, "strategy.clones");
        EXPECT_EQ(snap.iter_counters[0].value, 10u);
        ASSERT_EQ(snap.total_counters.size(), 1u);
        EXPECT_EQ(snap.total_counters[0].value, 10u);

        p.beginIteration(1);
        snap = p.snapshot();
        EXPECT_TRUE(snap.iter_counters.empty());
        ASSERT_EQ(snap.total_counters.size(), 1u);
        EXPECT_EQ(snap.total_counters[0].value, 10u);
    }

    TEST_F(VramProfilerMetricsTest, HistogramP95FromRollingSamples) {
        auto& p = VramProfiler::instance();
        for (int i = 1; i <= 100; ++i) {
            p.recordHistogram("loss", static_cast<double>(i));
        }
        const auto snap = p.snapshot();
        ASSERT_EQ(snap.histograms.size(), 1u);
        const auto& h = snap.histograms[0];
        EXPECT_EQ(h.count, 100u);
        EXPECT_DOUBLE_EQ(h.min_value, 1.0);
        EXPECT_DOUBLE_EQ(h.max_value, 100.0);
        EXPECT_NEAR(h.p50, 50.5, 1.5);
        EXPECT_NEAR(h.p95, 95.0, 2.0);
        EXPECT_NEAR(h.p99, 99.0, 2.0);
    }

    TEST_F(VramProfilerMetricsTest, TopLiveSortedByLiveBytes) {
        auto& p = VramProfiler::instance();
        // Use recordCurrentBytes (no real pointers needed).
        p.recordCurrentBytes("test.small", "tensor", 1024, VramAllocationMethod::External);
        p.recordCurrentBytes("test.large", "tensor", 1024 * 1024 * 32, VramAllocationMethod::External);
        p.recordCurrentBytes("test.medium", "tensor", 1024 * 1024, VramAllocationMethod::External);

        const auto snap = p.snapshot();
        ASSERT_GE(snap.top_live.size(), 3u);
        EXPECT_EQ(snap.top_live[0].scope, "test.large");
        EXPECT_EQ(snap.top_live[1].scope, "test.medium");
        EXPECT_EQ(snap.top_live[2].scope, "test.small");
    }

    TEST_F(VramProfilerMetricsTest, TimerSampleFillsWallPercentiles) {
        auto& p = VramProfiler::instance();
        for (int i = 1; i <= 200; ++i) {
            p.recordTimerSample("kernel.fake", static_cast<double>(i));
        }
        const auto snap = p.snapshot();
        bool found = false;
        for (const auto& node : snap.tree) {
            if (node.path == "kernel/fake") {
                found = true;
                EXPECT_GT(node.wall_p95_ms, node.wall_p50_ms);
                EXPECT_GT(node.wall_p99_ms, node.wall_p95_ms);
                EXPECT_NEAR(node.wall_p50_ms, 100.5, 5.0);
                EXPECT_NEAR(node.wall_p95_ms, 190.0, 5.0);
                EXPECT_NEAR(node.wall_p99_ms, 198.0, 3.0);
                break;
            }
        }
        EXPECT_TRUE(found);
    }

    TEST_F(VramProfilerMetricsTest, IterAllocFreeDeltasPerIteration) {
        auto& p = VramProfiler::instance();
        p.beginIteration(0);

        int dummy_a = 0;
        int dummy_b = 0;
        int dummy_c = 0;
        p.recordAllocation(&dummy_a, 64, VramAllocationMethod::External);
        p.recordAllocation(&dummy_b, 64, VramAllocationMethod::External);
        p.recordDeallocation(&dummy_a);

        auto snap = p.snapshot();
        EXPECT_EQ(snap.iter_allocation_events, 2u);
        EXPECT_EQ(snap.iter_free_events, 1u);

        p.beginIteration(1);
        p.recordAllocation(&dummy_c, 32, VramAllocationMethod::External);
        snap = p.snapshot();
        EXPECT_EQ(snap.iter_allocation_events, 1u);
        EXPECT_EQ(snap.iter_free_events, 0u);

        // Total counters keep accumulating.
        EXPECT_EQ(snap.allocation_events, 3u);
        EXPECT_EQ(snap.free_events, 1u);

        p.recordDeallocation(&dummy_b);
        p.recordDeallocation(&dummy_c);
    }

    TEST_F(VramProfilerMetricsTest, DisableWipesGaugesAndCounters) {
        auto& p = VramProfiler::instance();
        p.setGauge("g", 1.0);
        p.addCounter("c", 5, true);
        p.recordHistogram("h", 1.0);

        p.setEnabled(false);
        p.setEnabled(true);
        const auto snap = p.snapshot();
        EXPECT_TRUE(snap.gauges.empty());
        EXPECT_TRUE(snap.iter_counters.empty());
        EXPECT_TRUE(snap.total_counters.empty());
        EXPECT_TRUE(snap.histograms.empty());
    }

    TEST_F(VramProfilerMetricsTest, NoOpWhenDisabled) {
        auto& p = VramProfiler::instance();
        p.setEnabled(false);
        p.setGauge("g", 99.0);
        p.addCounter("c", 5, true);
        p.recordHistogram("h", 1.0);
        const auto snap = p.snapshot();
        EXPECT_TRUE(snap.gauges.empty());
        EXPECT_TRUE(snap.iter_counters.empty());
        EXPECT_TRUE(snap.histograms.empty());
    }

    TEST_F(VramProfilerMetricsTest, StaticBytesContributeToSampledLiveBytes) {
        auto& p = VramProfiler::instance();
        p.clearStaticScope("io.nvimagecodec");
        p.recordStaticBytes("io.nvimagecodec", "decoder_encoder_pool", 4096);

        auto snap = p.snapshot();
        bool found = false;
        for (const auto& row : snap.rows) {
            if (row.scope == "io.nvimagecodec" && row.label == "decoder_encoder_pool") {
                found = true;
                EXPECT_EQ(row.live_bytes, 4096u);
            }
        }

        EXPECT_TRUE(found);
        EXPECT_GE(snap.sampled_live_bytes, 4096u);

        p.clearStaticScope("io.nvimagecodec");
        snap = p.snapshot();
        for (const auto& row : snap.rows) {
            EXPECT_NE(row.scope, "io.nvimagecodec");
        }
    }

    TEST_F(VramProfilerMetricsTest, SlabReserveBytesRoundTripThroughProcessSnapshot) {
        auto& p = VramProfiler::instance();
        constexpr std::size_t kReservedBytes = 32ull << 20;
        p.setCudaSlabReservedBytes(kReservedBytes);

        const auto snap = p.snapshot();
        EXPECT_EQ(snap.process.cuda_slab_reserved_bytes, kReservedBytes);
    }

    TEST_F(VramProfilerMetricsTest, LiveBytesDoesNotAccumulateAcrossIterations) {
        // Regression: when allocations live under a deeply-nested scope path
        // ("Training execution/train.step/..."), beginIteration() previously
        // failed to reset metric.live_bytes (the brittle prefix match missed
        // the outer scope), then re-added the allocation bytes on every
        // rebuild — exploding live_bytes far past physical VRAM.
        auto& p = VramProfiler::instance();
        p.pushScope("Training execution");
        p.pushScope("train.step");
        int dummy = 0;
        p.recordAllocation(&dummy, 1024, VramAllocationMethod::Bucketed);

        for (int i = 0; i < 50; ++i) {
            p.beginIteration(i);
        }

        const auto snap = p.snapshot();
        std::size_t total_live_in_training = 0;
        for (const auto& row : snap.rows) {
            if (row.scope.find("Training execution") != std::string::npos) {
                total_live_in_training += row.live_bytes;
            }
        }
        EXPECT_EQ(total_live_in_training, 1024u);
        EXPECT_EQ(snap.accounted_live_bytes, 1024u);

        p.recordDeallocation(&dummy);
        p.popScope();
        p.popScope();
    }

    TEST_F(VramProfilerMetricsTest, TensorLabelFlowsToMetricRow) {
        auto& p = VramProfiler::instance();
        lfs::core::Tensor labeled;
        {
            LFS_LABEL_SCOPE("splat.positions.test");
            labeled = lfs::core::Tensor::zeros({64, 3}, lfs::core::Device::CUDA,
                                               lfs::core::DataType::Float32);
        }
        labeled.set_name("splat.positions.test");

        const auto snap = p.snapshot();
        bool found = false;
        for (const auto& row : snap.rows) {
            if (row.label == "splat.positions.test" && row.live_bytes > 0) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }

    TEST_F(VramProfilerMetricsTest, IterPerSecondGrowsAcrossIterations) {
        auto& p = VramProfiler::instance();
        p.beginIteration(0);
        for (int i = 1; i <= 5; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            p.beginIteration(i);
        }
        // We don't assert exact iter/s because timing is noisy, but it should be > 0
        // and iter_ms_last should be populated.
        const auto snap = p.snapshot();
        EXPECT_GT(snap.iter_ms_last, 0.0);
    }

} // namespace
