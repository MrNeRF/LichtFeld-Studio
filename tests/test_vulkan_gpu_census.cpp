/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Epic #1496 §5.4 — GpuObjectCensus live External* counts.

#include "window/gpu_object_census.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

    using lfs::vis::GpuObjectCensus;
    using lfs::vis::GpuObjectCensusRow;
    using lfs::vis::GpuObjectKind;

    [[nodiscard]] const GpuObjectCensusRow* findRow(const std::vector<GpuObjectCensusRow>& rows,
                                                    GpuObjectKind kind,
                                                    std::string_view scope) {
        for (const auto& row : rows) {
            if (row.kind == kind && row.scope == scope) {
                return &row;
            }
        }
        return nullptr;
    }

} // namespace

// Catches: onDestroy failing to decrement (balanced create/destroy leaving survivors).
TEST(GpuObjectCensus, BalancedCreateDestroyEmptyReport) {
    GpuObjectCensus census;
    census.onCreate(GpuObjectKind::ExternalImage, "scope.a");
    census.onCreate(GpuObjectKind::ExternalBuffer, "scope.b");
    census.onDestroy(GpuObjectKind::ExternalImage, "scope.a");
    census.onDestroy(GpuObjectKind::ExternalBuffer, "scope.b");
    EXPECT_TRUE(census.report().empty());
    EXPECT_FALSE(census.underflowFlagged());
}

// Catches: empty report ignoring live objects (no {kind, scope, count} rows).
TEST(GpuObjectCensus, SurvivorsReportKindScopeCount) {
    GpuObjectCensus census;
    census.onCreate(GpuObjectKind::ExternalImage, "vulkan.vksplat.output_image");
    census.onCreate(GpuObjectKind::ExternalImage, "vulkan.vksplat.output_image");
    census.onCreate(GpuObjectKind::ExternalSemaphore, "vulkan.gui.interop_semaphore");

    const auto rows = census.report();
    ASSERT_FALSE(rows.empty());

    const auto* images = findRow(rows, GpuObjectKind::ExternalImage, "vulkan.vksplat.output_image");
    ASSERT_NE(images, nullptr);
    EXPECT_EQ(images->count, 2);

    const auto* sems =
        findRow(rows, GpuObjectKind::ExternalSemaphore, "vulkan.gui.interop_semaphore");
    ASSERT_NE(sems, nullptr);
    EXPECT_EQ(sems->count, 1);
}

// Catches: counts collapsed across scopes (destroy one scope affecting another).
TEST(GpuObjectCensus, ScopeIsolation) {
    GpuObjectCensus census;
    census.onCreate(GpuObjectKind::ExternalBuffer, "scope.left");
    census.onCreate(GpuObjectKind::ExternalBuffer, "scope.right");
    census.onDestroy(GpuObjectKind::ExternalBuffer, "scope.left");

    const auto rows = census.report();
    EXPECT_EQ(findRow(rows, GpuObjectKind::ExternalBuffer, "scope.left"), nullptr);
    const auto* right = findRow(rows, GpuObjectKind::ExternalBuffer, "scope.right");
    ASSERT_NE(right, nullptr);
    EXPECT_EQ(right->count, 1);
}

// Catches: destroy-without-create going negative or silently ignoring underflow.
TEST(GpuObjectCensus, DestroyWithoutCreateClampsAndFlags) {
    GpuObjectCensus census;
    census.onDestroy(GpuObjectKind::ExternalImage, "orphan.scope");
    EXPECT_TRUE(census.underflowFlagged());
    // Clamped: report must not contain a negative count for that scope.
    for (const auto& row : census.report()) {
        EXPECT_GE(row.count, 0);
    }
}
