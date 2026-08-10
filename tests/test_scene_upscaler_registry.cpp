/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_upscaler_registry.hpp"

#include <gtest/gtest.h>

namespace lfs::vis {

    TEST(SceneUpscalerRegistry, RegistersInternalBackends) {
        const auto descriptors = sceneUpscalerDescriptors();
        ASSERT_EQ(descriptors.size(), 3u);
        EXPECT_EQ(descriptors.front().backend, SceneUpscalerBackend::Native);
        EXPECT_EQ(descriptors.front().id, "native");
        EXPECT_EQ(descriptors[1].backend, SceneUpscalerBackend::Spatial);
        EXPECT_EQ(descriptors[1].id, "spatial");
        EXPECT_EQ(descriptors.back().backend, SceneUpscalerBackend::Temporal);
        EXPECT_EQ(descriptors.back().id, "temporal");
    }

    TEST(SceneUpscalerRegistry, SpatialNeedsOnlyItsLazyAdapter) {
        const auto& spatial = spatialSceneUpscalerDescriptor();
        EXPECT_TRUE(spatial.available);
        EXPECT_TRUE(spatial.requires_adapter);
        EXPECT_FALSE(spatial.requirements.any());
        EXPECT_FALSE(spatial.requirements.temporal());
    }

    TEST(SceneUpscalerRegistry, NativeRequiresNoInputsAdapterOrOptionalRuntime) {
        const auto& native = nativeSceneUpscalerDescriptor();
        EXPECT_TRUE(native.available);
        EXPECT_FALSE(native.requires_adapter);
        EXPECT_FALSE(native.requirements.any());
        EXPECT_FALSE(native.requirements.temporal());
        EXPECT_FALSE(native.requirements.depth);
        EXPECT_FALSE(native.requirements.motion_vectors);
        EXPECT_FALSE(native.requirements.jitter);
        EXPECT_FALSE(native.requirements.history);
        EXPECT_FALSE(native.requirements.reactive_mask);
        EXPECT_FALSE(native.requirements.exposure);
    }

    TEST(SceneUpscalerRegistry, TemporalDeclaresOnlyImplementedRuntimeInputs) {
        const auto& temporal = temporalSceneUpscalerDescriptor();
        EXPECT_TRUE(temporal.available);
        EXPECT_TRUE(temporal.requires_adapter);
        EXPECT_TRUE(temporal.requirements.depth);
        EXPECT_TRUE(temporal.requirements.motion_vectors);
        EXPECT_TRUE(temporal.requirements.history);
        EXPECT_FALSE(temporal.requirements.jitter);
        EXPECT_FALSE(temporal.requirements.reactive_mask);
        EXPECT_FALSE(temporal.requirements.exposure);
    }

    TEST(SceneUpscalerRegistry, LookupRejectsUnknownOrUnavailableIds) {
        EXPECT_EQ(sceneUpscalerBackendFromId("native"), SceneUpscalerBackend::Native);
        EXPECT_EQ(sceneUpscalerBackendFromId("spatial"), SceneUpscalerBackend::Spatial);
        EXPECT_EQ(sceneUpscalerBackendFromId("temporal"), SceneUpscalerBackend::Temporal);
        EXPECT_FALSE(sceneUpscalerBackendFromId("nis").has_value());
        EXPECT_FALSE(sceneUpscalerBackendFromId("fsr").has_value());
        EXPECT_FALSE(sceneUpscalerBackendFromId("xess").has_value());
        EXPECT_FALSE(sceneUpscalerBackendFromId("dlss").has_value());
        EXPECT_FALSE(sceneUpscalerBackendFromId("").has_value());
    }

    TEST(SceneUpscalerRegistry, ResolvesEffectiveBackendAndFallbackExplicitly) {
        const auto native = resolveSceneUpscalerSelection(
            SceneUpscalerBackend::Native, false);
        EXPECT_EQ(native.requested, SceneUpscalerBackend::Native);
        EXPECT_EQ(native.effective, SceneUpscalerBackend::Native);
        EXPECT_FALSE(native.fallback);

        const auto spatial = resolveSceneUpscalerSelection(
            SceneUpscalerBackend::Spatial, true);
        EXPECT_EQ(spatial.requested, SceneUpscalerBackend::Spatial);
        EXPECT_EQ(spatial.effective, SceneUpscalerBackend::Spatial);
        EXPECT_FALSE(spatial.fallback);

        const auto fallback = resolveSceneUpscalerSelection(
            SceneUpscalerBackend::Spatial, false);
        EXPECT_EQ(fallback.requested, SceneUpscalerBackend::Spatial);
        EXPECT_EQ(fallback.effective, SceneUpscalerBackend::Native);
        EXPECT_TRUE(fallback.fallback);

        const auto temporal = resolveSceneUpscalerSelection(
            SceneUpscalerBackend::Temporal, true);
        EXPECT_EQ(temporal.effective, SceneUpscalerBackend::Temporal);
        EXPECT_FALSE(temporal.fallback);
    }
} // namespace lfs::vis
