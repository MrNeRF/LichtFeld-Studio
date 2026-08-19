/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_upscaler_registry.hpp"

#include <gtest/gtest.h>

namespace lfs::vis {

    TEST(SceneUpscalerRegistry, RegistersStableNativeAndSpatialIds) {
        const auto descriptors = sceneUpscalerDescriptors();
        ASSERT_EQ(descriptors.size(), 2u);
        EXPECT_EQ(descriptors[0].backend, SceneUpscalerBackend::Native);
        EXPECT_EQ(descriptors[0].id, "native");
        EXPECT_EQ(descriptors[0].label_key, "preferences.scene_reconstruction_off");
        EXPECT_EQ(descriptors[1].backend, SceneUpscalerBackend::Spatial);
        EXPECT_EQ(descriptors[1].id, "spatial");
        EXPECT_EQ(descriptors[1].label_key, "preferences.scene_reconstruction_spatial");
        EXPECT_EQ(sceneUpscalerBackendFromId("native"), SceneUpscalerBackend::Native);
        EXPECT_EQ(sceneUpscalerBackendFromId("spatial"), SceneUpscalerBackend::Spatial);
        EXPECT_FALSE(sceneUpscalerBackendFromId("dlss").has_value());
        EXPECT_FALSE(sceneUpscalerBackendFromId("").has_value());
    }

    TEST(SceneUpscalerRegistry, SpatialPresetsAreBackendSpecific) {
        const auto& native = sceneUpscalerDescriptor(SceneUpscalerBackend::Native);
        ASSERT_EQ(native.presets.size(), 1u);
        EXPECT_EQ(native.presets.front().id, "native");
        EXPECT_FLOAT_EQ(native.presets.front().input_scale, 1.0f);

        const auto& spatial = sceneUpscalerDescriptor(SceneUpscalerBackend::Spatial);
        ASSERT_EQ(spatial.presets.size(), 3u);
        EXPECT_EQ(defaultSceneUpscalerPreset(SceneUpscalerBackend::Spatial).id, "quality");
        EXPECT_FLOAT_EQ(sceneUpscalerPreset(SceneUpscalerBackend::Spatial, "balanced")->input_scale,
                        0.67f);
        EXPECT_FALSE(sceneUpscalerPreset(SceneUpscalerBackend::Native, "balanced").has_value());
    }

    TEST(SceneUpscalerRegistry, ReportsRequestedEffectiveAndFallbackSeparately) {
        const auto spatial = resolveSceneUpscalerSelection(SceneUpscalerBackend::Spatial, true);
        EXPECT_EQ(spatial.requested, SceneUpscalerBackend::Spatial);
        EXPECT_EQ(spatial.effective, SceneUpscalerBackend::Spatial);
        EXPECT_FALSE(spatial.fellBack());

        const auto fallback = resolveSceneUpscalerSelection(SceneUpscalerBackend::Spatial, false);
        EXPECT_EQ(fallback.requested, SceneUpscalerBackend::Spatial);
        EXPECT_EQ(fallback.effective, SceneUpscalerBackend::Native);
        EXPECT_EQ(fallback.fallback, SceneUpscalerFallback::RuntimeUnavailable);
        EXPECT_TRUE(fallback.fellBack());
    }

} // namespace lfs::vis
