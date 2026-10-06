/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_upscaler_plugin.hpp"
#include "visualizer/rendering/scene_upscaler_registry.hpp"

#include <algorithm>
#include <gtest/gtest.h>

namespace lfs::vis {

    TEST(SceneUpscalerRegistry, RegistersStableBuiltInIdsAndKeepsOptionalIdsAddressable) {
        const auto descriptors = sceneUpscalerDescriptors();
        ASSERT_GE(descriptors.size(), 3u);
        EXPECT_EQ(descriptors[0].backend, SceneUpscalerBackend::Native);
        EXPECT_EQ(descriptors[0].id, "native");
        EXPECT_EQ(descriptors[0].label_key, "preferences.scene_reconstruction_off");
        EXPECT_EQ(descriptors[1].backend, SceneUpscalerBackend::Spatial);
        EXPECT_EQ(descriptors[1].id, "spatial");
        EXPECT_EQ(descriptors[1].label_key, "preferences.scene_reconstruction_spatial");
        EXPECT_EQ(descriptors[2].backend, SceneUpscalerBackend::Temporal);
        EXPECT_EQ(descriptors[2].id, "temporal");
        EXPECT_EQ(descriptors[2].label_key, "preferences.scene_reconstruction_temporal");
        EXPECT_TRUE(sceneUpscalerBackendAvailable(SceneUpscalerBackend::Native));
        EXPECT_EQ(sceneUpscalerBackendFromId("native"), SceneUpscalerBackend::Native);
        EXPECT_EQ(sceneUpscalerBackendFromId("spatial"), SceneUpscalerBackend::Spatial);
        EXPECT_EQ(sceneUpscalerBackendFromId("temporal"), SceneUpscalerBackend::Temporal);
        EXPECT_FALSE(sceneUpscalerBackendFromId("unknown-provider").has_value());
        EXPECT_FALSE(sceneUpscalerBackendFromId("").has_value());
    }

    TEST(SceneUpscalerRegistry, ReconstructionPresetsAreBackendSpecific) {
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

        const auto& temporal = sceneUpscalerDescriptor(SceneUpscalerBackend::Temporal);
        ASSERT_EQ(temporal.presets.size(), 3u);
        EXPECT_EQ(defaultSceneUpscalerPreset(SceneUpscalerBackend::Temporal).id, "quality");
        EXPECT_FLOAT_EQ(sceneUpscalerPreset(SceneUpscalerBackend::Temporal, "balanced")->input_scale,
                        0.67f);
    }

    TEST(SceneUpscalerRegistry, EveryPluginHasMatchingDescriptor) {
        for (const auto* const plugin : sceneUpscalerPlugins()) {
            const auto& info = plugin->info();
            EXPECT_EQ(sceneUpscalerPlugin(info.backend), plugin);
            EXPECT_EQ(sceneUpscalerDescriptor(info.backend).id, info.id);
        }
        EXPECT_EQ(sceneUpscalerPlugin(SceneUpscalerBackend::Native), nullptr);
        EXPECT_EQ(sceneUpscalerPlugin(SceneUpscalerBackend::Temporal), nullptr);
    }

    TEST(SceneUpscalerRegistry, BackendOnlyUpdateRestoresRememberedPresetEvenWhenIdsOverlap) {
        const auto remembered = resolveSceneUpscalerPresetUpdate(
            SceneUpscalerBackend::Temporal, std::nullopt, "performance");
        ASSERT_TRUE(remembered.has_value());
        EXPECT_EQ(remembered->id, "performance");

        const auto explicitly_requested = resolveSceneUpscalerPresetUpdate(
            SceneUpscalerBackend::Temporal,
            std::optional<std::string_view>{"balanced"},
            "performance");
        ASSERT_TRUE(explicitly_requested.has_value());
        EXPECT_EQ(explicitly_requested->id, "balanced");

        EXPECT_FALSE(resolveSceneUpscalerPresetUpdate(
                         SceneUpscalerBackend::Temporal,
                         std::optional<std::string_view>{"native"},
                         "performance")
                         .has_value());
        EXPECT_EQ(resolveSceneUpscalerPresetUpdate(
                      SceneUpscalerBackend::Temporal, std::nullopt, "invalid")
                      ->id,
                  "quality");
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
        EXPECT_EQ(sceneUpscalerFallbackId(fallback.fallback), "runtime_unavailable");
        EXPECT_EQ(sceneUpscalerFallbackId(SceneUpscalerFallback::None), "none");

        const auto unsupported_mode = resolveSceneUpscalerSelection(
            SceneUpscalerBackend::Temporal, false, SceneUpscalerFallback::UnsupportedMode);
        EXPECT_EQ(unsupported_mode.requested, SceneUpscalerBackend::Temporal);
        EXPECT_EQ(unsupported_mode.effective, SceneUpscalerBackend::Native);
        EXPECT_TRUE(unsupported_mode.fellBack());
        EXPECT_EQ(unsupported_mode.fallback, SceneUpscalerFallback::UnsupportedMode);
        EXPECT_EQ(sceneUpscalerFallbackId(unsupported_mode.fallback), "unsupported_mode");

        const auto temporal = resolveSceneUpscalerSelection(SceneUpscalerBackend::Temporal, true);
        EXPECT_EQ(temporal.requested, SceneUpscalerBackend::Temporal);
        EXPECT_EQ(temporal.effective, SceneUpscalerBackend::Temporal);
        EXPECT_FALSE(temporal.fellBack());

        const auto external_fallback = resolveSceneUpscalerSelection(
            static_cast<SceneUpscalerBackend>(10), false);
        EXPECT_EQ(external_fallback.requested, static_cast<SceneUpscalerBackend>(10));
        EXPECT_EQ(external_fallback.effective, SceneUpscalerBackend::Native);
        EXPECT_TRUE(external_fallback.fellBack());

        const auto second_provider_fallback = resolveSceneUpscalerSelection(
            static_cast<SceneUpscalerBackend>(20), false);
        EXPECT_EQ(second_provider_fallback.requested, static_cast<SceneUpscalerBackend>(20));
        EXPECT_EQ(second_provider_fallback.effective, SceneUpscalerBackend::Native);
        EXPECT_TRUE(second_provider_fallback.fellBack());
    }

} // namespace lfs::vis
