/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_upscaler_registry.hpp"

#include <gtest/gtest.h>

namespace lfs::vis {

    TEST(SceneUpscalerRegistry, NativeIsTheOnlyRegisteredBackendInitially) {
        const auto descriptors = sceneUpscalerDescriptors();
        ASSERT_EQ(descriptors.size(), 1u);
        EXPECT_EQ(descriptors.front().backend, SceneUpscalerBackend::Native);
        EXPECT_EQ(descriptors.front().id, "native");
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

    TEST(SceneUpscalerRegistry, LookupRejectsUnknownOrUnavailableIds) {
        EXPECT_EQ(sceneUpscalerBackendFromId("native"), SceneUpscalerBackend::Native);
        EXPECT_FALSE(sceneUpscalerBackendFromId("nis").has_value());
        EXPECT_FALSE(sceneUpscalerBackendFromId("fsr").has_value());
        EXPECT_FALSE(sceneUpscalerBackendFromId("xess").has_value());
        EXPECT_FALSE(sceneUpscalerBackendFromId("dlss").has_value());
        EXPECT_FALSE(sceneUpscalerBackendFromId("").has_value());
    }
} // namespace lfs::vis
