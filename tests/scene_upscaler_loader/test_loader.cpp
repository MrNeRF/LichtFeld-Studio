/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/scene_upscaler_plugin.hpp"
#include "rendering/scene_upscaler_plugin_metadata.hpp"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>

namespace lfs::vis {
    TEST(GenericProviderLoader, DiscoversMultipleNamesAndPreservesIndependentContracts) {
        configureSceneUpscalerPluginLoading(false);
        EXPECT_EQ(sceneUpscalerDescriptors().size(), 3u);
        const auto alpha = sceneUpscalerBackendFromId("provider-alpha");
        const auto beta = sceneUpscalerBackendFromId("amd-fsr3");
        ASSERT_TRUE(alpha);
        ASSERT_TRUE(beta);
        EXPECT_NE(alpha, beta);
        EXPECT_FALSE(sceneUpscalerBackendFromId("duplicate-provider"));
        EXPECT_FALSE(sceneUpscalerBackendAvailable(*alpha));
        configureSceneUpscalerPluginLoading(true);
        const auto catalog = sceneUpscalerDescriptors();
        ASSERT_EQ(catalog.size(), 5u);
        EXPECT_EQ(sceneUpscalerDescriptor(*alpha).display_name, "Provider Alpha");
        EXPECT_EQ(sceneUpscalerDescriptor(*beta).display_name, "AMD FSR 3.1");
        EXPECT_EQ(sceneUpscalerBackendId(*alpha), "provider-alpha");
        EXPECT_FLOAT_EQ(sceneUpscalerPreset(*alpha, "balanced")->input_scale, 0.58f);
        EXPECT_FLOAT_EQ(sceneUpscalerPreset(*beta, "balanced")->input_scale, 1.0f / 1.7f);
        const auto bad = sceneUpscalerBackendFromId("bad-provider");
        ASSERT_TRUE(bad);
        EXPECT_FALSE(sceneUpscalerBackendAvailable(*bad));
        auto* first = sceneUpscalerPlugin(*alpha);
        auto* second = sceneUpscalerPlugin(*beta);
        ASSERT_NE(first, second);
        LfsSceneUpscalerRuntimeConfigV1 runtime{.struct_size = sizeof(runtime)};
        ASSERT_TRUE(first->initializeRuntime(runtime));
        ASSERT_TRUE(second->initializeRuntime(runtime));
        auto view_a = first->acquireViewIdentity();
        auto view_b = first->acquireViewIdentity();
        auto view_other = second->acquireViewIdentity();
        ASSERT_TRUE(view_a);
        ASSERT_TRUE(view_b);
        ASSERT_TRUE(view_other);
        EXPECT_NE(view_a, view_b);
        LfsSceneUpscalerFeatureConfigV1 feature{.struct_size = sizeof(feature), .view = *view_a};
        ASSERT_TRUE(first->createFeature(VK_NULL_HANDLE, feature));
        feature.view = *view_b;
        ASSERT_TRUE(first->createFeature(VK_NULL_HANDLE, feature));
        feature.view = *view_other;
        ASSERT_TRUE(second->createFeature(VK_NULL_HANDLE, feature));
        LfsSceneUpscalerEvaluateV1 evaluation{.struct_size = sizeof(evaluation), .view = *view_a};
        EXPECT_TRUE(first->evaluate(evaluation));
        first->releaseFeature(*view_a);
        evaluation.view = *view_b;
        EXPECT_TRUE(first->evaluate(evaluation));
        evaluation.view = *view_other;
        EXPECT_TRUE(second->evaluate(evaluation));
        evaluation.view = *view_a;
        EXPECT_FALSE(first->evaluate(evaluation));
        evaluation.view = *view_other;
        EXPECT_TRUE(second->evaluate(evaluation));
        auto fallback = resolveSceneUpscalerSelection(*alpha, false);
        EXPECT_EQ(fallback.requested, *alpha);
        EXPECT_EQ(fallback.effective, SceneUpscalerBackend::Native);
        first->shutdown();
        EXPECT_EQ(first->displayName(), "Provider Alpha");
        configureSceneUpscalerPluginLoading(false);
        EXPECT_EQ(sceneUpscalerDescriptors().size(), 3u);
    }

    TEST(GenericProviderMetadata, RejectsMalformedIdentityPathsAndPresetScales) {
        const auto file = std::filesystem::current_path() / "lfs-provider-metadata-test.txt";
        auto write = [&](const std::string& value) {std::ofstream out(file,std::ios::binary);out<<value; };
        for (const auto& value : {"native", "spatial", "temporal", "../outside", "two\nlines", "", "UPPER"}) {
            write(value);
            EXPECT_FALSE(readSceneUpscalerProviderId(file));
        }
        write(std::string(129, 'a'));
        EXPECT_FALSE(readSceneUpscalerProviderId(file));
        write("valid-provider\r\n");
        EXPECT_EQ(readSceneUpscalerProviderId(file), "valid-provider");
        write("../outside");
        EXPECT_FALSE(readSceneUpscalerModuleStem(file));
        write("Some_Module");
        EXPECT_EQ(readSceneUpscalerModuleStem(file), "Some_Module");
        for (const auto& value : {"0.5 0.6", "0.5 0.6 0", "0.5 0.6 2", "0.5 nan 0.5", "0.5 0.6 0.5 junk"}) {
            write(value);
            EXPECT_FALSE(readSceneUpscalerPresetScales(file));
        }
        write("0.6666666865348816 0.5882352590560913 0.5");
        ASSERT_TRUE(readSceneUpscalerPresetScales(file));
        EXPECT_FLOAT_EQ((*readSceneUpscalerPresetScales(file))[1], 1.0f / 1.7f);
        std::filesystem::remove(file);
    }
} // namespace lfs::vis
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
