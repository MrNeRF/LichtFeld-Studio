/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_backend.hpp"
#include "visualizer/rendering/metal_scene_upscaler.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace lfs::vis {
    namespace {
        using namespace core;
        class MetalFxContracts : public testing::TestWithParam<SceneUpscalerBackend> {
            void SetUp() override {
                if (!metalFxBackendAvailable(GetParam())) GTEST_SKIP() << "MetalFX capability unavailable";
            }
        };
        TensorSceneTemporalRequest request(TemporalViewId view = TemporalViewId::Main,
                                            glm::ivec2 input = {64,48}, glm::ivec2 output = {128,96},
                                            float value = .25f) {
            const GpuBackendScope scope(GpuBackend::Metal);
            auto color = std::make_shared<Tensor>(Tensor::full({size_t(input.y),size_t(input.x),4},value,Device::GPU));
            auto depth = std::make_shared<Tensor>(Tensor::full({size_t(input.y),size_t(input.x)},4.f,Device::GPU));
            return {.view = view,.color = color,.depth = depth,
                .frame = {.view = {.size = input},.output_extent = output,.render_scale = .5f,
                          .scene_generation = 1,.backend_key = 1},
                .render_extent = input,.output_extent = output};
        }
        void checkImage(const Tensor& image, glm::ivec2 extent, float expected) {
            ASSERT_EQ(image.size(0),size_t(extent.y)); ASSERT_EQ(image.size(1),size_t(extent.x));
            ASSERT_EQ(image.size(2),4u); ASSERT_EQ(gpu_backend_of(image),GpuBackend::Metal);
            // Download through the normal tensor timeline, without a native
            // command wait: this tests native-write -> tensor-read ordering.
            auto host = image.cpu();
            const auto values = host.ptr<float>();
            for (int y=8;y<extent.y-8;++y) for (int x=8;x<extent.x-8;++x)
                for (int c=0;c<4;++c) {
                    ASSERT_TRUE(std::isfinite(values[(y*extent.x+x)*4+c]));
                    ASSERT_NEAR(values[(y*extent.x+x)*4+c],expected,.035f);
                }
        }
        TEST_P(MetalFxContracts, ConstantColorAndDependentTensorWritesAreOrdered) {
            MetalSceneUpscaler pipeline;
            auto r = request();
            auto result = pipeline.resolve(GetParam(),r);
            ASSERT_TRUE(result) << (result ? "" : lfs::format_for_developer(result.error()));
            // Mutate/release the producer before waiting: the native reader must
            // retain its snapshot and serialize this write after its own reads.
            r.color->add_(.5f); r.color.reset(); r.depth.reset();
            checkImage(*result->color,{128,96},.25f);
        }
        TEST_P(MetalFxContracts, IndependentPanelsAndOwnersCannotAliasHistory) {
            MetalSceneUpscaler first, second;
            auto left = request(TemporalViewId::SplitLeft,{64,48},{128,96},.2f);
            auto right = request(TemporalViewId::SplitRight,{65,48},{129,96},.7f);
            for (int frame=0;frame<3;++frame) {
                auto a = first.resolve(GetParam(),left); auto b = first.resolve(GetParam(),right);
                auto c = second.resolve(GetParam(),left);
                ASSERT_TRUE(a); ASSERT_TRUE(b); ASSERT_TRUE(c);
                checkImage(*a->color,left.output_extent,.2f); checkImage(*b->color,right.output_extent,.7f);
                if (GetParam() == SceneUpscalerBackend::MetalFxTemporal) {
                    EXPECT_EQ(a->sequence,uint64_t(frame+1)); EXPECT_EQ(b->sequence,uint64_t(frame+1));
                    EXPECT_EQ(c->sequence,uint64_t(frame+1));
                    if (frame) EXPECT_EQ(a->reset_reasons,TemporalResetReason::None);
                }
            }
        }
        TEST_P(MetalFxContracts, ResizeResetAndOwnerDestructionRetainInflightOutput) {
            std::shared_ptr<Tensor> retained;
            {
                MetalSceneUpscaler pipeline;
                auto result = pipeline.resolve(GetParam(),request()); ASSERT_TRUE(result);
                retained = result->color;
                pipeline.resetAll();
                auto resized = pipeline.resolve(GetParam(),request(TemporalViewId::Main,{80,60},{160,120},.6f));
                ASSERT_TRUE(resized); checkImage(*resized->color,{160,120},.6f);
            }
            checkImage(*retained,{128,96},.25f);
        }
        TEST_P(MetalFxContracts, OutputPoolReusesReleasedBuffersWithoutOverwritingRetainedFrames) {
            MetalSceneUpscaler pipeline;
            auto r = request();
            auto first = pipeline.resolve(GetParam(),r); ASSERT_TRUE(first);
            auto retained = first->color;
            const auto* first_address = retained.get();
            first->color.reset();
            r.color->fill_(.7f);
            auto second = pipeline.resolve(GetParam(),r); ASSERT_TRUE(second);
            const auto* second_address = second->color.get();
            EXPECT_NE(first_address,second_address);
            auto third = pipeline.resolve(GetParam(),r); ASSERT_TRUE(third);
            EXPECT_NE(third->color.get(),first_address);
            EXPECT_NE(third->color.get(),second_address);
            second->color.reset(); third->color.reset();
            auto recycled = pipeline.resolve(GetParam(),r); ASSERT_TRUE(recycled);
            EXPECT_EQ(recycled->color.get(),second_address);
            checkImage(*retained,r.output_extent,.25f);
        }
        TEST_P(MetalFxContracts, PaddedUInt8RowsAndFlippedCoverageSurviveTextureRoundTrip) {
            const GpuBackendScope scope(GpuBackend::Metal);
            MetalSceneUpscaler pipeline;
            auto r = request(TemporalViewId::Main,{33,25},{66,50});
            std::vector<uint8_t> pixels(33*25*4,64);
            for (int y=0;y<25;++y) for (int x=0;x<33;++x) pixels[(y*33+x)*4+3]=uint8_t(30+y*5);
            auto cpu = Tensor::from_blob(pixels.data(),{25,33,4},Device::CPU,DataType::UInt8);
            r.color = std::make_shared<Tensor>(cpu.to(Device::GPU));
            r.flip_y = true;
            auto result = pipeline.resolve(GetParam(),r); ASSERT_TRUE(result);
            auto host = result->color->cpu();
            const auto* values = host.ptr<float>();
            for (int y=8;y<42;++y) {
                const float source_y = 24.f-((float(y)+.5f)*.5f-.5f);
                EXPECT_NEAR(values[(y*66+20)*4+3],(30.f+source_y*5.f)/255.f,1e-5f);
                EXPECT_NEAR(values[(y*66+20)*4],64.f/255.f,.04f);
            }
        }
        TEST(TensorTemporalContracts, RecycledDepthInputKeepsIndependentHistory) {
            if (!gpu_backend_available(GpuBackend::Metal)) GTEST_SKIP();
            TensorSceneTemporalPipeline recycled(GpuBackend::Metal), reference(GpuBackend::Metal);
            auto reused = request();
            for (int frame=0;frame<6;++frame) {
                const float depth = frame%2 ? 12.f:4.f;
                const float color = frame%2 ? .8f:.2f;
                reused.depth->fill_(depth); reused.color->fill_(color);
                auto stable = request(); stable.depth->fill_(depth); stable.color->fill_(color);
                auto a=recycled.resolve(reused), b=reference.resolve(stable);
                ASSERT_TRUE(a); ASSERT_TRUE(b);
                auto ah=a->color->cpu(), bh=b->color->cpu();
                const auto* av=ah.ptr<float>(); const auto* bv=bh.ptr<float>();
                for (size_t i=0;i<ah.numel();++i) ASSERT_NEAR(av[i],bv[i],1e-6f);
            }
        }
        TEST_P(MetalFxContracts, RejectsInvalidViewAndStorageBeforeSubmission) {
            MetalSceneUpscaler pipeline;
            auto r = request(); r.view = TemporalViewId::Count;
            EXPECT_FALSE(pipeline.resolve(GetParam(),r));
            r = request(); r.render_extent = {1000,1000}; EXPECT_FALSE(pipeline.resolve(GetParam(),r));
            r = request(); r.color = std::make_shared<Tensor>(r.color->cpu());
            EXPECT_FALSE(pipeline.resolve(GetParam(),r));
            r = request(); EXPECT_FALSE(pipeline.resolve(SceneUpscalerBackend::Temporal,r));
            if (GetParam() == SceneUpscalerBackend::MetalFxTemporal) {
                r = request(); r.depth.reset(); EXPECT_FALSE(pipeline.resolve(GetParam(),r));
            }
        }
        TEST(MetalFxTemporalContracts, CroppedCalibratedPanelsAndNonzeroJitter) {
            if (!metalFxBackendAvailable(SceneUpscalerBackend::MetalFxTemporal)) GTEST_SKIP();
            MetalSceneUpscaler pipeline; auto r = request(TemporalViewId::SplitRight);
            r.frame.view.subregion_full_size = {128,48}; r.frame.view.subregion_origin = {64,0};
            r.frame.view.intrinsics_override = rendering::CameraIntrinsics{90,85,65,23};
            for (int frame=0;frame<4;++frame) {
                r.frame.jitter = temporalJitterNdc(temporalJitterPixels(frame),r.render_extent);
                auto result = pipeline.resolve(SceneUpscalerBackend::MetalFxTemporal,r);
                ASSERT_TRUE(result) << (result ? "" : lfs::format_for_developer(result.error()));
                checkImage(*result->color,r.output_extent,.25f);
                if (frame) EXPECT_EQ(result->reset_reasons,TemporalResetReason::None);
            }
        }
        TEST(MetalFxTemporalContracts, ResetReasonsFollowSharedFrameContract) {
            if (!metalFxBackendAvailable(SceneUpscalerBackend::MetalFxTemporal)) GTEST_SKIP();
            MetalSceneUpscaler pipeline; auto r = request();
            auto resolve = [&] { return pipeline.resolve(SceneUpscalerBackend::MetalFxTemporal,r); };
            ASSERT_TRUE(resolve()); auto settled = resolve(); ASSERT_TRUE(settled);
            EXPECT_EQ(settled->reset_reasons,TemporalResetReason::None);
            r.frame.camera_cut = true; auto cut = resolve(); ASSERT_TRUE(cut);
            EXPECT_TRUE(hasTemporalResetReason(cut->reset_reasons,TemporalResetReason::CameraCut));
            r.frame.camera_cut = false; r.frame.scene_generation++; auto scene = resolve(); ASSERT_TRUE(scene);
            EXPECT_TRUE(hasTemporalResetReason(scene->reset_reasons,TemporalResetReason::Scene));
            r.frame.backend_key++; auto quality = resolve(); ASSERT_TRUE(quality);
            EXPECT_TRUE(hasTemporalResetReason(quality->reset_reasons,TemporalResetReason::Backend));
            r.frame.view.orthographic = true; auto projection = resolve(); ASSERT_TRUE(projection);
            EXPECT_TRUE(hasTemporalResetReason(projection->reset_reasons,TemporalResetReason::Projection));
        }
        TEST(MetalBaseReconstructionContracts, SpatialAndTemporalUseIndependentPanelHistory) {
            if (!metalFxBackendAvailable(SceneUpscalerBackend::MetalFxTemporal)) GTEST_SKIP();
            TensorSceneTemporalPipeline pipeline(GpuBackend::Metal);
            auto left = request(TemporalViewId::SplitLeft,{64,48},{128,96},.2f);
            auto right = request(TemporalViewId::SplitRight,{65,48},{129,96},.7f);
            auto spatial = pipeline.spatial(left.color,left.render_extent,left.output_extent);
            ASSERT_TRUE(spatial); checkImage(**spatial,left.output_extent,.2f);
            for (int frame=0;frame<3;++frame) {
                auto a = pipeline.resolve(left), b = pipeline.resolve(right);
                ASSERT_TRUE(a); ASSERT_TRUE(b);
                EXPECT_EQ(a->sequence,uint64_t(frame+1)); EXPECT_EQ(b->sequence,uint64_t(frame+1));
                checkImage(*a->color,left.output_extent,.2f); checkImage(*b->color,right.output_extent,.7f);
            }
            left.view = TemporalViewId::Count; EXPECT_FALSE(pipeline.resolve(left));
            pipeline.reset(TemporalViewId::Count);
        }
        INSTANTIATE_TEST_SUITE_P(Native,MetalFxContracts,testing::Values(
            SceneUpscalerBackend::MetalFxSpatial,SceneUpscalerBackend::MetalFxTemporal));
    }
}
