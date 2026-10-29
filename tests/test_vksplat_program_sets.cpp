/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#define LFS_USER_SHADER_PROGRAM_FIXTURE
#include "user_shader_images.hpp"
#include "user_shader_test_support.hpp"

namespace lfs::test::program_set {
    TEST_F(ProgramSetTest, LegacyPipelinesBuildThroughTheInMemoryPathUnchanged) {
        const auto files = initializeShipped();
        ASSERT_FALSE(HasFailure());
        ASSERT_NO_THROW(renderer_->setBandedExport(true));
        ASSERT_NO_THROW(renderer_->setRevealEffect(true));

        const auto& live = renderer_->live();
        EXPECT_TRUE(renderer_->pending().empty());
        EXPECT_EQ(Seam::live(), (std::array<int, 4>{static_cast<int>(live.size()), static_cast<int>(live.size()),
                                                    static_cast<int>(live.size()), static_cast<int>(live.size())}));
        EXPECT_EQ(Seam::module_code.size(), live.size());
        EXPECT_EQ(Seam::push_constant_sizes, (std::set<uint32_t>{sizeof(VulkanGSRendererUniforms)}));
        std::set<std::string> built;
        for (const auto* program : live) {
            ASSERT_NE(program->pipeline, VK_NULL_HANDLE) << program->diagnostic_name;
            EXPECT_EQ(Seam::set_layout_bindings.at(program->descriptor_set_layout), program->buffer_layouts)
                << program->diagnostic_name;
            EXPECT_TRUE(Seam::module_code.contains(readWords(program->spirv_path))) << program->spirv_path;
            built.insert(fs::path(program->spirv_path).lexically_normal().string());
        }
        const auto was_built = [&](const fs::path& file) { return built.contains(file.lexically_normal().string()); };
        for (const auto& file : files) {
            // A HiGS raster builds one staging variant, the one the device's shared memory fits; its banded twin follows.
            auto stem = file.stem().string();
            const std::string banded = stem.ends_with("_banded") ? "_banded" : "";
            stem.resize(stem.size() - banded.size());
            const auto other = (stem.ends_with("_lean") ? stem.substr(0, stem.size() - 5) : stem + "_lean") + banded;
            const auto staging = file.parent_path() / (other + ".spv");
            EXPECT_TRUE(was_built(file) || (std::ranges::contains(files, staging) && was_built(staging)))
                << "never built: " << file;
        }
        // Debug labels keep their historical names.
        EXPECT_TRUE(std::ranges::any_of(
            live, [](const auto* program) { return program->diagnostic_name == "projection_forward/standard"; }));
    }

    TEST_F(ProgramSetTest, InMemoryPathRejectsMalformedAndMismatchedSpirv) {
        const auto source = legacy();
        const auto before = Seam::live();
        const auto expect_rejected = [&](TestRenderer::ComputeProgramSpec spec, const std::vector<uint32_t>& spirv,
                                         const lfs::ErrorCode code) {
            auto& program = record(spec.bindings); // stays valid if a broken check registers it
            const auto status = renderer_->createComputePipeline(program, spec, spirv);
            ASSERT_FALSE(status);
            EXPECT_EQ(status.error().code(), code) << status.error().detail();
            EXPECT_NE(status.error().detail().find(spec.label), std::string_view::npos);
            EXPECT_EQ(program.shader, VK_NULL_HANDLE);
            EXPECT_TRUE(renderer_->pending().empty());
            EXPECT_EQ(Seam::live(), before);
        };
        const auto& words = source.spirv;
        expect_rejected(source.spec, {words.begin(), words.begin() + 3}, lfs::ErrorCode::InvalidArgument);
        // The two-word OpCapability that follows the header runs past the end.
        expect_rejected(source.spec, {words.begin(), words.begin() + 6}, lfs::ErrorCode::InvalidArgument);
        auto bad_magic = words;
        bad_magic[0] = 0x03022307u;
        expect_rejected(source.spec, bad_magic, lfs::ErrorCode::InvalidArgument);
        auto spec = source.spec;
        spec.workgroup = {64, 1, 1};
        expect_rejected(spec, words, lfs::ErrorCode::ContractViolation);
        spec = source.spec;
        spec.bindings = {0, 1};
        expect_rejected(spec, words, lfs::ErrorCode::ContractViolation);

        const auto device = renderer_->limits();
        renderer_->limits().maxWorkGroupInvocations = 64;
        expect_rejected(source.spec, words, lfs::ErrorCode::Unsupported);
        renderer_->limits() = device;
        renderer_->limits().subgroupSize = renderer_->limits().minSubgroupSize = 64;
        renderer_->limits().maxSubgroupSize = 64;
        expect_rejected(source.spec, words, lfs::ErrorCode::Unsupported);
        renderer_->limits() = device;

        auto& program = record(source.spec.bindings);
        ASSERT_TRUE(renderer_->createComputePipeline(program, source.spec, words));
        EXPECT_EQ(renderer_->pending().size(), 1u);
        EXPECT_TRUE(renderer_->live().empty());
        renderer_->createPendingComputePipelines();
        EXPECT_EQ(renderer_->live().size(), 1u);
        EXPECT_NE(program.pipeline, VK_NULL_HANDLE);
        EXPECT_EQ(program.diagnostic_name, "test/visible_flags");
    }

    TEST_F(ProgramSetTest, InjectedCreationFailuresLeaveTheLiveSetUnchanged) {
        const auto source = legacy();
        const auto& spirv = source.spirv;
        auto& live = record(source.spec.bindings);
        ASSERT_TRUE(renderer_->createComputePipeline(live, source.spec, spirv));
        renderer_->createPendingComputePipelines();
        const auto before = Seam::live();
        const auto expect_unchanged = [&] {
            EXPECT_EQ(Seam::live(), before);
            EXPECT_EQ(renderer_->live(), std::vector<TestRenderer::_ComputePipeline*>{&live});
            EXPECT_TRUE(renderer_->pending().empty());
            EXPECT_EQ(renderer_->currentProgramSet(), nullptr);
        };

        for (const auto& [name, step] : std::initializer_list<std::pair<const char*, Fail>>{
                 {"module", Fail::Module},
                 {"descriptor layout", Fail::SetLayout},
                 {"pipeline layout", Fail::Layout}}) {
            SCOPED_TRACE(name);
            Seam::fail = step;
            TestRenderer::_ComputePipeline program(source.spec.bindings);
            const auto status = renderer_->createComputePipeline(program, source.spec, spirv);
            ASSERT_FALSE(status);
            EXPECT_EQ(status.error().code(), lfs::ErrorCode::ResourceExhausted);
            EXPECT_NE(status.error().detail().find("test/visible_flags"), std::string_view::npos);
            expect_unchanged();

            ASSERT_TRUE(renderer_->requestProgramSet({source, source}));
            renderer_->waitForProgramSetBuilds();
            const auto published = renderer_->publishProgramSet();
            ASSERT_FALSE(published);
            EXPECT_EQ(published.error().code(), lfs::ErrorCode::ResourceExhausted);
            expect_unchanged();
        }

        Seam::fail = Fail::Pipelines;
        ASSERT_TRUE(renderer_->createComputePipeline(record(source.spec.bindings), source.spec, spirv));
        ASSERT_TRUE(renderer_->createComputePipeline(record(source.spec.bindings), source.spec, spirv));
        EXPECT_THROW(renderer_->createPendingComputePipelines(), lfs::Exception);
        expect_unchanged();
        ASSERT_TRUE(renderer_->requestProgramSet({source, source}));
        renderer_->waitForProgramSetBuilds();
        EXPECT_FALSE(renderer_->publishProgramSet());
        expect_unchanged();

        Seam::fail = Fail::ThrowPipelines;
        ASSERT_TRUE(renderer_->requestProgramSet({source}));
        renderer_->waitForProgramSetBuilds();
        const auto thrown = renderer_->publishProgramSet();
        ASSERT_FALSE(thrown);
        EXPECT_NE(lfs::format_for_developer(thrown.error()).find("pipeline fault"), std::string::npos);
        expect_unchanged();
        Seam::fail = Fail::None;
        ASSERT_TRUE(renderer_->requestProgramSet({source}));
        EXPECT_EQ(publish(), TestRenderer::ProgramSetPublication::Published);
        renderer_->releaseProgramSet();
        expect_unchanged();
    }

    TEST_F(ProgramSetTest, SetBuiltForAnEarlierDeviceGenerationIsRejected) {
        const auto baseline = Seam::live();
        ASSERT_TRUE(renderer_->requestProgramSet({legacy()}));
        renderer_->waitForProgramSetBuilds();
        renderer_->invalidateProgramSets();
        EXPECT_EQ(publish(), TestRenderer::ProgramSetPublication::Stale);
        EXPECT_EQ(renderer_->currentProgramSet(), nullptr);
        EXPECT_EQ(Seam::live(), baseline);

        ASSERT_TRUE(renderer_->requestProgramSet({legacy()}));
        EXPECT_EQ(publish(), TestRenderer::ProgramSetPublication::Published);
        ASSERT_NE(renderer_->currentProgramSet(), nullptr);
        EXPECT_NE(renderer_->currentProgramSet()->programs[0].pipeline, VK_NULL_HANDLE);
        EXPECT_EQ(publish(), TestRenderer::ProgramSetPublication::None);
    }

    TEST_F(ProgramSetTest, SetCreationRacingShutdownNeitherCrashesNorPublishes) {
        // Deterministic: shutdown starts while the worker is inside pipeline creation.
        Seam::hold = true;
        Seam::held = false;
        ASSERT_TRUE(renderer_->requestProgramSet({legacy(), legacy()}));
        while (!Seam::held)
            std::this_thread::sleep_for(1ms);
        std::jthread release([] {
            std::this_thread::sleep_for(50ms);
            Seam::hold = false;
        });
        renderer_->cleanup();
        EXPECT_EQ(Seam::live(), (std::array<int, 4>{}));
        EXPECT_EQ(publish(), TestRenderer::ProgramSetPublication::None);
        release.join();

        // Shutdown at varying points of a build.
        for (int i = 0; i < 16; ++i) {
            initialize();
            ASSERT_TRUE(renderer_->requestProgramSet({legacy(), legacy()}));
            std::this_thread::sleep_for(std::chrono::microseconds(i * 400));
            renderer_->cleanup();
            EXPECT_EQ(Seam::live(), (std::array<int, 4>{})) << "iteration " << i;
            EXPECT_EQ(renderer_->currentProgramSet(), nullptr);
            EXPECT_EQ(publish(), TestRenderer::ProgramSetPublication::None);
        }
        initialize();
    }

    TEST_F(ProgramSetTest, ReplacedSetIsFreedOnlyAfterItsSubmissionCompleted) {
        buffer(64 * 16);
        for (const bool release : {false, true}) {
            SCOPED_TRACE(release ? "release" : "replace");
            renderer_->releaseProgramSet();
            publish();
            const auto none = Seam::live();
            ASSERT_TRUE(renderer_->requestProgramSet({tiny()}));
            ASSERT_EQ(publish(), TestRenderer::ProgramSetPublication::Published);
            const auto one_set = Seam::live();
            const VulkanGSPipelineBuffers buffers;
            if (release)
                ASSERT_TRUE(renderer_->setCustomEffect(true, buffers, false, false).active);
            const uint64_t submission = release ? 2 : 1;
            submitBehindGate(submission, submission);
            if (release) {
                renderer_->releaseProgramSet();
                EXPECT_EQ(renderer_->currentProgramSet(), nullptr);
                EXPECT_FALSE(renderer_->setCustomEffect(true, buffers, false, false).active);
                EXPECT_EQ(Seam::live(), one_set);
            } else {
                // Publication refuses to run inside a command batch.
                ASSERT_TRUE(renderer_->requestProgramSet({tiny()}));
                renderer_->waitForProgramSetBuilds();
                renderer_->beginCommandBatch();
                const auto inside = renderer_->publishProgramSet();
                renderer_->cancelCommandBatch();
                ASSERT_FALSE(inside);
                EXPECT_EQ(inside.error().code(), lfs::ErrorCode::ContractViolation);
                ASSERT_EQ(publish(), TestRenderer::ProgramSetPublication::Published);
                EXPECT_EQ(renderer_->retiredProgramSetCount(), 1u);
                EXPECT_EQ(Seam::live()[3], 2 * one_set[3]);
                EXPECT_EQ(publish(), TestRenderer::ProgramSetPublication::None);
            }
            EXPECT_EQ(renderer_->retiredProgramSetCount(), 1u);
            signal(gate_, submission);
            awaitRender(submission);
            EXPECT_EQ(publish(), TestRenderer::ProgramSetPublication::None);
            EXPECT_EQ(renderer_->retiredProgramSetCount(), 0u);
            EXPECT_EQ(Seam::live(), release ? none : one_set);
        }
    }

    TEST_F(ProgramSetTest, RepeatedEditsKeepRetiredSetsBounded) {
        buffer(64 * 16);
        ASSERT_TRUE(renderer_->requestProgramSet({tiny()}));
        ASSERT_EQ(publish(), TestRenderer::ProgramSetPublication::Published);
        const int per_set = Seam::live()[3];
        submitBehindGate(1, 1);

        // Every replaced set waits on the gated submission; one past the bound
        // makes publication wait for it, which only the gate releases.
        std::atomic<bool> overflowing{false};
        std::jthread release([&](const std::stop_token stop) {
            while (!overflowing && !stop.stop_requested())
                std::this_thread::sleep_for(1ms);
            std::this_thread::sleep_for(100ms);
            signal(gate_, 1);
        });
        for (size_t edit = 1; edit <= 4; ++edit) {
            ASSERT_TRUE(renderer_->requestProgramSet({tiny()}));
            ASSERT_EQ(publish(), TestRenderer::ProgramSetPublication::Published);
            EXPECT_EQ(renderer_->retiredProgramSetCount(), edit);
            EXPECT_EQ(Seam::live()[3], static_cast<int>(edit + 1) * per_set);
        }
        overflowing = true;
        ASSERT_TRUE(renderer_->requestProgramSet({tiny()}));
        ASSERT_EQ(publish(), TestRenderer::ProgramSetPublication::Published);
        uint64_t gate = 0;
        vkGetSemaphoreCounterValue(gpu_.device, gate_, &gate);
        EXPECT_EQ(gate, 1u) << "publication past the bound must wait for the submission";
        EXPECT_EQ(renderer_->retiredProgramSetCount(), 0u);
        EXPECT_EQ(Seam::live()[3], per_set);
    }

    // A set holds any subset of the shipped projection hosts. A draw is custom
    // only while the set and the adopted offline variants hold every program
    // it runs; incomplete chains keep every built-in program. The first set with a
    // survivor or 3DGUT host also builds the cull or 3DGUT variants its draws
    // run; publication keeps them for the renderer's lifetime.
    TEST_F(ProgramSetTest, PlannedHostsMatchDispatchedChains) {
        initializeShipped();
        ASSERT_FALSE(HasFailure());
        const auto package = loadPackage(LFS_SHADER_PACKAGE_DIR);
        ASSERT_TRUE(package);
        lfs::vis::shaders::SlangCompiler compiler;
        // Publishes a set of `hosts` (all when empty); returns how many pipelines it built.
        const auto set = [&](const std::vector<std::string>& hosts) {
            const int before = Seam::live()[3];
            const int owner_before = Seam::owner_pipelines;
            EXPECT_TRUE(renderer_->requestProgramSet(lfs::test::compileHosts<TestRenderer::ComputeProgramSource>(
                compiler, package, kTinyEffect, hosts)));
            renderer_->waitForProgramSetBuilds();
            const int built = Seam::live()[3] - before;
            EXPECT_EQ(publish(), TestRenderer::ProgramSetPublication::Published);
            EXPECT_EQ(Seam::owner_pipelines, owner_before) << "program set pipelines created on the frame thread";
            return built;
        };
        const auto host = [&](const std::string_view stem) {
            return renderer_->customVariant(renderer_->base(stem)).diagnostic_name;
        };
        // A draw the set cannot run keeps every built-in program.
        const auto missing = [&](const VulkanGSPipelineBuffers& buffers, const bool gut, const bool survivors) {
            const auto draw = renderer_->setCustomEffect(true, buffers, gut, survivors);
            EXPECT_FALSE(draw.active);
            auto& projection = renderer_->base("projection_forward");
            EXPECT_EQ(&renderer_->customVariant(projection), &projection);
            return std::string(renderer_->hostsFor({buffers.quant_pool, buffers.shN_q16, buffers.shN_f16, gut, survivors, renderer_->banded()}).host);
        };
        VulkanGSPipelineBuffers fp32, q16;
        q16.shN_q16 = true;

        EXPECT_EQ(set({"projection_forward_custom"}), 1);
        EXPECT_TRUE(renderer_->setCustomEffect(true, fp32, false, false).active);
        EXPECT_EQ(host("projection_forward"), "projection_forward_custom");
        EXPECT_EQ(missing(q16, false, false), "projection_forward_shn_q16_custom");
        EXPECT_EQ(missing(fp32, true, false), "projection_forward_3dgut_custom");
        EXPECT_EQ(missing(fp32, false, true), "projection_forward_survivors_custom");

        // The cull and the eight 3DGUT rasters and outlines come with the first set that needs them.
        EXPECT_EQ(set({"projection_forward_3dgut_custom", "projection_forward_survivors_custom"}), 2 + 1 + 8);
        ASSERT_TRUE(renderer_->setCustomEffect(true, fp32, true, false).active);
        EXPECT_EQ(host("projection_forward_3dgut"), "projection_forward_3dgut_custom");
        for (const std::string program : kGutPrograms)
            EXPECT_EQ(host(program), program + "_custom");
        ASSERT_TRUE(renderer_->setCustomEffect(true, fp32, false, true).active);
        EXPECT_EQ(host("projection_forward_survivors"), "projection_forward_survivors_custom");
        EXPECT_EQ(host("cull_splats"), "cull_splats_custom");
        EXPECT_EQ(missing(fp32, false, false), "projection_forward_custom");
        EXPECT_EQ(set({"projection_forward_3dgut_custom", "projection_forward_survivors_custom"}), 2);
        ASSERT_TRUE(renderer_->setCustomEffect(true, fp32, true, false).active);
        EXPECT_EQ(host("rasterize_outline_gut_plain_rad"), "rasterize_outline_gut_plain_rad_custom");

        // Banded export runs banded projection and forward rasters; outlines have no banded variants.
        renderer_->setBandedExport(true);
        EXPECT_EQ(missing(fp32, true, false), "projection_forward_3dgut_custom_banded");
        EXPECT_EQ(set({"projection_forward_3dgut_custom_banded"}), 1 + 4);
        ASSERT_TRUE(renderer_->setCustomEffect(true, fp32, true, false).active);
        EXPECT_EQ(host("projection_forward_3dgut"), "projection_forward_3dgut_custom_banded");
        EXPECT_EQ(host("rasterize_forward_3dgut_rad"), "rasterize_forward_3dgut_rad_custom_banded");
        EXPECT_EQ(host("rasterize_outline_gut_rad"), "rasterize_outline_gut_rad_custom");
        renderer_->setBandedExport(false);

        // Only survivor and 3DGUT projections have banded hosts; a banded draw
        // through the 2D projection keeps the built-in programs.
        EXPECT_EQ(set({}), 20);
        for (const bool banded : {false, true}) {
            renderer_->setBandedExport(banded);
            for (const std::string encoding : {"", "_quant", "_shn_f16", "_shn_q16"}) {
                VulkanGSPipelineBuffers buffers;
                buffers.quant_pool = encoding == "_quant";
                buffers.shN_f16 = encoding == "_shn_f16";
                buffers.shN_q16 = encoding == "_shn_q16";
                for (const auto [gut, survivors] : {std::pair{false, false}, {true, false}, {false, true}}) {
                    auto stem = "projection_forward" + encoding;
                    if (gut)
                        stem += "_3dgut";
                    if (survivors)
                        stem += "_survivors";
                    SCOPED_TRACE(std::format("{} banded={}", stem, banded));
                    if (banded && !gut && !survivors) {
                        EXPECT_EQ(missing(buffers, gut, survivors), stem + "_custom_banded");
                        continue;
                    }
                    const auto plan = renderer_->hostsFor({buffers.quant_pool, buffers.shN_q16, buffers.shN_f16, gut, survivors, banded});
                    EXPECT_EQ(plan.programs[0], &renderer_->base(stem));
                    EXPECT_EQ(plan.count, gut ? 9u : survivors ? 2u
                                                               : 1u);
                    ASSERT_TRUE(renderer_->setCustomEffect(true, plan).active);
                    EXPECT_EQ(renderer_->customVariant(*plan.programs[0]).diagnostic_name, plan.host);
                    EXPECT_EQ(host(stem), stem + (banded ? "_custom_banded" : "_custom"));
                }
            }
        }
        renderer_->setBandedExport(false);
        EXPECT_FALSE(renderer_->setCustomEffect(false, fp32, false, false).active);
        auto& projection = renderer_->base("projection_forward");
        EXPECT_EQ(&renderer_->customVariant(projection), &projection);
        renderer_->releaseProgramSet();
        EXPECT_FALSE(renderer_->setCustomEffect(true, fp32, false, false).active);
    }

    TEST_F(ProgramSetTest, ExportLeaseIsIndependentAndComplete) {
        auto first = renderer_->prepareProgramSet({tiny()});
        auto second = renderer_->prepareProgramSet({tiny()});
        ASSERT_TRUE(first && second);
        EXPECT_NE((*first)->id, (*second)->id);
        renderer_->waitForProgramSetBuilds();
        ASSERT_TRUE(renderer_->programSetReady(*first).value_or(false));
        ASSERT_TRUE(renderer_->programSetReady(*second).value_or(false));
        auto* a = renderer_->programSet(*first);
        auto* b = renderer_->programSet(*second);
        ASSERT_NE(a, b);
        renderer_->useProgramSet(*first);
        buffer(64 * 16);
        // The submitted dispatch owns first independently of a later selection.
        renderer_->selected = *first;
        submitBehindGate(1, 1);
        renderer_->selected.reset();
        renderer_->useProgramSet(*second);
        const auto live = Seam::live();
        std::thread release([handle = std::move(*first)]() mutable { handle.reset(); });
        release.join();
        EXPECT_EQ(Seam::live(), live); // foreign release never calls Vulkan
        renderer_->drainRetiredProgramSets();
        EXPECT_EQ(renderer_->retiredProgramSetCount(), 1u);
        EXPECT_EQ(renderer_->programSet(*second), b);
        signal(gate_, 1);
        awaitRender(1);
        renderer_->drainRetiredProgramSets();
        EXPECT_EQ(renderer_->retiredProgramSetCount(), 0u);
        renderer_->invalidateProgramSets();
        EXPECT_FALSE(renderer_->programSetReady(*second).value_or(true));
        EXPECT_EQ(renderer_->programSet(*second), nullptr);
    }

    // Stage A output, a program set and a dispatch: the effect's values reach memory.
    TEST_F(ProgramSetTest, TinyCompiledEffectRunsEndToEnd) {
        constexpr size_t count = 64;
        auto& output = buffer(count * 4 * sizeof(float));
        ASSERT_TRUE(renderer_->requestProgramSet({tiny()}));
        ASSERT_EQ(publish(), TestRenderer::ProgramSetPublication::Published);
        renderer_->dispatch(renderer_->currentProgramSet()->programs[0], output, count);

        const auto values = lfs::test::readBuffer(*renderer_, output, gpu_.allocator, count * 4);
        ASSERT_EQ(values.size(), count * 4);
        for (size_t i = 0; i < count; ++i) {
            EXPECT_EQ(values[4 * i + 0], static_cast<float>(i) + 0.25f) << i;
            EXPECT_EQ(values[4 * i + 1], 0.5f) << i;
            EXPECT_EQ(values[4 * i + 2], 0.75f) << i;
            EXPECT_EQ(values[4 * i + 3], 0.5f) << i;
        }
    }

} // namespace lfs::test::program_set
