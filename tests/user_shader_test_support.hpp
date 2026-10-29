/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

// Opt in to PROGRAM, CUSTOM and/or SERVICE fixtures before this include.
// PROGRAM uses VkSplatTestDevice only; CUSTOM/SERVICE own a window context.
#include "custom_effect_hosts.hpp"
#include "user_shader_sources.hpp"
#include <algorithm>
#include <filesystem>
#include <format>
#include <gtest/gtest.h>
#include <vector>

#if defined(LFS_USER_SHADER_PROGRAM_FIXTURE)
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"
#include "shaders/slang_compiler.hpp"
#include "vksplat_test_device.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <stop_token>
#include <thread>

namespace lfs::test::program_set {
    namespace fs = std::filesystem;
    using namespace std::chrono_literals;

    enum class Fail { None,
                      Module,
                      SetLayout,
                      Layout,
                      Pipelines,
                      ThrowPipelines };

    // Counts live construction objects and records what reaches the driver.
    struct Seam {
        static inline std::atomic<int> modules{0}, set_layouts{0}, layouts{0}, pipelines{0};
        static inline std::atomic<Fail> fail{Fail::None};
        static inline std::atomic<bool> hold{false}, held{false};
        // Pipelines created on the thread that runs the test, the frame thread.
        static inline std::thread::id owner;
        static inline std::atomic<int> owner_pipelines{0};
        static inline std::mutex mutex;
        static inline std::multiset<std::vector<uint32_t>> module_code;
        static inline std::map<VkDescriptorSetLayout, std::vector<int>> set_layout_bindings;
        static inline std::set<uint32_t> push_constant_sizes;

        static std::array<int, 4> live() { return {modules, set_layouts, layouts, pipelines}; }

        static VKAPI_ATTR VkResult VKAPI_CALL create_module(VkDevice device, const VkShaderModuleCreateInfo* info,
                                                            const VkAllocationCallbacks* allocator, VkShaderModule* out) {
            if (fail == Fail::Module)
                return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            {
                const std::lock_guard lock(mutex);
                module_code.emplace(info->pCode, info->pCode + info->codeSize / 4);
            }
            const VkResult result = vkCreateShaderModule(device, info, allocator, out);
            modules += result == VK_SUCCESS;
            return result;
        }
        static VKAPI_ATTR void VKAPI_CALL destroy_module(VkDevice device, VkShaderModule module,
                                                         const VkAllocationCallbacks* allocator) {
            --modules;
            vkDestroyShaderModule(device, module, allocator);
        }
        static VKAPI_ATTR VkResult VKAPI_CALL create_set_layout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* info,
                                                                const VkAllocationCallbacks* allocator,
                                                                VkDescriptorSetLayout* out) {
            if (fail == Fail::SetLayout)
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            const VkResult result = vkCreateDescriptorSetLayout(device, info, allocator, out);
            if (result == VK_SUCCESS) {
                ++set_layouts;
                std::vector<int> bindings;
                for (uint32_t i = 0; i < info->bindingCount; ++i)
                    bindings.push_back(static_cast<int>(info->pBindings[i].binding));
                const std::lock_guard lock(mutex);
                set_layout_bindings[*out] = std::move(bindings);
            }
            return result;
        }
        static VKAPI_ATTR void VKAPI_CALL destroy_set_layout(VkDevice device, VkDescriptorSetLayout layout,
                                                             const VkAllocationCallbacks* allocator) {
            --set_layouts;
            vkDestroyDescriptorSetLayout(device, layout, allocator);
        }
        static VKAPI_ATTR VkResult VKAPI_CALL create_layout(VkDevice device, const VkPipelineLayoutCreateInfo* info,
                                                            const VkAllocationCallbacks* allocator, VkPipelineLayout* out) {
            if (fail == Fail::Layout)
                return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            {
                const std::lock_guard lock(mutex);
                for (uint32_t i = 0; i < info->pushConstantRangeCount; ++i)
                    push_constant_sizes.insert(info->pPushConstantRanges[i].size);
            }
            const VkResult result = vkCreatePipelineLayout(device, info, allocator, out);
            layouts += result == VK_SUCCESS;
            return result;
        }
        static VKAPI_ATTR void VKAPI_CALL destroy_layout(VkDevice device, VkPipelineLayout layout,
                                                         const VkAllocationCallbacks* allocator) {
            --layouts;
            vkDestroyPipelineLayout(device, layout, allocator);
        }
        // An injected failure still creates all but the last pipeline, as a
        // driver may on a partial batch failure.
        static VKAPI_ATTR VkResult VKAPI_CALL create_pipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                               const VkComputePipelineCreateInfo* infos,
                                                               const VkAllocationCallbacks* allocator, VkPipeline* out) {
            if (fail == Fail::ThrowPipelines)
                throw std::runtime_error("pipeline fault");
            if (hold) {
                held = true;
                while (hold)
                    std::this_thread::sleep_for(1ms);
            }
            if (std::this_thread::get_id() == owner)
                owner_pipelines += static_cast<int>(count);
            const bool failing = fail == Fail::Pipelines;
            const uint32_t created = failing ? count - 1 : count;
            VkResult result = created == 0 ? VK_SUCCESS : vkCreateComputePipelines(device, cache, created, infos, allocator, out);
            for (uint32_t i = 0; i < created; ++i)
                pipelines += out[i] != VK_NULL_HANDLE;
            if (failing) {
                out[count - 1] = VK_NULL_HANDLE;
                result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            }
            return result;
        }
        static VKAPI_ATTR void VKAPI_CALL destroy_pipeline(VkDevice device, VkPipeline pipeline,
                                                           const VkAllocationCallbacks* allocator) {
            --pipelines;
            vkDestroyPipeline(device, pipeline, allocator);
        }

        static lfs::rendering::VulkanDispatch dispatch() {
            auto d = lfs::rendering::VulkanDispatch::real();
            d.create_shader_module = &create_module;
            d.destroy_shader_module = &destroy_module;
            d.create_descriptor_set_layout = &create_set_layout;
            d.destroy_descriptor_set_layout = &destroy_set_layout;
            d.create_pipeline_layout = &create_layout;
            d.destroy_pipeline_layout = &destroy_layout;
            d.create_compute_pipelines = &create_pipelines;
            d.destroy_pipeline = &destroy_pipeline;
            return d;
        }
    };

    class TestRenderer : public VulkanGSRenderer {
    public:
        using VulkanGSPipeline::_ComputePipeline;
        using VulkanGSPipeline::ComputeProgramSource;
        using VulkanGSPipeline::ComputeProgramSpec;
        using VulkanGSPipeline::createComputePipeline;
        using VulkanGSPipeline::createPendingComputePipelines;
        using VulkanGSPipeline::customVariant;
        using VulkanGSPipeline::invalidateProgramSets;
        using VulkanGSPipeline::retiredProgramSetCount;
        using VulkanGSPipeline::waitForProgramSetBuilds;

        using VulkanGSRenderer::setCustomEffect;
        CustomEffectDraw setCustomEffect(bool enabled, const VulkanGSPipelineBuffers& buffers, bool gut, bool survivors) {
            return setCustomEffect(enabled, hostsFor({buffers.quant_pool, buffers.shN_q16, buffers.shN_f16, gut, survivors, banded_export_active_}));
        }
        using VulkanGSPipeline::programSet;
        // Test driver keeps its own submission handle. Production draws pass a keyed handle explicitly.
        enum class ProgramSetPublication { None,
                                           Published,
                                           Stale };
        ProgramSetHandle requested, selected;
        lfs::Status requestProgramSet(std::vector<ComputeProgramSource> sources) {
            auto result = prepareProgramSet(std::move(sources));
            if (!result)
                return lfs::Status::failure(result.error());
            requested = *result;
            return {};
        }
        lfs::Result<ProgramSetPublication> publishProgramSet() {
            drainRetiredProgramSets();
            if (!requested)
                return ProgramSetPublication::None;
            if (requested->generation != deviceGeneration()) {
                requested.reset();
                drainRetiredProgramSets();
                return ProgramSetPublication::Stale;
            }
            auto ready = programSetReady(requested);
            if (!ready) {
                if (ready.error().code() != lfs::ErrorCode::ContractViolation)
                    requested.reset();
                drainRetiredProgramSets();
                return ready.error();
            }
            if (!*ready)
                return ProgramSetPublication::None;
            selected = std::move(requested);
            useProgramSet(selected);
            drainRetiredProgramSets();
            return ProgramSetPublication::Published;
        }
        auto* currentProgramSet() { return programSet(selected); }
        void releaseProgramSet() {
            useProgramSet({});
            selected.reset();
            drainRetiredProgramSets();
        }
        void cleanup() {
            requested.reset();
            selected.reset();
            VulkanGSRenderer::cleanup();
        }

        bool banded() const { return banded_export_active_; }
        std::vector<_ComputePipeline*>& live() { return all_compute_pipelines; }
        // The registered program built from generated/<stem>.spv.
        _ComputePipeline& base(const std::string_view stem) {
            const auto found = std::ranges::find_if(all_compute_pipelines, [&](const _ComputePipeline* program) {
                return fs::path(program->spirv_path).stem() == stem;
            });
            if (found == all_compute_pipelines.end())
                throw std::runtime_error(std::format("no program {}", stem));
            return **found;
        }
        std::vector<_ComputePipeline*>& pending() { return pending_compute_pipelines; }
        DeviceInfo& limits() { return deviceInfo; }

        void dispatch(_ComputePipeline& program, const _VulkanBuffer& output, const size_t count) {
            executeCompute({{count, 1}}, nullptr, 0, program,
                           std::vector<TaggedBinding>{{output, lfs::rendering::vulkan::BufferUse::ComputeReadWrite}});
        }
    };

    inline std::vector<uint32_t> readWords(const fs::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        std::vector<uint32_t> words(static_cast<size_t>(file.tellg()) / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
        return words;
    }

    const fs::path kSpirvRoot = LFS_PROGRAM_SET_TEST_SPIRV_DIR;

    // The 3DGUT rasters and outlines a 3DGUT host's draws run as offline variants.
    constexpr const char* kGutPrograms[] = {"rasterize_forward_3dgut", "rasterize_forward_3dgut_plain",
                                            "rasterize_forward_3dgut_rad", "rasterize_forward_3dgut_plain_rad",
                                            "rasterize_outline_gut", "rasterize_outline_gut_plain",
                                            "rasterize_outline_gut_rad", "rasterize_outline_gut_plain_rad"};

    class ProgramSetTest : public ::testing::Test {
    protected:
        lfs::test::VkSplatTestDevice gpu_; // first member: outlives everything that uses it
        VkSemaphore gate_ = VK_NULL_HANDLE, render_ = VK_NULL_HANDLE;
        // Registered records outlive TearDown's cleanup, which destroys them.
        std::deque<TestRenderer::_ComputePipeline> records_;
        std::unique_ptr<TestRenderer> renderer_;
        std::vector<_VulkanBuffer> buffers_;

        TestRenderer::_ComputePipeline& record(const std::vector<int>& bindings) {
            return records_.emplace_back(bindings);
        }

        // A legacy module with three bindings, a 128-wide workgroup and push constants.
        TestRenderer::ComputeProgramSource legacy() const {
            return {{.label = "test/visible_flags", .bindings = {0, 1, 2}, .workgroup = {128, 1, 1}, .required_subgroup_size = SUBGROUP_SIZE},
                    readWords(kSpirvRoot / "generated/visible_flags.spv")};
        }

        // lfs_test_host bound to kTinyEffect by the stage A compiler.
        static TestRenderer::ComputeProgramSource tiny() {
            static const auto source = [] {
                lfs::vis::shaders::SlangCompiler compiler;
                const auto package = loadPackage(LFS_SHADER_TEST_PACKAGE_DIR);
                auto sources = package ? lfs::test::compileHosts<TestRenderer::ComputeProgramSource>(
                                             compiler, package, kTinyEffect, {"lfs_test_host"})
                                       : std::vector<TestRenderer::ComputeProgramSource>{};
                return sources.empty() ? TestRenderer::ComputeProgramSource{} : std::move(sources.front());
            }();
            return source;
        }

        // Every shipped legacy program, as the viewport's renderer builds
        // them; returns their files. Custom variants are built by program sets.
        std::vector<fs::path> initializeShipped() {
            std::map<std::string, std::string> paths;
            std::vector<fs::path> files;
            for (const char* directory : {"generated", "radix_sort"}) {
                for (const auto& entry : fs::directory_iterator(kSpirvRoot / directory)) {
                    const auto stem = entry.path().stem().string();
                    if (entry.path().extension() != ".spv" || stem.contains("_custom"))
                        continue;
                    files.push_back(entry.path());
                    if (!stem.ends_with("_banded") && !stem.ends_with("_reveal"))
                        paths[directory == std::string("radix_sort") ? "radix_sort/" + stem : stem] = entry.path().string();
                }
            }
            renderer_->cleanup();
            Seam::module_code.clear();
            Seam::set_layout_bindings.clear();
            Seam::push_constant_sizes.clear();
            EXPECT_NO_THROW(renderer_->initializeExternal(paths, gpu_.instance, gpu_.physical, gpu_.device, gpu_.queue,
                                                          gpu_.family, gpu_.allocator, VK_NULL_HANDLE, false, nullptr,
                                                          nullptr));
            return files;
        }

        void SetUp() override {
            Seam::fail = Fail::None;
            Seam::owner = std::this_thread::get_id();
            gpu_.create();
            if (IsSkipped() || HasFatalFailure())
                return;
            VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            semaphore_info.pNext = &timeline;
            ASSERT_EQ(vkCreateSemaphore(gpu_.device, &semaphore_info, nullptr, &gate_), VK_SUCCESS);
            ASSERT_EQ(vkCreateSemaphore(gpu_.device, &semaphore_info, nullptr, &render_), VK_SUCCESS);
            renderer_ = std::make_unique<TestRenderer>();
            renderer_->setVulkanDispatch(Seam::dispatch());
            initialize();
        }

        void initialize() {
            ASSERT_NO_THROW(renderer_->VulkanGSPipeline::initializeExternal(gpu_.instance, gpu_.physical, gpu_.device, gpu_.queue,
                                                                            gpu_.family, gpu_.allocator));
        }

        void TearDown() override {
            Seam::fail = Fail::None;
            Seam::hold = false;
            if (renderer_) {
                signal(gate_, 1u << 30);
                vkDeviceWaitIdle(gpu_.device);
                for (auto& buffer : buffers_)
                    renderer_->destroyBuffer(buffer);
                renderer_->cleanup();
                renderer_.reset();
                EXPECT_EQ(Seam::live(), (std::array<int, 4>{})) << "construction objects leaked";
            }
            for (const auto semaphore : {gate_, render_})
                if (semaphore)
                    vkDestroySemaphore(gpu_.device, semaphore, nullptr);
        }

        void signal(const VkSemaphore semaphore, const uint64_t value) {
            uint64_t current = 0;
            vkGetSemaphoreCounterValue(gpu_.device, semaphore, &current);
            if (current >= value)
                return;
            VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            info.semaphore = semaphore;
            info.value = value;
            ASSERT_EQ(vkSignalSemaphore(gpu_.device, &info), VK_SUCCESS);
        }

        TestRenderer::ProgramSetPublication publish() {
            renderer_->waitForProgramSetBuilds();
            auto published = renderer_->publishProgramSet();
            EXPECT_TRUE(published) << lfs::format_for_developer(published.error());
            return published ? *published : TestRenderer::ProgramSetPublication::None;
        }

        _VulkanBuffer& buffer(const size_t bytes) {
            buffers_.reserve(8);
            auto& result = buffers_.emplace_back();
            renderer_->createBuffer(bytes, result);
            return result;
        }

        void awaitRender(const uint64_t value) {
            VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wait.semaphoreCount = 1;
            wait.pSemaphores = &render_;
            wait.pValues = &value;
            ASSERT_EQ(vkWaitSemaphores(gpu_.device, &wait, 5'000'000'000ull), VK_SUCCESS);
        }

        // Submits one dispatch of the current set that the GPU cannot finish before the gate opens.
        void submitBehindGate(const uint64_t gate_value, const uint64_t signal_value) {
            renderer_->beginCommandBatch();
            renderer_->addTimelineWait(gate_, gate_value, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            renderer_->dispatch(renderer_->currentProgramSet()->programs[0], buffers_.front(), 64);
            renderer_->endCommandBatch(false, render_, signal_value);
        }
    };

    // Every shipped module, including the banded and reveal variants, reaches
    // the driver byte for byte through the delegating path overload.
} // namespace lfs::test::program_set

#endif

#if defined(LFS_USER_SHADER_CUSTOM_FIXTURE) || defined(LFS_USER_SHADER_SERVICE_FIXTURE)
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/services.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "gui/string_keys.hpp"
#include "gui/video_export_utils.hpp"
#include "io/loader.hpp"
#include "operation/undo_history.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rasterizer/vulkan/src/custom_effect_layout.h"
#include "rendering/rendering_manager.hpp"
#include "rendering/scene_overlay_params.hpp"
#include "rendering/vksplat_viewport_renderer.hpp"
#include "scene/scene_manager.hpp"
#include "screen/screen_service.hpp"
#include "selection/selection_service.hpp"
#include "sequencer/reveal_render_state.hpp"
#include "shaders/shader_service.hpp"
#include "shaders/slang_compiler.hpp"
#include "user_shader_images.hpp"
#include "user_shader_linker.hpp"
#include "user_shader_scene.hpp"
#include "user_shader_test_access.hpp"
#include "vulkan_external_tensor_test.hpp"
#include "window/vulkan_context.hpp"
#include <SDL3/SDL.h>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cuda_runtime.h>
#include <expected>
#include <functional>
#include <future>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

namespace lfs::test {
    // Window ownership is shared by the two viewport fixtures. Device-only
    // program tests never include this block or initialize SDL/CUDA.
    struct ShaderWindow {
        SDL_Window* window = nullptr;
        bool video = false;
        std::string start() {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
                return "CUDA device unavailable";
            if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
                return SDL_GetError();
            video = true;
            return {};
        }
        bool initialize(std::unique_ptr<lfs::vis::VulkanContext>& context, const int size) {
            window = SDL_CreateWindow("Shader test", size, size, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
            context = std::make_unique<lfs::vis::VulkanContext>();
            if (!window || !context->init(window, size, size))
                return false;
            return true;
        }
        // The viewer's graphics context: rendering managers draw through it.
        bool initialize(lfs::vis::VulkanGraphicsContext& graphics, const int size) {
            window = SDL_CreateWindow("Shader test", size, size, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
            if (!window || !graphics.initialize(window, size, size))
                return false;
            return true;
        }
        ~ShaderWindow() {
            if (window)
                SDL_DestroyWindow(window);
            if (video)
                SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }
    };
} // namespace lfs::test

#endif

#if defined(LFS_USER_SHADER_CUSTOM_FIXTURE)
namespace lfs::test::custom_effect {
    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    using lfs::rendering::GaussianCustomEffectState;
    using lfs::rendering::ViewportRenderRequest;
    using lfs::vis::VksplatViewportRenderer;
    using Access = lfs::vis::UserShaderTestAccess;
    using Slot = Access::Slot;
    using Parameters = GaussianCustomEffectState::Parameters;

    using lfs::test::kIdentity;
    // Each binding's parameter slots select its edit; all-zero slots keep the element.
    enum Slots { Offset,
                 OffsetPerRed,
                 Color,
                 Opacity,
                 Scale,
                 Rotation,
                 NonFinite };

    // Identity, a translation, and a rotation with nonuniform scale.
    inline std::vector<glm::mat4> nodeTransforms() {
        const glm::mat4 rotated = glm::translate(glm::mat4(1.0f), glm::vec3(-0.1f, 0.05f, 0.2f)) *
                                  glm::rotate(glm::mat4(1.0f), glm::radians(15.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
                                  glm::scale(glm::mat4(1.0f), glm::vec3(1.15f, 0.85f, 1.0f));
        return {glm::mat4(1.0f), glm::translate(glm::mat4(1.0f), glm::vec3(0.15f, 0.1f, -0.3f)), rotated};
    }

    enum class Encoding { Float32,
                          Float16,
                          Q16 };
    inline const char* name(const Encoding encoding) {
        return encoding == Encoding::Float32 ? "fp32" : encoding == Encoding::Float16 ? "f16"
                                                                                      : "q16";
    }

    enum class Path { Legacy,
                      Survivors,
                      Gut,
                      GutOutline,
                      BandedSurvivors,
                      BandedGut };
    inline const char* name(const Path path) {
        constexpr const char* names[] = {"legacy", "survivors", "3dgut", "3dgut-outline", "banded-survivors", "banded-3dgut"};
        return names[static_cast<int>(path)];
    }
    inline bool gut(const Path path) { return path == Path::Gut || path == Path::GutOutline || path == Path::BandedGut; }

    struct Frame {
        std::vector<float> pixels;
        bool geometry = false; // the derived 3DGUT geometry buffer was live
        bool custom = false;   // the projection ran the effect's hosts
        bool missing = false;  // the prepared custom chain was unavailable
    };

    inline GaussianCustomEffectState perNode(const std::array<Parameters, kNodes>& values);
    inline GaussianCustomEffectState global();

    inline void expectClose(const Frame& actual, const Frame& expected, const Frame& unedited);

    class CustomEffectTest : public ::testing::Test {
    protected:
        static inline std::unique_ptr<ShaderWindow> window_;
        static inline std::unique_ptr<lfs::vis::VulkanContext> context_;
        static inline std::string skip_;
        static inline std::map<std::string, std::vector<Access::Source>> programs_;
        static inline std::shared_ptr<const lfs::vis::shaders::ShaderPackage> package_;

        static void SetUpTestSuite() {
            window_ = std::make_unique<ShaderWindow>();
            skip_ = window_->start();
            if (skip_.empty() && !window_->initialize(context_, 64))
                skip_ = "Vulkan context unavailable";
        }

        static void TearDownTestSuite() {
            context_.reset();
            window_.reset();
        }

        // Every shipped host linked to `effect`, compiled once per effect.
        static const std::vector<Access::Source>& programs(const char* effect) {
            auto& programs = programs_[effect];
            if (!programs.empty())
                return programs;
            if (!package_) {
                package_ = lfs::test::loadPackage(LFS_SHADER_PACKAGE_DIR);
                if (!package_)
                    return programs;
            }
            static lfs::vis::shaders::SlangCompiler compiler;
            programs = lfs::test::compileHosts<Access::Source>(compiler, package_, effect);
            return programs;
        }

        std::unique_ptr<VksplatViewportRenderer> viewport_;
        std::vector<glm::mat4> transforms_ = nodeTransforms();
        std::shared_ptr<Tensor> transform_indices_;
        Arrays arrays_ = generate();

        void SetUp() override {
            if (!skip_.empty())
                GTEST_SKIP() << skip_;
            viewport_ = std::make_unique<VksplatViewportRenderer>();
            transform_indices_ = indices(arrays_);
        }
        static std::shared_ptr<Tensor> indices(const Arrays& a) {
            return std::make_shared<Tensor>(Tensor::from_vector(a.nodes, {a.size()}, Device::GPU));
        }
        void TearDown() override {
            if (context_)
                (void)context_->deviceWaitIdle();
            viewport_.reset();
        }

        Tensor external(const Tensor& values, const char* label) const {
            return externalTensor(*context_, values, values.shape()[0], label);
        }

        SplatData model(const Arrays& a, const Encoding encoding = Encoding::Float32) const {
            const std::size_t n = a.size();
            const auto cpu = [](const std::vector<float>& values, lfs::core::TensorShape shape) {
                return Tensor::from_vector(values, std::move(shape), Device::CPU);
            };
            const auto make = [&](Tensor shN, const SplatData::ShNLayout layout) {
                return SplatData(1, external(cpu(a.means, {n, 3}), "means"), external(cpu(a.sh0, {n, 1, 3}), "sh0"),
                                 std::move(shN), external(cpu(a.scales, {n, 3}), "scales"),
                                 external(cpu(a.rotations, {n, 4}), "rotations"),
                                 external(cpu(a.opacities, {n, 1}), "opacities"), 1.0f, layout);
            };
            // The renderer binds the resident swizzled SH rest in place.
            const SplatData canonical = make(cpu(a.shN, {n, 3, 3}).to(Device::GPU), SplatData::ShNLayout::Canonical);
            const Tensor& swizzled = canonical.shN_raw();
            if (encoding == Encoding::Float16)
                return make(external(swizzled.to(DataType::Float16), "shN"), SplatData::ShNLayout::Swizzled);
            SplatData result = make(external(swizzled, "shN"), SplatData::ShNLayout::Swizzled);
            if (encoding == Encoding::Q16) {
                result.set_tensor_allocator([this](lfs::core::TensorShape shape, const std::size_t capacity,
                                                   const DataType dtype, const std::string_view label) {
                    auto tensor = lfs::test::makeVulkanExternalTensor(*context_, std::move(shape), dtype, capacity,
                                                                      std::string(label).c_str());
                    if (!tensor)
                        throw std::runtime_error(tensor.error());
                    return std::move(*tensor);
                });
                lfs::core::sh_value_quant::set_enabled_for_testing(true);
                const bool encoded = result.apply_shN_value_quant();
                lfs::core::sh_value_quant::set_enabled_for_testing(std::nullopt);
                EXPECT_TRUE(encoded && result.shN_value_quantized());
            }
            return result;
        }

        // An in-core LOD model of leaves: draws with LOD indices take the
        // quantized page pool, whose pages are quantized from the resident
        // tensors, so the RAD source is never read. RAD opacity encoding
        // selects the RAD 3DGUT rasters and outlines.
        SplatData lodModel(const Arrays& a, const bool rad_opacity) const {
            SplatData result = model(a);
            auto tree = std::make_unique<lfs::core::SplatLodTree>();
            const std::size_t n = a.size();
            tree->child_count.assign(n, 0);
            tree->child_start.assign(n, 0);
            tree->lod_level.assign(n, 0);
            for (std::size_t i = 0; i < n; ++i)
                tree->centers.emplace_back(a.means[3 * i], a.means[3 * i + 1], a.means[3 * i + 2]);
            tree->sizes.assign(n, 0.05f);
            tree->rad_source.path = "in-core.rad";
            tree->rad_source.chunks.resize(tree->chunk_count());
            tree->lod_opacity_encoded = rad_opacity;
            result.lod_tree = std::move(tree);
            return result;
        }

        ViewportRenderRequest request(const Path path) const {
            ViewportRenderRequest r;
            r.frame_view.size = {96, 64};
            r.sh_degree = 1;
            r.scene.model_transforms = &transforms_;
            r.scene.transform_indices = transform_indices_;
            r.gut = gut(path);
            // Survivor projection is the HiGS chain, which deterministic
            // captures take from the first frame.
            r.deterministic_export = path == Path::Survivors || path == Path::BandedSurvivors;
            if (path == Path::GutOutline) {
                r.overlay.markers.show_object_outline = true;
                r.overlay.emphasis.emphasized_node_mask = {true, false, true};
            }
            if (path == Path::BandedSurvivors || path == Path::BandedGut) {
                r.frame_view.size = {96, 32};
                r.frame_view.subregion_origin = {0, 16};
                r.frame_view.subregion_full_size = {96, 64};
            }
            return r;
        }

        Frame render(const SplatData& model, const ViewportRenderRequest& r, const Slot slot, const bool upload,
                     const bool synchronize) {
            auto rendered = viewport_->render(*context_, model, r, upload, slot, synchronize);
            EXPECT_TRUE(rendered) << rendered.error();
            Frame frame;
            frame.geometry = Access::geometry(*viewport_).buffer != VK_NULL_HANDLE;
            frame.custom = Access::customProjection(*viewport_);
            auto image = viewport_->readOutputImageRgba(*context_, slot);
            EXPECT_TRUE(image) << image.error();
            if (rendered && image) {
                frame.missing = rendered->custom_pending;
                const auto* data = (*image)->ptr<float>();
                frame.pixels.assign(data, data + (*image)->numel());
            }
            return frame;
        }
        Frame draw(const SplatData& model, const ViewportRenderRequest& r, const Path path) {
            const bool banded = path == Path::BandedSurvivors || path == Path::BandedGut;
            // Live training uploads take the legacy projection chain.
            return render(model, r, banded ? Slot::Preview : Slot::Main, true, path == Path::Legacy);
        }

        // Publishes `effect` linked into every shipped host, or into the named ones.
        void publish(const char* effect, const std::vector<std::string>& hosts = {}) {
            auto sources = programs(effect);
            ASSERT_EQ(sources.size(), 20u);
            if (!hosts.empty()) {
                std::erase_if(sources, [&](const Access::Source& source) {
                    return std::ranges::find(hosts, source.spec.label) == hosts.end();
                });
                ASSERT_EQ(sources.size(), hosts.size());
            }
            // The renderer exists after its first draw.
            if (!initialized_) {
                (void)draw(model(arrays_), request(Path::Legacy), Path::Legacy);
                initialized_ = true;
            }
            Access::publish(*viewport_, std::move(sources));
        }
        template <class Measure>
        void dispatchCases(Measure&& measure) {
            auto dispatch = lfs::rendering::VulkanDispatch::real();
            dispatch.cmd_dispatch = &Access::dispatch;
            dispatch.cmd_dispatch_indirect = &Access::dispatchIndirect;
            Access::renderer(*viewport_).setVulkanDispatch(dispatch);
            const Arrays large = generate(256, 128);
            const auto scene = model(large);
            const auto indices = std::make_shared<Tensor>(Tensor::from_vector(large.nodes, {large.size()}, Device::GPU));
            const auto frame_request = [&](const Path path) {
                auto r = request(path);
                r.frame_view.size = {1280, 720};
                r.scene.transform_indices = indices;
                r.deterministic_export = false;
                return r;
            };
            // The renderer and its timer callbacks exist after the first draw.
            (void)draw(scene, frame_request(Path::Legacy), Path::Legacy);
            const auto countDispatches = [&](const Path path, const std::optional<GaussianCustomEffectState>& state) {
                auto r = frame_request(path);
                r.custom_effect = state;
                // Past the command batch ring, so earlier timestamps are collected.
                for (int warm = 0; warm < 16; ++warm)
                    (void)draw(scene, r, path);
                Access::dispatches = 0;
                (void)draw(scene, r, path);
                const std::size_t count = Access::dispatches;
                EXPECT_TRUE(context_->deviceWaitIdle());
                measure(path, count, [&] { (void)draw(scene, r, path); });
                return count;
            };
            std::array<Parameters, kNodes> moving{};
            for (auto& node : moving)
                node[Offset] = glm::vec4(0.05f, 0.0f, 0.0f, 0.0f);
            for (const auto path : {Path::Survivors, Path::Legacy, Path::Gut}) {
                const auto before = countDispatches(path, std::nullopt);
                publish(kIdentity);
                const auto unused = countDispatches(path, std::nullopt);
                const auto identity = countDispatches(path, global());
                publish(kEdit);
                const auto moved = countDispatches(path, perNode(moving));
                Access::release(*viewport_);
                EXPECT_EQ(unused, before) << "a published set must not change a default draw";
                EXPECT_EQ(identity, before);
                EXPECT_EQ(moved, before);
            }
        }

        void expectEdited(const Arrays& edited, const GaussianCustomEffectState& state) {
            const auto scene = model(arrays_);
            const auto oracle = model(edited);
            for (const auto path : {Path::Legacy, Path::Survivors, Path::Gut}) {
                SCOPED_TRACE(name(path));
                auto r = request(path);
                r.sh_degree = 0;
                const auto unedited = draw(scene, r, path);
                const auto expected = draw(oracle, r, path);
                r.custom_effect = state;
                expectClose(draw(scene, r, path), expected, unedited);
            }
        }
        bool initialized_ = false;
    };

    // Bindings: one per node, each with its own parameter block.
    inline GaussianCustomEffectState perNode(const std::array<Parameters, kNodes>& values) {
        GaussianCustomEffectState state;
        state.bounds_min = glm::vec3(-1.5f, -1.0f, -6.0f);
        state.bounds_max = glm::vec3(1.5f, 1.0f, -3.5f);
        for (std::uint32_t node = 0; node < kNodes; ++node) {
            state.bindings.push_back({values[node]});
            state.objects.push_back({.binding = node, .id = node});
        }
        return state;
    }
    inline GaussianCustomEffectState global() {
        GaussianCustomEffectState state;
        state.bindings.resize(1);
        state.global_binding = 0;
        state.objects.resize(kNodes);
        return state;
    }

    using lfs::test::compareImages;
    inline void expectSame(const Frame& actual, const Frame& expected) {
        ASSERT_FALSE(expected.pixels.empty());
        const auto d = compareImages(actual.pixels, expected.pixels);
        EXPECT_EQ(d.count, 0u) << "max difference " << d.max;
    }
    // Edits reach the stored domains through exp/sigmoid on one side only, so
    // a pixel of the 8-bit output may round one step apart.
    inline void expectClose(const Frame& actual, const Frame& expected, const Frame& unedited) {
        ASSERT_FALSE(expected.pixels.empty());
        EXPECT_LE(compareImages(actual.pixels, expected.pixels).max, 1.0f / 255.0f + 1e-6f);
        EXPECT_GT(compareImages(actual.pixels, unedited.pixels).max, 0.05f) << "the edit must change the image";
    }

    constexpr Encoding kEncodings[] = {Encoding::Float32, Encoding::Float16, Encoding::Q16};
    constexpr Path kPaths[] = {Path::Legacy, Path::Survivors, Path::Gut, Path::GutOutline,
                               Path::BandedSurvivors, Path::BandedGut};

} // namespace lfs::test::custom_effect

#endif

#if defined(LFS_USER_SHADER_SERVICE_FIXTURE)
namespace lfs::test::service_rendering {
    namespace fs = std::filesystem;
    namespace shaders = lfs::vis::shaders;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    using lfs::core::Uuid;
    namespace DirtyFlag = lfs::vis::DirtyFlag;
    using lfs::vis::FrameReason;
    using lfs::vis::RenderingManager;
    using shaders::ShaderService;
    using shaders::ShaderSlot;
    using shaders::ShaderState;
    using Access = lfs::vis::UserShaderTestAccess;
    using Slot = Access::Slot;
    using Kind = ShaderSlot::Kind;
    using namespace std::chrono_literals;

    constexpr int kSize = 128;
    // The preview the fixture's capture() and exportFrame() draw.
    constexpr RenderingManager::ExportFrame kFrame{.width = kSize, .height = kSize};
    const std::function<RenderingManager::ExportFrame()> kFrameOf = [] { return kFrame; };
    // The viewport output holds 8-bit channels; separate draws of one image
    // may round a channel differently.
    constexpr float kOneStep = 1.5f / 255.0f;
    constexpr std::array kRed{0.3f, -0.3f, -0.3f};

    using Image = std::expected<std::vector<float>, std::string>;
    // drawWithShaders' post on the thread that draws.
    constexpr auto kInline = [](auto draw) { return draw(); };

    class OffViewer;
    class ExportRun;

    class ShaderServiceRenderingTest : public ::testing::Test {
    protected:
        using RenderSettings = lfs::vis::RenderSettings;
        friend class OffViewer;
        friend class ExportRun;

        void SetUp() override {
            window_ = std::make_unique<ShaderWindow>();
            const auto unavailable = window_->start();
            if (!unavailable.empty())
                GTEST_SKIP() << unavailable;
            graphics_ = std::make_unique<lfs::vis::VulkanGraphicsContext>();
            ASSERT_TRUE(window_->initialize(*graphics_, kSize)) << SDL_GetError() << graphics_->lastError();
            context_ = &graphics_->vulkanContext();

            lfs::event::EventBridge::instance().clear_all();
            lfs::vis::services().clear();
            lfs::vis::op::undoHistory().clear();
            scene_ = std::make_unique<lfs::vis::SceneManager>();
            for (int i = 0; i < 3; ++i) {
                Uuid uuid;
                scene().addSplat(std::format("node{}", i), node(1.2f * static_cast<float>(i - 1)),
                                 lfs::core::NULL_NODE, &uuid);
                nodes_.push_back(uuid);
            }
            scene_->changeContentType(lfs::vis::SceneManager::ContentType::SplatFiles);
            manager_ = std::make_unique<RenderingManager>(views_);
            manager_->retainVisibleViews({views_.activeView()});
            lfs::vis::services().set(scene_.get());
            lfs::vis::services().set(manager_.get());
            Access::install(*manager_, *graphics_);
            scene().setCombinedModelAllocator(manager_->makeSplatTensorAllocator());
            manager_->setWakeCallback([this] { ++render_wakes_; });
            settings_ = manager_->getSettings();
            settings_.background_color = glm::vec3(0.0f);
            settings_.sh_degree = 0;
            settings_.show_grid = false;
            settings_.show_camera_frustums = false;
            manager_->updateSettings(settings_);
            viewport_.frameBufferSize = {kSize, kSize};
            look(viewport_, eye_);

            const fs::path manifest = fs::path(LFS_SHADER_PACKAGE_DIR) / "manifest.json";
            service_ = std::make_unique<ShaderService>(
                scene(),
                [this, slang = shaders::slangHostLinker(manifest)] {
                    ++linkers_;
                    return std::make_unique<ScheduledLinker>(schedule_, slang());
                },
                [this] { ++service_wakes_; });
            manager_->setShaderService(service_.get());
            // The first frames of a new viewport size wait for it to settle.
            ASSERT_TRUE(settle([&] { return frame().rendered; }));
        }

        void TearDown() override {
            Access::fail_pipeline = false;
            gate(false);
            manager_.reset();
            service_.reset();
            scene_.reset();
            lfs::vis::op::undoHistory().clear();
            lfs::vis::services().clear();
            lfs::event::EventBridge::instance().clear_all();
            context_ = nullptr;
            if (graphics_)
                graphics_->shutdown();
            graphics_.reset();
            window_.reset();
        }

        // A 5x5 grid of gray splats around (x, 0, 0) in fresh Vulkan memory.
        std::unique_ptr<SplatData> node(const float x) {
            const auto a = generate(5, 5, x);
            const auto n = a.size();
            const auto external = [&](const std::vector<float>& values, lfs::core::TensorShape shape) {
                const auto cpu = Tensor::from_vector(values, std::move(shape), Device::CPU);
                return externalTensor(*context_, cpu, cpu.numel(), "shader-service-fixture");
            };
            return std::make_unique<SplatData>(
                0, external(a.means, {n, 3}), external(a.sh0, {n, 1, 3}),
                Tensor::zeros({n, std::size_t{0}, std::size_t{3}}, Device::GPU),
                external(a.scales, {n, 3}), external(a.rotations, {n, 4}), external(a.opacities, {n, 1}), 1.0f);
        }

        static void look(Viewport& viewport, const glm::vec3& eye) {
            viewport.setViewMatrix(lfs::rendering::makeVisualizerLookAtRotation(eye, glm::vec3(0.0f)), eye);
        }

        lfs::core::Scene& scene() { return scene_->getScene(); }
        ShaderService& service() { return *service_; }
        lfs::vis::ShaderCommands commands() { return {service()}; }

        void gate(const bool closed) { schedule_.gate(closed); }

        // One viewer loop iteration: adopt the service's programs, then draw.
        lfs::vis::ViewportFrameResult frame(const bool on_screen = true) {
            manager_->syncCustomPrograms();
            (void)manager_->frameDemandLedger().plan(lfs::vis::FrameClock::now());
            return manager_->renderFrame({.view = views_.activeView(),
                                          .viewport = viewport_,
                                          .settings = settings_,
                                          .scene_manager = scene_.get(),
                                          .graphics_context = graphics_.get(),
                                          .on_screen = on_screen});
        }
        lfs::vis::ViewportFrameResult redraw() {
            manager_->markDirty(DirtyFlag::SPLATS, FrameReason::SettingsChange);
            return frame();
        }
        bool settle(const std::function<bool()>& done) {
            const auto deadline = std::chrono::steady_clock::now() + 30s;
            while (!done()) {
                if (std::chrono::steady_clock::now() > deadline)
                    return false;
                frame();
                std::this_thread::sleep_for(2ms);
            }
            return true;
        }
        bool applied(const Uuid& asset) {
            return settle([&] { return service().status(asset)->state == ShaderState::Applied; });
        }
        // Applied, and every draw of a frame runs it: a set that grows by a
        // family the frames need may still be building after `applied`.
        bool drawing(const Uuid& asset) {
            return settle([&] {
                redraw();
                return service().status(asset)->state == ShaderState::Applied && Access::ranEveryProgram(*manager_);
            });
        }
        std::uint64_t links(const Uuid& asset) { return service().status(asset)->links; }
        std::size_t shownHosts() {
            const auto shown = Access::displayPrograms(*manager_);
            return shown ? shown->hosts.size() : 0;
        }
        lfs::core::NodeId id(const Uuid& node) { return scene().getNodeByUuid(node)->id; }
        std::vector<float> image(const Slot slot = Slot::Main) { return Access::image(*manager_, *context_, slot); }
        std::vector<float> drawn() {
            redraw();
            return image();
        }
        void apply(RenderSettings settings) {
            // frame() draws these settings as given; the tests toggle GUT alone, which the manager turns into the raster backend.
            settings.raster_backend = lfs::rendering::viewerRasterBackendForGutMode(settings.gut);
            settings_ = settings;
            manager_->updateSettings(settings_);
        }

        Uuid bound(const std::string& source, const ShaderSlot& slot = {.kind = Kind::Global}) {
            auto asset = commands().createAsset("Effect", source);
            EXPECT_TRUE(asset);
            const auto status = commands().bind(slot, *asset);
            EXPECT_TRUE(status) << lfs::format_for_developer(status.error());
            return *asset;
        }
        void set(const Uuid& asset, const std::string_view name, const std::span<const float> value) {
            const auto status = commands().setParameter(asset, name, value);
            ASSERT_TRUE(status) << lfs::format_for_developer(status.error());
        }
        void shift(const Uuid& asset, const std::array<float, 3>& value) { set(asset, "shift", value); }

        // The viewport camera's image of `model` drawn with `state`.
        Image preview(const lfs::core::SplatData& model, lfs::vis::SceneRenderState state) {
            return rgbOf(manager_->renderPreviewImage(model, std::move(state), viewport_.getRotationMatrix(),
                                                      viewport_.getTranslation(), settings_.focal_length_mm, kSize,
                                                      kSize));
        }
        // The splat path of a video export frame: the scene of the export's
        // start with its reveal and the custom effect of its lease at `time`.
        std::vector<float> capture() {
            return rgbOf(manager_->renderPreviewImage(scene_.get(), viewport_.getRotationMatrix(),
                                                      viewport_.getTranslation(), settings_.focal_length_mm, kSize, kSize));
        }
        Image exportFrame(const lfs::vis::gui::VideoExportSceneSnapshot& snapshot,
                          const ShaderService::ExportLease& lease, const float time,
                          const std::optional<lfs::sequencer::RevealClip>& clip) {
            const auto shaders = lease.snapshot;
            return preview(*snapshot.gaussianModel(),
                           lfs::vis::gui::makeVideoExportGaussianSceneState(snapshot, clip, shaders.get(), time));
        }

        // Pixels of each node, from built-in draws of that node alone.
        std::vector<std::vector<bool>> nodeMasks() {
            std::vector<std::vector<bool>> masks;
            for (std::size_t shown = 0; shown < nodes_.size(); ++shown) {
                for (std::size_t i = 0; i < nodes_.size(); ++i)
                    scene().setNodeVisibility(scene().getNodeByUuid(nodes_[i])->id, i == shown);
                const auto rgb = drawn();
                auto& mask = masks.emplace_back(rgb.size() / 3);
                for (std::size_t p = 0; p < mask.size(); ++p)
                    mask[p] = rgb[3 * p] + rgb[3 * p + 1] + rgb[3 * p + 2] > 0.15f;
                EXPECT_GT(std::ranges::count(mask, true), 10) << "node " << shown << " is off screen";
            }
            for (const auto& uuid : nodes_)
                scene().setNodeVisibility(scene().getNodeByUuid(uuid)->id, true);
            return masks;
        }

        std::unique_ptr<ShaderWindow> window_;
        std::unique_ptr<lfs::vis::VulkanGraphicsContext> graphics_;
        lfs::vis::VulkanContext* context_ = nullptr;
        lfs::vis::screen::ScreenService views_;
        std::unique_ptr<lfs::vis::SceneManager> scene_;
        std::unique_ptr<RenderingManager> manager_;
        std::unique_ptr<ShaderService> service_;
        std::vector<Uuid> nodes_;
        RenderSettings settings_;
        Viewport viewport_{kSize, kSize};
        glm::vec3 eye_{0.0f, 0.0f, 6.0f};

        Schedule schedule_;
        std::atomic<int> linkers_ = 0;
        std::atomic<int> service_wakes_ = 0;
        std::atomic<int> render_wakes_ = 0;
    };

    class OffViewer {
    public:
        explicit OffViewer(ShaderServiceRenderingTest& test, std::function<void()> loop = {})
            : test_(test),
              loop_(std::move(loop)) {}

        template <class Work>
        std::invoke_result_t<Work&> run(Work work) {
            auto done = std::async(std::launch::async, std::move(work));
            const auto deadline = std::chrono::steady_clock::now() + 30s;
            while (done.wait_for(0ms) != std::future_status::ready) {
                drain();
                if (loop_)
                    loop_();
                else
                    test_.frame();
                if (std::chrono::steady_clock::now() > deadline) {
                    ADD_FAILURE() << "the work did not end";
                    test_.service().stop();
                }
                std::this_thread::sleep_for(1ms);
            }
            return done.get();
        }
        // drawWithShaders' post: runs `work` on the viewer thread.
        template <class Work>
        std::invoke_result_t<Work&> onViewer(Work work) {
            std::packaged_task<std::invoke_result_t<Work&>()> task(std::move(work));
            auto done = task.get_future();
            {
                const std::lock_guard lock(mutex_);
                tasks_.emplace_back([&task] { task(); });
            }
            return done.get();
        }
        auto post() {
            return [this](auto draw) { return onViewer(std::move(draw)); };
        }

    private:
        void drain() {
            std::vector<std::function<void()>> tasks;
            {
                const std::lock_guard lock(mutex_);
                tasks.swap(tasks_);
            }
            for (auto& task : tasks)
                task();
        }

        ShaderServiceRenderingTest& test_;
        std::function<void()> loop_;
        std::mutex mutex_;
        std::vector<std::function<void()>> tasks_;
    };

    // Exports from another thread as the video export does: through
    // drawWithShaders, each frame drawn on the viewer thread from the scene
    // snapshot of the export's start, while the viewer keeps drawing frames.
    class ExportRun {
    public:
        ExportRun(ShaderServiceRenderingTest& test, const ShaderService::ExportLease& lease,
                  std::optional<lfs::sequencer::RevealClip> clip = {})
            : test_(test),
              viewer_(test),
              lease_(lease),
              clip_(clip) {}

        // `edit` runs on the viewer thread after each exported frame.
        std::expected<std::vector<std::vector<float>>, std::string> run(const std::function<void(int)>& edit = {}) {
            return viewer_.run([&] { return exportFrames(edit); });
        }
        // Actual draw callbacks after shader preflight.
        int draws() const { return draws_; }

    private:
        std::expected<std::vector<std::vector<float>>, std::string> exportFrames(const std::function<void(int)>& edit) {
            const auto onViewer = [this](auto work) { return viewer_.onViewer(std::move(work)); };
            auto snapshot = onViewer([&] {
                auto captured = lfs::vis::gui::captureVideoExportSceneSnapshot(*test_.scene_);
                // The viewer moves the copy into Vulkan memory through its window; the test has none.
                if (captured) {
                    if (auto moved = lfs::io::migrateSplatTensorsToAllocator(*captured->combined_model,
                                                                             test_.manager_->makeSplatTensorAllocator());
                        !moved)
                        captured = std::unexpected(moved.error().format());
                }
                return captured;
            });
            if (!snapshot)
                return std::unexpected(snapshot.error());
            std::vector<std::vector<float>> frames;
            for (int index = 0; index < 3; ++index) {
                const float time = 0.5f * static_cast<float>(index);
                auto image = test_.manager_->drawWithShaders(&lease_, viewer_.post(), [&] {
                    ++draws_;
                    return test_.exportFrame(*snapshot, lease_, time, clip_);
                });
                if (!image)
                    return std::unexpected(image.error());
                frames.push_back(std::move(*image));
                if (edit)
                    onViewer([&] { edit(index); });
            }
            return frames;
        }

        ShaderServiceRenderingTest& test_;
        OffViewer viewer_;
        const ShaderService::ExportLease& lease_;
        std::optional<lfs::sequencer::RevealClip> clip_;
        std::atomic<int> draws_ = 0;
    };

} // namespace lfs::test::service_rendering

#endif
