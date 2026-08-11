/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/rendering/scene_upscaler_registry.hpp"
#include "visualizer/rendering/vulkan_scene_upscaler_adapter.hpp"
#include "visualizer/rendering/vulkan_scene_upscaler_controller.hpp"
#include "visualizer/window/vulkan_context.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace lfs::vis {
    namespace {
        int factory_calls = 0;
        int vulkan_initialize_calls = 0;
        int vulkan_record_calls = 0;
        int vulkan_reset_calls = 0;
        int vulkan_shutdown_calls = 0;
        bool vulkan_record_succeeds = true;

        class ReadyOptionalAdapter final : public SceneUpscalerAdapter {
        public:
            SceneUpscalerAvailability probe(
                const SceneUpscalerProbeContext& context) const noexcept override {
                if (context.graphics_api != SceneUpscalerGraphicsApi::Vulkan) {
                    return {.reason =
                                SceneUpscalerAvailabilityReason::GraphicsApiUnsupported};
                }
                if (context.vendor_id == 0) {
                    return {.reason = SceneUpscalerAvailabilityReason::DeviceUnsupported};
                }
                return {.reason = SceneUpscalerAvailabilityReason::Ready};
            }
        };

        class MissingRuntimeAdapter final : public SceneUpscalerAdapter {
        public:
            SceneUpscalerAvailability probe(
                const SceneUpscalerProbeContext&) const noexcept override {
                return {.reason = SceneUpscalerAvailabilityReason::RuntimeMissing};
            }
        };

        class ReadyVulkanAdapter final : public VulkanSceneUpscalerAdapter {
        public:
            SceneUpscalerAvailability probe(
                const SceneUpscalerProbeContext& context) const noexcept override {
                return {.reason = context.vendor_id == 1
                                      ? SceneUpscalerAvailabilityReason::Ready
                                      : SceneUpscalerAvailabilityReason::DeviceUnsupported};
            }

            SceneUpscalerAvailability initialize(VulkanContext&) noexcept override {
                ++vulkan_initialize_calls;
                return {.reason = SceneUpscalerAvailabilityReason::Ready};
            }

            bool record(VkCommandBuffer,
                        const VulkanSceneUpscalerDispatch& dispatch) noexcept override {
                ++vulkan_record_calls;
                output_ = {
                    .color = {
                        .image = reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(0x4000)),
                        .view = reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(0x4001)),
                        .layout = VK_IMAGE_LAYOUT_GENERAL,
                        .valid_extent = dispatch.output_extent,
                        .allocation_extent = dispatch.output_extent,
                    },
                };
                return vulkan_record_succeeds;
            }

            VulkanSceneUpscalerOutput output(TemporalViewId) const noexcept override {
                return output_;
            }

            void reset(TemporalViewId, TemporalResetReason) noexcept override {
                ++vulkan_reset_calls;
            }

            void shutdown() noexcept override {
                ++vulkan_shutdown_calls;
            }

        private:
            VulkanSceneUpscalerOutput output_{};
        };

        SceneUpscalerAdapterFactoryResult makeReadyAdapter() noexcept {
            ++factory_calls;
            auto adapter = std::unique_ptr<SceneUpscalerAdapter>(
                new (std::nothrow) ReadyOptionalAdapter());
            if (!adapter)
                return std::unexpected(SceneUpscalerAvailabilityReason::ProbeFailed);
            return adapter;
        }

        SceneUpscalerAdapterFactoryResult makeMissingRuntimeAdapter() noexcept {
            ++factory_calls;
            auto adapter = std::unique_ptr<SceneUpscalerAdapter>(
                new (std::nothrow) MissingRuntimeAdapter());
            if (!adapter)
                return std::unexpected(SceneUpscalerAvailabilityReason::ProbeFailed);
            return adapter;
        }

        SceneUpscalerAdapterFactoryResult makeReadyVulkanAdapter() noexcept {
            ++factory_calls;
            auto adapter = std::unique_ptr<SceneUpscalerAdapter>(
                new (std::nothrow) ReadyVulkanAdapter());
            if (!adapter)
                return std::unexpected(SceneUpscalerAvailabilityReason::ProbeFailed);
            return adapter;
        }

        void resetVulkanAdapterCounters() {
            factory_calls = 0;
            vulkan_initialize_calls = 0;
            vulkan_record_calls = 0;
            vulkan_reset_calls = 0;
            vulkan_shutdown_calls = 0;
            vulkan_record_succeeds = true;
        }

        VulkanSceneUpscalerResource testResource(const std::uintptr_t base,
                                                 const glm::ivec2 extent) {
            return {
                .image = reinterpret_cast<VkImage>(base),
                .view = reinterpret_cast<VkImageView>(base + 1),
                .layout = VK_IMAGE_LAYOUT_GENERAL,
                .valid_extent = extent,
                .allocation_extent = extent,
            };
        }

        VulkanContext& inertVulkanContext() {
            alignas(VulkanContext) static std::byte storage[sizeof(VulkanContext)];
            static auto* const context = new (storage) VulkanContext();
            return *context;
        }

        OptionalSceneUpscalerDescriptor optionalDescriptor(std::string id = "optional_test") {
            return {
                .id = std::move(id),
                .label_key = "preferences.scene_upscaler_optional_test",
                .requirements = {
                    .depth = true,
                    .motion_vectors = true,
                    .jitter = true,
                    .history = true,
                },
            };
        }
    } // namespace

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
        EXPECT_TRUE(temporal.requirements.jitter);
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

    TEST(SceneUpscalerAdapterRegistry, RejectsInvalidDuplicateAndBuiltinRegistrations) {
        OptionalSceneUpscalerRegistry registry;

        EXPECT_FALSE(registry.registerAdapter({}, makeReadyAdapter));
        EXPECT_FALSE(registry.registerAdapter(optionalDescriptor("native"), makeReadyAdapter));
        EXPECT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyAdapter));
        EXPECT_FALSE(registry.registerAdapter(optionalDescriptor(), makeReadyAdapter));
        ASSERT_EQ(registry.descriptors().size(), 1U);
    }

    TEST(SceneUpscalerAdapterRegistry, PreservesOwnedDescriptorAndRequirements) {
        OptionalSceneUpscalerRegistry registry;
        auto source = optionalDescriptor();
        ASSERT_TRUE(registry.registerAdapter(source, makeReadyAdapter));
        source.id = "changed_after_registration";

        const auto stored = registry.descriptor("optional_test");
        ASSERT_TRUE(stored.has_value());
        EXPECT_EQ(stored->label_key, "preferences.scene_upscaler_optional_test");
        EXPECT_TRUE(stored->requirements.depth);
        EXPECT_TRUE(stored->requirements.temporal());
        EXPECT_FALSE(registry.descriptor(source.id).has_value());
    }

    TEST(SceneUpscalerAdapterRegistry, SafeModeRejectsBeforeFactoryOrRuntimeProbe) {
        OptionalSceneUpscalerRegistry registry;
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyAdapter));
        factory_calls = 0;

        const auto availability = registry.probe(
            "optional_test", {.vendor_id = 1, .safe_mode = true});
        EXPECT_EQ(availability.reason, SceneUpscalerAvailabilityReason::SafeMode);
        EXPECT_FALSE(availability.available());
        EXPECT_EQ(factory_calls, 0);
        EXPECT_EQ(registry.createAvailable(
                      "optional_test", {.vendor_id = 1, .safe_mode = true}),
                  nullptr);
        EXPECT_EQ(factory_calls, 0);
    }

    TEST(SceneUpscalerAdapterRegistry, ReportsUnknownRuntimeAndDeviceFailuresExplicitly) {
        OptionalSceneUpscalerRegistry registry;
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyAdapter));
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor("missing_runtime"),
                                             makeMissingRuntimeAdapter));

        EXPECT_EQ(registry.probe("unknown", {}).reason,
                  SceneUpscalerAvailabilityReason::NotCompiled);
        EXPECT_EQ(registry.probe("optional_test", {}).reason,
                  SceneUpscalerAvailabilityReason::DeviceUnsupported);
        EXPECT_EQ(registry.probe("missing_runtime", {.vendor_id = 1}).reason,
                  SceneUpscalerAvailabilityReason::RuntimeMissing);
    }

    TEST(SceneUpscalerAdapterRegistry, CreatesOnlyAdaptersThatPassCapabilityProbe) {
        OptionalSceneUpscalerRegistry registry;
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyAdapter));
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor("missing_runtime"),
                                             makeMissingRuntimeAdapter));

        EXPECT_NE(registry.createAvailable("optional_test", {.vendor_id = 1}), nullptr);
        EXPECT_EQ(registry.createAvailable("optional_test", {}), nullptr);
        EXPECT_EQ(registry.createAvailable("missing_runtime", {.vendor_id = 1}), nullptr);
        EXPECT_EQ(registry.createAvailable("unknown", {.vendor_id = 1}), nullptr);
    }

    TEST(SceneUpscalerAdapterRegistry, TypedCreationPreservesFailureWithoutSecondFactoryCall) {
        OptionalSceneUpscalerRegistry registry;
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor("missing_runtime"),
                                             makeMissingRuntimeAdapter));
        factory_calls = 0;

        const auto result = registry.createAvailableResult("missing_runtime", {.vendor_id = 1});

        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), SceneUpscalerAvailabilityReason::RuntimeMissing);
        EXPECT_EQ(factory_calls, 1);
    }

    TEST(SceneUpscalerCatalog, CombinesInternalAndSelectableOptionalBackendsInStableOrder) {
        OptionalSceneUpscalerRegistry registry;
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyAdapter));

        const auto catalog = availableSceneUpscalerCatalog(registry, {.vendor_id = 1});
        ASSERT_EQ(catalog.size(), 4U);
        EXPECT_EQ(catalog[0].id, "native");
        EXPECT_EQ(catalog[1].id, "spatial");
        EXPECT_EQ(catalog[2].id, "temporal");
        EXPECT_EQ(catalog[3].id, "optional_test");
        EXPECT_TRUE(catalog[0].internal);
        EXPECT_FALSE(catalog[3].internal);
        EXPECT_TRUE(catalog[3].requirements.temporal());
    }

    TEST(SceneUpscalerCatalog, RetainsFailureReasonButFiltersUnavailableChoices) {
        OptionalSceneUpscalerRegistry registry;
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor("missing_runtime"),
                                             makeMissingRuntimeAdapter));

        const auto complete = buildSceneUpscalerCatalog(registry, {.vendor_id = 1});
        ASSERT_EQ(complete.size(), 4U);
        EXPECT_EQ(complete.back().availability.reason,
                  SceneUpscalerAvailabilityReason::RuntimeMissing);
        EXPECT_FALSE(complete.back().selectable());

        const auto selectable = availableSceneUpscalerCatalog(registry, {.vendor_id = 1});
        ASSERT_EQ(selectable.size(), 3U);
        EXPECT_EQ(selectable.back().id, "temporal");
    }

    TEST(SceneUpscalerCatalog, SafeModeKeepsOnlyNativeWithoutConstructingOptionalAdapters) {
        OptionalSceneUpscalerRegistry registry;
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyAdapter));
        factory_calls = 0;

        const auto complete = buildSceneUpscalerCatalog(
            registry, {.vendor_id = 1, .safe_mode = true});
        ASSERT_EQ(complete.size(), 4U);
        EXPECT_TRUE(complete[0].selectable());
        EXPECT_EQ(complete[1].availability.reason,
                  SceneUpscalerAvailabilityReason::SafeMode);
        EXPECT_EQ(complete[2].availability.reason,
                  SceneUpscalerAvailabilityReason::SafeMode);
        EXPECT_EQ(complete.back().availability.reason,
                  SceneUpscalerAvailabilityReason::SafeMode);
        EXPECT_EQ(factory_calls, 0);

        const auto selectable = availableSceneUpscalerCatalog(
            registry, {.vendor_id = 1, .safe_mode = true});
        ASSERT_EQ(selectable.size(), 1U);
        EXPECT_EQ(selectable.front().id, "native");
        EXPECT_EQ(factory_calls, 0);
    }

    TEST(SceneUpscalerAdapterContract, ValidatesOnlyResourcesRequiredByTheBackend) {
        const auto make_resource = [](const std::uintptr_t base) {
            return VulkanSceneUpscalerResource{
                .image = reinterpret_cast<VkImage>(base),
                .view = reinterpret_cast<VkImageView>(base + 1),
                .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .valid_extent = {1280, 720},
                .allocation_extent = {1280, 720},
                .generation = 1,
            };
        };
        VulkanSceneUpscalerDispatch dispatch{
            .color = make_resource(0x1000),
            .output_extent = {1920, 1080},
            .frame_time_seconds = 1.0f / 60.0f,
        };

        EXPECT_TRUE(dispatch.valid({}));
        EXPECT_FALSE(dispatch.valid({.depth = true}));
        dispatch.depth = make_resource(0x2000);
        EXPECT_TRUE(dispatch.valid({.depth = true}));
        EXPECT_FALSE(dispatch.valid({.depth = true, .motion_vectors = true}));
        dispatch.motion = make_resource(0x3000);
        EXPECT_TRUE(dispatch.valid({.depth = true, .motion_vectors = true}));
    }

    TEST(SceneUpscalerAdapterContract, RejectsMalformedExtentsAndPhysicalResources) {
        VulkanSceneUpscalerResource resource{
            .image = reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(0x1000)),
            .view = reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(0x1001)),
            .layout = VK_IMAGE_LAYOUT_GENERAL,
            .valid_extent = {1280, 720},
            .allocation_extent = {1280, 720},
        };
        EXPECT_TRUE(resource.valid());
        resource.allocation_extent = {1279, 720};
        EXPECT_FALSE(resource.valid());
        resource.allocation_extent = {1280, 720};
        resource.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        EXPECT_FALSE(resource.valid());
    }

    TEST(SceneUpscalerAdapterContract, OutputMustMatchTheRequestedPresentationExtent) {
        VulkanSceneUpscalerOutput output{
            .color = {
                .image = reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(0x1000)),
                .view = reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>(0x1001)),
                .layout = VK_IMAGE_LAYOUT_GENERAL,
                .valid_extent = {1920, 1080},
                .allocation_extent = {1920, 1080},
            },
        };
        EXPECT_TRUE(output.valid({1920, 1080}));
        EXPECT_FALSE(output.valid({1280, 720}));
    }

    TEST(VulkanSceneUpscalerController, LazilyOwnsAndReusesTheSelectedAdapter) {
        OptionalSceneUpscalerRegistry registry;
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyVulkanAdapter));
        auto& context = inertVulkanContext();
        VulkanSceneUpscalerController controller(registry);
        resetVulkanAdapterCounters();

        EXPECT_TRUE(controller.select("optional_test", {.vendor_id = 1}, context));
        EXPECT_TRUE(controller.status().active());
        EXPECT_EQ(controller.status().active_id, "optional_test");
        EXPECT_EQ(factory_calls, 1);
        EXPECT_EQ(vulkan_initialize_calls, 1);

        EXPECT_TRUE(controller.select("optional_test", {.vendor_id = 1}, context));
        EXPECT_EQ(factory_calls, 1);
        EXPECT_EQ(vulkan_initialize_calls, 1);

        EXPECT_FALSE(controller.select("native", {.vendor_id = 1}, context));
        EXPECT_FALSE(controller.status().active());
        EXPECT_EQ(controller.status().requested_id, "native");
        EXPECT_EQ(vulkan_shutdown_calls, 1);
    }

    TEST(VulkanSceneUpscalerController, ValidatesDispatchResetsAndOutputBeforeSuccess) {
        OptionalSceneUpscalerRegistry registry;
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyVulkanAdapter));
        auto& context = inertVulkanContext();
        VulkanSceneUpscalerController controller(registry);
        resetVulkanAdapterCounters();
        ASSERT_TRUE(controller.select("optional_test", {.vendor_id = 1}, context));

        const glm::ivec2 input_extent{1280, 720};
        const glm::ivec2 output_extent{1920, 1080};
        const VulkanSceneUpscalerDispatch dispatch{
            .color = testResource(0x1000, input_extent),
            .depth = testResource(0x2000, input_extent),
            .motion = testResource(0x3000, input_extent),
            .output_extent = output_extent,
            .frame_time_seconds = 1.0f / 60.0f,
            .reset_reasons = TemporalResetReason::Scene,
        };

        EXPECT_TRUE(controller.record(
            reinterpret_cast<VkCommandBuffer>(static_cast<std::uintptr_t>(0x5000)), dispatch));
        EXPECT_EQ(vulkan_record_calls, 1);
        EXPECT_EQ(vulkan_reset_calls, 1);
        EXPECT_TRUE(controller.output(TemporalViewId::Main).valid(output_extent));
    }

    TEST(VulkanSceneUpscalerController, FallsBackAndReleasesAdapterOnRecordFailure) {
        OptionalSceneUpscalerRegistry registry;
        ASSERT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyVulkanAdapter));
        auto& context = inertVulkanContext();
        VulkanSceneUpscalerController controller(registry);
        resetVulkanAdapterCounters();
        ASSERT_TRUE(controller.select("optional_test", {.vendor_id = 1}, context));
        vulkan_record_succeeds = false;

        const glm::ivec2 extent{1280, 720};
        const VulkanSceneUpscalerDispatch dispatch{
            .color = testResource(0x1000, extent),
            .depth = testResource(0x2000, extent),
            .motion = testResource(0x3000, extent),
            .output_extent = extent,
        };
        EXPECT_FALSE(controller.record(
            reinterpret_cast<VkCommandBuffer>(static_cast<std::uintptr_t>(0x5000)), dispatch));
        EXPECT_FALSE(controller.status().active());
        EXPECT_EQ(controller.status().failure, VulkanSceneUpscalerFailure::Record);
        EXPECT_EQ(controller.status().fallback_count, 1U);
        EXPECT_EQ(vulkan_shutdown_calls, 1);
    }

    TEST(VulkanSceneUpscalerController, RejectsSafeModeAndNonVulkanAdapters) {
        auto& context = inertVulkanContext();
        {
            OptionalSceneUpscalerRegistry registry;
            ASSERT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyVulkanAdapter));
            VulkanSceneUpscalerController controller(registry);
            resetVulkanAdapterCounters();
            EXPECT_FALSE(controller.select(
                "optional_test", {.vendor_id = 1, .safe_mode = true}, context));
            EXPECT_EQ(controller.status().availability,
                      SceneUpscalerAvailabilityReason::SafeMode);
            EXPECT_EQ(factory_calls, 0);
        }
        {
            OptionalSceneUpscalerRegistry registry;
            ASSERT_TRUE(registry.registerAdapter(optionalDescriptor(), makeReadyAdapter));
            VulkanSceneUpscalerController controller(registry);
            resetVulkanAdapterCounters();
            EXPECT_FALSE(controller.select("optional_test", {.vendor_id = 1}, context));
            EXPECT_EQ(controller.status().failure,
                      VulkanSceneUpscalerFailure::WrongAdapterType);
        }
    }
} // namespace lfs::vis
