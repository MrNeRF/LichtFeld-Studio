/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "visualizer/gpu_preferences_state.hpp"
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;
    using namespace lfs::vis;
    const std::string gpu_a(32, 'a'), gpu_b(32, 'b');

    class GpuConfigurationPolicy : public ::testing::Test {
    protected:
        std::vector<VulkanHardwareDevice> devices{
            {.name = "Identical GPU name", .uuid = gpu_a, .index = 0, .discrete = true},
            {.name = "Identical GPU name", .uuid = gpu_b, .index = 1, .discrete = true}};
        TensorPreferenceState startup;
        GpuRuntimeSelection running{
            .backend = GpuBackend::CUDA,
            .automatic_backend = GpuBackend::CUDA,
            .cuda_uuid = gpu_a,
            .vulkan_uuid = gpu_a,
            .rendering_uuid = gpu_a,
            .automatic_cuda_uuid = gpu_a,
            .automatic_vulkan_uuid = gpu_a,
            .automatic_rendering_uuid = gpu_a};
        bool pending(const TensorPreferenceState& requested) {
            return gpuPreferencesRequireRestart(requested, startup, running, devices, devices);
        }
    };

    TEST_F(GpuConfigurationPolicy, InactiveTensorGpuDoesNotSteerAutomaticRenderer) {
        TensorBackendOptions options{.vulkan_device = gpu_b};
        EXPECT_TRUE(rendering_device_preference(options, GpuBackend::CUDA).empty());
        EXPECT_TRUE(rendering_device_preference(options, GpuBackend::Metal).empty());
        EXPECT_EQ(rendering_device_preference(options, GpuBackend::Vulkan), gpu_b);
        options.rendering_vulkan_device = gpu_a;
        for (const auto backend : {GpuBackend::CUDA, GpuBackend::Vulkan, GpuBackend::Metal})
            EXPECT_EQ(rendering_device_preference(options, backend), gpu_a);
    }

    TEST_F(GpuConfigurationPolicy, StableIdentitySurvivesReorderingAndDuplicateNames) {
        std::swap(devices[0], devices[1]);
        devices[0].index = 0;
        devices[1].index = 1;
        EXPECT_EQ(find_gpu_device<VulkanHardwareDevice>(devices, gpu_a), 1u);
        EXPECT_EQ(find_gpu_device<VulkanHardwareDevice>(devices, "0"), 0u);
        EXPECT_EQ(find_gpu_device<VulkanHardwareDevice>(devices, "GPU-{AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA}"), 1u);
        EXPECT_FALSE(find_gpu_device<VulkanHardwareDevice>(devices, "missing"));
        EXPECT_FALSE(find_gpu_device<VulkanHardwareDevice>(devices, "99"));
    }

    TEST_F(GpuConfigurationPolicy, ActiveGpuNamesDistinguishIdenticalCards) {
        EXPECT_EQ(gpuPreferenceDeviceName(gpu_a, devices, "unknown"), "Identical GPU name (#0, aaaaaaaa)");
        EXPECT_EQ(gpuPreferenceDeviceName(gpu_b, devices, "unknown"), "Identical GPU name (#1, bbbbbbbb)");
        EXPECT_EQ(gpuPreferenceDeviceName("missing", devices, "driver name"), "driver name");
        devices.resize(1);
        EXPECT_EQ(gpuPreferenceDeviceName(gpu_a, devices, "unknown"), "Identical GPU name");
    }

    TEST_F(GpuConfigurationPolicy, BackendChangeRequiresRestartAndRestoringActiveClearsIt) {
        auto requested = startup;
        requested.backend = GpuBackend::Vulkan;
        EXPECT_TRUE(pending(requested));
        requested.backend = GpuBackend::CUDA;
        requested.options.cuda_device = gpu_a;
        EXPECT_FALSE(pending(requested));
        requested.backend.reset();
        EXPECT_FALSE(pending(requested));
    }

    TEST_F(GpuConfigurationPolicy, SameBackendDifferentGpuRequiresRestart) {
        auto requested = startup;
        requested.options.cuda_device = gpu_b;
        EXPECT_TRUE(pending(requested));
        requested.options.cuda_device = "0";
        EXPECT_FALSE(pending(requested));
        requested.options.cuda_device = "GPU-{AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA}";
        EXPECT_FALSE(pending(requested));
    }

    TEST_F(GpuConfigurationPolicy, VulkanGpuChangeAndRestoreUseActualIdentity) {
        startup.backend = running.backend = GpuBackend::Vulkan;
        auto requested = startup;
        requested.options.vulkan_device = gpu_b;
        EXPECT_TRUE(pending(requested));
        requested.options.vulkan_device = "0";
        EXPECT_FALSE(pending(requested));
    }

    TEST_F(GpuConfigurationPolicy, RemovedGpuDoesNotLoseSavedRequestOrCreateStartupRestartLoop) {
        startup.backend = GpuBackend::Vulkan;
        startup.options.vulkan_device = "removed-device";
        EXPECT_FALSE(pending(startup));
        auto requested = startup;
        requested.backend = GpuBackend::CUDA;
        requested.options.cuda_device = gpu_a;
        EXPECT_FALSE(pending(requested));
        EXPECT_EQ(startup.options.vulkan_device, "removed-device");
    }

    TEST_F(GpuConfigurationPolicy, ActivatingAnInactiveSavedDeviceIsComparedWithRunningGpu) {
        startup.backend = GpuBackend::Vulkan;
        startup.options.cuda_device = gpu_b;
        // Vulkan failed at startup; the current CUDA fallback uses GPU A.
        EXPECT_FALSE(pending(startup));
        auto requested = startup;
        requested.backend = GpuBackend::CUDA;
        EXPECT_TRUE(pending(requested));
        requested.options.cuda_device = gpu_a;
        EXPECT_FALSE(pending(requested));
    }

    TEST_F(GpuConfigurationPolicy, MissingNewDeviceRequiresRestart) {
        auto requested = startup;
        requested.options.cuda_device = "removed-device";
        EXPECT_TRUE(pending(requested));
    }

    TEST_F(GpuConfigurationPolicy, InactiveBackendGpuPreferenceDoesNotRequestRestart) {
        auto requested = startup;
        requested.options.vulkan_device = gpu_b;
        requested.options.force_fp32_half = true;
        EXPECT_FALSE(pending(requested));
    }

    TEST_F(GpuConfigurationPolicy, AutomaticDeviceIsComparedWithItsRealDefault) {
        startup.options.cuda_device = running.cuda_uuid = gpu_b;
        auto requested = startup;
        requested.options.cuda_device.clear();
        EXPECT_TRUE(pending(requested));
        requested.options.cuda_device = gpu_b;
        EXPECT_FALSE(pending(requested));
    }

    TEST_F(GpuConfigurationPolicy, RendererGpuIsIndependentAndRestoringActiveClearsIt) {
        auto requested = startup;
        requested.options.rendering_vulkan_device = gpu_b;
        EXPECT_TRUE(pending(requested));
        requested.options.rendering_vulkan_device = gpu_a;
        EXPECT_FALSE(pending(requested));
        requested.options.rendering_vulkan_device.clear();
        EXPECT_FALSE(pending(requested));
    }

    TEST_F(GpuConfigurationPolicy, DiagnosticsRestoreAndInactiveMetalOptionsAreHandled) {
        startup.backend = running.backend = GpuBackend::Vulkan;
        auto requested = startup;
        requested.options.force_no_atomic_float = true;
        EXPECT_TRUE(pending(requested));
        requested.options.force_no_atomic_float = false;
        EXPECT_FALSE(pending(requested));
        requested.options.vulkan_validation = 2;
        EXPECT_TRUE(pending(requested));
        requested.options.vulkan_validation = 0;
        EXPECT_FALSE(pending(requested));
        startup.backend = running.backend = GpuBackend::Metal;
        running.vulkan_renderer = false;
        requested = startup;
        requested.options.vulkan_validation = 2;
        requested.options.rendering_vulkan_device = gpu_b;
        EXPECT_FALSE(pending(requested));
    }
} // namespace
