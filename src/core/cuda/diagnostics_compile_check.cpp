/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_memory_guard.hpp"
#include "core/tensor/internal/gpu_config.hpp"
#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "core/tensor_trace.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "memory_arena.hpp"
#include "rendering/rasterizer/vulkan/src/config.h"
#include "visualizer/window/vulkan_result.hpp"

#include <cuda_runtime.h>

#include <string>

namespace lfs::core::compile_check {

    namespace {

        void compile_cpp_logger_arities(const int device) {
            LOG_TRACE("trace zero");
            LOG_TRACE("trace one={}", device);
            LOG_TRACE("trace many={}/{}/{}", device, 1, 2);
            LOG_DEBUG("debug zero");
            LOG_DEBUG("debug one={}", device);
            LOG_DEBUG("debug many={}/{}/{}", device, 1, 2);
            LOG_INFO("info zero");
            LOG_INFO("info one={}", device);
            LOG_INFO("info many={}/{}/{}", device, 1, 2);
            LOG_PERF("perf zero");
            LOG_PERF("perf one={}", device);
            LOG_PERF("perf many={}/{}/{}", device, 1, 2);
            LOG_WARN("warn zero");
            LOG_WARN("warn one={}", device);
            LOG_WARN("warn many={}/{}/{}", device, 1, 2);
            LOG_ERROR("error zero");
            LOG_ERROR("error one={}", device);
            LOG_ERROR("error many={}/{}/{}", device, 1, 2);
            LOG_CRITICAL("critical zero");
            LOG_CRITICAL("critical one={}", device);
            LOG_CRITICAL("critical many={}/{}/{}", device, 1, 2);
        }

        void compile_cpp_macro_surface(void* device_pointer,
                                       const int device) {
            LFS_ASSERT(device >= -1);
            LFS_ASSERT_MSG(device_pointer != nullptr, "compile-check pointer");
            LFS_DEBUG_ASSERT(device >= -1);
            LFS_DEBUG_ASSERT_MSG(device_pointer != nullptr, "compile-check debug pointer");

            LOG_TIMER("compile-check timer");
            LOG_TIMER_THRESHOLD("compile-check threshold timer", 0.0);
            LOG_TIMER_TRACE("compile-check trace timer");
            LOG_TIMER_DEBUG("compile-check debug timer");

            const TensorShape shape{1};
            {
                TRACE_OP("compile-check trace", shape);
                TRACE_OP_OUTPUT(shape);
            }
            {
                TRACE_OP2("compile-check trace pair", shape, shape);
                TRACE_OP_OUTPUT(shape);
            }
            TRACE_PRINT_STACK();
            TRACE_PRINT_HISTORY(1);

            LFS_VRAM_SCOPE("compile-check VRAM scope");
            LOG_VRAM_DIFF("compile-check VRAM delta");
            LFS_TRACE("compile-check CPU trace");
            LFS_GPU_TIME("compile-check GPU timer", static_cast<cudaStream_t>(nullptr));
            LFS_GAUGE("compile-check.gauge", device);
            LFS_COUNTER_ADD("compile-check.iter", 1);
            LFS_COUNTER_TOTAL_ADD("compile-check.total", 1);
            LFS_HIST("compile-check.histogram", device);

            LFS_CUDA_BREADCRUMB("compile-check breadcrumb");
            LFS_CUDA_BREADCRUMB_STREAM(
                "compile-check stream breadcrumb", static_cast<cudaStream_t>(nullptr));
        }

        [[nodiscard]] bool compile_vk_result_macros(const VkResult result,
                                                    const int value) {
            if (!lfs::vis::vk_try_bool(
                    result,
                    "result",
                    lfs::rendering::formatVulkanDiagnostic("compile-check Vulkan zero arguments"),
                    std::source_location::current())) {
                return false;
            }
            if (!lfs::vis::vk_try_bool(
                    result,
                    "result",
                    lfs::rendering::formatVulkanDiagnostic(
                        "compile-check Vulkan one={}",
                        value),
                    std::source_location::current())) {
                return false;
            }
            if (!lfs::vis::vk_try_bool(
                    result,
                    "result",
                    lfs::rendering::formatVulkanDiagnostic(
                        "compile-check Vulkan many={}/{}/{}",
                        value,
                        1,
                        2),
                    std::source_location::current())) {
                return false;
            }
            return true;
        }

        class VkContextCompileCheck {
        public:
            [[nodiscard]] bool compile(const VkResult result,
                                       const int value) {
                LFS_VK_CONTEXT_CHECK_MSG(
                    result, "compile-check Vulkan context zero arguments");
                LFS_VK_CONTEXT_CHECK_MSG(
                    result, "compile-check Vulkan context one={}", value);
                LFS_VK_CONTEXT_CHECK_MSG(
                    result, "compile-check Vulkan context many={}/{}/{}", value, 1, 2);
                return true;
            }

        private:
            [[nodiscard]] bool setVkFailure(std::string) { return false; }
        };

        void compile_vk_debug_assert_arities(const int value) {
            LFS_VK_DEBUG_ASSERT(value >= -1, "compile-check Vulkan assert zero arguments");
            LFS_VK_DEBUG_ASSERT(
                value >= -1, "compile-check Vulkan assert one={}", value);
            LFS_VK_DEBUG_ASSERT(
                value >= -1, "compile-check Vulkan assert many={}/{}/{}", value, 1, 2);
        }

    } // namespace

    // This function is intentionally never run. The source is compiled with
    // MSVC's traditional preprocessor even though production C++ targets use
    // /Zc:preprocessor, so every caller macro remains independently portable.
    void compile_cpp_host_diagnostics(void* device_pointer) {
        int device = -1;
        LFS_CUDA_CHECK(cudaGetDevice(&device));
        LFS_CUDA_CHECK_MSG(cudaSuccess, "compile-check zero arguments");
        LFS_CUDA_CHECK_MSG(cudaSuccess, "compile-check device={}", device);
        LFS_CUDA_CHECK_MSG(
            cudaGetLastError(), "compile-check many={}/{}/{}", device, 1, 2);

        compile_cpp_logger_arities(device);
        compile_cpp_macro_surface(device_pointer, device);
        (void)compile_vk_result_macros(VK_SUCCESS, device);
        VkContextCompileCheck vk_context;
        (void)vk_context.compile(VK_SUCCESS, device);
        compile_vk_debug_assert_arities(device);

        if (device < -1) {
            lfs::rendering::throw_renderer_contract(std::string{"compile-check Vulkan throw wrapper"}, LFS_SOURCE_SITE_CURRENT());
        }

        LFS_ENSURE_CUDA_SUCCESS(cudaSuccess, "compile-check status");
        LFS_ENSURE_CUDA_SUCCESS_MSG(
            cudaSuccess, "compile-check status with context", "context=MSVC host path");
        const CudaCheckState state{};
        LFS_ENSURE_CUDA_SUCCESS_STATE(
            cudaSuccess, state, "compile-check stateful status", "context=MSVC host path");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(device_pointer, "compile-check pointer");
        LFS_VALIDATE_CUDA_DEVICE_POINTER_OPTIONAL(nullptr, "compile-check optional pointer");

        static_assert(sizeof(RasterizerMemoryArena) > 0);
        static_assert(sizeof(CudaDeviceMemory<int>) > 0);
    }

} // namespace lfs::core::compile_check
