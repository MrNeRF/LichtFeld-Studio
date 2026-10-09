/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/session_vulkan.hpp"

#include "core/logger.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <format>
#include <mutex>
#include <utility>

namespace lfs::training {
    namespace {
        using namespace lfs::core::internal;

        struct Arena {
            std::mutex mutex;
            std::condition_variable released;
            std::shared_ptr<VulkanContext> context;
            StorageRef storage{};
            size_t capacity = 0;
            size_t required = 0;
            size_t current = 0;
            size_t peak = 0;
            uint64_t next_frame = 1;
            uint64_t active_frame = 0;
        };

        thread_local uint32_t arena_timeout_ms = 0;

        Arena& arena() {
            static Arena value;
            return value;
        }

        size_t bucket_bytes(const size_t bytes) {
            constexpr size_t k256K = 256 * 1024;
            constexpr size_t k1M = 1024 * 1024;
            constexpr size_t k16M = 16 * k1M;
            constexpr size_t k256M = 256 * k1M;
            constexpr size_t k1G = 1024 * k1M;
            constexpr size_t k8G = 8 * k1G;
            const auto round_up = [](const size_t value, const size_t multiple) {
                return ((value + multiple - 1) / multiple) * multiple;
            };
            if (bytes <= k256K)
                return k256K;
            if (bytes <= k1M)
                return round_up(bytes, k256K);
            if (bytes <= k16M)
                return round_up(bytes, k1M);
            if (bytes <= k256M)
                return round_up(bytes, k16M);
            if (bytes <= k1G)
                return round_up(bytes, 64 * k1M);
            if (bytes <= k8G)
                return round_up(bytes, 256 * k1M);
            return round_up(bytes, k1G);
        }

        // Vulkan has no capture-window API like the CUDA profiler start and stop;
        // GPU timing comes from timestamp queries instead.
        void profile(bool) {}
        void reset_arena();
        void resize_arena(std::string_view, const bool release_all) {
            if (release_all) {
                reset_arena();
                return;
            }
            Arena& state = arena();
            std::unique_lock lock(state.mutex);
            state.released.wait(lock, [&] { return state.active_frame == 0; });
            if (state.context)
                state.context->recorders().wait_all();
            const size_t target = state.required == 0
                                      ? 0
                                      : std::min(state.capacity, bucket_bytes(state.required));
            if (target == state.capacity)
                return;
            if (state.context && state.storage.meta != nullptr) {
                if (target == 0) {
                    state.context->memory().deallocate(state.storage);
                    state.storage = {};
                    state.context.reset();
                } else {
                    const StorageRef replacement = state.context->memory().allocate(target, 16, {});
                    state.context->memory().deallocate(state.storage);
                    state.storage = replacement;
                }
            }
            state.capacity = target;
        }
        void reset_arena() {
            Arena& state = arena();
            std::unique_lock lock(state.mutex);
            state.released.wait(lock, [&] { return state.active_frame == 0; });
            if (state.context && state.storage.meta != nullptr) {
                state.context->recorders().wait_all();
                // This arena owns its VMA block; dropping it releases the pool backing.
                state.context->memory().deallocate(state.storage);
            }
            state.storage = {};
            state.context.reset();
            state.capacity = 0;
            state.required = 0;
            state.current = 0;
            state.peak = 0;
        }
        std::optional<lfs::gpu_ops::ArenaMemoryInfo> arena_memory_info() {
            Arena& state = arena();
            std::lock_guard lock(state.mutex);
            if (state.context == nullptr && state.capacity == 0)
                return std::nullopt;
            return lfs::gpu_ops::ArenaMemoryInfo{
                state.capacity, state.required, state.current, state.peak};
        }
        void log_arena_failure(const char* label) {
            Arena& state = arena();
            std::lock_guard lock(state.mutex);
            const size_t allocated = state.context
                                         ? state.context->memory().stats().allocated_bytes
                                         : 0;
            const size_t cached = state.context ? state.context->memory().cached_bytes() : 0;
            LOG_ERROR("Vulkan training arena failure label={} capacity={} required={} current={} "
                      "peak={} device_heap_usage={} pool_cached={}",
                      label == nullptr ? "unnamed" : label, state.capacity,
                      state.required, state.current, state.peak,
                      allocated, cached);
        }
        uint32_t set_arena_timeout(const uint32_t value) {
            return std::exchange(arena_timeout_ms, value);
        }
        lfs::gpu_ops::IdleArenaBorrow borrow_idle_arena(
            const size_t wanted, const size_t minimum, core::TensorExecutionTarget) {
            Arena& state = arena();
            std::unique_lock lock(state.mutex);
            const size_t bytes = std::min(wanted, state.capacity);
            if (bytes == 0 || bytes < minimum)
                return {};
            const auto ready = [&] { return state.active_frame == 0; };
            if (arena_timeout_ms == 0) {
                state.released.wait(lock, ready);
            } else if (!state.released.wait_for(lock, std::chrono::milliseconds(arena_timeout_ms), ready)) {
                return {};
            }
            state.required = wanted;
            state.current = bytes;
            state.peak = std::max(state.peak, bytes);
            state.active_frame = state.next_frame++;
            return {&state, reinterpret_cast<char*>(vk::address(state.storage)), bytes, state.active_frame};
        }
        void release_idle_arena(const lfs::gpu_ops::IdleArenaBorrow& borrow, core::TensorExecutionTarget) {
            if (borrow.owner != &arena())
                return;
            Arena& state = arena();
            std::shared_ptr<VulkanContext> context;
            {
                std::lock_guard lock(state.mutex);
                if (borrow.frame != state.active_frame)
                    return;
                context = state.context;
            }
            if (context)
                context->recorders().wait_all();
            std::lock_guard lock(state.mutex);
            if (borrow.frame == state.active_frame) {
                state.current = 0;
                state.active_frame = 0;
                state.released.notify_all();
            }
        }
        void* zero_idle_arena(const lfs::gpu_ops::IdleArenaBorrow& borrow, const size_t bytes,
                              core::TensorExecutionTarget) {
            if (bytes == 0 || bytes > borrow.bytes || borrow.owner != &arena())
                return nullptr;
            Arena& state = arena();
            std::lock_guard lock(state.mutex);
            if (borrow.frame != state.active_frame || !state.context)
                return nullptr;
            state.context->memory().memset({
                .dst = state.storage,
                .bytes = bytes,
                .value = 0,
                .synchronous = false,
                .context = {},
                .operation = "training.session.zero_arena",
            });
            return borrow.data;
        }
        std::optional<std::string> last_error() {
            if (const auto context = try_live_vulkan_context()) {
                context->recorders().wait_all();
                const auto fault = context->consume_fault_record();
                if (fault[0] != 0)
                    return std::format("Vulkan device fault {} at {}", fault[0], fault[1]);
            }
            return std::nullopt;
        }
    } // namespace

    const lfs::gpu_ops::SessionOps& vulkan_session_ops() {
        static const lfs::gpu_ops::SessionOps ops{
            .profile = profile,
            .resize_arena = resize_arena,
            .reset_arena = reset_arena,
            .arena_memory_info = arena_memory_info,
            .log_arena_failure = log_arena_failure,
            .set_arena_timeout = set_arena_timeout,
            .borrow_idle_arena = borrow_idle_arena,
            .release_idle_arena = release_idle_arena,
            .zero_idle_arena = zero_idle_arena,
            .allocation_bytes = bucket_bytes,
            .direct_storage_live_bytes = core::Tensor::vulkan_external_storage_live_bytes,
            .last_error = last_error,
            .device_baseline_bytes = [] { return diagnostics::VramProfiler::instance().snapshot().process.vulkan_vma_used; },
            .sample_memory = [] {
                if (const auto context = try_live_vulkan_context()) {
                    const auto stats = context->memory().stats();
                    diagnostics::VramProfiler::instance().setVulkanVmaUsed(stats.allocated_bytes);
                } },
            .release_workspaces_before_evaluation = true,
            .release_workspaces_before_reorder = true,
        };
        return ops;
    }
} // namespace lfs::training
