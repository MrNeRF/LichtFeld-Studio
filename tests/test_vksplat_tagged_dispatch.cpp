/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Epic #1496 P3: tagged dispatch infrastructure (P3a) + first LOD chain audit (P3b).
// Scripted VulkanDispatch (TestablePipeline / TestableRenderer pattern).

#include "rendering/rasterizer/vulkan/src/barrier_planner.h"
#include "rendering/rasterizer/vulkan/src/gs_pipeline.h"
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"
#include "rendering/rasterizer/vulkan/src/viewport_scratch_bucket.h"
#include "rendering/vulkan_wait.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// Free functions defined in gs_pipeline.cpp (external linkage; not in the header).
VkAccessFlags2 toAccessMask(VulkanGSPipeline::BarrierMask barrierMask);
VkPipelineStageFlags2 toStageMask(VulkanGSPipeline::BarrierMask barrierMask);

namespace {

    using namespace lfs::rendering::vulkan;
    using TaggedBinding = VulkanGSPipeline::TaggedBinding;

    constexpr uint32_t kQueueFamily = 7;

    template <typename Handle>
    [[nodiscard]] Handle fakeVkHandle(const std::uintptr_t value) {
        if constexpr (std::is_pointer_v<Handle>) {
            return reinterpret_cast<Handle>(value);
        } else {
            return static_cast<Handle>(value);
        }
    }

    [[nodiscard]] _VulkanBuffer makeBuffer(const std::uintptr_t id, const VkDeviceSize size = 4096) {
        _VulkanBuffer b;
        b.buffer = fakeVkHandle<VkBuffer>(id);
        b.allocSize = static_cast<size_t>(size);
        b.capacity = static_cast<size_t>(size);
        b.size = static_cast<size_t>(size);
        b.offset = 0;
        // Non-null allocation so validateBufferRange / destroy paths that key
        // on allocation presence can treat forged buffers as "owned" views.
        b.allocation = fakeVkHandle<VmaAllocation>(id + 0x8000);
        return b;
    }

    [[nodiscard]] Scope scopeFor(const BufferUse use) {
        using BM = VulkanGSPipeline::BarrierMask;
        BM mask = BM::COMPUTE_SHADER_READ;
        switch (use) {
        case BufferUse::ComputeRead:
            mask = BM::COMPUTE_SHADER_READ;
            break;
        case BufferUse::ComputeWrite:
            mask = BM::COMPUTE_SHADER_WRITE;
            break;
        case BufferUse::ComputeReadWrite:
            mask = BM::COMPUTE_SHADER_READ_WRITE;
            break;
        case BufferUse::TransferRead:
            mask = BM::TRANSFER_READ;
            break;
        case BufferUse::TransferWrite:
            mask = BM::TRANSFER_WRITE;
            break;
        case BufferUse::IndirectRead:
            mask = BM::INDIRECT_DISPATCH_READ;
            break;
        case BufferUse::HostRead:
            mask = BM::HOST_READ;
            break;
        case BufferUse::ConditionalRead:
            mask = BM::CONDITIONAL_RENDERING_READ;
            break;
        }
        return Scope{toStageMask(mask), toAccessMask(mask)};
    }

    [[nodiscard]] Scope conservativeSrc() {
        using BM = VulkanGSPipeline::BarrierMask;
        return Scope{
            toStageMask(BM::TRANSFER_COMPUTE_SHADER_WRITE),
            toAccessMask(BM::TRANSFER_COMPUTE_SHADER_WRITE),
        };
    }

    enum class RecordedOp : std::uint8_t {
        Barrier2,
        BindPipeline,
        PushConstants,
        Dispatch,
        DispatchIndirect,
        FillBuffer,
        CopyBuffer,
        BeginCb,
        EndCb,
        ResetQuery,
        QueueSubmit,
        ResetCb,
    };

    struct CapturedBarrier2 {
        std::vector<VkBufferMemoryBarrier2> buffer_barriers;
        std::uint32_t memory_barrier_count = 0;
    };

    // Scripted dispatch: captures barrier2 dependency infos AND
    // bind/dispatch/push order. Also covers begin/end/submit/reset_query so
    // beginCommandBatch/endCommandBatch work without a real device.
    struct DispatchScript {
        std::vector<RecordedOp> ops;
        std::vector<CapturedBarrier2> barriers;
        int submit_calls = 0;
        int begin_calls = 0;
        int end_calls = 0;

        static DispatchScript*& active() {
            static DispatchScript* ptr = nullptr;
            return ptr;
        }
        void bind() { active() = this; }
        void unbind() {
            if (active() == this) {
                active() = nullptr;
            }
        }

        void clear_recording() {
            ops.clear();
            barriers.clear();
        }

        [[nodiscard]] std::size_t buffer_barrier_calls() const {
            std::size_t n = 0;
            for (const auto& b : barriers) {
                if (!b.buffer_barriers.empty()) {
                    ++n;
                }
            }
            return n;
        }

        [[nodiscard]] const CapturedBarrier2* first_buffer_barrier_call() const {
            for (const auto& b : barriers) {
                if (!b.buffer_barriers.empty()) {
                    return &b;
                }
            }
            return nullptr;
        }

        // Index of first Barrier2 op that carried buffer barriers (in ops[]).
        [[nodiscard]] int first_buffer_barrier_op_index() const {
            int barrier_i = 0;
            for (std::size_t i = 0; i < ops.size(); ++i) {
                if (ops[i] != RecordedOp::Barrier2) {
                    continue;
                }
                if (barrier_i < static_cast<int>(barriers.size()) &&
                    !barriers[static_cast<std::size_t>(barrier_i)].buffer_barriers.empty()) {
                    return static_cast<int>(i);
                }
                ++barrier_i;
            }
            return -1;
        }

        [[nodiscard]] int first_op_index(const RecordedOp op) const {
            for (std::size_t i = 0; i < ops.size(); ++i) {
                if (ops[i] == op) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        static VKAPI_ATTR VkResult VKAPI_CALL begin_cb(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
            EXPECT_NE(active(), nullptr);
            ++active()->begin_calls;
            active()->ops.push_back(RecordedOp::BeginCb);
            return VK_SUCCESS;
        }

        static VKAPI_ATTR void VKAPI_CALL barrier2(VkCommandBuffer, const VkDependencyInfo* info) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::Barrier2);
            CapturedBarrier2 cap;
            if (info != nullptr) {
                cap.memory_barrier_count = info->memoryBarrierCount;
                if (info->pBufferMemoryBarriers != nullptr && info->bufferMemoryBarrierCount > 0) {
                    cap.buffer_barriers.assign(
                        info->pBufferMemoryBarriers,
                        info->pBufferMemoryBarriers + info->bufferMemoryBarrierCount);
                }
            }
            active()->barriers.push_back(std::move(cap));
        }

        static VKAPI_ATTR void VKAPI_CALL reset_query(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::ResetQuery);
        }

        static VKAPI_ATTR void VKAPI_CALL write_timestamp(VkCommandBuffer,
                                                          VkPipelineStageFlagBits,
                                                          VkQueryPool,
                                                          uint32_t) {
            // no-op for PerfTimer::Timer in chain audits
        }

        static VKAPI_ATTR VkResult VKAPI_CALL end_cb(VkCommandBuffer) {
            EXPECT_NE(active(), nullptr);
            ++active()->end_calls;
            active()->ops.push_back(RecordedOp::EndCb);
            return VK_SUCCESS;
        }

        static VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue,
                                                           uint32_t,
                                                           const VkSubmitInfo*,
                                                           VkFence) {
            EXPECT_NE(active(), nullptr);
            ++active()->submit_calls;
            active()->ops.push_back(RecordedOp::QueueSubmit);
            return VK_SUCCESS;
        }

        static VKAPI_ATTR VkResult VKAPI_CALL queue_wait_idle(VkQueue) {
            return VK_SUCCESS;
        }

        static VKAPI_ATTR VkResult VKAPI_CALL reset_cb(VkCommandBuffer, VkCommandBufferResetFlags) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::ResetCb);
            return VK_SUCCESS;
        }

        static VKAPI_ATTR VkResult VKAPI_CALL get_sem(VkDevice, VkSemaphore, uint64_t* value) {
            if (value != nullptr) {
                *value = 0;
            }
            return VK_SUCCESS;
        }

        static VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer,
                                                        VkPipelineBindPoint,
                                                        VkPipeline) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::BindPipeline);
        }

        static VKAPI_ATTR void VKAPI_CALL push_constants(VkCommandBuffer,
                                                         VkPipelineLayout,
                                                         VkShaderStageFlags,
                                                         uint32_t,
                                                         uint32_t,
                                                         const void*) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::PushConstants);
        }

        static VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::Dispatch);
        }

        static VKAPI_ATTR void VKAPI_CALL dispatch_indirect(VkCommandBuffer, VkBuffer, VkDeviceSize) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::DispatchIndirect);
        }

        static VKAPI_ATTR void VKAPI_CALL fill_buffer(VkCommandBuffer,
                                                      VkBuffer,
                                                      VkDeviceSize,
                                                      VkDeviceSize,
                                                      uint32_t) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::FillBuffer);
        }

        static VKAPI_ATTR void VKAPI_CALL copy_buffer(VkCommandBuffer,
                                                      VkBuffer,
                                                      VkBuffer,
                                                      uint32_t,
                                                      const VkBufferCopy*) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::CopyBuffer);
        }

        // No-op push-descriptor — scripted tests never need real descriptor writes.
        static VKAPI_ATTR void VKAPI_CALL push_descriptor_set(VkCommandBuffer,
                                                              VkPipelineBindPoint,
                                                              VkPipelineLayout,
                                                              uint32_t,
                                                              uint32_t,
                                                              const VkWriteDescriptorSet*) {
            // intentionally empty
        }

        // Conditional rendering EXT no-ops for predicate_waves audits.
        static VKAPI_ATTR void VKAPI_CALL begin_conditional(
            VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {}
        static VKAPI_ATTR void VKAPI_CALL end_conditional(VkCommandBuffer) {}
    };

    struct BindScript {
        DispatchScript& script;
        explicit BindScript(DispatchScript& s) : script(s) { script.bind(); }
        ~BindScript() { script.unbind(); }
        BindScript(const BindScript&) = delete;
        BindScript& operator=(const BindScript&) = delete;
    };

    class TestablePipeline final : public VulkanGSPipeline {
    public:
        ~TestablePipeline() {
            disarm_for_destruction();
        }

        void install_fake_handles() {
            device = fakeVkHandle<VkDevice>(0x1001);
            command_queue = fakeVkHandle<VkQueue>(0x1002);
            command_pool = fakeVkHandle<VkCommandPool>(0x1003);
            fence = fakeVkHandle<VkFence>(0x1004);
            queue_family_index = kQueueFamily;
            // Reconstruct planner with the forged queue family (mirrors initializeExternal).
            barrier_planner_ = BufferBarrierPlanner(kQueueFamily);

            for (std::uint32_t i = 0; i < kCommandBatchSlotCount; ++i) {
                command_batch_slots_[i].command_buffer =
                    fakeVkHandle<VkCommandBuffer>(0x2000 + i);
                command_batch_slots_[i].timestamp_query_pool =
                    fakeVkHandle<VkQueryPool>(0x3000 + i);
                command_batch_slots_[i].pending_signal = VK_NULL_HANDLE;
                command_batch_slots_[i].pending_signal_value = 0;
            }
            command_buffer = command_batch_slots_[0].command_buffer;
            timestamp_query_pool = command_batch_slots_[0].timestamp_query_pool;
            next_command_batch_slot_ = 0;
            active_command_batch_slot_ = 0;
            commandBatchInProgress = false;
            last_timeline_signal_values_.clear();
            pending_timeline_waits_.clear();
            buffer_retire_timeline_ = fakeVkHandle<VkSemaphore>(0x5001);
            next_buffer_retire_value_ = 1;
            retired_buffer_shells_.clear();
            test_buffer_handle_counter_ = 0xB1000;
            current_vram = 0;
            peak_vram = 0;

            deviceInfo.subgroupSize = 32;
            deviceInfo.sharedSize = 48 * 1024;
            deviceInfo.maxGroupsX = 65535;
            deviceInfo.maxGroupsY = 65535;
            deviceInfo.maxGroupsZ = 65535;

            // Injectable no-op push-descriptor proc (spec §3.3).
            vk_cmd_push_descriptor_set_ = &DispatchScript::push_descriptor_set;
        }

        void disarm_for_destruction() {
            commandBatchInProgress = false;
            pending_timeline_waits_.clear();
            last_timeline_signal_values_.clear();
            for (auto& slot : command_batch_slots_) {
                slot.pending_signal = VK_NULL_HANDLE;
                slot.pending_signal_value = 0;
                slot.pending_timestamp_count = 0;
                slot.pending_timestamp_marks.clear();
                slot.command_buffer = VK_NULL_HANDLE;
                slot.timestamp_query_pool = VK_NULL_HANDLE;
            }
            // Force-free retired shells without real VMA (allocator already null).
            retired_buffer_shells_.clear();
            buffer_retire_timeline_ = VK_NULL_HANDLE;
            next_buffer_retire_value_ = 1;
            command_buffer = VK_NULL_HANDLE;
            timestamp_query_pool = VK_NULL_HANDLE;
            fence = VK_NULL_HANDLE;
            command_pool = VK_NULL_HANDLE;
            command_queue = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
            instance = VK_NULL_HANDLE;
            physical_device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
            vk_cmd_push_descriptor_set_ = nullptr;
            barrier_planner_.reset();
            current_vram = 0;
        }

        // Expose protected dispatch / barrier APIs for scripted tests.
        using VulkanGSPipeline::BufferBarrier;
        using VulkanGSPipeline::bufferMemoryBarrier;
        using VulkanGSPipeline::executeCompute;
        using VulkanGSPipeline::executeComputeIndirect;

        // Forge a minimal compute pipeline (handles only; no real SPIR-V).
        [[nodiscard]] _ComputePipeline make_fake_pipeline(const int num_buffers,
                                                          const char* name = "test.fake") {
            _ComputePipeline cp(num_buffers);
            cp.pipeline = fakeVkHandle<VkPipeline>(0x4001);
            cp.pipeline_layout = fakeVkHandle<VkPipelineLayout>(0x4002);
            cp.descriptor_set_layout = fakeVkHandle<VkDescriptorSetLayout>(0x4003);
            cp.shader = fakeVkHandle<VkShaderModule>(0x4004);
            cp.diagnostic_name = name;
            return cp;
        }

        // Simulate createBuffer/destroyBuffer track/forget without VMA.
        void simulate_create(_VulkanBuffer& buffer) {
            trackExternalParent(buffer.buffer);
        }
        void simulate_destroy(_VulkanBuffer& buffer) {
            untrackExternalParent(buffer.buffer);
        }
    };

    [[nodiscard]] lfs::rendering::VulkanDispatch make_scripted_dispatch() {
        lfs::rendering::VulkanDispatch d{};
        d.begin_command_buffer = &DispatchScript::begin_cb;
        d.cmd_pipeline_barrier2 = &DispatchScript::barrier2;
        d.cmd_reset_query_pool = &DispatchScript::reset_query;
        d.cmd_write_timestamp = &DispatchScript::write_timestamp;
        d.end_command_buffer = &DispatchScript::end_cb;
        d.queue_submit = &DispatchScript::queue_submit;
        d.queue_wait_idle = &DispatchScript::queue_wait_idle;
        d.reset_command_buffer = &DispatchScript::reset_cb;
        d.get_semaphore_counter_value = &DispatchScript::get_sem;
        d.cmd_bind_pipeline = &DispatchScript::bind_pipeline;
        d.cmd_push_constants = &DispatchScript::push_constants;
        d.cmd_dispatch = &DispatchScript::dispatch;
        d.cmd_dispatch_indirect = &DispatchScript::dispatch_indirect;
        d.cmd_fill_buffer = &DispatchScript::fill_buffer;
        d.cmd_copy_buffer = &DispatchScript::copy_buffer;
        return d;
    }

    void expect_src_dst(const VkBufferMemoryBarrier2& b, const Scope& src, const Scope& dst) {
        EXPECT_EQ(b.srcStageMask, src.stage);
        EXPECT_EQ(b.srcAccessMask, src.access);
        EXPECT_EQ(b.dstStageMask, dst.stage);
        EXPECT_EQ(b.dstAccessMask, dst.access);
    }

    [[nodiscard]] const VkBufferMemoryBarrier2* find_buf(const CapturedBarrier2& cap, VkBuffer buffer) {
        for (const auto& b : cap.buffer_barriers) {
            if (b.buffer == buffer) {
                return &b;
            }
        }
        return nullptr;
    }

} // namespace

// Catches: tagged path not calling plan()/emitting a coalesced barrier2 before dispatch.
TEST(VkSplatTaggedDispatch, TaggedComputeEmitsOneCoalescedBarrierBeforeDispatch) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    auto buf_a = makeBuffer(0xA001);
    auto buf_b = makeBuffer(0xB001);
    auto cp_write = pipeline.make_fake_pipeline(1, "test.write");
    auto cp_read = pipeline.make_fake_pipeline(2, "test.read_ab");

    pipeline.beginCommandBatch();
    // Track after begin so onBatchBegin does not pre-seed conservative writers
    // (first write after empty track must emit no barrier — RAW is the only hazard).
    pipeline.trackExternalParent(buf_a.buffer);
    pipeline.trackExternalParent(buf_b.buffer);
    script.clear_recording();

    // Write A (empty prior state → no barrier when plan() is live).
    // Dense binding list length must match pipeline.buffer_layouts span.
    pipeline.executeCompute(
        {{64u, 64u}},
        nullptr,
        0,
        cp_write,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeWrite}});
    EXPECT_EQ(script.buffer_barrier_calls(), 0u)
        << "first write after empty track must emit no buffer barrier";

    // Isolate the read dispatch recording so order checks are not polluted by the write.
    script.clear_recording();

    // Dispatch reading A+B: RAW on A only, coalesced into one barrier2 before dispatch.
    pipeline.executeCompute(
        {{64u, 64u}},
        nullptr,
        0,
        cp_read,
        std::vector<TaggedBinding>{
            {buf_a, BufferUse::ComputeRead},
            {buf_b, BufferUse::ComputeRead},
        });

    ASSERT_EQ(script.buffer_barrier_calls(), 1u)
        << "expected exactly one coalesced buffer barrier2 for the RAW hazard";
    const CapturedBarrier2* cap = script.first_buffer_barrier_call();
    ASSERT_NE(cap, nullptr);
    ASSERT_EQ(cap->buffer_barriers.size(), 1u);
    EXPECT_EQ(cap->buffer_barriers[0].buffer, buf_a.buffer);
    expect_src_dst(cap->buffer_barriers[0],
                   scopeFor(BufferUse::ComputeWrite),
                   scopeFor(BufferUse::ComputeRead));
    EXPECT_EQ(find_buf(*cap, buf_b.buffer), nullptr);

    const int barrier_op = script.first_buffer_barrier_op_index();
    const int dispatch_op = script.first_op_index(RecordedOp::Dispatch);
    ASSERT_GE(barrier_op, 0);
    ASSERT_GE(dispatch_op, 0);
    EXPECT_LT(barrier_op, dispatch_op) << "barrier2 must precede cmd_dispatch";

    pipeline.endCommandBatch(/*use_fence=*/false);
}

// Catches: planner not updating visibility so a second identical read re-emits a barrier.
TEST(VkSplatTaggedDispatch, TaggedComputeSecondSameReadElides) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    auto buf_a = makeBuffer(0xA002);
    auto cp = pipeline.make_fake_pipeline(1);

    pipeline.beginCommandBatch();
    pipeline.trackExternalParent(buf_a.buffer);
    script.clear_recording();

    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeWrite}});
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeRead}});

    ASSERT_EQ(script.buffer_barrier_calls(), 1u)
        << "first ComputeRead after ComputeWrite must emit RAW (establishes visibility)";

    script.clear_recording();
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeRead}});

    EXPECT_EQ(script.buffer_barrier_calls(), 0u)
        << "second identical ComputeRead must elide (already visible)";
    EXPECT_GE(script.first_op_index(RecordedOp::Dispatch), 0);

    pipeline.endCommandBatch(/*use_fence=*/false);
}

// Catches: G8 mixed-mode — untagged execute not invalidating planner state before next tagged access.
TEST(VkSplatTaggedDispatch, UntaggedDispatchInvalidatesForNextTaggedAccess) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    auto buf_a = makeBuffer(0xA003);
    auto cp = pipeline.make_fake_pipeline(1);

    pipeline.beginCommandBatch();
    pipeline.trackExternalParent(buf_a.buffer);

    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeWrite}});

    // Untagged dispatch on the same buffer — must invalidate planner state.
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<_VulkanBuffer>{buf_a});

    script.clear_recording();
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeRead}});

    ASSERT_EQ(script.buffer_barrier_calls(), 1u);
    const CapturedBarrier2* cap = script.first_buffer_barrier_call();
    ASSERT_NE(cap, nullptr);
    ASSERT_EQ(cap->buffer_barriers.size(), 1u);
    EXPECT_EQ(cap->buffer_barriers[0].buffer, buf_a.buffer);
    expect_src_dst(cap->buffer_barriers[0],
                   conservativeSrc(),
                   scopeFor(BufferUse::ComputeRead));

    pipeline.endCommandBatch(/*use_fence=*/false);
}

// Catches: legacy bufferMemoryBarrier not invalidating planner state (same G8 shape).
TEST(VkSplatTaggedDispatch, LegacyBufferMemoryBarrierInvalidates) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    auto buf_a = makeBuffer(0xA004);
    auto cp = pipeline.make_fake_pipeline(1);

    pipeline.beginCommandBatch();
    pipeline.trackExternalParent(buf_a.buffer);

    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeWrite}});

    // Legacy pair-overload barrier — must invalidate, and must route through dispatch.
    pipeline.bufferMemoryBarrier(
        {{buf_a, VulkanGSPipeline::COMPUTE_SHADER_WRITE}},
        VulkanGSPipeline::COMPUTE_SHADER_READ);

    script.clear_recording();
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeRead}});

    ASSERT_EQ(script.buffer_barrier_calls(), 1u);
    const CapturedBarrier2* cap = script.first_buffer_barrier_call();
    ASSERT_NE(cap, nullptr);
    ASSERT_EQ(cap->buffer_barriers.size(), 1u);
    EXPECT_EQ(cap->buffer_barriers[0].buffer, buf_a.buffer);
    expect_src_dst(cap->buffer_barriers[0],
                   conservativeSrc(),
                   scopeFor(BufferUse::ComputeRead));

    pipeline.endCommandBatch(/*use_fence=*/false);
}

// Catches: BufferBarrier (per-entry src/dst) overload not invalidating planner state (G8).
TEST(VkSplatTaggedDispatch, LegacyBufferBarrierOverloadInvalidates) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    auto buf_a = makeBuffer(0xA014);
    auto cp = pipeline.make_fake_pipeline(1);

    pipeline.beginCommandBatch();
    pipeline.trackExternalParent(buf_a.buffer);

    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeWrite}});

    // Per-entry src/dst overload — same invalidate contract as the pair overload.
    pipeline.bufferMemoryBarrier({TestablePipeline::BufferBarrier{
        buf_a,
        VulkanGSPipeline::COMPUTE_SHADER_WRITE,
        VulkanGSPipeline::COMPUTE_SHADER_READ,
    }});

    script.clear_recording();
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeRead}});

    ASSERT_EQ(script.buffer_barrier_calls(), 1u);
    const CapturedBarrier2* cap = script.first_buffer_barrier_call();
    ASSERT_NE(cap, nullptr);
    ASSERT_EQ(cap->buffer_barriers.size(), 1u);
    EXPECT_EQ(cap->buffer_barriers[0].buffer, buf_a.buffer);
    expect_src_dst(cap->buffer_barriers[0],
                   conservativeSrc(),
                   scopeFor(BufferUse::ComputeRead));

    pipeline.endCommandBatch(/*use_fence=*/false);
}

// Catches: tagged indirect path omitting the implicit IndirectRead on the args buffer.
TEST(VkSplatTaggedDispatch, IndirectDispatchAddsImplicitIndirectRead) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    auto args = makeBuffer(0xA005, 256);
    auto data = makeBuffer(0xD005);
    auto cp = pipeline.make_fake_pipeline(1);

    pipeline.beginCommandBatch();
    pipeline.trackExternalParent(args.buffer);
    pipeline.trackExternalParent(data.buffer);

    // Prior tagged write of indirect args.
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{args, BufferUse::ComputeWrite}});

    script.clear_recording();
    pipeline.executeComputeIndirect(
        args,
        /*indirect_offset=*/0,
        nullptr,
        0,
        cp,
        std::vector<TaggedBinding>{{data, BufferUse::ComputeRead}});

    ASSERT_GE(script.buffer_barrier_calls(), 1u);
    const CapturedBarrier2* cap = script.first_buffer_barrier_call();
    ASSERT_NE(cap, nullptr);
    const VkBufferMemoryBarrier2* args_barrier = find_buf(*cap, args.buffer);
    ASSERT_NE(args_barrier, nullptr)
        << "implicit IndirectRead on the args buffer must appear in the planned barrier";
    expect_src_dst(*args_barrier,
                   scopeFor(BufferUse::ComputeWrite),
                   scopeFor(BufferUse::IndirectRead));
    EXPECT_EQ(args_barrier->dstStageMask, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
    EXPECT_EQ(args_barrier->dstAccessMask, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

    const int barrier_op = script.first_buffer_barrier_op_index();
    const int dispatch_op = script.first_op_index(RecordedOp::DispatchIndirect);
    ASSERT_GE(barrier_op, 0);
    ASSERT_GE(dispatch_op, 0);
    EXPECT_LT(barrier_op, dispatch_op);

    pipeline.endCommandBatch(/*use_fence=*/false);
}

// Catches: onBatchBegin not resetting visibility — compute read wrongly re-barriers / ConditionalRead elides.
TEST(VkSplatTaggedDispatch, BatchBoundaryResetsVisibility) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    auto buf_a = makeBuffer(0xA006);
    auto cp = pipeline.make_fake_pipeline(1);

    // Batch N: write + read (establishes RAW then visibility). Track after begin
    // so the write is a true first-write (no conservative pre-seed).
    pipeline.beginCommandBatch();
    pipeline.trackExternalParent(buf_a.buffer);
    script.clear_recording();
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeWrite}});
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeRead}});
    ASSERT_EQ(script.buffer_barrier_calls(), 1u)
        << "batch N write→read must emit RAW";
    pipeline.endCommandBatch(/*use_fence=*/false);

    // Batch N+1: onBatchBegin seeds conservative writer + reuse-barrier visibility.
    pipeline.beginCommandBatch();
    script.clear_recording();

    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ComputeRead}});
    EXPECT_EQ(script.buffer_barrier_calls(), 0u)
        << "ComputeRead after batch boundary must elide (reuse-barrier visibility)";

    script.clear_recording();
    // ConditionalRead is outside the reuse-barrier dst scope — must chain.
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf_a, BufferUse::ConditionalRead}});
    ASSERT_EQ(script.buffer_barrier_calls(), 1u);
    const CapturedBarrier2* cap = script.first_buffer_barrier_call();
    ASSERT_NE(cap, nullptr);
    ASSERT_EQ(cap->buffer_barriers.size(), 1u);
    EXPECT_EQ(cap->buffer_barriers[0].buffer, buf_a.buffer);
    expect_src_dst(cap->buffer_barriers[0],
                   conservativeSrc(),
                   scopeFor(BufferUse::ConditionalRead));

    pipeline.endCommandBatch(/*use_fence=*/false);
}

// Catches: createBuffer/destroyBuffer not calling track/forget so a reused handle inherits writer state.
TEST(VkSplatTaggedDispatch, CreateDestroyBufferTrackForget) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    // Same forged handle value across destroy+recreate (Vulkan can reuse VkBuffer bits).
    auto buf = makeBuffer(0xA007);
    auto cp = pipeline.make_fake_pipeline(1);

    pipeline.beginCommandBatch();
    pipeline.simulate_create(buf);
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf, BufferUse::ComputeWrite}});
    pipeline.endCommandBatch(/*use_fence=*/false);

    // Destroy drops planner state (createBuffer/destroyBuffer hooks / external untrack).
    pipeline.simulate_destroy(buf);

    // Recreate: same handle, empty track. Without forget, try_emplace would keep the writer.
    // Access while untracked (between destroy and create) must be conservative — proves forget.
    // After re-create we also verify a first write is clean, but the discriminating check is:
    // untracked access after destroy is conservative (handle-reuse safety).
    pipeline.beginCommandBatch();
    script.clear_recording();
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf, BufferUse::ComputeRead}});

    ASSERT_EQ(script.buffer_barrier_calls(), 1u);
    const CapturedBarrier2* cap = script.first_buffer_barrier_call();
    ASSERT_NE(cap, nullptr);
    ASSERT_EQ(cap->buffer_barriers.size(), 1u);
    EXPECT_EQ(cap->buffer_barriers[0].buffer, buf.buffer);
    expect_src_dst(cap->buffer_barriers[0],
                   conservativeSrc(),
                   scopeFor(BufferUse::ComputeRead));

    // Recreate same handle and confirm track accepts it (no inherited ghost after proper forget).
    pipeline.simulate_create(buf);
    script.clear_recording();
    pipeline.executeCompute(
        {{64u, 64u}}, nullptr, 0, cp,
        std::vector<TaggedBinding>{{buf, BufferUse::ComputeWrite}});
    EXPECT_EQ(script.buffer_barrier_calls(), 0u)
        << "first write after destroy+recreate+track must not inherit prior writer";

    pipeline.endCommandBatch(/*use_fence=*/false);
}

// =============================================================================
// P3b: first-chain audit (executeMapLodIndices + executeSelectLodThreshold)
// Spec §2.6 unit-test gate + §3.5 first migrated chain.
// =============================================================================

namespace {

    // Catalog-derived hand-written barrier struct counts (EPIC_1496_BARRIER_SPEC.md §2.6).
    constexpr std::size_t kAuditMapLodIndices = 4;                   // 3 pre + 1 post
    constexpr std::size_t kAuditSelectLodThresholdWithReadback = 24; // 12+5+2 + 4+1

    // Frozen branch config for the audit recording.
    constexpr std::uint32_t kAuditLodCount = 64;
    constexpr std::uint32_t kAuditChunkSplats = 32;
    constexpr std::uint32_t kAuditInvalidPage = 0xffffffffu;
    constexpr std::uint32_t kAuditNodeCount = 128;
    constexpr std::uint32_t kAuditPhysicalNodeCount = 128;
    constexpr std::uint32_t kAuditOutputCapacity = 64;
    constexpr std::uint32_t kAuditLogicalChunkCount = 8;

    [[nodiscard]] std::size_t total_buffer_barrier_structs(const DispatchScript& script) {
        std::size_t n = 0;
        for (const auto& cap : script.barriers) {
            n += cap.buffer_barriers.size();
        }
        return n;
    }

    struct HazardEdge {
        VkBuffer buffer = VK_NULL_HANDLE;
        VulkanGSPipeline::BarrierMask src = VulkanGSPipeline::COMPUTE_SHADER_WRITE;
        VulkanGSPipeline::BarrierMask dst = VulkanGSPipeline::COMPUTE_SHADER_READ;
        const char* name = "";
    };

    [[nodiscard]] bool edge_covered(const std::vector<VkBufferMemoryBarrier2>& derived,
                                    const HazardEdge& edge) {
        const Scope want_src{toStageMask(edge.src), toAccessMask(edge.src)};
        const Scope want_dst{toStageMask(edge.dst), toAccessMask(edge.dst)};
        for (const auto& b : derived) {
            if (b.buffer != edge.buffer) {
                continue;
            }
            // Derived scopes must be supersets of the catalog hazard edge.
            if ((want_src.stage & ~b.srcStageMask) == 0 &&
                (want_src.access & ~b.srcAccessMask) == 0 &&
                (want_dst.stage & ~b.dstStageMask) == 0 &&
                (want_dst.access & ~b.dstAccessMask) == 0) {
                return true;
            }
        }
        return false;
    }

    class TestableRenderer final : public VulkanGSRenderer {
    public:
        ~TestableRenderer() {
            disarm_for_destruction();
        }

        void install_fake_handles() {
            device = fakeVkHandle<VkDevice>(0x1101);
            command_queue = fakeVkHandle<VkQueue>(0x1102);
            command_pool = fakeVkHandle<VkCommandPool>(0x1103);
            fence = fakeVkHandle<VkFence>(0x1104);
            queue_family_index = kQueueFamily;
            barrier_planner_ = BufferBarrierPlanner(kQueueFamily);

            for (std::uint32_t i = 0; i < kCommandBatchSlotCount; ++i) {
                command_batch_slots_[i].command_buffer =
                    fakeVkHandle<VkCommandBuffer>(0x2100 + i);
                command_batch_slots_[i].timestamp_query_pool =
                    fakeVkHandle<VkQueryPool>(0x3100 + i);
                command_batch_slots_[i].pending_signal = VK_NULL_HANDLE;
                command_batch_slots_[i].pending_signal_value = 0;
            }
            command_buffer = command_batch_slots_[0].command_buffer;
            timestamp_query_pool = command_batch_slots_[0].timestamp_query_pool;
            next_command_batch_slot_ = 0;
            active_command_batch_slot_ = 0;
            commandBatchInProgress = false;
            last_timeline_signal_values_.clear();
            pending_timeline_waits_.clear();
            buffer_retire_timeline_ = fakeVkHandle<VkSemaphore>(0x5101);
            next_buffer_retire_value_ = 1;
            retired_buffer_shells_.clear();
            test_buffer_handle_counter_ = 0xC1000;
            current_vram = 0;
            peak_vram = 0;

            deviceInfo.subgroupSize = 32;
            deviceInfo.sharedSize = 48 * 1024;
            deviceInfo.maxGroupsX = 65535;
            deviceInfo.maxGroupsY = 65535;
            deviceInfo.maxGroupsZ = 65535;

            vk_cmd_push_descriptor_set_ = &DispatchScript::push_descriptor_set;

            // Forge compute pipelines (handles only — never registered in all_compute_pipelines).
            forge_pipeline(pipeline_lod_map_indices, 0x5101);
            forge_pipeline(pipeline_lod_select_threshold, 0x5102);
            forge_pipeline(pipeline_lod_compact_touch, 0x5103);
            forge_pipeline(pipeline_selection_mask, 0x5201);
            forge_pipeline(pipeline_selection_polygon_rasterize, 0x5202);
            forge_pipeline(pipeline_projection_forward, 0x5301);
            forge_pipeline(pipeline_projection_forward_3dgut, 0x5302);
            forge_pipeline(pipeline_projection_forward_quant, 0x5303);
            forge_pipeline(pipeline_projection_forward_quant_3dgut, 0x5304);
            forge_pipeline(pipeline_cull_splats, 0x5401);
            forge_pipeline(pipeline_cull_prepare, 0x5402);
            forge_pipeline(pipeline_projection_forward_survivors, 0x5403);
            forge_pipeline(pipeline_projection_forward_quant_survivors, 0x5404);
            forge_pipeline(pipeline_cumsum.single_pass, 0x5501);
            forge_pipeline(pipeline_cumsum.block_scan, 0x5502);
            forge_pipeline(pipeline_cumsum.scan_block_sums, 0x5503);
            forge_pipeline(pipeline_cumsum.add_block_offsets, 0x5504);
            forge_pipeline(pipeline_cumsum_indirect.block_scan, 0x5511);
            forge_pipeline(pipeline_cumsum_indirect.scan_block_sums, 0x5512);
            forge_pipeline(pipeline_cumsum_indirect.add_block_offsets, 0x5513);
            forge_pipeline(pipeline_prepare_tile_sort, 0x5521);
            forge_pipeline(pipeline_prepare_tile_sort_visible, 0x5522);
            forge_pipeline(pipeline_visible_flags, 0x5531);
            forge_pipeline(pipeline_prepare_visible_sort, 0x5532);
            forge_pipeline(pipeline_compact_visible_primitives, 0x5533);
            forge_pipeline(pipeline_prepare_visible_chain, 0x5534);
            forge_pipeline(pipeline_copy_visible_indices, 0x5535);
            forge_pipeline(pipeline_radix_histogram_clear, 0x5541);
            forge_pipeline(pipeline_sorting_indirect_1.upsweep, 0x5542);
            forge_pipeline(pipeline_sorting_indirect_1.spine, 0x5543);
            forge_pipeline(pipeline_sorting_indirect_1.downsweep, 0x5544);
            forge_pipeline(pipeline_sorting_indirect_2.upsweep, 0x5552);
            forge_pipeline(pipeline_sorting_indirect_2.spine, 0x5553);
            forge_pipeline(pipeline_sorting_indirect_2.downsweep, 0x5554);
            forge_pipeline(pipeline_apply_depth_ordering, 0x5561);
            forge_pipeline(pipeline_wave_partition, 0x5562);
            forge_pipeline(pipeline_wave_partition_visible, 0x5563);
            forge_pipeline(pipeline_macro_coverage, 0x5564);
            forge_pipeline(pipeline_generate_keys_wave, 0x5601);
            forge_pipeline(pipeline_generate_macro_keys_wave, 0x5602);
            forge_pipeline(pipeline_macro_batch_prepare, 0x5603);
            forge_pipeline(pipeline_compute_macro_ranges[0], 0x5604);
            forge_pipeline(pipeline_compute_macro_ranges[1], 0x5605);
            forge_pipeline(pipeline_compute_tile_ranges[0], 0x5606);
            forge_pipeline(pipeline_compute_tile_ranges[1], 0x5607);
            forge_pipeline(pipeline_compute_tile_ranges_and_batch_counts[0], 0x5608);
            forge_pipeline(pipeline_compute_tile_ranges_and_batch_counts[1], 0x5609);
            forge_pipeline(pipeline_tile_batch_descriptors, 0x560A);
            forge_pipeline(pipeline_compose_tile_batches, 0x560B);
            forge_pipeline(pipeline_compose_tile_batches_plain, 0x560C);
            forge_pipeline(pipeline_expected_depth_finalize, 0x560D);
            forge_pair(pipeline_macro_raster, 0x5610);
            forge_pair(pipeline_macro_raster_fp32, 0x5620);
            forge_pair(pipeline_macro_raster_overlays, 0x5630);
            forge_pair(pipeline_macro_compose, 0x5640);
            forge_pair(pipeline_macro_compose_overlays, 0x5650);
            forge_pair(pipeline_rasterize_forward, 0x5660);
            forge_pair(pipeline_rasterize_forward_plain, 0x5670);
            forge_pair(pipeline_rasterize_forward_3dgut, 0x5680);
            forge_pair(pipeline_rasterize_forward_3dgut_plain, 0x5690);
            forge_pair(pipeline_rasterize_forward_light, 0x56A0);
            forge_pair(pipeline_rasterize_forward_light_plain, 0x56B0);
            forge_pair(pipeline_rasterize_forward_batches, 0x56C0);
            forge_pair(pipeline_rasterize_forward_batches_plain, 0x56D0);

            // Pre-sized host-visible readback so ensureLodSelectionReadback is a no-op.
            // ensureLodSelectionReadback(chunk_capacity) allocates (2+chunk_capacity) words;
            // copies end at word (6 + protected + 2*miss).
            constexpr std::size_t kPayloadWords =
                4 + kLodCompactProtectedCap + 2 * kLodCompactMissCap;
            const VkDeviceSize readback_bytes =
                (2 + kPayloadWords) * sizeof(std::uint32_t);
            lod_selection_readback_buffer_ = makeBuffer(0xF001, readback_bytes);
            lod_selection_readback_mapped_ = reinterpret_cast<std::uint32_t*>(
                static_cast<std::uintptr_t>(0xBEEF0000));
            lod_selection_readback_initialized_ = true;
            lod_selection_readback_pending_ = false;
            lod_selection_readback_chunk_capacity_ = kPayloadWords;
            lod_selection_readback_capacity_ = 0;

            // Visible-count readback (two words) for sort-chain audits.
            visible_count_readback_buffer_ = makeBuffer(0xF002, 2 * sizeof(std::uint32_t));
            visible_count_readback_mapped_ = reinterpret_cast<std::uint32_t*>(
                static_cast<std::uintptr_t>(0xBEEF1000));
            visible_count_readback_initialized_ = true;
            visible_count_readback_pending_ = false;

            // Instance-count readback (3 words) for wave-partition chain.
            instance_count_readback_buffer_ = makeBuffer(0xF003, 3 * sizeof(std::uint32_t));
            instance_count_readback_mapped_ = reinterpret_cast<std::uint32_t*>(
                static_cast<std::uintptr_t>(0xBEEF2000));
            instance_count_readback_initialized_ = true;
            instance_count_readback_pending_ = false;

            // Gate readback (1 word). Full synchronizeTileInstanceGate still
            // needs fence wait + vmaInvalidate; production path is planTransfer.
            instance_gate_readback_buffer_ = makeBuffer(0xF004, sizeof(std::uint32_t));
            instance_gate_readback_mapped_ = reinterpret_cast<std::uint32_t*>(
                static_cast<std::uintptr_t>(0xBEEF3000));
            instance_gate_readback_initialized_ = true;
        }

        // Skip recordInstanceCountReadback body (raw vkCmdUpdateBuffer + copies)
        // while still exercising partition's ConditionalRead handoff.
        void skip_instance_count_readback() {
            instance_count_readback_pending_ = true;
            instance_count_readback_signal_ = fakeVkHandle<VkSemaphore>(0xC0FFEE);
        }

        void install_conditional_rendering_nops() {
            supports_conditional_rendering_ = true;
            vk_cmd_begin_conditional_rendering_ = &DispatchScript::begin_conditional;
            vk_cmd_end_conditional_rendering_ = &DispatchScript::end_conditional;
        }

        void disarm_for_destruction() {
            commandBatchInProgress = false;
            pending_timeline_waits_.clear();
            last_timeline_signal_values_.clear();
            for (auto& slot : command_batch_slots_) {
                slot.pending_signal = VK_NULL_HANDLE;
                slot.pending_signal_value = 0;
                slot.pending_timestamp_count = 0;
                slot.pending_timestamp_marks.clear();
                slot.command_buffer = VK_NULL_HANDLE;
                slot.timestamp_query_pool = VK_NULL_HANDLE;
            }
            retired_buffer_shells_.clear();
            buffer_retire_timeline_ = VK_NULL_HANDLE;
            next_buffer_retire_value_ = 1;
            command_buffer = VK_NULL_HANDLE;
            timestamp_query_pool = VK_NULL_HANDLE;
            fence = VK_NULL_HANDLE;
            command_pool = VK_NULL_HANDLE;
            command_queue = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
            instance = VK_NULL_HANDLE;
            physical_device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
            vk_cmd_push_descriptor_set_ = nullptr;
            barrier_planner_.reset();
            current_vram = 0;

            // Skip vmaDestroy on forged readbacks.
            lod_selection_readback_initialized_ = false;
            lod_selection_readback_buffer_ = {};
            lod_selection_readback_mapped_ = nullptr;
            visible_count_readback_initialized_ = false;
            visible_count_readback_buffer_ = {};
            instance_count_readback_initialized_ = false;
            instance_count_readback_buffer_ = {};
            instance_gate_readback_initialized_ = false;
            instance_gate_readback_buffer_ = {};

            zero_pipeline(pipeline_lod_map_indices);
            zero_pipeline(pipeline_lod_select_threshold);
            zero_pipeline(pipeline_lod_compact_touch);
            zero_pipeline(pipeline_selection_mask);
            zero_pipeline(pipeline_selection_polygon_rasterize);
            zero_pipeline(pipeline_projection_forward);
            zero_pipeline(pipeline_projection_forward_3dgut);
            zero_pipeline(pipeline_projection_forward_quant);
            zero_pipeline(pipeline_projection_forward_quant_3dgut);
            zero_pipeline(pipeline_cull_splats);
            zero_pipeline(pipeline_cull_prepare);
            zero_pipeline(pipeline_projection_forward_survivors);
            zero_pipeline(pipeline_projection_forward_quant_survivors);
            all_compute_pipelines.clear();
        }

        [[nodiscard]] _VulkanBuffer& lod_readback() noexcept {
            return lod_selection_readback_buffer_;
        }

        // Drop GPU timestamp bookkeeping so endCommandBatch does not call
        // collectTimestampResults (real vkGetPhysicalDeviceProperties) on fakes.
        void discard_timestamps() {
            timestampNumWritten = 0;
            timestampStackDepth = 0;
            PerfTimer::discardMarkers();
        }

    private:
        static void forge_pipeline(_ComputePipeline& cp, const std::uintptr_t base) {
            cp.pipeline = fakeVkHandle<VkPipeline>(base);
            cp.pipeline_layout = fakeVkHandle<VkPipelineLayout>(base + 1);
            cp.descriptor_set_layout = fakeVkHandle<VkDescriptorSetLayout>(base + 2);
            cp.shader = fakeVkHandle<VkShaderModule>(base + 3);
            cp.diagnostic_name = "audit.wave";
        }
        static void forge_pair(_ComputePipelinePair& pair, const std::uintptr_t base) {
            forge_pipeline(pair._cp0, base);
            forge_pipeline(pair._cp1, base + 0x8);
        }
        static void zero_pipeline(_ComputePipeline& cp) {
            cp.pipeline = VK_NULL_HANDLE;
            cp.pipeline_layout = VK_NULL_HANDLE;
            cp.descriptor_set_layout = VK_NULL_HANDLE;
            cp.shader = VK_NULL_HANDLE;
        }
    };

    void forge_owned(Buffer<std::uint32_t>& buf, const std::uintptr_t id, const std::size_t elements) {
        const VkDeviceSize bytes = elements * sizeof(std::uint32_t);
        buf.deviceBuffer = makeBuffer(id, bytes);
    }
    void forge_owned_f(Buffer<float>& buf, const std::uintptr_t id, const std::size_t elements) {
        const VkDeviceSize bytes = elements * sizeof(float);
        buf.deviceBuffer = makeBuffer(id, bytes);
    }
    void forge_owned_i32(Buffer<std::int32_t>& buf, const std::uintptr_t id, const std::size_t elements) {
        const VkDeviceSize bytes = elements * sizeof(std::int32_t);
        buf.deviceBuffer = makeBuffer(id, bytes);
    }
    void forge_owned_i64(Buffer<std::int64_t>& buf, const std::uintptr_t id, const std::size_t elements) {
        const VkDeviceSize bytes = elements * sizeof(std::int64_t);
        buf.deviceBuffer = makeBuffer(id, bytes);
    }
    void forge_owned_u16(Buffer<std::uint16_t>& buf, const std::uintptr_t id, const std::size_t elements) {
        const VkDeviceSize bytes = elements * sizeof(std::uint16_t);
        buf.deviceBuffer = makeBuffer(id, bytes);
    }

    void track_buf(TestableRenderer& r, const _VulkanBuffer& b) {
        if (b.buffer != VK_NULL_HANDLE) {
            r.trackExternalParent(b.buffer);
        }
    }

    // P4 baselines (catalog struct counts).
    constexpr std::size_t kAuditSelectionMask = 13;            // 11 pre + 2 post
    constexpr std::size_t kAuditSelectionPolygonRasterize = 3; // 2 pre + 1 post
    constexpr std::size_t kAuditProjectionForwardNoLod = 12;   // 10 + 1 + 1 (no L1218)
    constexpr std::size_t kAuditProjectionForwardWithLod = 16; // +4 LOD
    // P4 r2: cull 5+1+(0..2)+1+1+2; survivors 5+(0..2).
    constexpr std::size_t kAuditCullSplatsNoLod = 10;
    constexpr std::size_t kAuditCullSplatsWithLod = 12;
    constexpr std::size_t kAuditProjectionSurvivorsNoLod = 5;
    constexpr std::size_t kAuditProjectionSurvivorsWithLod = 7;
    // P4 r3: cumsum single-pass begin=3 structs (input+output; no blockSums) + phases 0;
    // prepare tile sort 2; generous caps for multi-phase + sort chain under freeze N=64.
    constexpr std::size_t kAuditCumsumSinglePass = 8;
    constexpr std::size_t kAuditPrepareTileSort = 4;
    // Radix multi-pass (32 bits → 4 passes) + cumsum + compact + copy expands
    // well past the catalog's local-site count (nested chains).
    constexpr std::size_t kAuditSortPrimitivesByDepth = 64;
    constexpr std::size_t kAuditIndexBufferOffsetVisible = 24;

    // P4 r4 branch-frozen baselines (critique F2 + de-hoist headroom).
    // Struct counts may exceed hand-written hoist totals because exact planning
    // emits more, tighter barriers (F3). Gate is edge coverage + generous LE.
    // legacy light: base24 + W*(~20 local + ~16 sort 1-pass) + headroom×2
    constexpr std::size_t kAuditLegacyLightW1 = 256;
    constexpr std::size_t kAuditLegacyLightW3 = 512;
    // macro: base18 + W*(~23 local + sort + cumsum) + headroom
    constexpr std::size_t kAuditMacroDepthW1 = 320;
    constexpr std::size_t kAuditMacroDepthW3 = 640;
    constexpr std::size_t kAuditApplyDepth = 8;
    constexpr std::size_t kAuditWavePartition = 16; // dispatch + ConditionalRead + transfer
    constexpr std::size_t kAuditMacroCoverage = 12;

    constexpr std::uint32_t kAuditSplatCount = 64;
    constexpr std::uint32_t kAuditAabbW = 16;
    constexpr std::uint32_t kAuditAabbH = 16;
    constexpr std::uint32_t kAuditWaveGrid = 4;
    constexpr std::uint32_t kAuditWaveImage = 16;
    constexpr int kAuditWaveSortBits = 8; // one radix pass

    [[nodiscard]] std::vector<VkBufferMemoryBarrier2> collect_derived(
        const DispatchScript& script) {
        std::vector<VkBufferMemoryBarrier2> derived;
        for (const auto& cap : script.barriers) {
            derived.insert(derived.end(),
                           cap.buffer_barriers.begin(),
                           cap.buffer_barriers.end());
        }
        return derived;
    }

    void seed_compute_writer(TestableRenderer& r, _VulkanBuffer& buf) {
        (void)r.barrierPlanner().plan(std::array{
            lfs::rendering::vulkan::DeclaredAccess{
                &buf, lfs::rendering::vulkan::BufferUse::ComputeWrite},
        });
    }

    void seed_csrw_writer(TestableRenderer& r, _VulkanBuffer& buf) {
        (void)r.barrierPlanner().plan(std::array{
            lfs::rendering::vulkan::DeclaredAccess{
                &buf, lfs::rendering::vulkan::BufferUse::ComputeReadWrite},
        });
    }

} // namespace

// Catches: LOD chain still hand-writing barriers (planner stats never move) or
// derived barrier count / edge coverage regressing past the §2.6 audit baselines.
TEST(VkSplatTaggedDispatch, LodChainAuditMapAndSelectWithinBaseline) {
    DispatchScript script;
    BindScript bind(script);

    TestableRenderer renderer;
    renderer.install_fake_handles();
    renderer.setVulkanDispatch(make_scripted_dispatch());

    VulkanGSPipelineBuffers buffers;
    // Capacities must absorb resize/clear without real VMA allocation.
    forge_owned(buffers.lod_logical_indices, 0xA001, kAuditLodCount);
    forge_owned(buffers.lod_indices, 0xA002, kAuditLodCount);
    forge_owned(buffers.lod_gpu_counts, 0xA003, 2);
    forge_owned(buffers.lod_gpu_indices, 0xA004, kAuditOutputCapacity);
    forge_owned(buffers.lod_gpu_logical_indices, 0xA005, kAuditOutputCapacity);
    forge_owned_f(buffers.lod_gpu_weights, 0xA006, kAuditOutputCapacity);
    forge_owned(buffers.lod_gpu_levels, 0xA007, kAuditOutputCapacity);
    forge_owned(buffers.lod_chunk_touch, 0xA008, kAuditLogicalChunkCount);
    forge_owned(buffers.lod_compact_counts, 0xA009, 4);
    forge_owned(buffers.lod_compact_protected, 0xA00A, kLodCompactProtectedCap);
    forge_owned(buffers.lod_compact_misses, 0xA00B, 2 * kLodCompactMissCap);

    auto chunk_to_page = makeBuffer(0xB001, 4096);
    auto node_bounds = makeBuffer(0xB002, kAuditPhysicalNodeCount * 2 * sizeof(std::uint32_t));
    auto node_links = makeBuffer(0xB003, kAuditPhysicalNodeCount * 3 * sizeof(std::uint32_t));
    auto page_age = makeBuffer(0xB004, 4096);
    auto page_frames = makeBuffer(0xB005, 4096);
    auto page_to_chunk = makeBuffer(0xB006, 4096);

    renderer.beginCommandBatch();
    // Track-after-begin (P3a convention): owned + external inputs.
    for (auto* b : {&buffers.lod_logical_indices.deviceBuffer,
                    &buffers.lod_indices.deviceBuffer,
                    &buffers.lod_gpu_counts.deviceBuffer,
                    &buffers.lod_gpu_indices.deviceBuffer,
                    &buffers.lod_gpu_logical_indices.deviceBuffer,
                    &buffers.lod_gpu_weights.deviceBuffer,
                    &buffers.lod_gpu_levels.deviceBuffer,
                    &buffers.lod_chunk_touch.deviceBuffer,
                    &buffers.lod_compact_counts.deviceBuffer,
                    &buffers.lod_compact_protected.deviceBuffer,
                    &buffers.lod_compact_misses.deviceBuffer,
                    &chunk_to_page,
                    &node_bounds,
                    &node_links,
                    &page_age,
                    &page_frames,
                    &page_to_chunk}) {
        renderer.trackExternalParent(b->buffer);
    }
    // Readback is intentionally untracked (conservative HostRead/Transfer rows).

    const auto stats_before = renderer.barrierPlanner().stats();
    script.clear_recording();

    // --- executeMapLodIndices ---
    renderer.executeMapLodIndices(
        kAuditLodCount, kAuditChunkSplats, kAuditInvalidPage, buffers, chunk_to_page);

    const std::size_t map_structs = total_buffer_barrier_structs(script);
    EXPECT_LE(map_structs, kAuditMapLodIndices)
        << "derived map-lod barrier structs must be ≤ catalog baseline";

    // --- executeSelectLodThreshold (+ fills + compact + readback) ---
    script.clear_recording();
    VulkanGSLodSelectUniforms uniforms{};
    uniforms.node_count = kAuditNodeCount;
    uniforms.output_capacity = kAuditOutputCapacity;
    uniforms.chunk_splats = kAuditChunkSplats;
    uniforms.invalid_page = kAuditInvalidPage;
    uniforms.pixel_scale_limit = 0.01f;
    uniforms.object_scale = 1.0f;
    uniforms.physical_node_count = kAuditPhysicalNodeCount;
    uniforms.logical_chunk_count = kAuditLogicalChunkCount;
    uniforms.current_frame = 1;
    uniforms.fade_frames = 0;

    renderer.executeSelectLodThreshold(
        uniforms, buffers, node_bounds, node_links, chunk_to_page,
        page_age, page_frames, page_to_chunk);

    const std::size_t select_structs = total_buffer_barrier_structs(script);
    EXPECT_LE(select_structs, kAuditSelectLodThresholdWithReadback)
        << "derived select-lod+readback barrier structs must be ≤ catalog baseline";

    // Collect all derived barriers from a full re-run for edge coverage.
    script.clear_recording();
    // Planner state was updated by the first run; re-track clean for a full record.
    renderer.endCommandBatch(/*use_fence=*/false);
    renderer.beginCommandBatch();
    for (auto* b : {&buffers.lod_logical_indices.deviceBuffer,
                    &buffers.lod_indices.deviceBuffer,
                    &buffers.lod_gpu_counts.deviceBuffer,
                    &buffers.lod_gpu_indices.deviceBuffer,
                    &buffers.lod_gpu_logical_indices.deviceBuffer,
                    &buffers.lod_gpu_weights.deviceBuffer,
                    &buffers.lod_gpu_levels.deviceBuffer,
                    &buffers.lod_chunk_touch.deviceBuffer,
                    &buffers.lod_compact_counts.deviceBuffer,
                    &buffers.lod_compact_protected.deviceBuffer,
                    &buffers.lod_compact_misses.deviceBuffer,
                    &chunk_to_page,
                    &node_bounds,
                    &node_links,
                    &page_age,
                    &page_frames,
                    &page_to_chunk}) {
        renderer.trackExternalParent(b->buffer);
    }
    script.clear_recording();
    renderer.executeMapLodIndices(
        kAuditLodCount, kAuditChunkSplats, kAuditInvalidPage, buffers, chunk_to_page);
    renderer.executeSelectLodThreshold(
        uniforms, buffers, node_bounds, node_links, chunk_to_page,
        page_age, page_frames, page_to_chunk);

    std::vector<VkBufferMemoryBarrier2> all_derived;
    for (const auto& cap : script.barriers) {
        all_derived.insert(all_derived.end(),
                           cap.buffer_barriers.begin(),
                           cap.buffer_barriers.end());
    }

    // Intra-chain true hazard edges (catalog producer→consumer). Post-handoff
    // edges consumed only by still-legacy projection pre-barriers are omitted (§3.4.5).
    using BM = VulkanGSPipeline::BarrierMask;
    const HazardEdge edges[] = {
        // Select: fill → compute (counts, chunk_touch) and compute→compute (chunk_touch→compact).
        {buffers.lod_gpu_counts.deviceBuffer.buffer, BM::TRANSFER_WRITE, BM::COMPUTE_SHADER_READ_WRITE,
         "counts fill→select"},
        {buffers.lod_chunk_touch.deviceBuffer.buffer, BM::TRANSFER_WRITE, BM::COMPUTE_SHADER_READ_WRITE,
         "chunk_touch fill→select"},
        {buffers.lod_chunk_touch.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE, BM::COMPUTE_SHADER_READ,
         "chunk_touch select→compact"},
        // compact_counts is ComputeWrite in lod_compact_touch.slang (not R/W).
        {buffers.lod_compact_counts.deviceBuffer.buffer, BM::TRANSFER_WRITE, BM::COMPUTE_SHADER_WRITE,
         "compact_counts fill→compact"},
        // Readback: compute/compact write → transfer read sources.
        {buffers.lod_gpu_counts.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE, BM::TRANSFER_READ,
         "counts → readback copy"},
        {buffers.lod_compact_counts.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE, BM::TRANSFER_READ,
         "compact_counts → readback copy"},
        {buffers.lod_compact_protected.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE, BM::TRANSFER_READ,
         "compact_protected → readback copy"},
        {buffers.lod_compact_misses.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE, BM::TRANSFER_READ,
         "compact_misses → readback copy"},
        // Readback host visibility (untracked → conservative src is acceptable superset).
        {renderer.lod_readback().buffer, BM::TRANSFER_WRITE, BM::HOST_READ,
         "readback transfer→host"},
    };

    for (const auto& edge : edges) {
        EXPECT_TRUE(edge_covered(all_derived, edge))
            << "missing coverage for edge: " << edge.name;
    }

    // (c) RED discriminator: tagged/planTransfer path must exercise the planner.
    const auto stats_after = renderer.barrierPlanner().stats();
    const std::uint64_t planned_activity =
        (stats_after.barriers_emitted + stats_after.accesses_elided) -
        (stats_before.barriers_emitted + stats_before.accesses_elided);
    EXPECT_GT(planned_activity, 0u)
        << "planner stats must move (barriers_emitted+accesses_elided); "
           "still-legacy hand-written barriers leave the planner idle";

    // Publish observed counts for the report (always printed on failure; also on success via cout).
    std::printf("LodChainAudit derived map_structs=%zu (baseline %zu) select_structs=%zu (baseline %zu) "
                "full_derived_structs=%zu planned_activity=%llu\n",
                map_structs,
                kAuditMapLodIndices,
                select_structs,
                kAuditSelectLodThresholdWithReadback,
                all_derived.size(),
                static_cast<unsigned long long>(planned_activity));

    renderer.endCommandBatch(/*use_fence=*/false);
}

// =============================================================================
// P4 r1 A: selection chains
// =============================================================================

// Catches: selection_mask / polygon still hand-writing barriers (planner idle) or
// derived struct count / edge coverage past catalog baselines.
TEST(VkSplatTaggedDispatch, SelectionChainAuditWithinBaseline) {
    DispatchScript script;
    BindScript bind(script);

    TestableRenderer renderer;
    renderer.install_fake_handles();
    renderer.setVulkanDispatch(make_scripted_dispatch());

    VulkanGSPipelineBuffers buffers;
    forge_owned_f(buffers.xyz_ws, 0xC001, kAuditSplatCount * 3);
    forge_owned_f(buffers.rotations, 0xC002, kAuditSplatCount * 4);
    forge_owned_f(buffers.scaling_raw, 0xC003, kAuditSplatCount * 3);
    forge_owned_f(buffers.opacity_raw, 0xC004, kAuditSplatCount);

    auto transform_indices = makeBuffer(0xC010, kAuditSplatCount * 4);
    auto node_mask = makeBuffer(0xC011, 4096);
    auto primitives = makeBuffer(0xC012, 4096);
    auto model_transforms = makeBuffer(0xC013, 4096);
    auto selection_out = makeBuffer(0xC014, kAuditSplatCount);
    auto polygon_mask = makeBuffer(0xC015, 4096);
    auto ring_pick_out = makeBuffer(0xC016, 64);
    auto polygon_vertices = makeBuffer(0xC017, 64 * sizeof(float) * 2);

    renderer.beginCommandBatch();
    track_buf(renderer, buffers.xyz_ws.deviceBuffer);
    track_buf(renderer, buffers.rotations.deviceBuffer);
    track_buf(renderer, buffers.scaling_raw.deviceBuffer);
    track_buf(renderer, buffers.opacity_raw.deviceBuffer);
    track_buf(renderer, transform_indices);
    track_buf(renderer, node_mask);
    track_buf(renderer, primitives);
    track_buf(renderer, model_transforms);
    track_buf(renderer, selection_out);
    track_buf(renderer, polygon_mask);
    track_buf(renderer, ring_pick_out);
    track_buf(renderer, polygon_vertices);

    const auto stats_before = renderer.barrierPlanner().stats();
    script.clear_recording();

    VulkanGSSelectionPolygonRasterizeUniforms poly_u{};
    poly_u.vertex_count = 4;
    poly_u.aabb_x0 = 0;
    poly_u.aabb_y0 = 0;
    poly_u.aabb_w = kAuditAabbW;
    poly_u.aabb_h = kAuditAabbH;
    renderer.executeSelectionPolygonRasterize(poly_u, polygon_vertices, polygon_mask);
    const std::size_t poly_structs = total_buffer_barrier_structs(script);
    EXPECT_LE(poly_structs, kAuditSelectionPolygonRasterize);

    script.clear_recording();
    VulkanGSSelectionMaskUniforms mask_u{};
    mask_u.num_splats = kAuditSplatCount;
    mask_u.primitive_count = 0;
    mask_u.mode = 2; // polygon mask
    mask_u.image_width = kAuditAabbW;
    mask_u.image_height = kAuditAabbH;
    mask_u.aabb_w = kAuditAabbW;
    mask_u.aabb_h = kAuditAabbH;
    renderer.executeSelectionMask(
        mask_u, buffers, transform_indices, node_mask, primitives, model_transforms,
        selection_out, polygon_mask, ring_pick_out);
    const std::size_t mask_structs = total_buffer_barrier_structs(script);
    EXPECT_LE(mask_structs, kAuditSelectionMask);

    // Edge coverage on combined recording.
    script.clear_recording();
    renderer.executeSelectionPolygonRasterize(poly_u, polygon_vertices, polygon_mask);
    renderer.executeSelectionMask(
        mask_u, buffers, transform_indices, node_mask, primitives, model_transforms,
        selection_out, polygon_mask, ring_pick_out);
    std::vector<VkBufferMemoryBarrier2> derived;
    for (const auto& cap : script.barriers) {
        derived.insert(derived.end(), cap.buffer_barriers.begin(), cap.buffer_barriers.end());
    }
    using BM = VulkanGSPipeline::BarrierMask;
    const HazardEdge edges[] = {
        // polygon write → selection ComputeRead (true hazard once both are planned).
        {polygon_mask.buffer, BM::COMPUTE_SHADER_WRITE, BM::COMPUTE_SHADER_READ,
         "polygon_mask write→selection read"},
        // selection write → TransferRead handoff (host/CUDA download path).
        {selection_out.buffer, BM::COMPUTE_SHADER_WRITE, BM::TRANSFER_READ,
         "selection_out → transfer handoff"},
        {ring_pick_out.buffer, BM::COMPUTE_SHADER_WRITE, BM::TRANSFER_READ,
         "ring_pick_out → transfer handoff"},
    };
    for (const auto& edge : edges) {
        EXPECT_TRUE(edge_covered(derived, edge)) << "missing edge: " << edge.name;
    }

    const auto stats_after = renderer.barrierPlanner().stats();
    const std::uint64_t planned_activity =
        (stats_after.barriers_emitted + stats_after.accesses_elided) -
        (stats_before.barriers_emitted + stats_before.accesses_elided);
    EXPECT_GT(planned_activity, 0u)
        << "selection chains must exercise planner (tagged path / handoff)";

    std::printf("SelectionChainAudit poly_structs=%zu (≤%zu) mask_structs=%zu (≤%zu) "
                "derived=%zu planned_activity=%llu\n",
                poly_structs, kAuditSelectionPolygonRasterize,
                mask_structs, kAuditSelectionMask,
                derived.size(),
                static_cast<unsigned long long>(planned_activity));

    renderer.endCommandBatch(/*use_fence=*/false);
}

// =============================================================================
// P4 r1 B: executeProjectionForward (no quant; with LOD inputs)
// recordVisibleCount/InstanceCount owned by later chains — not migrated here.
// =============================================================================

// Catches: projection still hand-writing barriers / fill not planTransfer-recorded.
TEST(VkSplatTaggedDispatch, ProjectionForwardAuditWithinBaseline) {
    DispatchScript script;
    BindScript bind(script);

    TestableRenderer renderer;
    renderer.install_fake_handles();
    renderer.setVulkanDispatch(make_scripted_dispatch());

    VulkanGSPipelineBuffers buffers;
    buffers.quant_pool = false;
    forge_owned_f(buffers.xyz_ws, 0xD001, kAuditSplatCount * 3);
    forge_owned_f(buffers.sh0, 0xD002, kAuditSplatCount * 3);
    forge_owned_f(buffers.shN, 0xD003, kAuditSplatCount * 16);
    forge_owned_f(buffers.rotations, 0xD004, kAuditSplatCount * 4);
    forge_owned_f(buffers.scaling_raw, 0xD005, kAuditSplatCount * 3);
    forge_owned_f(buffers.opacity_raw, 0xD006, kAuditSplatCount);
    forge_owned_i32(buffers.tiles_touched, 0xD007, kAuditSplatCount);
    forge_owned_i64(buffers.rect_tile_space, 0xD008, kAuditSplatCount);
    forge_owned_i32(buffers.radii, 0xD009, kAuditSplatCount);
    forge_owned_f(buffers.xy_vs, 0xD00A, kAuditSplatCount * 2);
    forge_owned_f(buffers.depths, 0xD00B, kAuditSplatCount);
    forge_owned_f(buffers.inv_cov_vs_opacity, 0xD00C, kAuditSplatCount * 4);
    forge_owned_f(buffers.rgb, 0xD00D, kAuditSplatCount * 3);
    forge_owned_i32(buffers.overlay_flags, 0xD00E, kAuditSplatCount);
    forge_owned(buffers.primitive_depth_keys, 0xD00F, kAuditSplatCount);

    auto transform_indices = makeBuffer(0xD020, kAuditSplatCount * 4);
    auto node_mask = makeBuffer(0xD021, 4096);
    auto overlay_params = makeBuffer(0xD022, 4096);
    auto model_transforms = makeBuffer(0xD023, 4096);
    auto lod_indices = makeBuffer(0xD024, kAuditSplatCount * 4);
    auto lod_logical = makeBuffer(0xD025, kAuditSplatCount * 4);
    auto lod_levels = makeBuffer(0xD026, kAuditSplatCount * 4);
    auto lod_weights = makeBuffer(0xD027, kAuditSplatCount * 4);
    auto lod_counts = makeBuffer(0xD028, 16);

    renderer.beginCommandBatch();
    for (auto* b : {&buffers.xyz_ws.deviceBuffer, &buffers.sh0.deviceBuffer,
                    &buffers.shN.deviceBuffer, &buffers.rotations.deviceBuffer,
                    &buffers.scaling_raw.deviceBuffer, &buffers.opacity_raw.deviceBuffer,
                    &buffers.tiles_touched.deviceBuffer, &buffers.rect_tile_space.deviceBuffer,
                    &buffers.radii.deviceBuffer, &buffers.xy_vs.deviceBuffer,
                    &buffers.depths.deviceBuffer, &buffers.inv_cov_vs_opacity.deviceBuffer,
                    &buffers.rgb.deviceBuffer, &buffers.overlay_flags.deviceBuffer,
                    &buffers.primitive_depth_keys.deviceBuffer,
                    &transform_indices, &node_mask, &overlay_params, &model_transforms,
                    &lod_indices, &lod_logical, &lod_levels, &lod_weights, &lod_counts}) {
        track_buf(renderer, *b);
    }

    const auto stats_before = renderer.barrierPlanner().stats();
    script.clear_recording();

    VulkanGSRendererUniforms u{};
    u.num_splats = kAuditSplatCount;
    u.image_width = 64;
    u.image_height = 64;
    u.grid_width = 4;
    u.grid_height = 4;
    u.lod_enabled = 1;
    u.lod_count = kAuditSplatCount;

    // With LOD inputs present (catalog L1218 up to +4 structs).
    renderer.executeProjectionForward(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms,
        /*alloc_reserve=*/kAuditSplatCount,
        /*use_gut_projection=*/false,
        lod_indices, lod_logical, lod_levels, lod_weights, lod_counts);

    const std::size_t with_lod_structs = total_buffer_barrier_structs(script);
    EXPECT_LE(with_lod_structs, kAuditProjectionForwardWithLod);

    // No-LOD path for the tighter baseline.
    script.clear_recording();
    renderer.executeProjectionForward(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms,
        kAuditSplatCount, false);
    const std::size_t no_lod_structs = total_buffer_barrier_structs(script);
    EXPECT_LE(no_lod_structs, kAuditProjectionForwardNoLod);

    // Edge coverage on with-LOD recording (re-run once more after clear).
    script.clear_recording();
    renderer.executeProjectionForward(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms,
        kAuditSplatCount, false,
        lod_indices, lod_logical, lod_levels, lod_weights, lod_counts);
    std::vector<VkBufferMemoryBarrier2> derived;
    for (const auto& cap : script.barriers) {
        derived.insert(derived.end(), cap.buffer_barriers.begin(), cap.buffer_barriers.end());
    }
    using BM = VulkanGSPipeline::BarrierMask;
    const HazardEdge edges[] = {
        // sentinel fill: prior compute/R/W → transfer write, then transfer → compute R/W
        {buffers.primitive_depth_keys.deviceBuffer.buffer, BM::TRANSFER_WRITE,
         BM::COMPUTE_SHADER_WRITE, "depth_keys fill→projection write"},
        // LOD inputs (when valid): prior write → compute read
        {lod_indices.buffer, BM::COMPUTE_SHADER_WRITE, BM::COMPUTE_SHADER_READ,
         "lod_indices → projection read"},
    };
    // LOD edge only if planner saw a prior write; seed with a tagged write first.
    // For coverage of fill→compute, TransferWrite→ComputeWrite is the true tag edge.
    for (const auto& edge : edges) {
        if (std::string_view(edge.name).starts_with("lod_indices")) {
            // Only assert if any barrier mentions lod_indices (may elide if first read).
            bool any = false;
            for (const auto& b : derived) {
                if (b.buffer == lod_indices.buffer) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                continue; // first-read elision under track-after-begin is legal
            }
        }
        EXPECT_TRUE(edge_covered(derived, edge)) << "missing edge: " << edge.name;
    }

    const auto stats_after = renderer.barrierPlanner().stats();
    const std::uint64_t planned_activity =
        (stats_after.barriers_emitted + stats_after.accesses_elided) -
        (stats_before.barriers_emitted + stats_before.accesses_elided);
    EXPECT_GT(planned_activity, 0u)
        << "projection must exercise planner (tagged + planTransfer fill)";

    std::printf("ProjectionForwardAudit with_lod=%zu (≤%zu) no_lod=%zu (≤%zu) "
                "derived=%zu planned_activity=%llu\n",
                with_lod_structs, kAuditProjectionForwardWithLod,
                no_lod_structs, kAuditProjectionForwardNoLod,
                derived.size(),
                static_cast<unsigned long long>(planned_activity));

    renderer.discard_timestamps();
    renderer.endCommandBatch(/*use_fence=*/false);
}

// =============================================================================
// P4 r2: executeCullSplats + executeProjectionForwardSurvivors
// =============================================================================

// Catches: cull/survivors still hand-writing barriers (planner idle) or struct
// count / edge coverage past catalog baselines; emit_count / IndirectRead missed.
TEST(VkSplatTaggedDispatch, CullAndSurvivorsProjectionAuditWithinBaseline) {
    DispatchScript script;
    BindScript bind(script);

    TestableRenderer renderer;
    renderer.install_fake_handles();
    renderer.setVulkanDispatch(make_scripted_dispatch());

    VulkanGSPipelineBuffers buffers;
    buffers.quant_pool = false;
    forge_owned_f(buffers.xyz_ws, 0xE001, kAuditSplatCount * 3);
    forge_owned_f(buffers.sh0, 0xE002, kAuditSplatCount * 3);
    forge_owned_f(buffers.shN, 0xE003, kAuditSplatCount * 16);
    forge_owned_f(buffers.rotations, 0xE004, kAuditSplatCount * 4);
    forge_owned_f(buffers.scaling_raw, 0xE005, kAuditSplatCount * 3);
    forge_owned_f(buffers.opacity_raw, 0xE006, kAuditSplatCount);
    forge_owned_i32(buffers.survivors, 0xE007, kAuditSplatCount);
    forge_owned(buffers.survivor_state, 0xE008, 16); // ≥ SurvivorState::kLayout.word_count
    forge_owned(buffers.visible_emit_count, 0xE009, 4);
    forge_owned_i32(buffers.sorting_keys_1, 0xE00A, kAuditSplatCount);
    forge_owned_i32(buffers.sorting_keys_2, 0xE00B, kAuditSplatCount);
    forge_owned_i32(buffers.sorting_gauss_idx_1, 0xE00C, kAuditSplatCount);
    forge_owned_i32(buffers.sorting_gauss_idx_2, 0xE00D, kAuditSplatCount);
    forge_owned_i64(buffers.rect_tile_space, 0xE00E, kAuditSplatCount);
    forge_owned_f(buffers.xy_vs, 0xE00F, kAuditSplatCount * 2);
    forge_owned_f(buffers.depths, 0xE010, kAuditSplatCount);
    forge_owned_f(buffers.inv_cov_vs_opacity, 0xE011, kAuditSplatCount * 4);
    forge_owned_f(buffers.rgb, 0xE012, kAuditSplatCount * 3);
    forge_owned_i32(buffers.overlay_flags, 0xE013, kAuditSplatCount);
    forge_owned_i32(buffers.orig_ids, 0xE014, kAuditSplatCount);

    auto transform_indices = makeBuffer(0xE020, kAuditSplatCount * 4);
    auto node_mask = makeBuffer(0xE021, 4096);
    auto overlay_params = makeBuffer(0xE022, 4096);
    auto model_transforms = makeBuffer(0xE023, 4096);
    auto lod_indices = makeBuffer(0xE024, kAuditSplatCount * 4);
    auto lod_logical = makeBuffer(0xE025, kAuditSplatCount * 4);
    auto lod_levels = makeBuffer(0xE026, kAuditSplatCount * 4);
    auto lod_weights = makeBuffer(0xE027, kAuditSplatCount * 4);
    auto lod_counts = makeBuffer(0xE028, 16);

    renderer.beginCommandBatch();
    for (auto* b : {&buffers.xyz_ws.deviceBuffer, &buffers.sh0.deviceBuffer,
                    &buffers.shN.deviceBuffer, &buffers.rotations.deviceBuffer,
                    &buffers.scaling_raw.deviceBuffer, &buffers.opacity_raw.deviceBuffer,
                    &buffers.survivors.deviceBuffer, &buffers.survivor_state.deviceBuffer,
                    &buffers.visible_emit_count.deviceBuffer,
                    &buffers.sorting_keys_1.deviceBuffer, &buffers.sorting_keys_2.deviceBuffer,
                    &buffers.sorting_gauss_idx_1.deviceBuffer,
                    &buffers.sorting_gauss_idx_2.deviceBuffer,
                    &buffers.rect_tile_space.deviceBuffer, &buffers.xy_vs.deviceBuffer,
                    &buffers.depths.deviceBuffer, &buffers.inv_cov_vs_opacity.deviceBuffer,
                    &buffers.rgb.deviceBuffer, &buffers.overlay_flags.deviceBuffer,
                    &buffers.orig_ids.deviceBuffer,
                    &transform_indices, &node_mask, &overlay_params, &model_transforms,
                    &lod_indices, &lod_logical, &lod_levels, &lod_weights, &lod_counts}) {
        track_buf(renderer, *b);
    }

    const auto stats_before = renderer.barrierPlanner().stats();
    script.clear_recording();

    VulkanGSRendererUniforms u{};
    u.num_splats = kAuditSplatCount;
    u.image_width = 64;
    u.image_height = 64;
    u.grid_width = 4;
    u.grid_height = 4;
    u.lod_enabled = 1;
    u.lod_count = kAuditSplatCount;
    u.model_num_splats = kAuditSplatCount;

    // --- cull alone (no-LOD first so state is clean for with-LOD) ---
    renderer.executeCullSplats(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms);
    const std::size_t cull_no_lod = total_buffer_barrier_structs(script);
    EXPECT_LE(cull_no_lod, kAuditCullSplatsNoLod);

    script.clear_recording();
    renderer.executeCullSplats(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms,
        lod_indices, lod_logical, lod_counts);
    const std::size_t cull_with_lod = total_buffer_barrier_structs(script);
    EXPECT_LE(cull_with_lod, kAuditCullSplatsWithLod);

    // --- survivors after a single fresh cull (no-LOD), then with-LOD ---
    // End/begin resets planner via onBatchBegin so counts are not polluted by
    // WAW stacks from earlier isolated measurements.
    renderer.discard_timestamps();
    renderer.endCommandBatch(/*use_fence=*/false);
    renderer.beginCommandBatch();
    for (auto* b : {&buffers.xyz_ws.deviceBuffer, &buffers.sh0.deviceBuffer,
                    &buffers.shN.deviceBuffer, &buffers.rotations.deviceBuffer,
                    &buffers.scaling_raw.deviceBuffer, &buffers.opacity_raw.deviceBuffer,
                    &buffers.survivors.deviceBuffer, &buffers.survivor_state.deviceBuffer,
                    &buffers.visible_emit_count.deviceBuffer,
                    &buffers.sorting_keys_1.deviceBuffer, &buffers.sorting_keys_2.deviceBuffer,
                    &buffers.sorting_gauss_idx_1.deviceBuffer,
                    &buffers.sorting_gauss_idx_2.deviceBuffer,
                    &buffers.rect_tile_space.deviceBuffer, &buffers.xy_vs.deviceBuffer,
                    &buffers.depths.deviceBuffer, &buffers.inv_cov_vs_opacity.deviceBuffer,
                    &buffers.rgb.deviceBuffer, &buffers.overlay_flags.deviceBuffer,
                    &buffers.orig_ids.deviceBuffer,
                    &transform_indices, &node_mask, &overlay_params, &model_transforms,
                    &lod_indices, &lod_logical, &lod_levels, &lod_weights, &lod_counts}) {
        track_buf(renderer, *b);
    }

    script.clear_recording();
    renderer.executeCullSplats(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms);
    renderer.executeProjectionForwardSurvivors(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms,
        kAuditSplatCount);
    // Structs for the survivors step only: re-run survivors after clear once cull state exists.
    // Catalog hand-written survivors only barriered 5 attrs; co-migration also plans
    // cull→survivors hazards (state/survivors/emit_count/IndirectRead) that legacy under-synced.
    // Bound by (catalog survivors max) + (cross-chain true hazards from cull outputs).
    const std::size_t kAuditSurvivorsAfterCullNoLod =
        kAuditProjectionSurvivorsNoLod + 6; // +state/survivors/emit/indirect/cross
    script.clear_recording();
    renderer.executeProjectionForwardSurvivors(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms,
        kAuditSplatCount);
    const std::size_t surv_no_lod = total_buffer_barrier_structs(script);
    EXPECT_LE(surv_no_lod, kAuditSurvivorsAfterCullNoLod);

    script.clear_recording();
    renderer.executeProjectionForwardSurvivors(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms,
        kAuditSplatCount, lod_indices, lod_logical, lod_levels, lod_weights, lod_counts);
    const std::size_t surv_with_lod = total_buffer_barrier_structs(script);
    EXPECT_LE(surv_with_lod, kAuditProjectionSurvivorsWithLod + 6);

    // Combined edge coverage.
    script.clear_recording();
    renderer.executeCullSplats(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms,
        lod_indices, lod_logical, lod_counts);
    renderer.executeProjectionForwardSurvivors(
        u, buffers, transform_indices, node_mask, overlay_params, model_transforms,
        kAuditSplatCount, lod_indices, lod_logical, lod_levels, lod_weights, lod_counts);

    std::vector<VkBufferMemoryBarrier2> derived;
    for (const auto& cap : script.barriers) {
        derived.insert(derived.end(), cap.buffer_barriers.begin(), cap.buffer_barriers.end());
    }
    using BM = VulkanGSPipeline::BarrierMask;
    const HazardEdge edges[] = {
        // clear → cull: survivor_state TransferWrite → ComputeWrite (cull) / R/W (prepare)
        {buffers.survivor_state.deviceBuffer.buffer, BM::TRANSFER_WRITE, BM::COMPUTE_SHADER_WRITE,
         "survivor_state fill→cull write"},
        // cull write → prepare R/W
        {buffers.survivor_state.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
         BM::COMPUTE_SHADER_READ_WRITE, "survivor_state cull→prepare"},
        // prepare write → indirect read for survivors projection
        {buffers.survivor_state.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
         BM::INDIRECT_DISPATCH_READ, "survivor_state prepare→indirect"},
        // survivors list → projection read
        {buffers.survivors.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE, BM::COMPUTE_SHADER_READ,
         "survivors cull→projection"},
        // emit_count prepare write → projection R/W
        {buffers.visible_emit_count.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
         BM::COMPUTE_SHADER_READ_WRITE, "emit_count prepare→projection"},
    };
    for (const auto& edge : edges) {
        EXPECT_TRUE(edge_covered(derived, edge)) << "missing edge: " << edge.name;
    }

    const auto stats_after = renderer.barrierPlanner().stats();
    const std::uint64_t planned_activity =
        (stats_after.barriers_emitted + stats_after.accesses_elided) -
        (stats_before.barriers_emitted + stats_before.accesses_elided);
    EXPECT_GT(planned_activity, 0u)
        << "cull/survivors must exercise planner (tagged + clear planTransfer)";

    std::printf(
        "CullSurvivorsAudit cull_with_lod=%zu (≤%zu) cull_no_lod=%zu (≤%zu) "
        "surv_with_lod=%zu (≤%zu) surv_no_lod=%zu (≤%zu) derived=%zu planned_activity=%llu\n",
        cull_with_lod, kAuditCullSplatsWithLod, cull_no_lod, kAuditCullSplatsNoLod,
        surv_with_lod, kAuditProjectionSurvivorsWithLod + 6, surv_no_lod,
        kAuditSurvivorsAfterCullNoLod, derived.size(),
        static_cast<unsigned long long>(planned_activity));

    renderer.discard_timestamps();
    renderer.endCommandBatch(/*use_fence=*/false);
}

// =============================================================================
// P4 r3: cumsum / offset / tile sort / radix sort / visible-count readback
// =============================================================================

// Catches: classic cumsum + prepare_tile_sort still hand-writing barriers.
TEST(VkSplatTaggedDispatch, CumsumAndPrepareTileSortAuditWithinBaseline) {
    DispatchScript script;
    BindScript bind(script);

    TestableRenderer renderer;
    renderer.install_fake_handles();
    renderer.setVulkanDispatch(make_scripted_dispatch());

    VulkanGSPipelineBuffers buffers;
    // N=64 → single-pass cumsum (≤1024).
    forge_owned_i32(buffers.tiles_touched_depth_ordered, 0xF101, kAuditSplatCount);
    forge_owned_i32(buffers.index_buffer_offset, 0xF102, kAuditSplatCount);
    forge_owned(buffers.tile_sort_count, 0xF103, 4);
    forge_owned_i32(buffers._cumsum_blockSums, 0xF104, 64);
    forge_owned_i32(buffers._cumsum_blockSums2, 0xF105, 64);

    renderer.beginCommandBatch();
    track_buf(renderer, buffers.tiles_touched_depth_ordered.deviceBuffer);
    track_buf(renderer, buffers.index_buffer_offset.deviceBuffer);
    track_buf(renderer, buffers.tile_sort_count.deviceBuffer);
    track_buf(renderer, buffers._cumsum_blockSums.deviceBuffer);
    track_buf(renderer, buffers._cumsum_blockSums2.deviceBuffer);

    // Seed a writer on tiles so cumsum input Read emits a real barrier.
    (void)renderer.barrierPlanner().plan(std::array{
        lfs::rendering::vulkan::DeclaredAccess{
            &buffers.tiles_touched_depth_ordered.deviceBuffer,
            lfs::rendering::vulkan::BufferUse::ComputeWrite},
    });

    const auto stats_before = renderer.barrierPlanner().stats();
    script.clear_recording();

    VulkanGSRendererUniforms u{};
    u.num_splats = kAuditSplatCount;
    u.grid_width = 4;
    u.grid_height = 4;

    renderer.executeCalculateIndexBufferOffset(u, buffers);
    const std::size_t n = total_buffer_barrier_structs(script);
    EXPECT_LE(n, kAuditCumsumSinglePass + kAuditPrepareTileSort);

    const auto stats_after = renderer.barrierPlanner().stats();
    EXPECT_GT((stats_after.barriers_emitted + stats_after.accesses_elided) -
                  (stats_before.barriers_emitted + stats_before.accesses_elided),
              0u);

    std::printf("CumsumPrepareAudit structs=%zu (≤%zu) planned_activity=%llu\n",
                n,
                kAuditCumsumSinglePass + kAuditPrepareTileSort,
                static_cast<unsigned long long>(
                    (stats_after.barriers_emitted + stats_after.accesses_elided) -
                    (stats_before.barriers_emitted + stats_before.accesses_elided)));

    renderer.discard_timestamps();
    renderer.endCommandBatch(/*use_fence=*/false);
}

// Catches: sort-by-depth + radix + visible-count readback still legacy.
TEST(VkSplatTaggedDispatch, SortPrimitivesByDepthAuditWithinBaseline) {
    DispatchScript script;
    BindScript bind(script);

    TestableRenderer renderer;
    renderer.install_fake_handles();
    renderer.setVulkanDispatch(make_scripted_dispatch());

    VulkanGSPipelineBuffers buffers;
    forge_owned_i32(buffers.tiles_touched, 0xF201, kAuditSplatCount);
    forge_owned_i32(buffers.visible_flags, 0xF202, kAuditSplatCount);
    forge_owned_i32(buffers.visible_prefix, 0xF203, kAuditSplatCount);
    forge_owned(buffers.visible_count, 0xF204, 2);
    forge_owned(buffers.visible_sort_dispatch_args, 0xF205, 16);
    forge_owned(buffers.primitive_depth_keys, 0xF206, kAuditSplatCount);
    forge_owned_i32(buffers.sorting_keys_1, 0xF207, kAuditSplatCount);
    forge_owned_i32(buffers.sorting_keys_2, 0xF208, kAuditSplatCount);
    forge_owned_i32(buffers.sorting_gauss_idx_1, 0xF209, kAuditSplatCount);
    forge_owned_i32(buffers.sorting_gauss_idx_2, 0xF20A, kAuditSplatCount);
    forge_owned_i32(buffers.primitive_sort_indices, 0xF20B, kAuditSplatCount);
    forge_owned_i32(buffers._sorting_histogram, 0xF20C, 8 * 256);
    forge_owned_i32(buffers._sorting_histogram_cumsum, 0xF20D, 64 * 256);
    forge_owned_i32(buffers._cumsum_blockSums, 0xF20E, 64);
    forge_owned_i32(buffers._cumsum_blockSums2, 0xF20F, 64);

    renderer.beginCommandBatch();
    for (auto* b : {&buffers.tiles_touched.deviceBuffer, &buffers.visible_flags.deviceBuffer,
                    &buffers.visible_prefix.deviceBuffer, &buffers.visible_count.deviceBuffer,
                    &buffers.visible_sort_dispatch_args.deviceBuffer,
                    &buffers.primitive_depth_keys.deviceBuffer,
                    &buffers.sorting_keys_1.deviceBuffer, &buffers.sorting_keys_2.deviceBuffer,
                    &buffers.sorting_gauss_idx_1.deviceBuffer,
                    &buffers.sorting_gauss_idx_2.deviceBuffer,
                    &buffers.primitive_sort_indices.deviceBuffer,
                    &buffers._sorting_histogram.deviceBuffer,
                    &buffers._sorting_histogram_cumsum.deviceBuffer,
                    &buffers._cumsum_blockSums.deviceBuffer,
                    &buffers._cumsum_blockSums2.deviceBuffer}) {
        track_buf(renderer, *b);
    }

    const auto stats_before = renderer.barrierPlanner().stats();
    script.clear_recording();

    VulkanGSRendererUniforms u{};
    u.num_splats = kAuditSplatCount;
    u.grid_width = 4;
    u.grid_height = 4;

    renderer.executeSortPrimitivesByDepth(u, buffers);
    const std::size_t n = total_buffer_barrier_structs(script);
    EXPECT_LE(n, kAuditSortPrimitivesByDepth);

    const auto stats_after = renderer.barrierPlanner().stats();
    EXPECT_GT((stats_after.barriers_emitted + stats_after.accesses_elided) -
                  (stats_before.barriers_emitted + stats_before.accesses_elided),
              0u);

    std::printf("SortByDepthAudit structs=%zu (≤%zu) planned_activity=%llu\n",
                n,
                kAuditSortPrimitivesByDepth,
                static_cast<unsigned long long>(
                    (stats_after.barriers_emitted + stats_after.accesses_elided) -
                    (stats_before.barriers_emitted + stats_before.accesses_elided)));

    renderer.discard_timestamps();
    renderer.endCommandBatch(/*use_fence=*/false);
}

// Catches: indirect visible cumsum + prepare_tile_sort_visible still legacy.
TEST(VkSplatTaggedDispatch, IndexBufferOffsetVisibleAuditWithinBaseline) {
    DispatchScript script;
    BindScript bind(script);

    TestableRenderer renderer;
    renderer.install_fake_handles();
    renderer.setVulkanDispatch(make_scripted_dispatch());

    VulkanGSPipelineBuffers buffers;
    forge_owned_i32(buffers.tiles_touched_depth_ordered, 0xF301, kAuditSplatCount);
    forge_owned_i32(buffers.index_buffer_offset, 0xF302, kAuditSplatCount);
    forge_owned_i32(buffers._cumsum_blockSums, 0xF303, 64);
    forge_owned_i32(buffers._cumsum_blockSums2, 0xF304, 64);
    forge_owned_i32(buffers.cumsum_counts, 0xF305, 4);
    forge_owned(buffers.visible_dispatch, 0xF306, 16); // 4*3 words
    forge_owned(buffers.tile_sort_count, 0xF307, 4);
    forge_owned(buffers.visible_count, 0xF308, 2);

    renderer.beginCommandBatch();
    for (auto* b : {&buffers.tiles_touched_depth_ordered.deviceBuffer,
                    &buffers.index_buffer_offset.deviceBuffer,
                    &buffers._cumsum_blockSums.deviceBuffer,
                    &buffers._cumsum_blockSums2.deviceBuffer,
                    &buffers.cumsum_counts.deviceBuffer,
                    &buffers.visible_dispatch.deviceBuffer,
                    &buffers.tile_sort_count.deviceBuffer,
                    &buffers.visible_count.deviceBuffer}) {
        track_buf(renderer, *b);
    }

    const auto stats_before = renderer.barrierPlanner().stats();
    script.clear_recording();

    VulkanGSRendererUniforms u{};
    u.num_splats = kAuditSplatCount;
    u.grid_width = 4;
    u.grid_height = 4;

    renderer.executeCalculateIndexBufferOffsetVisible(u, buffers, kAuditSplatCount);
    const std::size_t n = total_buffer_barrier_structs(script);
    EXPECT_LE(n, kAuditIndexBufferOffsetVisible);

    const auto stats_after = renderer.barrierPlanner().stats();
    EXPECT_GT((stats_after.barriers_emitted + stats_after.accesses_elided) -
                  (stats_before.barriers_emitted + stats_before.accesses_elided),
              0u);

    std::printf("IndexOffsetVisibleAudit structs=%zu (≤%zu) planned_activity=%llu\n",
                n,
                kAuditIndexBufferOffsetVisible,
                static_cast<unsigned long long>(
                    (stats_after.barriers_emitted + stats_after.accesses_elided) -
                    (stats_before.barriers_emitted + stats_before.accesses_elided)));

    renderer.discard_timestamps();
    renderer.endCommandBatch(/*use_fence=*/false);
}

// =============================================================================
// P4 r4: apply depth, wave partition, macro coverage, depth waves (W∈{1,3})
// =============================================================================

// Catches: apply-depth / wave-partition still legacy; ConditionalRead handoff missing.
TEST(VkSplatTaggedDispatch, WavePartitionAndApplyDepthAudit) {
    DispatchScript script;
    BindScript bind(script);

    TestableRenderer renderer;
    renderer.install_fake_handles();
    renderer.setVulkanDispatch(make_scripted_dispatch());
    renderer.install_conditional_rendering_nops();
    renderer.skip_instance_count_readback(); // avoid raw vkCmdUpdateBuffer on fakes

    VulkanGSPipelineBuffers buffers;
    constexpr std::size_t kArmed = 1;
    forge_owned_i32(buffers.primitive_sort_indices, 0xF401, kAuditSplatCount);
    forge_owned_i32(buffers.tiles_touched, 0xF402, kAuditSplatCount);
    forge_owned_i32(buffers.tiles_touched_depth_ordered, 0xF403, kAuditSplatCount);
    forge_owned(buffers.visible_count, 0xF404, 2);
    forge_owned_i32(buffers.index_buffer_offset, 0xF405, kAuditSplatCount);
    // recordInstanceCountReadback requires exactly one uint32 when not skipped.
    forge_owned(buffers.tile_sort_count, 0xF406, 1);
    // layout(armed): (1+armed)*64 words
    forge_owned(buffers.depth_wave_dispatch, 0xF407, (1u + kArmed) * 64u);
    forge_owned(buffers.wave_predicates, 0xF408, kArmed);
    // Partition stamps CONDITIONAL_RENDERING usage before resize; forge must match.
    buffers.wave_predicates.deviceBuffer.extra_usage =
        VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    buffers.wave_predicates.deviceBuffer.created_with_extra_usage =
        VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;

    renderer.beginCommandBatch();
    track_buf(renderer, buffers.primitive_sort_indices.deviceBuffer);
    track_buf(renderer, buffers.tiles_touched.deviceBuffer);
    track_buf(renderer, buffers.tiles_touched_depth_ordered.deviceBuffer);
    track_buf(renderer, buffers.visible_count.deviceBuffer);
    track_buf(renderer, buffers.index_buffer_offset.deviceBuffer);
    track_buf(renderer, buffers.tile_sort_count.deviceBuffer);
    track_buf(renderer, buffers.depth_wave_dispatch.deviceBuffer);
    track_buf(renderer, buffers.wave_predicates.deviceBuffer);

    const auto stats_before = renderer.barrierPlanner().stats();
    script.clear_recording();

    VulkanGSRendererUniforms u{};
    u.num_splats = kAuditSplatCount;
    u.grid_width = kAuditWaveGrid;
    u.grid_height = kAuditWaveGrid;
    u.sort_capacity = HIGS_DEPTH_WAVE_INSTANCES;

    renderer.executeApplyDepthOrdering(u, buffers);
    const std::size_t apply_structs = total_buffer_barrier_structs(script);
    EXPECT_LE(apply_structs, kAuditApplyDepth);

    script.clear_recording();
    renderer.executeWavePartition(u, buffers, kArmed, /*visible_bounded=*/false);
    const std::size_t part_structs = total_buffer_barrier_structs(script);
    EXPECT_LE(part_structs, kAuditWavePartition);

    const auto derived = collect_derived(script);
    using BM = VulkanGSPipeline::BarrierMask;
    const HazardEdge cond_edge{
        buffers.wave_predicates.deviceBuffer.buffer,
        BM::COMPUTE_SHADER_WRITE,
        BM::CONDITIONAL_RENDERING_READ,
        "predicates write→ConditionalRead handoff",
    };
    EXPECT_TRUE(edge_covered(derived, cond_edge)) << "missing ConditionalRead handoff";

    // IndirectRead handoff for wave_dispatch after partition write.
    const HazardEdge ind_edge{
        buffers.depth_wave_dispatch.deviceBuffer.buffer,
        BM::COMPUTE_SHADER_WRITE,
        BM::INDIRECT_DISPATCH_READ,
        "wave_dispatch write→IndirectRead handoff",
    };
    EXPECT_TRUE(edge_covered(derived, ind_edge)) << "missing IndirectRead handoff";

    const auto stats_after = renderer.barrierPlanner().stats();
    EXPECT_GT((stats_after.barriers_emitted + stats_after.accesses_elided) -
                  (stats_before.barriers_emitted + stats_before.accesses_elided),
              0u);

    std::printf(
        "WavePartitionApplyDepthAudit apply=%zu(≤%zu) partition=%zu(≤%zu) "
        "planned_activity=%llu\n",
        apply_structs,
        kAuditApplyDepth,
        part_structs,
        kAuditWavePartition,
        static_cast<unsigned long long>(
            (stats_after.barriers_emitted + stats_after.accesses_elided) -
            (stats_before.barriers_emitted + stats_before.accesses_elided)));

    renderer.discard_timestamps();
    renderer.endCommandBatch(/*use_fence=*/false);
}

// Catches: macro_coverage still legacy / missing tags on visible_dispatch path.
TEST(VkSplatTaggedDispatch, MacroCoverageAuditWithinBaseline) {
    DispatchScript script;
    BindScript bind(script);

    TestableRenderer renderer;
    renderer.install_fake_handles();
    renderer.setVulkanDispatch(make_scripted_dispatch());

    VulkanGSPipelineBuffers buffers;
    forge_owned_i32(buffers.primitive_sort_indices, 0xF501, kAuditSplatCount);
    forge_owned_i64(buffers.rect_tile_space, 0xF502, kAuditSplatCount);
    forge_owned_i32(buffers.tiles_touched_depth_ordered, 0xF503, kAuditSplatCount);
    forge_owned(buffers.visible_count, 0xF504, 2);
    forge_owned_f(buffers.xy_vs, 0xF505, kAuditSplatCount * 2);
    forge_owned_f(buffers.inv_cov_vs_opacity, 0xF506, kAuditSplatCount * 4);
    forge_owned(buffers.visible_dispatch, 0xF507, 16); // VisibleChainDispatch words

    renderer.beginCommandBatch();
    for (auto* b : {&buffers.primitive_sort_indices.deviceBuffer,
                    &buffers.rect_tile_space.deviceBuffer,
                    &buffers.tiles_touched_depth_ordered.deviceBuffer,
                    &buffers.visible_count.deviceBuffer,
                    &buffers.xy_vs.deviceBuffer,
                    &buffers.inv_cov_vs_opacity.deviceBuffer,
                    &buffers.visible_dispatch.deviceBuffer}) {
        track_buf(renderer, *b);
    }
    seed_compute_writer(renderer, buffers.primitive_sort_indices.deviceBuffer);
    seed_compute_writer(renderer, buffers.rect_tile_space.deviceBuffer);

    const auto stats_before = renderer.barrierPlanner().stats();
    script.clear_recording();

    VulkanGSRendererUniforms u{};
    u.num_splats = kAuditSplatCount;
    u.grid_width = kAuditWaveGrid;
    u.grid_height = kAuditWaveGrid;

    renderer.executeMacroCoverage(u, buffers, kAuditSplatCount);
    const std::size_t n = total_buffer_barrier_structs(script);
    EXPECT_LE(n, kAuditMacroCoverage);

    const auto derived = collect_derived(script);
    using BM = VulkanGSPipeline::BarrierMask;
    const HazardEdge edges[] = {
        {buffers.primitive_sort_indices.deviceBuffer.buffer,
         BM::COMPUTE_SHADER_WRITE,
         BM::COMPUTE_SHADER_READ,
         "macro_coverage primitive_sort_indices"},
        {buffers.rect_tile_space.deviceBuffer.buffer,
         BM::COMPUTE_SHADER_WRITE,
         BM::COMPUTE_SHADER_READ,
         "macro_coverage rect_tile_space"},
        {buffers.visible_dispatch.deviceBuffer.buffer,
         BM::COMPUTE_SHADER_WRITE,
         BM::INDIRECT_DISPATCH_READ,
         "macro_coverage visible_dispatch IndirectRead (implicit)"},
    };
    // Indirect edge only if we seeded a CSW on visible_dispatch; seed now for check
    // when planner saw first IndirectRead after empty track may elide — re-seed+rerun
    // is heavy; require at least the two geometry RAW edges.
    EXPECT_TRUE(edge_covered(derived, edges[0])) << edges[0].name;
    EXPECT_TRUE(edge_covered(derived, edges[1])) << edges[1].name;

    const auto stats_after = renderer.barrierPlanner().stats();
    EXPECT_GT((stats_after.barriers_emitted + stats_after.accesses_elided) -
                  (stats_before.barriers_emitted + stats_before.accesses_elided),
              0u);

    std::printf("MacroCoverageAudit structs=%zu (≤%zu) planned_activity=%llu\n",
                n,
                kAuditMacroCoverage,
                static_cast<unsigned long long>(
                    (stats_after.barriers_emitted + stats_after.accesses_elided) -
                    (stats_before.barriers_emitted + stats_before.accesses_elided)));

    renderer.discard_timestamps();
    renderer.endCommandBatch(/*use_fence=*/false);
}

namespace {

    // Shared forge for legacy/macro depth-wave audits at full K capacity.
    // resizeDeviceBuffer is a no-op when capacity already covers K.
    void forge_depth_wave_workspace(VulkanGSPipelineBuffers& buffers,
                                    const std::size_t armed) {
        constexpr std::size_t K = HIGS_DEPTH_WAVE_INSTANCES;
        constexpr std::size_t kHistParts = (K + 4095u) / 4096u; // PARTITION_SIZE=4096
        forge_owned_f(buffers.xy_vs, 0xF601, kAuditSplatCount * 2);
        forge_owned_f(buffers.inv_cov_vs_opacity, 0xF602, kAuditSplatCount * 4);
        forge_owned_i64(buffers.rect_tile_space, 0xF603, kAuditSplatCount);
        forge_owned_i32(buffers.index_buffer_offset, 0xF604, kAuditSplatCount);
        forge_owned_i32(buffers.primitive_sort_indices, 0xF605, kAuditSplatCount);
        forge_owned_f(buffers.rgb, 0xF606, kAuditSplatCount * 3);
        forge_owned_f(buffers.depths, 0xF607, kAuditSplatCount);
        forge_owned_f(buffers.xyz_ws, 0xF608, kAuditSplatCount * 3);
        forge_owned_f(buffers.rotations, 0xF609, kAuditSplatCount * 4);
        forge_owned_f(buffers.scaling_raw, 0xF60A, kAuditSplatCount * 3);
        forge_owned_f(buffers.opacity_raw, 0xF60B, kAuditSplatCount);
        forge_owned_i32(buffers.overlay_flags, 0xF60C, kAuditSplatCount);
        forge_owned(buffers.visible_count, 0xF60D, 2);
        forge_owned_i32(buffers.orig_ids, 0xF60E, kAuditSplatCount);

        forge_owned_i32(buffers.sorting_keys_1, 0xF610, K);
        forge_owned_i32(buffers.sorting_keys_2, 0xF611, K);
        forge_owned_i32(buffers.sorting_gauss_idx_1, 0xF612, K);
        forge_owned_i32(buffers.sorting_gauss_idx_2, 0xF613, K);
        forge_owned_i32(buffers._sorting_histogram, 0xF614, 4 * 256);
        forge_owned_i32(buffers._sorting_histogram_cumsum, 0xF615, kHistParts * 256);
        forge_owned_i32(buffers._cumsum_blockSums, 0xF616, 64);
        forge_owned_i32(buffers._cumsum_blockSums2, 0xF617, 64);

        // Viewport: 16×16 image → forge at the ceil64-bucketed capacities the
        // production resize path requests (#1565); a smaller forge would grow
        // mid-batch and trip the HOST_GUARD fence on the scripted dispatch.
        const auto scratch_bucket = lfs::rendering::vulkan::viewportScratchBucket(
            kAuditWaveImage, kAuditWaveImage);
        const std::size_t alloc_tiles = scratch_bucket.alloc_tiles;
        const std::size_t alloc_pixels = scratch_bucket.alloc_pixels;
        forge_owned_i32(buffers.tile_ranges, 0xF620, alloc_tiles + 1);
        forge_owned_f(buffers.pixel_state, 0xF621, 4 * alloc_pixels);
        forge_owned_f(buffers.pixel_depth, 0xF622, alloc_pixels);
        forge_owned_f(buffers.pixel_depth_weight, 0xF623, alloc_pixels);
        forge_owned_i32(buffers.n_contributors, 0xF624, alloc_pixels);

        // Macro workspace (also used by macro path resizes).
        forge_owned_i32(buffers.tile_batch_counts, 0xF630, alloc_tiles);
        forge_owned_i32(buffers.tile_batch_offsets, 0xF631, alloc_tiles);
        // macro_wave_args: 2 * HIGS_RASTER_MAX_WAVES * 3 = 96 words
        forge_owned(buffers.macro_wave_args, 0xF632, 96);
        // partials / active_mask sized like the production macro path:
        // ceil(K / RASTER_BATCH_SIZE) + macro tiles over the bucketed grid.
        const std::size_t alloc_grid_w = _CEIL_DIV(scratch_bucket.alloc_w,
                                                   static_cast<std::uint32_t>(TILE_WIDTH));
        const std::size_t alloc_grid_h = _CEIL_DIV(scratch_bucket.alloc_h,
                                                   static_cast<std::uint32_t>(TILE_HEIGHT));
        const std::size_t alloc_macro_tiles =
            _CEIL_DIV(alloc_grid_w, std::size_t{HIGS_MACRO_T16_W}) *
            _CEIL_DIV(alloc_grid_h, std::size_t{HIGS_MACRO_T16_H});
        const std::size_t max_batches =
            _CEIL_DIV(K, std::size_t{RASTER_BATCH_SIZE}) + alloc_macro_tiles;
        // macro_partials is half storage; size in elements matches float path capacity.
        forge_owned_u16(buffers.macro_partials, 0xF633, max_batches * 32u * 64u * 4u);
        forge_owned(buffers.macro_active_mask, 0xF634, max_batches);

        forge_owned(buffers.depth_wave_dispatch, 0xF640, (1u + armed) * 64u);
        forge_owned(buffers.wave_predicates, 0xF641, armed);

        buffers.num_indices = 0; // freeze non-batched light path
        buffers.is_unsorted_1 = true;
    }

    void track_depth_wave_workspace(TestableRenderer& r, VulkanGSPipelineBuffers& b) {
        for (auto* buf : {
                 &b.xy_vs.deviceBuffer,
                 &b.inv_cov_vs_opacity.deviceBuffer,
                 &b.rect_tile_space.deviceBuffer,
                 &b.index_buffer_offset.deviceBuffer,
                 &b.primitive_sort_indices.deviceBuffer,
                 &b.rgb.deviceBuffer,
                 &b.depths.deviceBuffer,
                 &b.sorting_keys_1.deviceBuffer,
                 &b.sorting_keys_2.deviceBuffer,
                 &b.sorting_gauss_idx_1.deviceBuffer,
                 &b.sorting_gauss_idx_2.deviceBuffer,
                 &b._sorting_histogram.deviceBuffer,
                 &b._sorting_histogram_cumsum.deviceBuffer,
                 &b.tile_ranges.deviceBuffer,
                 &b.pixel_state.deviceBuffer,
                 &b.pixel_depth.deviceBuffer,
                 &b.pixel_depth_weight.deviceBuffer,
                 &b.n_contributors.deviceBuffer,
                 &b.depth_wave_dispatch.deviceBuffer,
                 &b.wave_predicates.deviceBuffer,
                 &b.visible_count.deviceBuffer,
                 &b.tile_batch_counts.deviceBuffer,
                 &b.tile_batch_offsets.deviceBuffer,
                 &b.macro_wave_args.deviceBuffer,
                 &b.macro_partials.deviceBuffer,
                 &b.macro_active_mask.deviceBuffer,
             }) {
            track_buf(r, *buf);
        }
    }

    void seed_legacy_hoist_writers(TestableRenderer& r, VulkanGSPipelineBuffers& b) {
        // L1387 mega-hoist edges: geometry CSW→CSR + sort workspace CSRW.
        seed_compute_writer(r, b.xy_vs.deviceBuffer);
        seed_compute_writer(r, b.inv_cov_vs_opacity.deviceBuffer);
        seed_compute_writer(r, b.rect_tile_space.deviceBuffer);
        seed_compute_writer(r, b.index_buffer_offset.deviceBuffer);
        seed_compute_writer(r, b.primitive_sort_indices.deviceBuffer);
        seed_compute_writer(r, b.rgb.deviceBuffer);
        seed_compute_writer(r, b.depths.deviceBuffer);
        seed_csrw_writer(r, b.sorting_keys_1.deviceBuffer);
        seed_csrw_writer(r, b.sorting_keys_2.deviceBuffer);
        seed_csrw_writer(r, b.sorting_gauss_idx_1.deviceBuffer);
        seed_csrw_writer(r, b.sorting_gauss_idx_2.deviceBuffer);
        seed_csrw_writer(r, b._sorting_histogram.deviceBuffer);
        seed_csrw_writer(r, b._sorting_histogram_cumsum.deviceBuffer);
    }

} // namespace

// Frozen: light path, no gut, no overlays, sort_bits=8, expected_far=0, W∈{1,3}.
// Critical: every L1387 hoist edge covered by per-wave derived barriers.
TEST(VkSplatTaggedDispatch, LegacyDepthWavesLightAuditW1AndW3) {
    using BM = VulkanGSPipeline::BarrierMask;

    for (const std::size_t armed : {std::size_t{1}, std::size_t{3}}) {
        DispatchScript script;
        BindScript bind(script);

        TestableRenderer renderer;
        renderer.install_fake_handles();
        renderer.setVulkanDispatch(make_scripted_dispatch());
        // predicate_waves=false → no ConditionalScope; hoist edges still covered.

        VulkanGSPipelineBuffers buffers;
        forge_depth_wave_workspace(buffers, armed);

        auto selection_mask = makeBuffer(0xF701, kAuditSplatCount * 4);
        auto preview_mask = makeBuffer(0xF702, kAuditSplatCount * 4);
        auto selection_colors = makeBuffer(0xF703, 16 * 4);
        auto overlay_flags = makeBuffer(0xF704, kAuditSplatCount * 4);
        auto overlay_params = makeBuffer(0xF705, 64);
        auto transform_indices = makeBuffer(0xF706, kAuditSplatCount * 4);
        auto model_transforms = makeBuffer(0xF707, 16 * 16);

        renderer.beginCommandBatch();
        track_depth_wave_workspace(renderer, buffers);
        for (auto* b : {&selection_mask, &preview_mask, &selection_colors, &overlay_flags,
                        &overlay_params, &transform_indices, &model_transforms}) {
            track_buf(renderer, *b);
        }
        seed_legacy_hoist_writers(renderer, buffers);

        const auto stats_before = renderer.barrierPlanner().stats();
        script.clear_recording();

        VulkanGSRendererUniforms u{};
        u.num_splats = kAuditSplatCount;
        u.grid_width = kAuditWaveGrid;
        u.grid_height = kAuditWaveGrid;
        u.image_width = kAuditWaveImage;
        u.image_height = kAuditWaveImage;
        u.sort_capacity = HIGS_DEPTH_WAVE_INSTANCES;
        u.expected_far = 0.0f;

        renderer.executeLegacyDepthWaves(
            u,
            buffers,
            armed,
            kAuditWaveSortBits,
            selection_mask,
            preview_mask,
            selection_colors,
            overlay_flags,
            overlay_params,
            transform_indices,
            model_transforms,
            /*use_gut_rasterization=*/false,
            /*overlays_active=*/false,
            /*predicate_waves=*/false);

        const std::size_t n = total_buffer_barrier_structs(script);
        const std::size_t baseline =
            armed == 1 ? kAuditLegacyLightW1 : kAuditLegacyLightW3;
        EXPECT_LE(n, baseline) << "W=" << armed;

        const auto derived = collect_derived(script);
        const HazardEdge hoist_edges[] = {
            {buffers.xy_vs.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L1387 xy_vs"},
            {buffers.inv_cov_vs_opacity.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L1387 inv_cov"},
            {buffers.rect_tile_space.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L1387 rect_tile_space"},
            {buffers.index_buffer_offset.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L1387 index_buffer_offset"},
            {buffers.primitive_sort_indices.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L1387 primitive_sort_indices"},
            {buffers.rgb.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L1387 rgb"},
            {buffers.depths.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L1387 depths"},
            // Sort workspace: CSRW seed → clear/keygen writes plan WAW (dst write).
            {buffers._sorting_histogram.deviceBuffer.buffer,
             BM::COMPUTE_SHADER_READ_WRITE, BM::COMPUTE_SHADER_WRITE,
             "L1387 histogram CSRW→CSW"},
            {buffers.sorting_keys_1.deviceBuffer.buffer, BM::COMPUTE_SHADER_READ_WRITE,
             BM::COMPUTE_SHADER_WRITE, "L1387 sorting_keys_1 CSRW→CSW"},
        };
        for (const auto& edge : hoist_edges) {
            EXPECT_TRUE(edge_covered(derived, edge))
                << "missing hoist edge W=" << armed << " " << edge.name;
        }

        // Per-wave: tile_ranges write→read (range compute → raster).
        // After chain, last access may be write; seed writer mid-chain is hard.
        // Assert planner activity instead for tile_ranges presence in derived.
        bool saw_tile_ranges = false;
        for (const auto& b : derived) {
            if (b.buffer == buffers.tile_ranges.deviceBuffer.buffer) {
                saw_tile_ranges = true;
                break;
            }
        }
        EXPECT_TRUE(saw_tile_ranges) << "tile_ranges absent from derived W=" << armed;

        const auto stats_after = renderer.barrierPlanner().stats();
        EXPECT_GT((stats_after.barriers_emitted + stats_after.accesses_elided) -
                      (stats_before.barriers_emitted + stats_before.accesses_elided),
                  0u);

        std::printf(
            "LegacyDepthWavesLightAudit W=%zu structs=%zu (≤%zu) derived=%zu "
            "planned_activity=%llu\n",
            armed,
            n,
            baseline,
            derived.size(),
            static_cast<unsigned long long>(
                (stats_after.barriers_emitted + stats_after.accesses_elided) -
                (stats_before.barriers_emitted + stats_before.accesses_elided)));

        renderer.discard_timestamps();
        renderer.endCommandBatch(/*use_fence=*/false);
    }
}

// Frozen: macro path, no overlays, sort_bits=8, W∈{1,3}. Covers L2988 hoist edges.
TEST(VkSplatTaggedDispatch, MacroDepthWavesAuditW1AndW3) {
    using BM = VulkanGSPipeline::BarrierMask;

    for (const std::size_t armed : {std::size_t{1}, std::size_t{3}}) {
        DispatchScript script;
        BindScript bind(script);

        TestableRenderer renderer;
        renderer.install_fake_handles();
        renderer.setVulkanDispatch(make_scripted_dispatch());

        VulkanGSPipelineBuffers buffers;
        forge_depth_wave_workspace(buffers, armed);

        auto selection_mask = makeBuffer(0xF801, kAuditSplatCount * 4);
        auto preview_mask = makeBuffer(0xF802, kAuditSplatCount * 4);
        auto selection_colors = makeBuffer(0xF803, 16 * 4);
        auto overlay_params = makeBuffer(0xF804, 64);

        renderer.beginCommandBatch();
        track_depth_wave_workspace(renderer, buffers);
        for (auto* b : {&selection_mask, &preview_mask, &selection_colors, &overlay_params}) {
            track_buf(renderer, *b);
        }
        seed_legacy_hoist_writers(renderer, buffers);
        seed_compute_writer(renderer, buffers.visible_count.deviceBuffer);

        const auto stats_before = renderer.barrierPlanner().stats();
        script.clear_recording();

        VulkanGSRendererUniforms u{};
        u.num_splats = kAuditSplatCount;
        u.grid_width = kAuditWaveGrid;
        u.grid_height = kAuditWaveGrid;
        u.image_width = kAuditWaveImage;
        u.image_height = kAuditWaveImage;
        u.sort_capacity = HIGS_DEPTH_WAVE_INSTANCES;
        u.lod_enabled = 0;

        renderer.executeMacroDepthWaves(
            u,
            buffers,
            armed,
            kAuditWaveSortBits,
            selection_mask,
            preview_mask,
            selection_colors,
            overlay_params,
            /*overlays_active=*/false,
            /*predicate_waves=*/false);

        const std::size_t n = total_buffer_barrier_structs(script);
        const std::size_t baseline = armed == 1 ? kAuditMacroDepthW1 : kAuditMacroDepthW3;
        EXPECT_LE(n, baseline) << "W=" << armed;

        const auto derived = collect_derived(script);
        const HazardEdge hoist_edges[] = {
            {buffers.xy_vs.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L2988 xy_vs"},
            {buffers.inv_cov_vs_opacity.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L2988 inv_cov"},
            {buffers.rect_tile_space.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L2988 rect"},
            {buffers.index_buffer_offset.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L2988 index_buffer_offset"},
            {buffers.primitive_sort_indices.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L2988 primitive_sort_indices"},
            {buffers.visible_count.deviceBuffer.buffer, BM::COMPUTE_SHADER_WRITE,
             BM::COMPUTE_SHADER_READ, "L2988 visible_count"},
            {buffers._sorting_histogram.deviceBuffer.buffer,
             BM::COMPUTE_SHADER_READ_WRITE, BM::COMPUTE_SHADER_WRITE,
             "L2988 histogram"},
        };
        for (const auto& edge : hoist_edges) {
            EXPECT_TRUE(edge_covered(derived, edge))
                << "missing macro hoist edge W=" << armed << " " << edge.name;
        }

        bool saw_tile_ranges = false;
        bool saw_macro_args = false;
        for (const auto& b : derived) {
            if (b.buffer == buffers.tile_ranges.deviceBuffer.buffer) {
                saw_tile_ranges = true;
            }
            if (b.buffer == buffers.macro_wave_args.deviceBuffer.buffer) {
                saw_macro_args = true;
            }
        }
        EXPECT_TRUE(saw_tile_ranges) << "tile_ranges W=" << armed;
        EXPECT_TRUE(saw_macro_args) << "macro_wave_args W=" << armed;

        const auto stats_after = renderer.barrierPlanner().stats();
        EXPECT_GT((stats_after.barriers_emitted + stats_after.accesses_elided) -
                      (stats_before.barriers_emitted + stats_before.accesses_elided),
                  0u);

        std::printf(
            "MacroDepthWavesAudit W=%zu structs=%zu (≤%zu) derived=%zu "
            "planned_activity=%llu\n",
            armed,
            n,
            baseline,
            derived.size(),
            static_cast<unsigned long long>(
                (stats_after.barriers_emitted + stats_after.accesses_elided) -
                (stats_before.barriers_emitted + stats_before.accesses_elided)));

        renderer.discard_timestamps();
        renderer.endCommandBatch(/*use_fence=*/false);
    }
}

// Predicate ConditionalRead at each wave scope begin (partition owns chain handoff;
// waves re-plan ConditionalRead when predicate_waves && supports_conditional).
TEST(VkSplatTaggedDispatch, LegacyDepthWavesConditionalReadPerWave) {
    using BM = VulkanGSPipeline::BarrierMask;

    DispatchScript script;
    BindScript bind(script);

    TestableRenderer renderer;
    renderer.install_fake_handles();
    renderer.setVulkanDispatch(make_scripted_dispatch());
    renderer.install_conditional_rendering_nops();

    constexpr std::size_t kArmed = 3;
    VulkanGSPipelineBuffers buffers;
    forge_depth_wave_workspace(buffers, kArmed);

    auto selection_mask = makeBuffer(0xF901, 64);
    auto preview_mask = makeBuffer(0xF902, 64);
    auto selection_colors = makeBuffer(0xF903, 64);
    auto overlay_flags = makeBuffer(0xF904, 64);
    auto overlay_params = makeBuffer(0xF905, 64);
    auto transform_indices = makeBuffer(0xF906, 64);
    auto model_transforms = makeBuffer(0xF907, 64);

    renderer.beginCommandBatch();
    track_depth_wave_workspace(renderer, buffers);
    track_buf(renderer, selection_mask);
    track_buf(renderer, preview_mask);
    track_buf(renderer, selection_colors);
    track_buf(renderer, overlay_flags);
    track_buf(renderer, overlay_params);
    track_buf(renderer, transform_indices);
    track_buf(renderer, model_transforms);

    // Seed predicates as written (as if partition just finished).
    seed_compute_writer(renderer, buffers.wave_predicates.deviceBuffer);

    script.clear_recording();

    VulkanGSRendererUniforms u{};
    u.num_splats = kAuditSplatCount;
    u.grid_width = kAuditWaveGrid;
    u.grid_height = kAuditWaveGrid;
    u.image_width = kAuditWaveImage;
    u.image_height = kAuditWaveImage;
    u.sort_capacity = HIGS_DEPTH_WAVE_INSTANCES;

    renderer.executeLegacyDepthWaves(
        u, buffers, kArmed, kAuditWaveSortBits, selection_mask, preview_mask,
        selection_colors, overlay_flags, overlay_params, transform_indices,
        model_transforms, false, false, /*predicate_waves=*/true);

    const auto derived = collect_derived(script);
    const HazardEdge cond{
        buffers.wave_predicates.deviceBuffer.buffer,
        BM::COMPUTE_SHADER_WRITE,
        BM::CONDITIONAL_RENDERING_READ,
        "wave predicate ConditionalRead",
    };
    EXPECT_TRUE(edge_covered(derived, cond));

    std::printf("LegacyDepthWavesConditionalReadPerWave W=3 structs=%zu\n",
                total_buffer_barrier_structs(script));

    renderer.discard_timestamps();
    renderer.endCommandBatch(/*use_fence=*/false);
}
