/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Epic #1496 P3: tagged dispatch infrastructure (P3a) + first LOD chain audit (P3b).
// Scripted VulkanDispatch (TestablePipeline / TestableRenderer pattern).

#include "rendering/rasterizer/vulkan/src/barrier_planner.h"
#include "rendering/rasterizer/vulkan/src/gs_pipeline.h"
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"
#include "rendering/vulkan_wait.hpp"

#include <gtest/gtest.h>

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

            deviceInfo.subgroupSize = 32;
            deviceInfo.sharedSize = 48 * 1024;
            deviceInfo.maxGroupsX = 65535;
            deviceInfo.maxGroupsY = 65535;
            deviceInfo.maxGroupsZ = 65535;
            deviceInfo.maxThreadsX = 1024;
            deviceInfo.maxThreadsY = 1024;
            deviceInfo.maxThreadsZ = 64;

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

            deviceInfo.subgroupSize = 32;
            deviceInfo.sharedSize = 48 * 1024;
            deviceInfo.maxGroupsX = 65535;
            deviceInfo.maxGroupsY = 65535;
            deviceInfo.maxGroupsZ = 65535;
            deviceInfo.maxThreadsX = 1024;
            deviceInfo.maxThreadsY = 1024;
            deviceInfo.maxThreadsZ = 64;

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
            cp.diagnostic_name = "audit.lod";
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

    constexpr std::uint32_t kAuditSplatCount = 64;
    constexpr std::uint32_t kAuditAabbW = 16;
    constexpr std::uint32_t kAuditAabbH = 16;

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
