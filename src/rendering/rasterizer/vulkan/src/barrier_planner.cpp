/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "barrier_planner.h"

#include <vector>

namespace lfs::rendering::vulkan {
    namespace {

        [[nodiscard]] bool isNullWriter(const Scope& writer) noexcept {
            return writer.stage == VK_PIPELINE_STAGE_2_NONE && writer.access == VK_ACCESS_2_NONE;
        }

    } // namespace

    BufferBarrierPlanner::BufferBarrierPlanner(uint32_t queue_family_index)
        : queue_family_index_(queue_family_index) {}

    Scope BufferBarrierPlanner::scopeFor(BufferUse use) noexcept {
        switch (use) {
        case BufferUse::ComputeRead:
            return Scope{VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT};
        case BufferUse::ComputeWrite:
            return Scope{VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT};
        case BufferUse::ComputeReadWrite:
            return Scope{
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            };
        case BufferUse::TransferRead:
            return Scope{VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT};
        case BufferUse::TransferWrite:
            return Scope{VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
        case BufferUse::IndirectRead:
            return Scope{
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
            };
        case BufferUse::HostRead:
            return Scope{VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT};
        case BufferUse::ConditionalRead:
            return Scope{
                VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT,
                VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT,
            };
        }
        return Scope{};
    }

    Scope BufferBarrierPlanner::conservativeWriterScope() noexcept {
        // Matches toStageMask/toAccessMask(TRANSFER_COMPUTE_SHADER_WRITE).
        return Scope{
            VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        };
    }

    Scope BufferBarrierPlanner::reuseBarrierDstScope() noexcept {
        // Matches beginCommandBatch reuse barrier dst (gs_pipeline.cpp).
        return Scope{
            VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT |
                VK_ACCESS_2_TRANSFER_WRITE_BIT |
                VK_ACCESS_2_SHADER_READ_BIT |
                VK_ACCESS_2_SHADER_WRITE_BIT |
                VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
        };
    }

    bool BufferBarrierPlanner::isWriteUse(BufferUse use) noexcept {
        switch (use) {
        case BufferUse::ComputeWrite:
        case BufferUse::ComputeReadWrite:
        case BufferUse::TransferWrite:
            return true;
        case BufferUse::ComputeRead:
        case BufferUse::TransferRead:
        case BufferUse::IndirectRead:
        case BufferUse::HostRead:
        case BufferUse::ConditionalRead:
            return false;
        }
        return false;
    }

    bool BufferBarrierPlanner::isVisible(const Scope& a, const BufferState& state) noexcept {
        return (a.stage & ~state.visible_stages) == 0 && (a.access & ~state.visible_access) == 0;
    }

    bool BufferBarrierPlanner::hasWriter(const BufferState& state) noexcept {
        return !isNullWriter(state.writer);
    }

    VkBufferMemoryBarrier2 BufferBarrierPlanner::makeBarrier(VkBuffer buffer, Scope src, Scope dst) const {
        return VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = src.stage,
            .srcAccessMask = src.access,
            .dstStageMask = dst.stage,
            .dstAccessMask = dst.access,
            .srcQueueFamilyIndex = queue_family_index_,
            .dstQueueFamilyIndex = queue_family_index_,
            .buffer = buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
    }

    void BufferBarrierPlanner::track(VkBuffer buffer) {
        if (buffer == VK_NULL_HANDLE) {
            return;
        }
        tracked_.insert(buffer);
        force_conservative_.erase(buffer);
        // Registered-but-never-accessed is EMPTY; insert only if absent so re-track of a live
        // handle does not wipe outstanding state unless the caller forgot first.
        states_.try_emplace(buffer);
    }

    void BufferBarrierPlanner::forget(VkBuffer buffer) {
        if (buffer == VK_NULL_HANDLE) {
            return;
        }
        tracked_.erase(buffer);
        force_conservative_.erase(buffer);
        states_.erase(buffer);
    }

    void BufferBarrierPlanner::invalidate(VkBuffer buffer) {
        if (buffer == VK_NULL_HANDLE) {
            return;
        }
        // Next plan() access takes the conservative row (UNTRACKED-equivalent). One-shot:
        // cleared after that access; tracked handles stay registered.
        force_conservative_.insert(buffer);
        states_.erase(buffer);
    }

    void BufferBarrierPlanner::onBatchBegin() {
        const Scope writer = conservativeWriterScope();
        const Scope visible = reuseBarrierDstScope();
        for (const VkBuffer buffer : tracked_) {
            force_conservative_.erase(buffer);
            states_[buffer] = BufferState{
                .writer = writer,
                .reader_stages = VK_PIPELINE_STAGE_2_NONE,
                .visible_stages = visible.stage,
                .visible_access = visible.access,
            };
        }
    }

    void BufferBarrierPlanner::reset() {
        tracked_.clear();
        force_conservative_.clear();
        states_.clear();
        stats_ = {};
    }

    std::vector<VkBufferMemoryBarrier2> BufferBarrierPlanner::plan(
        std::span<const DeclaredAccess> accesses) {
        // 1) Merge per VkBuffer (simultaneity rule). Preserve first-seen order.
        std::vector<MergedAccess> merged;
        merged.reserve(accesses.size());
        std::unordered_map<VkBuffer, size_t> index_by_buffer;

        for (const DeclaredAccess& access : accesses) {
            if (access.buffer == nullptr || access.buffer->buffer == VK_NULL_HANDLE) {
                continue;
            }
            const VkBuffer handle = access.buffer->buffer;
            const Scope use_scope = scopeFor(access.use);
            const bool write = isWriteUse(access.use);

            const auto it = index_by_buffer.find(handle);
            if (it == index_by_buffer.end()) {
                index_by_buffer.emplace(handle, merged.size());
                merged.push_back(MergedAccess{
                    .handle = handle,
                    .scope = use_scope,
                    .is_write = write,
                });
            } else {
                MergedAccess& m = merged[it->second];
                m.scope.stage |= use_scope.stage;
                m.scope.access |= use_scope.access;
                m.is_write = m.is_write || write;
            }
        }

        // 2) Hazard-check each merged access against pre-plan state; defer state writes.
        struct PendingUpdate {
            VkBuffer handle = VK_NULL_HANDLE;
            BufferState state;
            bool erase_state = false;
            bool clear_force = false;
        };
        std::vector<PendingUpdate> updates;
        updates.reserve(merged.size());

        std::vector<VkBufferMemoryBarrier2> barriers;
        barriers.reserve(merged.size());

        for (const MergedAccess& a : merged) {
            const bool tracked = tracked_.contains(a.handle);
            const bool force = force_conservative_.contains(a.handle);
            const bool conservative_row = !tracked || force;

            if (conservative_row) {
                barriers.push_back(makeBarrier(a.handle, conservativeWriterScope(), a.scope));
                ++stats_.barriers_emitted;
                ++stats_.conservative_fallbacks;

                PendingUpdate u{.handle = a.handle, .clear_force = true};
                if (tracked) {
                    // Resume exact tracking after the conservative barrier. WRITE replaces the
                    // unknown producer; READ keeps the unknown (conservative) writer and records
                    // only the scope the barrier just unlocked — matching onBatchBegin shape.
                    BufferState after{};
                    if (a.is_write) {
                        after.writer = a.scope;
                    } else {
                        after.writer = conservativeWriterScope();
                        after.visible_stages = a.scope.stage;
                        after.visible_access = a.scope.access;
                        after.reader_stages = a.scope.stage;
                    }
                    u.state = after;
                } else {
                    // Untracked: none kept.
                    u.erase_state = true;
                }
                updates.push_back(u);
                continue;
            }

            // Tracked + exact state (missing map entry = EMPTY).
            BufferState prior{};
            if (const auto sit = states_.find(a.handle); sit != states_.end()) {
                prior = sit->second;
            }

            BufferState next = prior;
            bool emit = false;
            Scope src{};

            if (!a.is_write) {
                // Read path.
                if (hasWriter(prior)) {
                    if (isVisible(a.scope, prior)) {
                        emit = false;
                        next.reader_stages |= a.scope.stage;
                    } else {
                        emit = true;
                        src = prior.writer;
                        next.visible_stages |= a.scope.stage;
                        next.visible_access |= a.scope.access;
                        next.reader_stages |= a.scope.stage;
                    }
                } else {
                    // writer=NONE: no barrier.
                    emit = false;
                    next.reader_stages |= a.scope.stage;
                }
            } else {
                // Write / read-write path.
                if (hasWriter(prior)) {
                    emit = true;
                    src = Scope{
                        prior.writer.stage | prior.reader_stages,
                        prior.writer.access,
                    };
                    next.writer = a.scope;
                    next.reader_stages = VK_PIPELINE_STAGE_2_NONE;
                    next.visible_stages = VK_PIPELINE_STAGE_2_NONE;
                    next.visible_access = VK_ACCESS_2_NONE;
                } else if (prior.reader_stages != VK_PIPELINE_STAGE_2_NONE) {
                    emit = true;
                    src = Scope{prior.reader_stages, VK_ACCESS_2_NONE};
                    next.writer = a.scope;
                    next.reader_stages = VK_PIPELINE_STAGE_2_NONE;
                    next.visible_stages = VK_PIPELINE_STAGE_2_NONE;
                    next.visible_access = VK_ACCESS_2_NONE;
                } else {
                    // writer=NONE, readers=NONE: no barrier.
                    emit = false;
                    next.writer = a.scope;
                    next.reader_stages = VK_PIPELINE_STAGE_2_NONE;
                    next.visible_stages = VK_PIPELINE_STAGE_2_NONE;
                    next.visible_access = VK_ACCESS_2_NONE;
                }
            }

            if (emit) {
                barriers.push_back(makeBarrier(a.handle, src, a.scope));
                ++stats_.barriers_emitted;
            } else {
                ++stats_.accesses_elided;
            }
            updates.push_back(PendingUpdate{.handle = a.handle, .state = next});
        }

        // 3) Apply state updates after the full barrier set is determined.
        for (const PendingUpdate& u : updates) {
            if (u.clear_force) {
                force_conservative_.erase(u.handle);
            }
            if (u.erase_state) {
                states_.erase(u.handle);
                continue;
            }
            states_[u.handle] = u.state;
        }

        return barriers;
    }

    BufferBarrierPlanner::Stats BufferBarrierPlanner::stats() const {
        return stats_;
    }

} // namespace lfs::rendering::vulkan
