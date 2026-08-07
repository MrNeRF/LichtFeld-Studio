/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    // Host bookkeeping for the 3-deep viewport readback ticket ring (#1574).
    // GPU-free: timeline complete/wait and Vulkan resources live on the renderer.
    //
    // Producer-side pins (the correctness core after removing the host fence wait):
    // - Ring-cell pin: max outstanding/failed ticket sourcing a frame-ring cell blocks reuse.
    // - Pool pin: non-Free tickets hold source VkImage handles against drain until freeCell.
    //   Production drain pins by image handle (OutputImagePool::consumer_pred receives the
    //   consumer *frame* serial, not the acquisition serial — see drainOutputImagePool).
    //
    // Failed lifecycle: markFailed keeps pins until the ticket timeline is observed complete,
    // then freeCell reclaims the slot (see reclaimFailedIf). reset() fails + frees all.
    class LFS_VIS_API ReadbackTicketRing {
    public:
        static constexpr std::size_t kRingSize = 3;

        enum class State : std::uint8_t {
            Free = 0,
            Outstanding = 1,
            Failed = 2,
        };

        enum class DeliveryKind : std::uint8_t {
            ColorHwc = 1,
            DepthFloatPlane = 2,
            DepthSample = 3,
        };

        enum class PollStatus : std::uint8_t {
            NotReady = 0,
            Ready = 1,
            Failed = 2,
            UnknownTicket = 3,
        };

        struct TicketMeta {
            std::uint64_t ticket_value = 0;
            std::size_t ring_cell = 0; // OutputSlotRing frame cell (0..2)
            // Source VkImage(s) for pool-pin matching (1:1 with acquisition while retired).
            VkImage source_image = VK_NULL_HANDLE;
            VkImage source_depth_image = VK_NULL_HANDLE;
            std::uint64_t byte_count = 0;
            int width = 0;
            int height = 0;
            int dest_x = 0;
            int dest_y = 0;
            int dest_width = 0;
            int dest_channels = 0;
            bool dest_is_float = false;
            DeliveryKind delivery = DeliveryKind::ColorHwc;
            State state = State::Free;
            std::string error;
            // Non-owning destination filled on Ready delivery.
            void* dest = nullptr;
        };

        // First free cell, or nullopt when the ring is full.
        [[nodiscard]] std::optional<std::size_t> tryAcquireCell() const noexcept;

        // Oldest Outstanding cell by ticket_value (nullopt if none outstanding).
        [[nodiscard]] std::optional<std::size_t> oldestOutstandingCell() const noexcept;

        // Oldest non-Free cell (Outstanding or Failed) by ticket_value — for ring-full wait.
        [[nodiscard]] std::optional<std::size_t> oldestActiveCell() const noexcept;

        // Mark cell as outstanding with the given meta (ticket_value must be > 0).
        void markSubmitted(std::size_t cell, TicketMeta meta);

        // Mark cell Failed with a diagnostic (reset / wait failure). Leaves meta queryable
        // and keeps producer pins until freeCell (GPU work may still be in flight).
        void markFailed(std::size_t cell, std::string error);

        // Free a cell after successful delivery or safe abandon (timeline complete / idle).
        void freeCell(std::size_t cell) noexcept;

        // Free every Failed cell whose ticket_value satisfies is_complete(ticket).
        // Returns the number of cells freed.
        std::size_t reclaimFailedIf(bool (*is_complete)(std::uint64_t ticket, void* ctx),
                                    void* ctx) noexcept;

        // Fail every Outstanding cell (device idle / teardown). Returns count failed.
        std::size_t failAllOutstanding(std::string_view reason);

        [[nodiscard]] TicketMeta* findByTicket(std::uint64_t ticket) noexcept;
        [[nodiscard]] const TicketMeta* findByTicket(std::uint64_t ticket) const noexcept;
        [[nodiscard]] TicketMeta& cell(std::size_t index);
        [[nodiscard]] const TicketMeta& cell(std::size_t index) const;

        // Max non-Free ticket_value sourcing OutputSlotRing frame cell `ring_cell`.
        // 0 means no pin. Includes Failed so pins hold until freeCell.
        [[nodiscard]] std::uint64_t maxTicketForFrameRingCell(std::size_t ring_cell) const noexcept;

        // True if any non-Free ticket sources this VkImage (pool drain pin).
        [[nodiscard]] bool hasOutstandingForImage(VkImage image) const noexcept;

        [[nodiscard]] std::size_t outstandingCount() const noexcept;
        [[nodiscard]] std::size_t failedCount() const noexcept;
        [[nodiscard]] std::uint64_t ringFullWaitCount() const noexcept { return ring_full_wait_count_; }
        [[nodiscard]] std::uint64_t cellPinWaitCount() const noexcept { return cell_pin_wait_count_; }

        void noteRingFullWait() noexcept { ++ring_full_wait_count_; }
        void noteCellPinWait() noexcept { ++cell_pin_wait_count_; }

        void reset() noexcept;

    private:
        void checkCell(std::size_t cell, std::string_view what) const;
        [[nodiscard]] static bool pinsActive(State state) noexcept {
            return state == State::Outstanding || state == State::Failed;
        }

        std::array<TicketMeta, kRingSize> cells_{};
        std::uint64_t ring_full_wait_count_ = 0;
        std::uint64_t cell_pin_wait_count_ = 0;
    };

} // namespace lfs::vis
