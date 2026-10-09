/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <stdexcept>
#include <utility>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace lfs::vis {

    // Host bookkeeping for the 3-deep viewport readback ticket ring (#1574).
    // GPU-free: timeline complete/wait and Vulkan resources live on the renderer.
    //
    // Producer-side pins (the correctness core after removing the host fence wait):
    // - Ring-cell pin: max outstanding/failed ticket sourcing a frame-ring cell blocks reuse.
    // - Pool pin: non-Free tickets hold source ResourceKey handles against drain until freeCell.
    //   Production drain pins by image handle (OutputImagePool::consumer_pred receives the
    //   consumer *frame* serial, not the acquisition serial — see drainOutputImagePool).
    //
    // Failed lifecycle: markFailed keeps pins until the ticket timeline is observed complete,
    // then freeCell reclaims the slot (see reclaimFailedIf). reset() fails + frees all.
    template <typename ResourceKey>
    class BasicReadbackTicketRing {
    public:
        static constexpr std::size_t kRingSize = 3;

        enum class State : std::uint8_t {
            Free = 0,
            Outstanding = 1,
            Failed = 2,
        };

        enum class DeliveryKind : std::uint8_t {
            ColorHwc = 1,
            FloatBuffer = 2,
            DepthSample = 3,
        };

        struct TicketMeta {
            std::uint64_t ticket_value = 0;
            std::size_t ring_cell = 0; // Globally unique submission cell across target columns.
            // Source ResourceKey(s) for pool-pin matching (1:1 with acquisition while retired).
            ResourceKey source_image = ResourceKey{};
            ResourceKey source_depth_image = ResourceKey{};
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

        // True if any non-Free ticket sources this ResourceKey (pool drain pin).
        [[nodiscard]] bool hasOutstandingForImage(ResourceKey image) const noexcept;

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

    template <typename ResourceKey>
    void BasicReadbackTicketRing<ResourceKey>::checkCell(const std::size_t cell, const std::string_view what) const {
        if (cell >= kRingSize) {
            throw std::out_of_range(std::string(what) + ": readback ticket cell out of range");
        }
    }

    template <typename ResourceKey>
    std::optional<std::size_t> BasicReadbackTicketRing<ResourceKey>::tryAcquireCell() const noexcept {
        for (std::size_t i = 0; i < kRingSize; ++i) {
            if (cells_[i].state == State::Free) {
                return i;
            }
        }
        return std::nullopt;
    }

    template <typename ResourceKey>
    std::optional<std::size_t> BasicReadbackTicketRing<ResourceKey>::oldestOutstandingCell() const noexcept {
        std::optional<std::size_t> oldest;
        std::uint64_t oldest_ticket = 0;
        for (std::size_t i = 0; i < kRingSize; ++i) {
            if (cells_[i].state != State::Outstanding) {
                continue;
            }
            if (!oldest.has_value() || cells_[i].ticket_value < oldest_ticket) {
                oldest = i;
                oldest_ticket = cells_[i].ticket_value;
            }
        }
        return oldest;
    }

    template <typename ResourceKey>
    std::optional<std::size_t> BasicReadbackTicketRing<ResourceKey>::oldestActiveCell() const noexcept {
        std::optional<std::size_t> oldest;
        std::uint64_t oldest_ticket = 0;
        for (std::size_t i = 0; i < kRingSize; ++i) {
            if (!pinsActive(cells_[i].state)) {
                continue;
            }
            if (!oldest.has_value() || cells_[i].ticket_value < oldest_ticket) {
                oldest = i;
                oldest_ticket = cells_[i].ticket_value;
            }
        }
        return oldest;
    }

    template <typename ResourceKey>
    void BasicReadbackTicketRing<ResourceKey>::markSubmitted(const std::size_t cell, TicketMeta meta) {
        checkCell(cell, "markSubmitted");
        if (meta.ticket_value == 0) {
            throw std::invalid_argument("markSubmitted requires a non-zero ticket_value");
        }
        if (cells_[cell].state != State::Free) {
            throw std::logic_error("markSubmitted on non-free readback cell");
        }
        meta.state = State::Outstanding;
        cells_[cell] = std::move(meta);
    }

    template <typename ResourceKey>
    void BasicReadbackTicketRing<ResourceKey>::markFailed(const std::size_t cell, std::string error) {
        checkCell(cell, "markFailed");
        // Keep pins (images / ring_cell) until freeCell — GPU copy may still be live.
        cells_[cell].state = State::Failed;
        cells_[cell].error = std::move(error);
    }

    template <typename ResourceKey>
    void BasicReadbackTicketRing<ResourceKey>::freeCell(const std::size_t cell) noexcept {
        if (cell >= kRingSize) {
            return;
        }
        cells_[cell] = TicketMeta{};
    }

    template <typename ResourceKey>
    std::size_t BasicReadbackTicketRing<ResourceKey>::reclaimFailedIf(bool (*is_complete)(std::uint64_t ticket, void* ctx),
                                                                      void* ctx) noexcept {
        if (is_complete == nullptr) {
            return 0;
        }
        std::size_t freed = 0;
        for (std::size_t i = 0; i < kRingSize; ++i) {
            if (cells_[i].state != State::Failed) {
                continue;
            }
            const std::uint64_t ticket = cells_[i].ticket_value;
            if (ticket == 0 || !is_complete(ticket, ctx)) {
                continue;
            }
            freeCell(i);
            ++freed;
        }
        return freed;
    }

    template <typename ResourceKey>
    std::size_t BasicReadbackTicketRing<ResourceKey>::failAllOutstanding(const std::string_view reason) {
        std::size_t failed = 0;
        for (std::size_t i = 0; i < kRingSize; ++i) {
            if (cells_[i].state == State::Outstanding) {
                markFailed(i, std::string(reason));
                ++failed;
            }
        }
        return failed;
    }

    template <typename ResourceKey>
    typename BasicReadbackTicketRing<ResourceKey>::TicketMeta* BasicReadbackTicketRing<ResourceKey>::findByTicket(
        const std::uint64_t ticket) noexcept {
        if (ticket == 0) {
            return nullptr;
        }
        for (auto& cell : cells_) {
            if (cell.ticket_value == ticket && cell.state != State::Free) {
                return &cell;
            }
        }
        return nullptr;
    }

    template <typename ResourceKey>
    const typename BasicReadbackTicketRing<ResourceKey>::TicketMeta* BasicReadbackTicketRing<ResourceKey>::findByTicket(
        const std::uint64_t ticket) const noexcept {
        return const_cast<BasicReadbackTicketRing<ResourceKey>*>(this)->findByTicket(ticket);
    }

    template <typename ResourceKey>
    typename BasicReadbackTicketRing<ResourceKey>::TicketMeta& BasicReadbackTicketRing<ResourceKey>::cell(const std::size_t index) {
        checkCell(index, "cell");
        return cells_[index];
    }

    template <typename ResourceKey>
    const typename BasicReadbackTicketRing<ResourceKey>::TicketMeta& BasicReadbackTicketRing<ResourceKey>::cell(const std::size_t index) const {
        checkCell(index, "cell");
        return cells_[index];
    }

    template <typename ResourceKey>
    std::uint64_t BasicReadbackTicketRing<ResourceKey>::maxTicketForFrameRingCell(
        const std::size_t ring_cell) const noexcept {
        std::uint64_t max_ticket = 0;
        for (const auto& cell : cells_) {
            if (!pinsActive(cell.state)) {
                continue;
            }
            if (cell.ring_cell == ring_cell && cell.ticket_value > max_ticket) {
                max_ticket = cell.ticket_value;
            }
        }
        return max_ticket;
    }

    template <typename ResourceKey>
    bool BasicReadbackTicketRing<ResourceKey>::hasOutstandingForImage(const ResourceKey image) const noexcept {
        if (image == ResourceKey{}) {
            return false;
        }
        for (const auto& cell : cells_) {
            if (!pinsActive(cell.state)) {
                continue;
            }
            if (cell.source_image == image || cell.source_depth_image == image) {
                return true;
            }
        }
        return false;
    }

    template <typename ResourceKey>
    std::size_t BasicReadbackTicketRing<ResourceKey>::outstandingCount() const noexcept {
        std::size_t count = 0;
        for (const auto& cell : cells_) {
            if (cell.state == State::Outstanding) {
                ++count;
            }
        }
        return count;
    }

    template <typename ResourceKey>
    std::size_t BasicReadbackTicketRing<ResourceKey>::failedCount() const noexcept {
        std::size_t count = 0;
        for (const auto& cell : cells_) {
            if (cell.state == State::Failed) {
                ++count;
            }
        }
        return count;
    }

    template <typename ResourceKey>
    void BasicReadbackTicketRing<ResourceKey>::reset() noexcept {
        cells_ = {};
        ring_full_wait_count_ = 0;
        cell_pin_wait_count_ = 0;
    }

} // namespace lfs::vis
