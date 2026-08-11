/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/readback_ticket_ring.hpp"

#include <stdexcept>
#include <utility>

namespace lfs::vis {

    void ReadbackTicketRing::checkCell(const std::size_t cell, const std::string_view what) const {
        if (cell >= kRingSize) {
            throw std::out_of_range(std::string(what) + ": readback ticket cell out of range");
        }
    }

    std::optional<std::size_t> ReadbackTicketRing::tryAcquireCell() const noexcept {
        for (std::size_t i = 0; i < kRingSize; ++i) {
            if (cells_[i].state == State::Free) {
                return i;
            }
        }
        return std::nullopt;
    }

    std::optional<std::size_t> ReadbackTicketRing::oldestOutstandingCell() const noexcept {
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

    std::optional<std::size_t> ReadbackTicketRing::oldestActiveCell() const noexcept {
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

    void ReadbackTicketRing::markSubmitted(const std::size_t cell, TicketMeta meta) {
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

    void ReadbackTicketRing::markFailed(const std::size_t cell, std::string error) {
        checkCell(cell, "markFailed");
        // Keep pins (images / ring_cell) until freeCell — GPU copy may still be live.
        cells_[cell].state = State::Failed;
        cells_[cell].error = std::move(error);
    }

    void ReadbackTicketRing::freeCell(const std::size_t cell) noexcept {
        if (cell >= kRingSize) {
            return;
        }
        cells_[cell] = TicketMeta{};
    }

    std::size_t ReadbackTicketRing::reclaimFailedIf(bool (*is_complete)(std::uint64_t ticket, void* ctx),
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

    std::size_t ReadbackTicketRing::failAllOutstanding(const std::string_view reason) {
        std::size_t failed = 0;
        for (std::size_t i = 0; i < kRingSize; ++i) {
            if (cells_[i].state == State::Outstanding) {
                markFailed(i, std::string(reason));
                ++failed;
            }
        }
        return failed;
    }

    ReadbackTicketRing::TicketMeta* ReadbackTicketRing::findByTicket(
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

    const ReadbackTicketRing::TicketMeta* ReadbackTicketRing::findByTicket(
        const std::uint64_t ticket) const noexcept {
        return const_cast<ReadbackTicketRing*>(this)->findByTicket(ticket);
    }

    ReadbackTicketRing::TicketMeta& ReadbackTicketRing::cell(const std::size_t index) {
        checkCell(index, "cell");
        return cells_[index];
    }

    const ReadbackTicketRing::TicketMeta& ReadbackTicketRing::cell(const std::size_t index) const {
        checkCell(index, "cell");
        return cells_[index];
    }

    std::uint64_t ReadbackTicketRing::maxTicketForFrameRingCell(
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

    bool ReadbackTicketRing::hasOutstandingForImage(const VkImage image) const noexcept {
        if (image == VK_NULL_HANDLE) {
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

    std::size_t ReadbackTicketRing::outstandingCount() const noexcept {
        std::size_t count = 0;
        for (const auto& cell : cells_) {
            if (cell.state == State::Outstanding) {
                ++count;
            }
        }
        return count;
    }

    std::size_t ReadbackTicketRing::failedCount() const noexcept {
        std::size_t count = 0;
        for (const auto& cell : cells_) {
            if (cell.state == State::Failed) {
                ++count;
            }
        }
        return count;
    }

    void ReadbackTicketRing::reset() noexcept {
        cells_ = {};
        ring_full_wait_count_ = 0;
        cell_pin_wait_count_ = 0;
    }

} // namespace lfs::vis
