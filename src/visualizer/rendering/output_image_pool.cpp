/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "output_image_pool.hpp"

#include "window/vulkan_result.hpp"

#include <algorithm>
#include <utility>

namespace lfs::vis {

    std::uint64_t OutputImagePool::nextSerial() {
        return next_serial_++;
    }

    void OutputImagePool::destroyEntry(Entry& entry, const DestroyFn& destroy) {
        if (destroy) {
            destroy(entry.image);
        }
        entry.image = {};
    }

    std::optional<OutputImagePool::Acquired> OutputImagePool::acquire(const Key& key) {
        for (auto it = free_serials_.begin(); it != free_serials_.end(); ++it) {
            const auto entry_it = entries_.find(*it);
            if (entry_it == entries_.end() || entry_it->second.state != State::Free) {
                continue;
            }
            if (!(entry_it->second.key == key)) {
                continue;
            }
            free_serials_.erase(it);
            Entry& entry = entry_it->second;
            entry.acquisition_serial = nextSerial();
            entry.state = State::Live;
            entry.producer_value = 0;
            entry.consumer_serial = 0;
            entry.free_since_tick = 0;
            ++live_count_;

            // Re-key map under the fresh serial (serial is never reused).
            const std::uint64_t serial = entry.acquisition_serial;
            Entry moved = std::move(entry);
            entries_.erase(entry_it);
            entries_.emplace(serial, std::move(moved));

            Acquired out;
            out.image = entries_.at(serial).image;
            out.acquisition_serial = serial;
            return out;
        }
        return std::nullopt;
    }

    OutputImagePool::Acquired OutputImagePool::registerCreated(const Key& key,
                                                               VulkanContext::ExternalImage&& image) {
        const std::uint64_t serial = nextSerial();
        Entry entry;
        entry.key = key;
        entry.image = std::move(image);
        entry.acquisition_serial = serial;
        entry.state = State::Live;
        entries_.emplace(serial, std::move(entry));
        ++live_count_;

        Acquired out;
        out.image = entries_.at(serial).image;
        out.acquisition_serial = serial;
        return out;
    }

    void OutputImagePool::release(const std::uint64_t acquisition_serial,
                                  const std::uint64_t producer_value,
                                  const std::uint64_t consumer_serial) {
        const auto it = entries_.find(acquisition_serial);
        if (it == entries_.end() || it->second.state != State::Live) {
            misuse_flagged_ = true;
            LFS_VK_DEBUG_ASSERT(false, "OutputImagePool::release: unknown or non-live serial {}",
                                acquisition_serial);
            return;
        }
        Entry& entry = it->second;
        entry.producer_value = producer_value;
        entry.consumer_serial = consumer_serial;
        entry.state = State::Retired;
        --live_count_;
        ++retired_count_;
    }

    void OutputImagePool::drain(const bool force,
                                const TimelinePred& producer_done,
                                const FramePred& consumer_done,
                                const DestroyFn& destroy) {
        ++drain_tick_;

        if (force) {
            std::vector<std::uint64_t> to_erase;
            to_erase.reserve(entries_.size());
            for (auto& [serial, entry] : entries_) {
                if (entry.state == State::Live) {
                    continue;
                }
                destroyEntry(entry, destroy);
                if (entry.state == State::Retired) {
                    --retired_count_;
                }
                to_erase.push_back(serial);
            }
            free_serials_.clear();
            for (const std::uint64_t serial : to_erase) {
                entries_.erase(serial);
            }
            return;
        }

        std::vector<std::uint64_t> freed;
        for (auto& [serial, entry] : entries_) {
            if (entry.state != State::Retired) {
                continue;
            }
            const bool prod_ok = !producer_done || producer_done(entry.producer_value);
            const bool cons_ok = !consumer_done || consumer_done(entry.consumer_serial);
            if (!prod_ok || !cons_ok) {
                continue;
            }
            entry.state = State::Free;
            entry.free_since_tick = drain_tick_;
            --retired_count_;
            freed.push_back(serial);
        }
        free_serials_.insert(free_serials_.end(), freed.begin(), freed.end());
    }

    void OutputImagePool::trimIdle(const DestroyFn& destroy) {
        for (const std::uint64_t serial : free_serials_) {
            const auto it = entries_.find(serial);
            if (it == entries_.end() || it->second.state != State::Free) {
                continue;
            }
            destroyEntry(it->second, destroy);
            entries_.erase(it);
        }
        free_serials_.clear();
    }

    void OutputImagePool::trimAged(const DestroyFn& destroy) {
        std::vector<std::uint64_t> keep;
        keep.reserve(free_serials_.size());
        for (const std::uint64_t serial : free_serials_) {
            const auto it = entries_.find(serial);
            if (it == entries_.end() || it->second.state != State::Free) {
                continue;
            }
            const std::uint64_t age = drain_tick_ - it->second.free_since_tick;
            if (age > kIdleTrimTicks) {
                destroyEntry(it->second, destroy);
                entries_.erase(it);
            } else {
                keep.push_back(serial);
            }
        }
        free_serials_ = std::move(keep);
    }

    std::size_t OutputImagePool::idleBytes() const {
        std::size_t bytes = 0;
        for (const auto& [serial, entry] : entries_) {
            if (entry.state != State::Live) {
                bytes += static_cast<std::size_t>(entry.image.allocation_size);
            }
        }
        return bytes;
    }

    std::size_t OutputImagePool::liveCount() const {
        return live_count_;
    }

    std::size_t OutputImagePool::retiredCount() const {
        return retired_count_;
    }

    std::size_t OutputImagePool::freeCount() const {
        return free_serials_.size();
    }

    bool OutputImagePool::misuseFlagged() const {
        return misuse_flagged_;
    }

} // namespace lfs::vis
