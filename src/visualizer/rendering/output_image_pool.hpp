/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_result.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    // Bucket allocation extent for viewport output images (64-px).
    [[nodiscard]] constexpr std::uint32_t ceil64(const std::uint32_t v) noexcept {
        return ((v + 63u) / 64u) * 64u;
    }

    // Map full-image UV [0,1] onto the valid top-left sub-rect of a padded output.
    // Per axis: scale = valid/alloc; 1.0 when alloc<=0 or valid>=alloc.
    [[nodiscard]] inline glm::vec2 outputUvScale(const glm::ivec2 valid,
                                                 const glm::ivec2 alloc) noexcept {
        glm::vec2 scale{1.0f, 1.0f};
        if (alloc.x > 0 && valid.x > 0 && valid.x < alloc.x) {
            scale.x = static_cast<float>(valid.x) / static_cast<float>(alloc.x);
        }
        if (alloc.y > 0 && valid.y > 0 && valid.y < alloc.y) {
            scale.y = static_cast<float>(valid.y) / static_cast<float>(alloc.y);
        }
        return scale;
    }

    // Half-texel max UV to stop linear-filter bleed from padding.
    // Per axis: (valid - 0.5) / alloc; 1.0 when alloc<=0 or valid<=0.
    [[nodiscard]] inline glm::vec2 outputUvClampMax(const glm::ivec2 valid,
                                                    const glm::ivec2 alloc) noexcept {
        glm::vec2 clamp_max{1.0f, 1.0f};
        if (alloc.x > 0 && valid.x > 0) {
            clamp_max.x = (static_cast<float>(valid.x) - 0.5f) / static_cast<float>(alloc.x);
        }
        if (alloc.y > 0 && valid.y > 0) {
            clamp_max.y = (static_cast<float>(valid.y) - 0.5f) / static_cast<float>(alloc.y);
        }
        return clamp_max;
    }

    // Shared key for GPU resource free-lists (format/extent/usage/external).
    struct GpuResourcePoolKey {
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{0, 0};
        VkImageUsageFlags usage = 0;
        bool external = false;

        [[nodiscard]] bool operator==(const GpuResourcePoolKey& other) const noexcept {
            return format == other.format && extent.width == other.extent.width &&
                   extent.height == other.extent.height && usage == other.usage &&
                   external == other.external;
        }
    };

    // Host-only Live/Retired/Free state machine for arbitrary payloads.
    // Payload is owned exclusively by the pool (unique_ptr); Acquired holds a
    // stable non-owning pointer valid while the entry is Live under that serial.
    // Producer drain predicate is payload-aware; consumer is a frame serial.
    template <typename Payload>
    class GpuResourcePool {
    public:
        static constexpr std::uint64_t kIdleTrimTicks = 240;

        using Key = GpuResourcePoolKey;

        struct Acquired {
            Payload* payload = nullptr;
            std::uint64_t acquisition_serial = 0;
        };

        using DestroyFn = std::function<void(Payload&)>;
        using ProducerPred = std::function<bool(const Payload&, std::uint64_t)>;
        using FramePred = std::function<bool(std::uint64_t)>;
        using BytesFn = std::function<std::size_t(const Payload&)>;

        explicit GpuResourcePool(BytesFn bytes_fn = {})
            : bytes_fn_(std::move(bytes_fn)) {}

        [[nodiscard]] std::optional<Acquired> acquire(const Key& key) {
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
                entry.evict = false;
                ++live_count_;

                // Re-key map under the fresh serial (serial is never reused).
                const std::uint64_t serial = entry.acquisition_serial;
                Entry moved = std::move(entry);
                entries_.erase(entry_it);
                entries_.emplace(serial, std::move(moved));

                Acquired out;
                out.payload = entries_.at(serial).payload.get();
                out.acquisition_serial = serial;
                return out;
            }
            return std::nullopt;
        }

        [[nodiscard]] Acquired registerCreated(const Key& key, Payload&& payload) {
            const std::uint64_t serial = nextSerial();
            Entry entry;
            entry.key = key;
            entry.payload = std::make_unique<Payload>(std::move(payload));
            entry.acquisition_serial = serial;
            entry.state = State::Live;
            entries_.emplace(serial, std::move(entry));
            ++live_count_;

            Acquired out;
            out.payload = entries_.at(serial).payload.get();
            out.acquisition_serial = serial;
            return out;
        }

        // Double/unknown release: debug assert + misuseFlagged() + ignore.
        // evict=true: on successful non-force drain, destroy instead of free-listing.
        void release(const std::uint64_t acquisition_serial,
                     const std::uint64_t producer_value,
                     const std::uint64_t consumer_serial,
                     const bool evict = false) {
            const auto it = entries_.find(acquisition_serial);
            if (it == entries_.end() || it->second.state != State::Live) {
                misuse_flagged_ = true;
                LFS_VK_DEBUG_ASSERT(false,
                                    "GpuResourcePool::release: unknown or non-live serial {}",
                                    acquisition_serial);
                return;
            }
            Entry& entry = it->second;
            entry.producer_value = producer_value;
            entry.consumer_serial = consumer_serial;
            entry.evict = evict;
            entry.state = State::Retired;
            --live_count_;
            ++retired_count_;
        }

        // force=true destroys retired+free (never live). Otherwise both predicates must pass
        // to move retired → free (or destroy if evict).
        void drain(const bool force,
                   const ProducerPred& producer_done,
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
            std::vector<std::uint64_t> evicted;
            for (auto& [serial, entry] : entries_) {
                if (entry.state != State::Retired) {
                    continue;
                }
                const Payload& payload_ref = *entry.payload;
                const bool prod_ok =
                    !producer_done || producer_done(payload_ref, entry.producer_value);
                const bool cons_ok = !consumer_done || consumer_done(entry.consumer_serial);
                if (!prod_ok || !cons_ok) {
                    continue;
                }
                --retired_count_;
                if (entry.evict) {
                    destroyEntry(entry, destroy);
                    evicted.push_back(serial);
                } else {
                    entry.state = State::Free;
                    entry.free_since_tick = drain_tick_;
                    freed.push_back(serial);
                }
            }
            free_serials_.insert(free_serials_.end(), freed.begin(), freed.end());
            for (const std::uint64_t serial : evicted) {
                entries_.erase(serial);
            }
        }

        void trimIdle(const DestroyFn& destroy) {
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

        // Destroy free entries idle for more than kIdleTrimTicks drain ticks.
        void trimAged(const DestroyFn& destroy) {
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

        // Bytes held by entries not bound to any slot (retired + free).
        [[nodiscard]] std::size_t idleBytes() const {
            if (!bytes_fn_) {
                return 0;
            }
            std::size_t bytes = 0;
            for (const auto& [serial, entry] : entries_) {
                if (entry.state != State::Live && entry.payload) {
                    bytes += bytes_fn_(*entry.payload);
                }
            }
            return bytes;
        }

        [[nodiscard]] std::size_t liveCount() const { return live_count_; }
        [[nodiscard]] std::size_t retiredCount() const { return retired_count_; }
        [[nodiscard]] std::size_t freeCount() const { return free_serials_.size(); }
        [[nodiscard]] bool misuseFlagged() const { return misuse_flagged_; }

        // Non-owning lookup; valid only while entry is Live under this serial.
        [[nodiscard]] Payload* tryGetLive(const std::uint64_t acquisition_serial) {
            const auto it = entries_.find(acquisition_serial);
            if (it == entries_.end() || it->second.state != State::Live) {
                return nullptr;
            }
            return it->second.payload.get();
        }

        [[nodiscard]] const Payload* tryGetLive(const std::uint64_t acquisition_serial) const {
            const auto it = entries_.find(acquisition_serial);
            if (it == entries_.end() || it->second.state != State::Live) {
                return nullptr;
            }
            return it->second.payload.get();
        }

    private:
        enum class State : std::uint8_t {
            Live,
            Retired,
            Free,
        };

        struct Entry {
            Key key{};
            std::unique_ptr<Payload> payload;
            std::uint64_t acquisition_serial = 0;
            std::uint64_t producer_value = 0;
            std::uint64_t consumer_serial = 0;
            std::uint64_t free_since_tick = 0;
            State state = State::Live;
            bool evict = false;
        };

        [[nodiscard]] std::uint64_t nextSerial() { return next_serial_++; }

        void destroyEntry(Entry& entry, const DestroyFn& destroy) {
            if (destroy && entry.payload) {
                destroy(*entry.payload);
            }
            entry.payload.reset();
        }

        std::unordered_map<std::uint64_t, Entry> entries_;
        std::vector<std::uint64_t> free_serials_;
        std::uint64_t next_serial_ = 1;
        std::uint64_t drain_tick_ = 0;
        std::size_t live_count_ = 0;
        std::size_t retired_count_ = 0;
        bool misuse_flagged_ = false;
        BytesFn bytes_fn_;
    };

    // Thin wrapper over GpuResourcePool<ExternalImage> with the historical public API
    // (Acquired.image by value, Key alias, idleBytes from allocation_size).
    class LFS_VIS_API OutputImagePool {
    public:
        static constexpr std::uint64_t kIdleTrimTicks = GpuResourcePool<VulkanContext::ExternalImage>::kIdleTrimTicks;

        using Key = GpuResourcePoolKey;
        using DestroyFn = GpuResourcePool<VulkanContext::ExternalImage>::DestroyFn;
        // Payload-aware producer predicate (mechanical upgrade from value-only).
        using TimelinePred = GpuResourcePool<VulkanContext::ExternalImage>::ProducerPred;
        using FramePred = GpuResourcePool<VulkanContext::ExternalImage>::FramePred;

        struct Acquired {
            VulkanContext::ExternalImage image{};
            std::uint64_t acquisition_serial = 0;
        };

        OutputImagePool();

        [[nodiscard]] std::optional<Acquired> acquire(const Key& key);

        [[nodiscard]] Acquired registerCreated(const Key& key, VulkanContext::ExternalImage&& image);

        void release(std::uint64_t acquisition_serial,
                     std::uint64_t producer_value,
                     std::uint64_t consumer_serial,
                     bool evict = false);

        void drain(bool force,
                   const TimelinePred& producer_done,
                   const FramePred& consumer_done,
                   const DestroyFn& destroy);

        void trimIdle(const DestroyFn& destroy);
        void trimAged(const DestroyFn& destroy);

        [[nodiscard]] std::size_t idleBytes() const;
        [[nodiscard]] std::size_t liveCount() const;
        [[nodiscard]] std::size_t retiredCount() const;
        [[nodiscard]] std::size_t freeCount() const;
        [[nodiscard]] bool misuseFlagged() const;

    private:
        GpuResourcePool<VulkanContext::ExternalImage> pool_;
    };

} // namespace lfs::vis
