/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "output_image_pool.hpp"

namespace lfs::vis {

    OutputImagePool::OutputImagePool()
        : pool_([](const VulkanContext::ExternalImage& image) {
              return static_cast<std::size_t>(image.allocation_size);
          }) {}

    std::optional<OutputImagePool::Acquired> OutputImagePool::acquire(const Key& key) {
        auto hit = pool_.acquire(key);
        if (!hit.has_value() || hit->payload == nullptr) {
            return std::nullopt;
        }
        Acquired out;
        out.image = *hit->payload;
        out.acquisition_serial = hit->acquisition_serial;
        return out;
    }

    OutputImagePool::Acquired OutputImagePool::registerCreated(const Key& key,
                                                               VulkanContext::ExternalImage&& image) {
        auto reg = pool_.registerCreated(key, std::move(image));
        Acquired out;
        out.image = reg.payload ? *reg.payload : VulkanContext::ExternalImage{};
        out.acquisition_serial = reg.acquisition_serial;
        return out;
    }

    void OutputImagePool::release(const std::uint64_t acquisition_serial,
                                  const std::uint64_t producer_value,
                                  const std::uint64_t consumer_serial,
                                  const bool evict) {
        pool_.release(acquisition_serial, producer_value, consumer_serial, evict);
    }

    void OutputImagePool::drain(const bool force,
                                const TimelinePred& producer_done,
                                const FramePred& consumer_done,
                                const DestroyFn& destroy) {
        pool_.drain(force, producer_done, consumer_done, destroy);
    }

    void OutputImagePool::trimIdle(const DestroyFn& destroy) {
        pool_.trimIdle(destroy);
    }

    void OutputImagePool::trimAged(const DestroyFn& destroy) {
        pool_.trimAged(destroy);
    }

    std::size_t OutputImagePool::idleBytes() const {
        return pool_.idleBytes();
    }

    std::size_t OutputImagePool::liveCount() const {
        return pool_.liveCount();
    }

    std::size_t OutputImagePool::retiredCount() const {
        return pool_.retiredCount();
    }

    std::size_t OutputImagePool::freeCount() const {
        return pool_.freeCount();
    }

    bool OutputImagePool::misuseFlagged() const {
        return pool_.misuseFlagged();
    }

} // namespace lfs::vis
