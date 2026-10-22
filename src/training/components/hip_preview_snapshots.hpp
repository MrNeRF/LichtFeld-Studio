/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_upload.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>

namespace lfs::training {
    // advance/reset belong to the producer thread; request only touches CPU
    // state. Copies are enqueued before the producer's next parameter writes.
    class HipPreviewSnapshots {
        using Clock = std::chrono::steady_clock;
        using Model = lfs::core::SplatData;
        using Tensor = lfs::core::Tensor;
        static constexpr size_t kBudget = 256u * 1024u * 1024u;
        inline static std::atomic<uint64_t> next_generation_{1};
        struct Budget { std::atomic<size_t> cpu_bytes{0}; };
        std::shared_ptr<Budget> budget_ = std::make_shared<Budget>();
        std::mutex mutex_;
        std::shared_ptr<const Model> published_;
        std::atomic<int64_t> requested_{0};
        std::atomic<int64_t> retry_after_{0};
        std::unique_ptr<lfs::core::TensorReadbackRing> ring_;
        std::unique_ptr<Model> pending_;
        std::array<size_t, 8> offsets_{};
        size_t ring_bytes_ = 0;
        size_t pending_bytes_ = 0;
        Clock::time_point next_capture_{};
        bool warned_ = false;
        bool logged_publication_ = false;

        static auto fields(Model& model) {
            return std::array<Tensor*, 8>{&model.means_raw(), &model.sh0_raw(),
                &model.rotation_raw(), &model.scaling_raw(), &model.opacity_raw(),
                &model.shN_raw(), &model.shN_value_bounds(), &model.deleted()};
        }
        static int64_t ticks(Clock::time_point time) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
        }
        void warn(const char* reason) {
            { std::lock_guard lock(mutex_); published_.reset(); }
            logged_publication_ = false;
            if (!warned_) {
                LOG_WARN("GPU transfer fallback: role=HIP training preview, requested=asynchronous snapshot, effective=synchronized live model, reason={}", reason);
                warned_ = true;
            }
        }
    public:
        std::shared_ptr<const Model> request() {
            const auto now = ticks(Clock::now());
            if (now < retry_after_.load(std::memory_order_acquire))
                return nullptr;
            requested_.store(now, std::memory_order_release);
            std::lock_guard lock(mutex_);
            return published_;
        }
        void suspend(std::string_view reason) {
            const auto now = ticks(Clock::now());
            const auto retry = now + 2'000'000'000ll;
            const auto previous = retry_after_.exchange(retry, std::memory_order_acq_rel);
            { std::lock_guard lock(mutex_); published_.reset(); }
            if (previous <= now)
                LOG_WARN("GPU transfer fallback: role=HIP training preview, requested=asynchronous snapshot, effective=synchronized live model, reason={}", reason);
        }
        void reset() {
            // Destroy the ring before releasing captured source ownership and
            // before the trainer destroys its queue/context.
            ring_.reset();
            pending_.reset();
            ring_bytes_ = pending_bytes_ = 0;
            next_capture_ = {};
            warned_ = logged_publication_ = false;
            requested_.store(0, std::memory_order_release);
            retry_after_.store(0, std::memory_order_release);
            std::lock_guard lock(mutex_);
            published_.reset();
        }
        void advance(const Model& source, lfs::core::TensorWorkQueue& queue) {
            using namespace lfs::core;
            const auto now = Clock::now();
            try {
                if (ticks(now) < retry_after_.load(std::memory_order_acquire)) {
                    ring_.reset();
                    pending_.reset();
                    ring_bytes_ = pending_bytes_ = 0;
                    logged_publication_ = false;
                    return;
                }
                if (pending_ && ring_->poll(0)) {
                    const auto held = budget_->cpu_bytes.load(std::memory_order_acquire);
                    if (pending_bytes_ > kBudget - ring_bytes_ || held > kBudget - ring_bytes_ - pending_bytes_)
                        return; // Readers still own older complete generations.
                    const auto bytes = ring_->slot_bytes(0);
                    for (size_t i = 0; auto* field : fields(*pending_)) {
                        if (field->is_valid() && gpu_backend_of(*field)) {
                            auto host = Tensor::empty(field->shape(), Device::CPU, field->dtype());
                            if (field->bytes())
                                std::memcpy(host.data_ptr(), bytes.data() + offsets_[i], field->bytes());
                            *field = std::move(host);
                        }
                        ++i;
                    }
                    const auto payload = pending_bytes_;
                    pending_->set_host_snapshot_generation(next_generation_.fetch_add(1, std::memory_order_relaxed));
                    budget_->cpu_bytes.fetch_add(payload, std::memory_order_acq_rel);
                    std::shared_ptr<const Model> complete(pending_.release(),
                        [budget = budget_, payload](const Model* model) {
                            delete model;
                            budget->cpu_bytes.fetch_sub(payload, std::memory_order_acq_rel);
                        });
                    ring_->release(0);
                    { std::lock_guard lock(mutex_); published_ = std::move(complete); }
                    if (!logged_publication_) {
                        LOG_INFO("HIP training preview ready: complete asynchronous snapshot, payload={} bytes, capture cadence=100ms, host payload budget={} bytes", payload, kBudget);
                        logged_publication_ = true;
                    }
                    warned_ = false;
                }
                if (pending_ || now < next_capture_)
                    return;
                const auto requested = requested_.load(std::memory_order_acquire);
                if (!requested || ticks(now) - requested > 2'000'000'000ll)
                    return;
                next_capture_ = now + std::chrono::milliseconds(100);
                auto snapshot = std::make_unique<Model>(source.readOnlySnapshot());
                size_t bytes = 0;
                auto inputs = fields(*snapshot);
                for (size_t i = 0; i < inputs.size(); ++i) {
                    auto& field = *inputs[i];
                    offsets_[i] = bytes;
                    if (!field.is_valid())
                        continue;
                    if (!field.is_contiguous() || field.bytes() > kBudget - bytes) {
                        warn("snapshot layout exceeds the bounded preview budget");
                        return;
                    }
                    bytes += field.bytes();
                    const auto backend = gpu_backend_of(field);
                    if (backend && *backend != GpuBackend::HIP) {
                        warn("snapshot contains storage from a different tensor backend");
                        return;
                    }
                    if (!backend)
                        field = field.clone();
                }
                // Reserve one pinned capture and two complete CPU generations.
                if (!bytes || bytes > kBudget / 3) {
                    warn("snapshot payload exceeds the bounded preview budget");
                    return;
                }
                if (!ring_ || ring_bytes_ != bytes) {
                    ring_.reset();
                    ring_bytes_ = 0;
                    if (budget_->cpu_bytes.load(std::memory_order_acquire) > kBudget - bytes)
                        return;
                    ring_ = std::make_unique<TensorReadbackRing>(GpuBackend::HIP, 1, bytes, queue);
                    ring_bytes_ = bytes;
                }
                pending_bytes_ = bytes;
                pending_ = std::move(snapshot);
                for (size_t i = 0; i < inputs.size(); ++i)
                    if (inputs[i]->is_valid() && inputs[i]->bytes() && gpu_backend_of(*inputs[i]))
                        ring_->enqueue(*inputs[i], 0, inputs[i]->bytes(), 0, offsets_[i]);
                ring_->seal(0);
            } catch (const std::exception& error) {
                // Optional captures never replay an optimizer step. The ring
                // drains any partially submitted copies before releasing input.
                ring_.reset();
                pending_.reset();
                ring_bytes_ = pending_bytes_ = 0;
                next_capture_ = now + std::chrono::seconds(2);
                warn(error.what());
            }
        }
    };
} // namespace lfs::training
