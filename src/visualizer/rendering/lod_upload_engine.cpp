/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lod_upload_engine.hpp"

#include "core/logger.hpp"
#include "core/tensor/internal/memory_pool.hpp"

#include "lod_page_dequant_cuda.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <thread>
#include <utility>

namespace lfs::vis {
    namespace {

        constexpr std::size_t kPageSplats = LodPageCache::kChunkSplats;

        std::size_t stagingRingDepth() {
            const std::size_t hw = std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
            return std::max<std::size_t>(8, std::clamp<std::size_t>(hw / 2, 2, 8) + 2);
        }

        LodUploadEngine::StagingLayout stagingLayoutFor(const bool has_meta) {
            // Worst case per splat: every payload property stored f32 at SH
            // degree 3 (means 12 + sh0 12 + scales 12 + orientation xyz 12 +
            // alpha 4 + SH bands 45*4), plus the sidecar planes (8 + 12) and
            // per-plane 16-byte alignment slack. Quantized profiles use a
            // fraction; only used_bytes is ever copied.
            constexpr std::size_t kWorstPayloadPerSplat = 232;
            constexpr std::size_t kMetaPerSplat = 20;
            constexpr std::size_t kAlignSlack = 16u * (lfs::io::kRadPackedMaxProps + 2u);
            LodUploadEngine::StagingLayout layout{};
            layout.total_bytes =
                kPageSplats * (kWorstPayloadPerSplat + (has_meta ? kMetaPerSplat : 0u)) + kAlignSlack;
            return layout;
        }

    } // namespace

    LodUploadEngine::LodUploadEngine() = default;

    LodUploadEngine::~LodUploadEngine() {
        (void)drainAndSync();
        std::lock_guard lock(mutex_);
        shutdown_ = true;
        slot_cv_.notify_all();
        for (const cudaEvent_t event : event_pool_) {
            (void)cudaEventDestroy(event);
        }
        event_pool_.clear();
        releaseStagingRingLocked();
        if (stream_ != nullptr) {
            // The staging copies and dequant launches only touch raw cudaMalloc /
            // cudaHostAlloc + Vulkan-external memory, never the tensor pool — but
            // sever the stream from the pool anyway so the engine obeys the same
            // lifetime contract as every other long-lived stream and stays UAF-safe
            // if pool-backed memory ever flows through it.
            lfs::core::CudaMemoryPool::instance().release_stream(stream_);
            (void)cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    std::vector<LodPageCache::PendingUpload>
    LodUploadEngine::configure(const DeviceLayout& layout,
                               const lfs::rendering::CudaTimelineSemaphore* const timeline) {
        {
            std::lock_guard lock(mutex_);
            if (layout_ == layout && timeline_ == timeline) {
                return {};
            }
        }
        auto drained = drainAndSync();
        std::lock_guard lock(mutex_);
        layout_ = layout;
        timeline_ = timeline;
        if (stream_ == nullptr && layout_.valid()) {
            if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
                stream_ = nullptr;
            }
        }
        releaseStagingRingLocked();
        staging_layout_ = stagingLayoutFor(layout_.meta_base != nullptr);
        if (layout_.valid() && stream_ != nullptr) {
            staging_ring_.resize(stagingRingDepth());
            for (auto& slot : staging_ring_) {
                if (cudaHostAlloc(reinterpret_cast<void**>(&slot.data),
                                  staging_layout_.total_bytes,
                                  cudaHostAllocDefault) != cudaSuccess) {
                    slot.data = nullptr;
                }
                if (cudaMalloc(reinterpret_cast<void**>(&slot.device_data),
                               staging_layout_.total_bytes) != cudaSuccess) {
                    slot.device_data = nullptr;
                }
                if (cudaEventCreateWithFlags(&slot.last_use, cudaEventDisableTiming) != cudaSuccess) {
                    slot.last_use = nullptr;
                }
            }
            // Partially-allocated slots can never be acquired; keep only
            // complete ones so an allocation-starved ring reads as
            // unconfigured instead of parking decode workers forever in
            // acquireStagingSlot (cache reset would then hang on the join).
            std::erase_if(staging_ring_, [](StagingSlot& slot) {
                const bool complete = slot.data != nullptr &&
                                      slot.device_data != nullptr &&
                                      slot.last_use != nullptr;
                if (!complete) {
                    if (slot.data != nullptr) {
                        (void)cudaFreeHost(slot.data);
                    }
                    if (slot.device_data != nullptr) {
                        (void)cudaFree(slot.device_data);
                    }
                    if (slot.last_use != nullptr) {
                        (void)cudaEventDestroy(slot.last_use);
                    }
                }
                return !complete;
            });
            if (staging_ring_.empty()) {
                LOG_ERROR("LOD upload engine: no staging slot survived allocation; "
                          "engine stays unconfigured");
                layout_ = {};
            }
        }
        slot_cv_.notify_all();
        return drained;
    }

    bool LodUploadEngine::configured() const {
        std::lock_guard lock(mutex_);
        return layout_.valid() && stream_ != nullptr && !staging_ring_.empty();
    }

    LodUploadEngine::StagingLayout LodUploadEngine::stagingLayout() const {
        std::lock_guard lock(mutex_);
        return staging_layout_;
    }

    bool LodUploadEngine::idle() const {
        std::lock_guard lock(mutex_);
        if (!in_flight_.empty()) {
            return false;
        }
        return std::none_of(staging_ring_.begin(), staging_ring_.end(),
                            [](const StagingSlot& slot) { return slot.acquired; });
    }

    std::uint64_t LodUploadEngine::lastPublishedSignalValue() const {
        std::lock_guard lock(mutex_);
        return last_published_signal_;
    }

    LodUploadEngine::StagingSlot* LodUploadEngine::acquireStagingSlot() {
        std::unique_lock lock(mutex_);
        while (true) {
            if (shutdown_ || staging_ring_.empty()) {
                return nullptr;
            }
            StagingSlot* candidate = nullptr;
            for (std::size_t probe = 0; probe < staging_ring_.size(); ++probe) {
                StagingSlot& slot = staging_ring_[(staging_cursor_ + probe) % staging_ring_.size()];
                if (!slot.acquired && slot.data != nullptr && slot.device_data != nullptr &&
                    slot.last_use != nullptr) {
                    candidate = &slot;
                    staging_cursor_ = (staging_cursor_ + probe + 1) % staging_ring_.size();
                    break;
                }
            }
            if (candidate == nullptr) {
                slot_cv_.wait(lock);
                continue;
            }
            candidate->acquired = true;
            if (candidate->used) {
                // The slot's previous copies may still be in flight; wait off
                // the lock so other workers keep packing.
                const cudaEvent_t guard = candidate->last_use;
                lock.unlock();
                if (cudaEventSynchronize(guard) != cudaSuccess) {
                    lock.lock();
                    candidate->acquired = false;
                    slot_cv_.notify_all();
                    return nullptr;
                }
                lock.lock();
            }
            return candidate;
        }
    }

    void LodUploadEngine::releaseSlot(StagingSlot* const slot) {
        if (slot == nullptr) {
            return;
        }
        std::lock_guard lock(mutex_);
        slot->acquired = false;
        slot_cv_.notify_all();
    }

    void LodUploadEngine::submitPackedPage(StagingSlot* const slot,
                                           const lfs::io::RadPagePackedDesc& desc,
                                           const std::uint32_t page,
                                           const std::uint64_t generation) {
        Job job{
            .upload = {
                .page = page,
                .chunk = desc.chunk,
                .generation = generation,
                .error = {},
            },
        };

        // One lock section covers copy + kernel + timeline signal + event
        // record so timeline values stay monotone in stream order across
        // workers.
        std::lock_guard lock(mutex_);
        const auto guard_slot_reuse = [&]() -> std::string {
            if (const cudaError_t status = cudaEventRecord(slot->last_use, stream_);
                status == cudaSuccess) {
                slot->used = true;
                return {};
            } else {
                // A failed guard must not expose pinned host/device scratch to
                // another decoder while the H2D copy or dequant kernel may
                // still be using it. This is an error path, so draining the
                // dedicated upload stream is preferable to corrupting a page.
                const cudaError_t sync_status = cudaStreamSynchronize(stream_);
                slot->used = false;
                if (sync_status != cudaSuccess) {
                    return std::format(
                        "LOD staging-slot event record failed: {} ({}); stream sync also failed: "
                        "{} ({})",
                        cudaGetErrorName(status),
                        cudaGetErrorString(status),
                        cudaGetErrorName(sync_status),
                        cudaGetErrorString(sync_status));
                }
                return std::format("LOD staging-slot event record failed: {} ({})",
                                   cudaGetErrorName(status),
                                   cudaGetErrorString(status));
            }
        };
        const auto submit = [&]() -> std::string {
            if (!layout_.valid() || stream_ == nullptr) {
                return "LOD upload engine is not configured";
            }
            const std::size_t dst_start = static_cast<std::size_t>(page) * kPageSplats;
            if (dst_start + kPageSplats > layout_.splat_capacity || desc.count == 0 ||
                desc.count > kPageSplats) {
                return std::format("LOD upload page {} exceeds splat capacity {}",
                                   page, layout_.splat_capacity);
            }
            if (layout_.meta_base != nullptr &&
                dst_start + kPageSplats > layout_.meta_capacity_nodes) {
                return std::format("LOD upload page {} exceeds metadata capacity {}",
                                   page, layout_.meta_capacity_nodes);
            }
            if (desc.used_bytes == 0 || desc.used_bytes > staging_layout_.total_bytes) {
                return std::format("LOD packed page descriptor spans {} bytes, slot holds {}",
                                   desc.used_bytes, staging_layout_.total_bytes);
            }
            if (const cudaError_t status = cudaMemcpyAsync(slot->device_data, slot->data,
                                                           desc.used_bytes,
                                                           cudaMemcpyHostToDevice, stream_);
                status != cudaSuccess) {
                return std::format("LOD page slot H2D copy failed: {} ({})",
                                   cudaGetErrorName(status),
                                   cudaGetErrorString(status));
            }

            auto* const device_base = static_cast<std::uint8_t*>(layout_.device_base);
            LodPoolDeviceView view{};
            view.means = reinterpret_cast<float*>(device_base + layout_.region_offset[0]);
            view.sh0 = reinterpret_cast<uint2*>(device_base + layout_.region_offset[1]);
            view.shN = reinterpret_cast<std::uint32_t*>(device_base + layout_.region_offset[2]);
            view.rotation = reinterpret_cast<uint2*>(device_base + layout_.region_offset[3]);
            view.scaling = reinterpret_cast<uint2*>(device_base + layout_.region_offset[4]);
            view.opacity = reinterpret_cast<std::uint16_t*>(device_base + layout_.region_offset[5]);
            view.page_frames = reinterpret_cast<float4*>(device_base + layout_.region_offset[6]);
            view.dst_rest = layout_.dst_rest;
            view.dst_slots = layout_.dst_slots;
            if (layout_.meta_base != nullptr) {
                auto* const meta_base = static_cast<std::uint8_t*>(layout_.meta_base);
                view.meta_bounds =
                    reinterpret_cast<uint2*>(meta_base + layout_.meta_bounds_offset);
                view.meta_links =
                    reinterpret_cast<std::uint32_t*>(meta_base + layout_.meta_links_offset);
            }
            if (const cudaError_t status =
                    launchLodPageDequant(slot->device_data, desc, view, page,
                                         static_cast<std::uint32_t>(kPageSplats), stream_);
                status != cudaSuccess) {
                std::string error =
                    std::format("LOD page dequant kernel launch failed: {} ({})",
                                cudaGetErrorName(status),
                                cudaGetErrorString(status));
                if (const std::string guard_error = guard_slot_reuse(); !guard_error.empty()) {
                    error += std::format("; {}", guard_error);
                }
                return error;
            }
            if (const std::string guard_error = guard_slot_reuse(); !guard_error.empty()) {
                return guard_error;
            }

            const cudaEvent_t event = acquireEventLocked();
            if (event == nullptr) {
                return "LOD upload failed to create a CUDA event";
            }
            const std::uint64_t signal_value = ++signal_counter_;
            if (timeline_ != nullptr && timeline_->valid() &&
                !timeline_->cudaSignal(signal_value, stream_)) {
                // Publishing without the signal would hand the renderer a
                // timeline value CUDA never reaches - a render-queue hang.
                // The skipped value is safe: any later successful signal is
                // larger and satisfies waits at or below it.
                event_pool_.push_back(event);
                return "LOD upload timeline signal failed";
            }
            if (const cudaError_t status = cudaEventRecord(event, stream_); status != cudaSuccess) {
                event_pool_.push_back(event);
                return std::format("LOD upload event record failed: {} ({})",
                                   cudaGetErrorName(status),
                                   cudaGetErrorString(status));
            }
            job.event = event;
            job.signal_value = signal_value;
            return {};
        };

        job.upload.error = submit();
        slot->acquired = false;
        slot_cv_.notify_all();
        in_flight_.push_back(std::move(job));
    }

    cudaEvent_t LodUploadEngine::acquireEventLocked() {
        if (!event_pool_.empty()) {
            const cudaEvent_t event = event_pool_.back();
            event_pool_.pop_back();
            return event;
        }
        cudaEvent_t event = nullptr;
        if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
            return nullptr;
        }
        return event;
    }

    std::vector<LodPageCache::PendingUpload>
    LodUploadEngine::takeCompletedLocked(const bool wait_for_all) {
        std::vector<LodPageCache::PendingUpload> published;
        while (!in_flight_.empty()) {
            Job& job = in_flight_.front();
            if (job.event != nullptr) {
                if (!wait_for_all) {
                    const cudaError_t status = cudaEventQuery(job.event);
                    if (status == cudaErrorNotReady) {
                        break;
                    }
                    if (status != cudaSuccess) {
                        job.upload.error = std::format("LOD upload event query failed: {} ({})",
                                                       cudaGetErrorName(status),
                                                       cudaGetErrorString(status));
                    }
                }
                event_pool_.push_back(job.event);
                job.event = nullptr;
                last_published_signal_ = std::max(last_published_signal_, job.signal_value);
            }
            published.push_back(std::move(job.upload));
            in_flight_.pop_front();
        }
        return published;
    }

    std::vector<LodPageCache::PendingUpload> LodUploadEngine::collectPublished() {
        std::lock_guard lock(mutex_);
        return takeCompletedLocked(false);
    }

    std::vector<LodPageCache::PendingUpload> LodUploadEngine::drainAndSync() {
        {
            std::unique_lock lock(mutex_);
            slot_cv_.wait(lock, [this] {
                return std::none_of(staging_ring_.begin(), staging_ring_.end(),
                                    [](const StagingSlot& slot) { return slot.acquired; });
            });
        }
        if (stream_ != nullptr) {
            (void)cudaStreamSynchronize(stream_);
        }
        std::lock_guard lock(mutex_);
        return takeCompletedLocked(true);
    }

    void LodUploadEngine::releaseStagingRingLocked() {
        for (auto& slot : staging_ring_) {
            if (slot.data != nullptr) {
                (void)cudaFreeHost(slot.data);
            }
            if (slot.device_data != nullptr) {
                (void)cudaFree(slot.device_data);
            }
            if (slot.last_use != nullptr) {
                (void)cudaEventDestroy(slot.last_use);
            }
        }
        staging_ring_.clear();
        staging_cursor_ = 0;
    }

} // namespace lfs::vis
