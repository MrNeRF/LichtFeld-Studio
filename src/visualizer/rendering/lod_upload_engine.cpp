/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lod_upload_engine.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

namespace lfs::vis {
    using namespace lfs::core;
    namespace {
        constexpr size_t kPageSplats = LodPageCache::kChunkSplats;
        constexpr size_t kDescriptorBytes = sizeof(RadPagePackedDesc);
        constexpr size_t kStagingBytes = rad_page_staging_bytes(kPageSplats);
        constexpr size_t kRingDepth = 16;
    } // namespace
    struct LodUploadEngine::Impl {
        struct Slot : StagingSlot {
            Tensor host, scratch;
            TensorUpload upload;
            TensorCompletion completion;
            bool acquired = false, queued = false;
        };
        struct Job {
            Slot* slot;
            RadPagePackedDesc desc;
            LodPageCache::PendingUpload result;
        };
        struct Batch {
            TensorCompletion completion;
            std::vector<Job> jobs;
        };
        DeviceLayout layout;
        std::unique_ptr<TensorWorkQueue> queue;
        std::vector<Slot> slots;
        std::deque<Job> pending;
        std::deque<Batch> submitted;
        std::vector<LodPageCache::PendingUpload> finished;
        mutable std::mutex mutex;
        std::mutex queue_mutex;
        std::condition_variable cv;
        std::thread worker;
        bool stopping = false, working = false;
        uint64_t published = 0, consumer = 0;
        size_t cursor = 0;

        void run() {
            for (;;) {
                std::vector<Job> jobs;
                uint64_t wait_value;
                {
                    std::unique_lock lock(mutex);
                    cv.wait(lock, [&] { return stopping || !pending.empty(); });
                    if (stopping && pending.empty())
                        return;
                    while (!pending.empty()) {
                        jobs.push_back(std::move(pending.front()));
                        pending.pop_front();
                    }
                    working = true;
                    wait_value = consumer;
                }
                TensorCompletion completion;
                try {
                    std::lock_guard queue_lock(queue_mutex);
                    completion = queue->execute([&] {
                        for (auto& job : jobs) {
                            auto& slot = *job.slot;
                            const size_t bytes = (kDescriptorBytes + job.desc.used_bytes + 3) & ~size_t(3);
                            slot.upload.enqueue(slot.scratch.slice(0, 0, bytes), slot.host.slice(0, 0, bytes));
                            rad_page_dequant(slot.scratch, job.desc, layout.pool, job.result.page);
                        }
                    },
                                                wait_value);
                } catch (const std::exception& e) {
                    for (auto& job : jobs)
                        job.result.error = e.what();
                    LOG_ERROR("LOD tensor upload batch failed: {}", e.what());
                }
                {
                    std::lock_guard lock(mutex);
                    for (auto& job : jobs) {
                        job.slot->completion = completion;
                        job.slot->queued = false;
                    }
                    submitted.push_back({completion, std::move(jobs)});
                    working = false;
                    cv.notify_all();
                }
            }
        }
        void collect() {
            while (!submitted.empty() && submitted.front().completion.ready()) {
                auto& batch = submitted.front();
                published = std::max(published, batch.completion.timeline().value);
                for (auto& job : batch.jobs)
                    finished.push_back(std::move(job.result));
                submitted.pop_front();
            }
        }
    };
    LodUploadEngine::LodUploadEngine() : impl_(std::make_unique<Impl>()) {}
    LodUploadEngine::~LodUploadEngine() {
        try {
            (void)configure({});
        } catch (const std::exception& e) {
            LOG_ERROR("LOD tensor engine retained after shutdown failure: {}", e.what());
            {
                std::lock_guard lock(impl_->mutex);
                impl_->stopping = true;
                impl_->cv.notify_all();
            }
            if (impl_->worker.joinable())
                impl_->worker.join();
            (void)impl_.release();
        }
    }
    std::vector<LodPageCache::PendingUpload> LodUploadEngine::configure(DeviceLayout layout, void* device, void* consumer) {
        auto results = drainAndSync();
        auto& s = *impl_;
        {
            std::lock_guard lock(s.mutex);
            s.stopping = true;
            s.cv.notify_all();
        }
        if (s.worker.joinable())
            s.worker.join();
        s.slots.clear();
        s.queue.reset();
        s.layout = std::move(layout);
        s.stopping = false;
        s.published = 0;
        s.consumer = 0;
        s.cursor = 0;
        if (!s.layout.valid())
            return results;
        const auto backend = *gpu_backend_of(s.layout.pool.regions[0]);
        const GpuBackendScope scope(backend);
        s.queue = std::make_unique<TensorWorkQueue>(backend, device, consumer);
        s.slots.resize(kRingDepth);
        for (auto& slot : s.slots) {
            slot.host = Tensor::empty({kDescriptorBytes + kStagingBytes}, Device::CPU, DataType::UInt8, true);
            slot.scratch = Tensor::empty({kDescriptorBytes + kStagingBytes}, Device::GPU, DataType::UInt8);
            slot.data = slot.host.ptr<uint8_t>() + kDescriptorBytes;
        }
        s.worker = std::thread([&s] { s.run(); });
        return results;
    }
    bool LodUploadEngine::configured() const {
        std::lock_guard lock(impl_->mutex);
        return impl_->layout.valid();
    }
    size_t LodUploadEngine::stagingBytes() const { return kStagingBytes; }
    LodUploadEngine::StagingSlot* LodUploadEngine::acquireStagingSlot() {
        auto& s = *impl_;
        std::unique_lock lock(s.mutex);
        for (;;) {
            if (s.stopping || !s.layout.valid())
                return nullptr;
            s.collect();
            Impl::Slot* waiting = nullptr;
            for (size_t i = 0; i < s.slots.size(); ++i) {
                auto& slot = s.slots[s.cursor++ % s.slots.size()];
                if (slot.acquired || slot.queued)
                    continue;
                if (slot.completion.ready()) {
                    slot.upload.wait();
                    slot.acquired = true;
                    return &slot;
                }
                if (!waiting)
                    waiting = &slot;
            }
            if (waiting) {
                // Reserve before dropping the lock so another decode worker
                // cannot reuse the slot during the completion wait.
                waiting->acquired = true;
                lock.unlock();
                waiting->completion.wait();
                waiting->upload.wait();
                return waiting;
            }
            s.cv.wait(lock);
        }
    }
    void LodUploadEngine::releaseSlot(StagingSlot* raw) {
        if (!raw)
            return;
        std::lock_guard lock(impl_->mutex);
        static_cast<Impl::Slot*>(raw)->acquired = false;
        impl_->cv.notify_all();
    }
    void LodUploadEngine::submitPackedPage(StagingSlot* raw, const lfs::core::RadPagePackedDesc& desc, uint32_t page, uint64_t generation) {
        auto& s = *impl_;
        auto& slot = *static_cast<Impl::Slot*>(raw);
        std::lock_guard lock(s.mutex);
        LodPageCache::PendingUpload result{.page = page, .chunk = desc.chunk, .generation = generation, .error = {}};
        try {
            if (!slot.acquired)
                throw std::logic_error("LOD staging slot was not acquired");
            rad_page_validate(s.layout.pool, desc, kDescriptorBytes + kStagingBytes, page);
            std::memcpy(slot.host.data_ptr(), &desc, kDescriptorBytes);
            const size_t padded = (desc.used_bytes + 3) & ~size_t(3);
            std::memset(slot.data + desc.used_bytes, 0, padded - desc.used_bytes);
            slot.queued = true;
            s.pending.push_back({&slot, desc, std::move(result)});
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): the error is published with the page result.
            result.error = e.what();
            s.finished.push_back(std::move(result));
        }
        slot.acquired = false;
        s.cv.notify_all();
    }
    std::vector<LodPageCache::PendingUpload> LodUploadEngine::collectPublished() {
        std::lock_guard lock(impl_->mutex);
        impl_->collect();
        return std::exchange(impl_->finished, {});
    }
    std::vector<LodPageCache::PendingUpload> LodUploadEngine::drainAndSync() {
        auto& s = *impl_;
        std::unique_lock lock(s.mutex);
        s.cv.wait(lock, [&] { return s.pending.empty() && !s.working &&
                                     std::none_of(s.slots.begin(), s.slots.end(), [](const auto& slot) { return slot.acquired; }); });
        for (auto& batch : s.submitted)
            batch.completion.wait();
        s.collect();
        return std::exchange(s.finished, {});
    }
    bool LodUploadEngine::idle() const {
        std::lock_guard lock(impl_->mutex);
        const auto& s = *impl_;
        return s.pending.empty() && s.submitted.empty() && s.finished.empty() && !s.working &&
               std::none_of(s.slots.begin(), s.slots.end(), [](const auto& slot) { return slot.acquired; });
    }
    uint64_t LodUploadEngine::lastPublishedSignalValue() const {
        std::lock_guard lock(impl_->mutex);
        return impl_->published;
    }
    void* LodUploadEngine::timeline() const { return impl_->queue ? impl_->queue->timeline() : nullptr; }
    void LodUploadEngine::noteRendererCompletion(uint64_t value) {
        std::lock_guard lock(impl_->mutex);
        impl_->consumer = std::max(impl_->consumer, value);
    }
    TensorCompletion LodUploadEngine::quantizeResident(const RadPageSources& source,
                                                       std::span<const ResidentPage> pages) {
        auto& s = *impl_;
        uint64_t value;
        {
            std::lock_guard lock(s.mutex);
            value = s.consumer;
        }
        std::lock_guard lock(s.queue_mutex);
        return s.queue->execute([&] {
            auto inputs = source;
            for (const auto& page : pages) {
                inputs.offset = page.offset;
                inputs.count = page.count;
                rad_page_quantize(inputs, s.layout.pool, page.page);
            }
        },
                                value);
    }
} // namespace lfs::vis
