/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>
#include "core/export.hpp"
#include "runtime_api.h"

namespace lfs::core::internal {
// GPU completion and queue-family ownership live in the storage implementation.
// This coordinator closes the CPU gap between binding addresses and submitting
// work. A consumer scope must cover command recording AND queue submission.
class HipExternalStorage {
public:
    virtual ~HipExternalStorage() = default;
    virtual void release_to_hip() = 0;
    virtual void acquire_to_vulkan() = 0;
};

class LFS_CORE_API HipExternalAccess {
public:
    void register_storage(const std::shared_ptr<HipExternalStorage>&);
    void begin_hip();
    void end_hip() noexcept;
    void begin_consumer(std::span<const std::shared_ptr<HipExternalStorage>>,
                        const std::function<void()>& synchronize_hip);
    void end_consumer() noexcept;
private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<std::weak_ptr<HipExternalStorage>> storages_;
    size_t active_hip_ = 0;
    size_t waiting_hip_ = 0, waiting_consumers_ = 0;
    bool prefer_consumer_ = true;
    bool transitioning_ = false, consumer_ = false;
};

LFS_CORE_API HipExternalAccess& hip_external_access();
LFS_CORE_API int hip_external_call_begin(void*, LfsHipError*) noexcept;
LFS_CORE_API void hip_external_call_end(void*) noexcept;

// Native Tensor entry points enqueue under this lock. Application kernels use
// the same coordinator through the private module's C callbacks.
class HipDispatchMutex {
public:
    void lock() {
        auto& access=hip_external_access();
        access.begin_hip();
        try { mutex_.lock(); } catch (...) { access.end_hip(); throw; }
    }
    void unlock() noexcept {
        mutex_.unlock();
        hip_external_access().end_hip();
    }
private:
    std::mutex mutex_;
};
}
