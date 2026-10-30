/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_module.hpp"
#include <memory>
#include <mutex>
#include <stdexcept>

namespace lfs::core::internal {
    class HipRuntimeError : public std::runtime_error {
    public:
        HipRuntimeError(int code, const char* operation, const LfsHipError& error);
        int native_code() const noexcept { return native_code_; }

    private:
        int native_code_;
    };
    void hip_runtime_check(int status, const LfsHipError& error, const char* operation);

    struct HipContextState;
    struct HipBufferState;
    class HipRuntimeContext;

    // Allocate the node before acquiring a native resource. Destruction can
    // then retain a failed release without allocating in a noexcept path.
    struct HipRetiredResource {
        enum class Kind { Readback,
                          Timing } kind = Kind::Readback;
        void* handle = nullptr;
        std::unique_ptr<HipRetiredResource> next;
    };

    class HipRuntimeBuffer {
    public:
        HipRuntimeBuffer() = default;
        size_t bytes() const noexcept;
        void upload(size_t offset, const void* source, size_t bytes) const;
        void download(size_t offset, void* destination, size_t bytes) const;
        void close(); // Retryable on failure; other copies share the same state.

    private:
        friend class HipRuntimeContext;
        explicit HipRuntimeBuffer(std::shared_ptr<HipBufferState> state) : state_(std::move(state)) {}
        std::shared_ptr<HipBufferState> state_;
    };

    class HipRuntimeContext {
    public:
        class Lock {
        public:
            const HipRuntimeApi& api() const noexcept;
            LfsHipContext* handle() const noexcept;
            LfsHipView view(const HipRuntimeBuffer& buffer, size_t offset, size_t bytes) const;

        private:
            friend class HipRuntimeContext;
            friend class HipRuntimeBuffer;
            explicit Lock(std::shared_ptr<HipContextState> state);
            std::shared_ptr<HipContextState> state_;
            std::unique_lock<std::mutex> lock_;
        };

        HipRuntimeContext(std::shared_ptr<HipRuntimeModule> module, int ordinal);
        HipRuntimeContext(const HipRuntimeContext&) = delete;
        HipRuntimeContext& operator=(const HipRuntimeContext&) = delete;
        HipRuntimeBuffer allocate(size_t bytes);
        Lock lock() const;
        void synchronize() const;
        void close(); // Rejects live buffer leases, rather than invalidating them.
        static size_t retry_retired_contexts();
        void release_readback(LfsHipReadback*, std::unique_ptr<HipRetiredResource>) noexcept;
        void release_timing(LfsHipTiming*, std::unique_ptr<HipRetiredResource>) noexcept;

    private:
        std::shared_ptr<HipContextState> state_;
    };
} // namespace lfs::core::internal
