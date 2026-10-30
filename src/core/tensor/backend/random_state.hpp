/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
namespace lfs::core::internal {
    // Backing state for the existing explicit CUDA generator API. CMake selects
    // the native implementation or an unavailable implementation; SDK handles
    // never enter the shared random-generator object.
    class NativeRandomState {
    public:
        virtual ~NativeRandomState() = default;
        virtual void reseed(uint64_t seed) = 0;
        virtual void normal(float* output, size_t count, float mean, float deviation, void* stream) = 0;
        virtual void* handle_address() = 0;
    };
    std::unique_ptr<NativeRandomState> make_native_random_state(uint64_t seed, const char* unavailable_message);
} // namespace lfs::core::internal
