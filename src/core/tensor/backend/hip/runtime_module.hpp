/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "gemm_api.h"
#include "integer_reduce_api.h"
#include "predicate_api.h"
#include "runtime_api.h"
#include "where_api.h"

#include <filesystem>

namespace lfs::core::internal {

    // SDK-free, typed dispatch. No import library is needed by the host adapter.
    struct HipRuntimeApi {
        decltype(&lfs_hip_runtime_abi) runtime_abi = nullptr;
#define LFS_HIP_RUNTIME_EXPORT(Name) decltype(&lfs_hip_##Name) Name = nullptr;
#include "runtime_exports.def"
#undef LFS_HIP_RUNTIME_EXPORT
    };

    // Contexts and resources retain shared ownership of this module. Loading
    // resolves symbols and validates the ABI but never enumerates devices.
    class HipRuntimeModule final {
    public:
        explicit HipRuntimeModule(const std::filesystem::path& path);
        ~HipRuntimeModule();
        HipRuntimeModule(const HipRuntimeModule&) = delete;
        HipRuntimeModule& operator=(const HipRuntimeModule&) = delete;
        HipRuntimeModule(HipRuntimeModule&&) = delete;
        HipRuntimeModule& operator=(HipRuntimeModule&&) = delete;

        const HipRuntimeApi& api() const noexcept { return api_; }

    private:
        void close() noexcept;
        void* symbol(const char* name) const;
        void* module_ = nullptr;
        HipRuntimeApi api_{};
    };

} // namespace lfs::core::internal
