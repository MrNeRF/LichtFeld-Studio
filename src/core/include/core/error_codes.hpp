/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>

// CUDA-safe taxonomy for the lfs error architecture (Section 7.2.1 of
// .codex_tmp/error-architecture-analysis.md). This header carries only fixed
// underlying-type enums with no owning members, so a .cu/.cuh translation
// unit may reference ErrorCode/ErrorDomain (for example to tag a
// DeviceFaultRecord) without ever parsing the host-only rich Error/Result
// API in core/error.hpp. Do not add strings, containers, or any type that
// owns memory to this file.

namespace lfs {

    // Stable, cross-module failure category. Meanings must survive GUI,
    // Python, MCP, TCP, and log translation. Do not encode every native
    // CUDA/Vulkan/errno value here; that detail belongs in NativeError
    // (core/error.hpp), keyed off ErrorDomain plus a native numeric code.
    enum class ErrorCode : std::uint16_t {
        Cancelled,
        InvalidArgument,
        BoundsViolation,
        FailedPrecondition,
        NotFound,
        PermissionDenied,
        AlreadyExists,
        ResourceExhausted, // host/GPU memory, disk, queue capacity
        DeadlineExceeded,
        Unavailable, // service/device/driver temporarily unavailable
        DataLoss,    // truncated/corrupt persisted data
        Unsupported,
        DeviceLost,
        Internal,
        ContractViolation,
    };

    // Owning module/subsystem of a failure's detection site.
    enum class ErrorDomain : std::uint16_t {
        Core,
        Tensor,
        IO,
        Training,
        Rendering,
        Vulkan,
        CUDA,
        Python,
        MCP,
        TCP,
        Preprocess,
        Sequencer,
        App,
    };

    [[nodiscard]] constexpr const char* to_string(const ErrorCode code) noexcept {
        switch (code) {
        case ErrorCode::Cancelled: return "Cancelled";
        case ErrorCode::InvalidArgument: return "InvalidArgument";
        case ErrorCode::BoundsViolation: return "BoundsViolation";
        case ErrorCode::FailedPrecondition: return "FailedPrecondition";
        case ErrorCode::NotFound: return "NotFound";
        case ErrorCode::PermissionDenied: return "PermissionDenied";
        case ErrorCode::AlreadyExists: return "AlreadyExists";
        case ErrorCode::ResourceExhausted: return "ResourceExhausted";
        case ErrorCode::DeadlineExceeded: return "DeadlineExceeded";
        case ErrorCode::Unavailable: return "Unavailable";
        case ErrorCode::DataLoss: return "DataLoss";
        case ErrorCode::Unsupported: return "Unsupported";
        case ErrorCode::DeviceLost: return "DeviceLost";
        case ErrorCode::Internal: return "Internal";
        case ErrorCode::ContractViolation: return "ContractViolation";
        }
        return "Unknown";
    }

    [[nodiscard]] constexpr const char* to_string(const ErrorDomain domain) noexcept {
        switch (domain) {
        case ErrorDomain::Core: return "Core";
        case ErrorDomain::Tensor: return "Tensor";
        case ErrorDomain::IO: return "IO";
        case ErrorDomain::Training: return "Training";
        case ErrorDomain::Rendering: return "Rendering";
        case ErrorDomain::Vulkan: return "Vulkan";
        case ErrorDomain::CUDA: return "CUDA";
        case ErrorDomain::Python: return "Python";
        case ErrorDomain::MCP: return "MCP";
        case ErrorDomain::TCP: return "TCP";
        case ErrorDomain::Preprocess: return "Preprocess";
        case ErrorDomain::Sequencer: return "Sequencer";
        case ErrorDomain::App: return "App";
        }
        return "Unknown";
    }

} // namespace lfs
