/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor_protocol_export.hpp"
#include <format>
#include <stdexcept>
namespace lfs::core::internal {
    [[noreturn]] inline void reject_dlpack_export(const char* name) {
        throw std::runtime_error(std::format("{} DLPack export is not supported; call .cpu() first", name));
    }
    TensorExportDevice cuda_dlpack_device(const Tensor&);
    Tensor cuda_prepare_dlpack_export(const Tensor&, std::optional<int64_t>);
    TensorExportCapabilities vulkan_dlpack_capabilities() noexcept;
    TensorExportDevice vulkan_dlpack_device(const Tensor&);
    Tensor vulkan_prepare_dlpack_export(const Tensor&, std::optional<int64_t>);
} // namespace lfs::core::internal
