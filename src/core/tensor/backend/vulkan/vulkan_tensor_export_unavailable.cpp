/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../native_export.hpp"
#include "core/tensor.hpp"
namespace lfs::core::internal {
    TensorExportCapabilities vulkan_dlpack_capabilities() noexcept { return {}; }
    TensorExportDevice vulkan_dlpack_device(const Tensor&) { reject_dlpack_export("Vulkan"); }
    Tensor vulkan_prepare_dlpack_export(const Tensor&, std::optional<int64_t>) { reject_dlpack_export("Vulkan"); }
} // namespace lfs::core::internal
