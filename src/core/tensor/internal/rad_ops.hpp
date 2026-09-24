/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_rad.hpp"
namespace lfs::core::internal {
    void cuda_rad_page_dequant(const Tensor&, const RadPagePackedDesc&, const RadPagePool&, uint32_t);
    void cuda_rad_page_quantize(const RadPageSources&, const RadPagePool&, uint32_t);
    void vulkan_rad_page_dequant(const Tensor&, const RadPagePool&, uint32_t);
    void vulkan_rad_page_quantize(const RadPageSources&, const RadPagePool&, uint32_t);
} // namespace lfs::core::internal
