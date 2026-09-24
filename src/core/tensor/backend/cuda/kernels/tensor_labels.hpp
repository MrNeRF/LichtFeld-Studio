/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "internal/point_labels.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
namespace lfs::core::tensor_ops {
    using internal::LabelUpdateParams;
    void launch_update_labels(const LabelUpdateParams& params, cudaStream_t stream);
} // namespace lfs::core::tensor_ops
