/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_types.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"

namespace lfs::core {

    Tensor undistort_image(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

    Tensor undistort_mask(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

} // namespace lfs::core
