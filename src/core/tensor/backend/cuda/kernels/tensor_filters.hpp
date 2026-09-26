/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../../../internal/point_filter.hpp"
#include "core/tensor/internal/private_access.hpp"
#include <cuda_runtime.h>
namespace lfs::core::tensor_ops {
    void launch_filter_points(const internal::PointFilterArgs& args, cudaStream_t stream);
}
