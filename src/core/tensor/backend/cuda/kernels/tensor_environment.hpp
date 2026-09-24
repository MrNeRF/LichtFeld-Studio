/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_environment.hpp"
#include <cuda_runtime.h>
namespace lfs::core::tensor_ops {
    void launch_environment_composite(const EnvironmentCompositeParams& p, const float* environment,
                                      const float* rgb, const float* alpha, unsigned char* output, cudaStream_t stream);
}
