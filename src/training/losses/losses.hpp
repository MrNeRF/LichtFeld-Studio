/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file losses.hpp
 * @brief Main header for libtorch-free loss functions
 *
 * All losses follow the pattern:
 * - Params struct for configuration
 * - Context struct for backward pass data (if needed)
 * - Static forward() method for computing loss and gradients
 *
 * These implementations wrap existing CUDA kernels but use lfs::core::Tensor
 * instead of torch::Tensor at the API level.
 */

#include "mask_loss.hpp"
#include "photometric_loss.hpp"
#include "regularization.hpp"
