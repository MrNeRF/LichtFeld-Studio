/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file tensor.hpp
 * @brief LichtFeld Studio Tensor Library - High-performance CPU/GPU tensor operations
 *
 * This is the public API for the lfs_tensor library. For forward declarations only,
 * use <core/tensor_fwd.hpp> instead.
 *
 * @section features Features
 * - Expression templates for lazy evaluation and automatic kernel fusion
 * - Optimized CUDA kernels with warp-level reductions
 * - Broadcasting, reductions, masking, and advanced indexing
 * - PyTorch-compatible semantics (shallow copy, type promotion)
 * - Memory-efficient: no LibTorch dependency
 *
 * @section quick_start Quick Start
 * @code
 * #include <core/tensor.hpp>
 * using namespace lfs::core;
 *
 * // Create tensors
 * auto a = Tensor::randn({1000, 3}, Device::GPU);
 * auto b = Tensor::ones({1000, 3}, Device::GPU);
 *
 * // Lazy evaluation - fused into single kernel
 * Tensor c = (a + b).mul(2.0f).relu();
 *
 * // Reductions
 * float sum = c.sum_scalar();
 * Tensor row_means = c.mean(1, true);
 *
 * // Indexing
 * auto mask = a.gt(0.0f);
 * auto selected = a.masked_select(mask);
 *
 * // Serialization
 * save_tensor(c, "tensor.bin");
 * auto loaded = load_tensor("tensor.bin");
 * @endcode
 *
 * @section memory Memory Management
 * Tensors use reference counting with copy-on-write semantics:
 * - Assignment creates a shallow copy (shared data)
 * - Use .clone() for deep copy
 * - Views share underlying storage
 *
 * @section namespace Namespace
 * All types are in `lfs::core`. For convenience:
 * @code
 * using namespace lfs::core;
 * // or
 * using lfs::core::Tensor;
 * @endcode
 */

#include <core/tensor/internal/tensor_impl.hpp>
#include <core/tensor/internal/tensor_serialization.hpp>
