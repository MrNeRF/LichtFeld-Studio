/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// ============================================================================
// Public Tensor Library Interface
// ============================================================================
//
// This is the single public header for the Tensor library.
// All implementation details are private and located in src/core_new/tensor/internal/
//
// Usage:
//   #include "core_new/tensor.hpp"
//
// The tensor library provides:
// - Tensor class: Multi-dimensional arrays with CPU/CUDA support
// - Expression templates: Lazy evaluation and kernel fusion
// - Rich API: Math operations, indexing, broadcasting, reductions, etc.
//
// Implementation is completely hidden - users only see this clean public API.
// ============================================================================

#include "../../src/core_new/tensor/internal/tensor_impl.hpp"
