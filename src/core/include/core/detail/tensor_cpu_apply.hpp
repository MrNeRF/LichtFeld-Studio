/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstddef>

namespace lfs::core {
    // CPU helper for unary operations
    template <typename T, typename OutT, typename Op>
    void apply_unary_cpu(const T* input, OutT* output, size_t n, Op op) {
        for (size_t i = 0; i < n; ++i) {
            output[i] = op(input[i]);
        }
    }

    // CPU helper for binary operations
    template <typename T, typename OutputT, typename Op>
    void apply_binary_cpu(const T* a, const T* b, OutputT* c, size_t n, Op op) {
        for (size_t i = 0; i < n; ++i) {
            c[i] = op(a[i], b[i]);
        }
    }
} // namespace lfs::core
