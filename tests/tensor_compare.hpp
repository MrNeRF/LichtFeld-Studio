/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <cmath>

namespace lfs::test {
    inline bool tensor_values_close(const core::Tensor& actual, const core::Tensor& expected,
                                    float rtol = 1e-5f, float atol = 1e-8f) {
        if (actual.shape() != expected.shape())
            return false;
        const auto a = actual.cpu().to_vector();
        const auto b = expected.cpu().to_vector();
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] == b[i])
                continue;
            if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
                std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i]))
                return false;
        }
        return true;
    }
} // namespace lfs::test
