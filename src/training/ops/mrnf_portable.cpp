/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/mrnf.hpp"

#include <limits>
#include <stdexcept>

namespace lfs::training {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    float portable_mrnf_kth_largest(const Tensor& values, const Tensor& include, const size_t k) {
        const auto flat = values.reshape({-1});
        if (k == 0 || k > flat.numel())
            throw std::invalid_argument("k-th largest needs 0 < k <= the rows");
        const auto candidates = include.is_valid()
                                    ? flat.masked_fill(include.reshape({-1}).logical_not(),
                                                       -std::numeric_limits<float>::infinity())
                                    : flat;
        return candidates.sort(0, /*descending=*/true).first.slice(0, k - 1, k).item<float>();
    }

    void portable_mrnf_rank_exchange_rows(const Tensor& primary, const Tensor& secondary, const Tensor& eligible,
                                          Tensor& rows) {
        const size_t n = primary.numel();
        const size_t count = rows.numel();
        if (n == 0 || count == 0)
            return;
        const auto ineligible = eligible.logical_not();
        const auto order = [&](const Tensor& importance) {
            return importance.masked_fill(ineligible, std::numeric_limits<float>::infinity())
                .sort(0, /*descending=*/false)
                .second;
        };
        // A row's rank is its later position in the two orders; ineligible rows sort last.
        // Float ranks are exact up to 2^24 rows; beyond that only near-ties reorder.
        const auto positions = Tensor::arange(0.0f, static_cast<float>(n), 1.0f, Device::GPU);
        const auto inverse_order = [&](const Tensor& sorted_rows) {
            auto ranks = Tensor::zeros({n}, Device::GPU, DataType::Float32);
            ranks.index_put_(sorted_rows, positions);
            return ranks;
        };
        Tensor ranks = inverse_order(order(primary));
        if (secondary.is_valid())
            ranks = ranks.maximum(inverse_order(order(secondary)));
        const auto ranked_rows = ranks.masked_fill(ineligible, std::numeric_limits<float>::infinity())
                                     .sort(0, /*descending=*/false)
                                     .second;
        rows.copy_from(ranked_rows.slice(0, 0, count));
    }
} // namespace lfs::training
