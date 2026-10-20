/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/parameters.hpp"
#include "core/tensor.hpp"

#include <functional>

namespace lfs::core {
    class SplatData;
}

namespace lfs::training {

    class IStrategy;

    // Stage the retained model and fresh strategy state before committing either.
    // Completed densification state and Adam moments restart at this boundary.
    // The caller owns the renderer/exportable-storage topology barrier.
    [[nodiscard]] lfs::Status compact_gaussians_for_postprocess(
        IStrategy& strategy,
        const core::param::OptimizationParameters& params,
        const core::Tensor& keep_indices,
        const std::function<lfs::Status(core::SplatData&)>& prepare_state = {});

} // namespace lfs::training
