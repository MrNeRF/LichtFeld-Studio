/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/kernels/loss_tensor_contract.hpp"

#include <format>

namespace lfs::training::kernels::loss_contract_detail {

    std::string format_loss_weight(const float weight) {
        return std::format(
            "SSIM loss weight must be finite and in [0,1] (weight={})", weight);
    }

} // namespace lfs::training::kernels::loss_contract_detail
