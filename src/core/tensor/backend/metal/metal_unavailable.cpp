/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "metal_module.hpp"

#include <stdexcept>

namespace lfs::core::internal {
    std::unique_ptr<MetalModule> make_metal_module(std::string, bool) {
        throw std::runtime_error("Metal kernel modules are unavailable in this build");
    }
} // namespace lfs::core::internal
