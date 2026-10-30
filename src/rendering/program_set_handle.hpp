/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>
#include <memory>
namespace lfs::rendering {
    struct ProgramSetKey {
        std::uint64_t id, generation;
    };
    using ProgramSetHandle = std::shared_ptr<const ProgramSetKey>;
} // namespace lfs::rendering
