/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor_fwd.hpp"
#include "passes/vulkan_split_view_pass.hpp"

#include <glm/glm.hpp>
#include <memory>

namespace lfs::vis {

    // Production screenshot/capture counterpart of split_view.frag. This is
    // intentionally separate from frame rendering so CPU composition remains
    // lazy and directly testable.
    [[nodiscard]] LFS_VIS_API std::shared_ptr<lfs::core::Tensor>
    composeSplitCaptureCpu(const VulkanSplitViewParams& params,
                           const glm::ivec2& output_size);

} // namespace lfs::vis
