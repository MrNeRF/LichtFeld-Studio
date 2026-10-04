/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/nodes/registry.hpp"

namespace lfs::nodes {

    struct Node;

    LFS_CORE_API void register_builtin_nodes(NodeTypeRegistry& registry);
    LFS_CORE_API void set_builtin_node_text(NodeTypeInfo& info);
    LFS_CORE_API void set_stored_selection(Node& node, const core::Tensor& selection);
    LFS_CORE_API Field stored_selection_field(const Node& node);

} // namespace lfs::nodes
