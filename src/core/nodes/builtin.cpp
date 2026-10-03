/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"

namespace lfs::nodes {
    void register_builtin_nodes(NodeTypeRegistry& registry) {
        builtin::register_input(registry);
        builtin::register_utilities(registry);
        builtin::register_selection(registry);
        builtin::register_geometry(registry);
        builtin::register_splat(registry);
        builtin::register_cleanup(registry);
        builtin::register_conversion(registry);
    }
} // namespace lfs::nodes
