/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/nodes/tree.hpp"

namespace Rml {
    class Element;
}

namespace lfs::vis::gui::node_widgets {
    void layoutCard(Rml::Element& card, float zoom, float dp_ratio);
    std::string escape(std::string_view text);
    LFS_VIS_API std::string nonBreakingStatus(std::string_view text);
    std::string categoryIcon(std::string_view category);
    LFS_VIS_API nlohmann::json valuePayload(const lfs::nodes::Node& node, std::string_view identifier,
                                            const lfs::nodes::Value& fallback);
    bool linked(const lfs::nodes::NodeTree& tree, const lfs::nodes::Node& node,
                const lfs::nodes::SocketDecl& socket);
    bool settingsExpanded(const lfs::nodes::Node& node);
    bool singleValue(const lfs::nodes::SocketDecl& socket);
    bool inputVisible(const lfs::nodes::NodeTree& tree, const lfs::nodes::Node& node,
                      const lfs::nodes::SocketDecl& socket);
    int inputRows(const lfs::nodes::NodeTree& tree, const lfs::nodes::Node& node,
                  const lfs::nodes::SocketDecl& socket);
    LFS_VIS_API std::string input(const lfs::nodes::Node& node, const lfs::nodes::SocketDecl& socket,
                                  bool inline_value);
    std::string property(const lfs::nodes::Node& node, const lfs::nodes::PropertyDecl& property);
} // namespace lfs::vis::gui::node_widgets
