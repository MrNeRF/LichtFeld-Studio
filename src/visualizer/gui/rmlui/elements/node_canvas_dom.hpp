/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <string>

namespace Rml {
    class Element;
}

namespace lfs::vis::gui::node_widgets {
    // Reconcile generated node/inspector markup while retaining existing controls,
    // focus and event state. Only changed attributes, text and children are patched.
    void patchMarkup(Rml::Element& parent, const std::string& markup);
} // namespace lfs::vis::gui::node_widgets
