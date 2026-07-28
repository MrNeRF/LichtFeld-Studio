/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/export.hpp"

#include <string>

namespace Rml {
    class Context;
}

namespace lfs::vis::gui {

    class RmlUIManager;

    // Owns the lifetime of one RmlUI context independently of the document
    // hosted inside it. RmlPanelHost still uses one owner today; keeping the
    // boundary explicit is a prerequisite for a later shared-context host.
    class LFS_VIS_API RmlContextOwner {
    public:
        RmlContextOwner(RmlUIManager* manager, std::string context_name);
        explicit RmlContextOwner(Rml::Context* borrowed_context);
        ~RmlContextOwner();

        RmlContextOwner(const RmlContextOwner&) = delete;
        RmlContextOwner& operator=(const RmlContextOwner&) = delete;

        bool ensureContext(int width, int height);
        [[nodiscard]] Rml::Context* get() const { return context_; }

    private:
        RmlUIManager* manager_ = nullptr;
        std::string context_name_;
        Rml::Context* context_ = nullptr;
        bool owns_context_ = true;
    };

} // namespace lfs::vis::gui
