/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gui/rmlui/rml_context_owner.hpp"

#include "gui/rmlui/rmlui_manager.hpp"

#include <cassert>
#include <utility>

namespace lfs::vis::gui {

    RmlContextOwner::RmlContextOwner(RmlUIManager* manager, std::string context_name)
        : manager_(manager), context_name_(std::move(context_name)) {
        assert(manager_);
    }

    RmlContextOwner::RmlContextOwner(Rml::Context* borrowed_context)
        : context_(borrowed_context), owns_context_(false) {
        assert(context_);
    }

    RmlContextOwner::~RmlContextOwner() {
        if (owns_context_ && manager_ && manager_->isInitialized())
            manager_->destroyContext(context_name_);
        context_ = nullptr;
    }

    bool RmlContextOwner::ensureContext(const int width, const int height) {
        if (context_)
            return true;

        context_ = manager_->createContext(context_name_, width, height);
        return context_ != nullptr;
    }

} // namespace lfs::vis::gui
