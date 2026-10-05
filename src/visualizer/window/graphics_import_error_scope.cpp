/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "graphics_import_error_scope.hpp"

namespace lfs::vis {
    namespace {
        thread_local std::string* current = nullptr;
    } // namespace

    GraphicsImportErrorScope::GraphicsImportErrorScope(std::string& error) : previous_(current) { current = &error; }

    GraphicsImportErrorScope::~GraphicsImportErrorScope() {
        if (previous_ && previous_->empty() && current)
            *previous_ = *current;
        current = previous_;
    }

    void GraphicsImportErrorScope::record(const std::string& error) {
        if (current && current->empty())
            *current = error;
    }

} // namespace lfs::vis
