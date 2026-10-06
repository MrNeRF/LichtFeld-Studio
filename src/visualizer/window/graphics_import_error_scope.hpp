/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <string>
namespace lfs::vis {
    // Capture bool-returning allocation failures without interrupting their
    // cleanup paths. Queued-attachment validation consumes the captured error.
    // Header-only and not exported: MSVC rejects thread_local data in a DLL
    // interface (C2492).
    class GraphicsImportErrorScope {
        inline static thread_local std::string* current_ = nullptr;
        std::string* previous_;

    public:
        explicit GraphicsImportErrorScope(std::string& error) : previous_(current_) { current_ = &error; }
        ~GraphicsImportErrorScope() {
            if (previous_ && previous_->empty() && current_)
                *previous_ = *current_;
            current_ = previous_;
        }
        static void record(const std::string& error) {
            if (current_ && current_->empty())
                *current_ = error;
        }
    };

} // namespace lfs::vis
