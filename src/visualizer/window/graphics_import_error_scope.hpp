/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include <string>
namespace lfs::vis {
    // Capture bool-returning allocation failures without interrupting their
    // cleanup paths. Queued-attachment validation consumes the captured error.
    // The per-thread current scope lives in the visualizer library, so scopes and
    // records meet across module boundaries.
    class LFS_VIS_API GraphicsImportErrorScope {
        std::string* previous_;

    public:
        explicit GraphicsImportErrorScope(std::string& error);
        ~GraphicsImportErrorScope();
        static void record(const std::string& error);
    };

} // namespace lfs::vis
