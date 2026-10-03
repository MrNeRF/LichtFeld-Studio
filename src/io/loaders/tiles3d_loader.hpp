/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "io/loader_interface.hpp"

namespace lfs::io {
    // Loads a 3D Tiles splat tileset flat when its full detail fits the GPU,
    // otherwise returns the coarsest cut plus the tile source for streaming.
    class Tiles3dLoader final : public IDataLoader {
    public:
        Result<LoadResult> load(const std::filesystem::path&, const LoadOptions& = {}) override;
        bool canLoad(const std::filesystem::path&) const override;
        std::string name() const override { return "3D Tiles"; }
        std::vector<std::string> supportedExtensions() const override { return {".json"}; }
        int priority() const override { return 19; }
    };
} // namespace lfs::io
