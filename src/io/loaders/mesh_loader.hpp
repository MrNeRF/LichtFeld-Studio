/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/loader_interface.hpp"

namespace lfs::io {

    class MeshLoader : public IDataLoader {
    public:
        MeshLoader() = default;
        ~MeshLoader() override = default;

        [[nodiscard]] Result<LoadResult> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) override;

        bool canLoad(const std::filesystem::path& path) const override;
        std::string name() const override;
        std::vector<std::string> supportedExtensions() const override;
        int priority() const override;
    };

} // namespace lfs::io
