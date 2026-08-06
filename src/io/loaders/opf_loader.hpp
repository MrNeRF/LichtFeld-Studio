/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/loader_interface.hpp"

namespace lfs::io {
    class OpfLoader final : public IDataLoader {
    public:
        [[nodiscard]] Result<LoadResult> load(const std::filesystem::path& path,
                                              const LoadOptions& options = {}) override;
        bool canLoad(const std::filesystem::path& path) const override;
        std::string name() const override { return "OPF"; }
        std::vector<std::string> supportedExtensions() const override { return {".opf", ".OPF"}; }
        int priority() const override { return 50; }
    };
} // namespace lfs::io
