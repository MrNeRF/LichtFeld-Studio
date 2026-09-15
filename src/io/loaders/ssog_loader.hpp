/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "io/loader_interface.hpp"
namespace lfs::io {
    class SsogLoader final : public IDataLoader {
    public:
        Result<LoadResult> load(const std::filesystem::path&, const LoadOptions& = {}) override;
        bool canLoad(const std::filesystem::path&) const override;
        std::string name() const override { return "SSOG"; }
        std::vector<std::string> supportedExtensions() const override { return {".ssog"}; }
        int priority() const override { return 18; }
    };
} // namespace lfs::io
