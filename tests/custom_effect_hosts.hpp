/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "shaders/slang_compiler.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lfs::test {

    inline std::shared_ptr<const lfs::vis::shaders::ShaderPackage> loadPackage(const std::filesystem::path& directory) {
        auto package = lfs::vis::shaders::SlangCompiler::load_package(directory / "manifest.json");
        if (!package) {
            ADD_FAILURE() << lfs::format_for_developer(package.error());
            return nullptr;
        }
        return *package;
    }

    // Links `effect` into package hosts through the runtime compiler, as
    // program set sources: every host, or the named ones in package order.
    // `Source` is the renderer's ComputeProgramSource, which only its test
    // subclasses can name. Returns no sources after reporting a failure.
    template <class Source>
    std::vector<Source> compileHosts(lfs::vis::shaders::SlangCompiler& compiler,
                                     const std::shared_ptr<const lfs::vis::shaders::ShaderPackage>& package,
                                     const std::string& effect, const std::vector<std::string>& names = {}) {
        std::vector<Source> sources;
        for (const auto& host : package->hosts) {
            if (!names.empty() && std::ranges::find(names, host.module.name) == names.end())
                continue;
            lfs::vis::shaders::CompileDiagnostics diagnostics;
            auto output = compiler.compile({.effects = {{effect}}, .package = package, .host = host.module.name},
                                           diagnostics);
            if (!output) {
                ADD_FAILURE() << host.module.name << ": " << diagnostics.raw
                              << lfs::format_for_developer(output.error());
                return {};
            }
            Source source{{.label = host.module.name,
                           .workgroup = host.workgroup,
                           .required_subgroup_size = host.subgroup_size},
                          std::move(output->spirv)};
            for (const auto& binding : host.bindings) {
                EXPECT_EQ(binding.set, 0u) << host.module.name;
                source.spec.bindings.push_back(static_cast<int>(binding.binding));
            }
            sources.push_back(std::move(source));
        }
        EXPECT_EQ(sources.size(), names.empty() ? package->hosts.size() : names.size());
        return sources;
    }

} // namespace lfs::test
