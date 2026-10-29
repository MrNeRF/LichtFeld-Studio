/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// User shaders for builds without the Vulkan viewport renderer, which runs them.
// Shader assets still load, save and edit; linking reports why they cannot draw.
#include "slang_compiler.hpp"

#include <string>
#include <utility>

namespace lfs::vis::shaders {

    namespace {
        constexpr std::string_view kUnavailable =
            "User shaders need the Vulkan viewport renderer, which this build does not include.";

        Error unavailable() {
            return make_error(ErrorInit{
                .code = ErrorCode::Unavailable,
                .domain = ErrorDomain::Rendering,
                .user_message = std::string(kUnavailable),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
    } // namespace

    struct SlangCompiler::Impl {};

    SlangCompiler::SlangCompiler() = default;
    SlangCompiler::~SlangCompiler() = default;

    Result<CompileOutput> SlangCompiler::compile(const CompileRequest&, CompileDiagnostics& diagnostics) {
        diagnostics.raw = std::string(kUnavailable);
        Diagnostic entry;
        entry.code = "LFS";
        entry.message = std::string(kUnavailable);
        diagnostics.entries.push_back(std::move(entry));
        return unavailable();
    }

    Result<std::shared_ptr<const ShaderPackage>> SlangCompiler::load_package(const std::filesystem::path&) {
        return unavailable();
    }

    std::string_view SlangCompiler::abi_hash() { return {}; }
    std::string_view SlangCompiler::compiler_build() { return {}; }

} // namespace lfs::vis::shaders
