/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis::shaders {

    enum class DiagnosticSeverity : uint8_t { Note,
                                              Warning,
                                              Error };

    // Generated adapter code reports its own virtual file, so locations in the
    // user's file are never shifted by it.
    struct Diagnostic {
        DiagnosticSeverity severity = DiagnosticSeverity::Error;
        std::string code;    // compiler code such as "E30015"; "LFS" for contract checks
        std::string file;    // virtual file name; a contract check names its effect's, if any
        uint32_t line = 0;   // one-based; zero without a location
        uint32_t column = 0; // one-based in code points, a tab counts as one
        std::string message;
    };

    struct CompileDiagnostics {
        std::string raw; // complete compiler output in call order
        std::vector<Diagnostic> entries;
    };

    // One field of the effect's optional `Params` struct.
    struct Parameter {
        std::string name;
        uint32_t slot = 0;       // declaration order
        uint32_t components = 1; // float .. float4
        float minimum = 0.0f;
        float maximum = 0.0f;
        float initial = 0.0f; // broadcast to every component
        bool operator==(const Parameter&) const = default;
    };

    struct Binding {
        std::string name;
        uint32_t set = 0;
        uint32_t binding = 0;
        uint32_t count = 1;
        bool operator==(const Binding&) const = default;
    };

    struct ModuleIR {
        std::string name; // the name `import` resolves
        std::string sha256;
        std::vector<std::byte> bytes;
    };

    // A precompiled host stage and the interface the renderer provides for it.
    struct HostStage {
        ModuleIR module;
        std::vector<size_t> imports; // ShaderPackage::modules loaded before the host, in order
        std::string stage;
        std::string entry;
        std::vector<std::string> defines; // offline variant selection
        std::vector<Binding> bindings;    // global resources in declaration order
        uint32_t push_constant_bytes = 0;
        std::array<uint32_t, 3> workgroup = {1, 1, 1};
        uint32_t subgroup_size = 0; // the renderer pins it; a declared WaveSize must match; 0 = none
    };

    // API and host IR plus the code generation options they were built for,
    // as written by cmake/GenerateUserShaderIR.cmake.
    struct ShaderPackage {
        uint32_t version = 0;
        std::string abi_hash;
        std::string compiler_build;
        std::string profile;
        uint32_t optimization = 0;
        bool fast_math = false;
        bool column_major = false;
        std::vector<ModuleIR> api;     // importable by effects, in dependency order
        std::vector<ModuleIR> modules; // imported by hosts only; one name may have variants
        std::vector<HostStage> hosts;
    };

    struct EffectSource {
        std::string source;
        std::string name = "effect.slang"; // virtual file name in diagnostics, unique in a request
    };

    struct CompileRequest {
        // Each effect compiles as a module of its own. A host draw runs one by
        // its index, which a generated dispatcher switches on.
        std::vector<EffectSource> effects;
        std::shared_ptr<const ShaderPackage> package;
        std::string host; // module name of one package host
    };

    struct CompileOutput {
        std::vector<uint32_t> spirv;
        std::vector<std::vector<Parameter>> parameters; // per effect
        std::string cache_key;
        bool cache_hit = false;
    };

    // Owns the Slang global session. Construct, use and destroy it on one
    // compiler worker; compile() rejects other threads. Nothing is initialized
    // before the first compile(), and every cache miss uses fresh sessions.
    class LFS_VIS_API SlangCompiler {
    public:
        SlangCompiler();
        ~SlangCompiler();
        SlangCompiler(const SlangCompiler&) = delete;
        SlangCompiler& operator=(const SlangCompiler&) = delete;

        // Fills diagnostics on success (warnings) and on failure.
        [[nodiscard]] Result<CompileOutput> compile(const CompileRequest& request,
                                                    CompileDiagnostics& diagnostics);

        // Reads and verifies manifest.json and the IR files beside it.
        [[nodiscard]] static Result<std::shared_ptr<const ShaderPackage>> load_package(
            const std::filesystem::path& manifest);

        // Identity of the lfs API sources this binary was built against.
        [[nodiscard]] static std::string_view abi_hash();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis::shaders
