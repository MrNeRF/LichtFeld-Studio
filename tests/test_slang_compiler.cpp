/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/rasterizer/vulkan/src/gs_renderer.h"
#include "shaders/slang_compiler.hpp"
#include "user_shader_sources.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace {
    namespace fs = std::filesystem;
    using namespace lfs::vis::shaders;

    using lfs::test::kIdentity;

    fs::path package_dir() { return LFS_SHADER_TEST_PACKAGE_DIR; }

    class SlangCompilerTest : public ::testing::Test {
    protected:
        void SetUp() override {
            auto loaded = SlangCompiler::load_package(package_dir() / "manifest.json");
            ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
            package = *loaded;
            request = {.effects = {{kIdentity}}, .package = package, .host = "lfs_test_host"};
        }

        // Successful outputs are kept for spirv-val in validation builds.
        lfs::Result<CompileOutput> compile() {
            auto result = compiler.compile(request, diagnostics);
            if (result) {
                fs::create_directories(LFS_SHADER_TEST_SPIRV_DIR);
                std::ofstream out(fs::path(LFS_SHADER_TEST_SPIRV_DIR) / (result->cache_key + ".spv"), std::ios::binary);
                out.write(reinterpret_cast<const char*>(result->spirv.data()),
                          static_cast<std::streamsize>(result->spirv.size() * sizeof(uint32_t)));
                EXPECT_TRUE(out.good());
            }
            return result;
        }

        void expect_rejected(std::string source) {
            request.effects[0].source = std::move(source);
            const auto result = compile();
            EXPECT_FALSE(result) << request.effects[0].source;
            EXPECT_FALSE(diagnostics.entries.empty());
        }

        // A modified copy of the loaded package for contract tests.
        template <class F>
        void with_package(F&& change) {
            auto copy = std::make_shared<ShaderPackage>(*package);
            change(*copy);
            request.package = copy;
        }

        const Diagnostic* find_error(const std::string_view text) const {
            for (const auto& d : diagnostics.entries)
                if (d.severity == DiagnosticSeverity::Error && d.message.find(text) != std::string::npos)
                    return &d;
            return nullptr;
        }

        std::shared_ptr<const ShaderPackage> package;
        CompileRequest request;
        SlangCompiler compiler;
        CompileDiagnostics diagnostics;
    };

    TEST_F(SlangCompilerTest, CompilesAndReflectsIsolatedEdits) {
        {
            SCOPED_TRACE("identity and signature forms");
            const auto result = compile();
            ASSERT_TRUE(result) << diagnostics.raw << lfs::format_for_developer(result.error());
            ASSERT_GE(result->spirv.size(), 5u);
            EXPECT_EQ(result->spirv[0], 0x07230203u);
            EXPECT_EQ(result->spirv[1], 0x00010500u); // SPIR-V 1.5, as the legacy recipes
            EXPECT_EQ(reinterpret_cast<uintptr_t>(result->spirv.data()) % alignof(uint32_t), 0u);
            EXPECT_TRUE(result->parameters[0].empty());
            EXPECT_FALSE(result->cache_hit);
            EXPECT_EQ(result->cache_key.size(), 64u);
            request.effects[0].source = R"(import lfs;
public struct Params {
    [LfsParam(0.0, 5.0, 1.0)] float speed;
};
LfsElement effect(LfsElement e, LfsContext ctx, Params p) {
    e.position.y += sin(ctx.time * p.speed);
    return e;
})";
            auto with_params = compile();
            ASSERT_TRUE(with_params) << diagnostics.raw << lfs::format_for_developer(with_params.error());
            ASSERT_EQ(with_params->parameters[0].size(), 1u);
            EXPECT_NE(result->spirv, with_params->spirv);
            // Params may be declared and read through slots directly.
            request.effects[0].source = R"(import lfs;
struct Params { [LfsParam(0.0, 1.0, 0.5)] float amount; };
LfsElement effect(LfsElement e, LfsContext ctx) {
    e.opacity *= ctx.param(0).x;
    return e;
})";
            auto low_level = compile();
            ASSERT_TRUE(low_level) << diagnostics.raw;
            EXPECT_EQ(low_level->parameters[0].size(), 1u);
        }
        {
            SCOPED_TRACE("declaration order");
            request.effects[0].source = R"(import lfs;
struct Params {
    [LfsParam(-1.0, 2.0, 0.5)] float z;
    [LfsParam(0, 3, 1)] float2 offset;
    [LfsParam(0.0, 1.0, 1.0)] float3 tint;
    [LfsParam(-4.0, 4.0, -4.0)] float4 axis;
};
LfsElement effect(LfsElement e, LfsContext ctx, Params p) {
    e.position += p.z + p.offset.x + p.axis.w;
    e.color *= p.tint;
    return e;
})";
            const auto result = compile();
            ASSERT_TRUE(result) << diagnostics.raw << lfs::format_for_developer(result.error());
            const std::vector<Parameter> expected = {{"z", 0, 1, -1.0f, 2.0f, 0.5f},
                                                     {"offset", 1, 2, 0.0f, 3.0f, 1.0f},
                                                     {"tint", 2, 3, 0.0f, 1.0f, 1.0f},
                                                     {"axis", 3, 4, -4.0f, 4.0f, -4.0f}};
            EXPECT_EQ(result->parameters[0], expected);
        }
        {
            SCOPED_TRACE("fresh sessions after edits and failure");
            std::vector<uint32_t> previous;
            for (int i = 1; i <= 6; ++i) {
                request.effects[0].source = "import lfs;\nLfsElement effect(LfsElement e, LfsContext ctx) { e.opacity = " +
                                            std::to_string(i) + ".0; return e; }\n";
                const auto result = compile();
                ASSERT_TRUE(result) << diagnostics.raw;
                EXPECT_FALSE(result->cache_hit);
                EXPECT_NE(result->spirv, previous);
                previous = result->spirv;
            }
            // A failed edit leaves no module behind that a later compile could reuse.
            expect_rejected("import lfs;\nLfsElement effect(LfsElement e, LfsContext ctx) { return broken; }\n");
            request.effects[0].source = std::string(kIdentity) + "// recovery after failure\n";
            const auto recovered = compile();
            ASSERT_TRUE(recovered) << diagnostics.raw;
            EXPECT_FALSE(recovered->cache_hit);
            // Switching signatures keeps working with the same module name.
            request.effects[0].source = "import lfs;\nstruct Params { [LfsParam(0.0, 1.0, 0.25)] float a; };\n"
                                        "LfsElement effect(LfsElement e, LfsContext ctx, Params p) { e.opacity = p.a; return e; }\n";
            const auto with_params = compile();
            ASSERT_TRUE(with_params) << diagnostics.raw;
            EXPECT_EQ(with_params->parameters[0].size(), 1u);
        }
    }

    TEST_F(SlangCompilerTest, AcceptsSixteenParamsAndRejectsSeventeen) {
        std::string fields;
        for (int i = 0; i < 16; ++i)
            fields += "[LfsParam(0.0, 1.0, 0.0)] float p" + std::to_string(i) + ";\n";
        const std::string body = "};\nLfsElement effect(LfsElement e, LfsContext ctx, Params p) "
                                 "{ e.opacity = p.p0 + p.p15; return e; }\n";
        request.effects[0].source = "import lfs;\nstruct Params {\n" + fields + body;
        const auto result = compile();
        ASSERT_TRUE(result) << diagnostics.raw;
        ASSERT_EQ(result->parameters[0].size(), 16u);
        EXPECT_EQ(result->parameters[0].back().slot, 15u);
        expect_rejected("import lfs;\nstruct Params {\n" + fields + "[LfsParam(0.0, 1.0, 0.0)] float p16;\n" + body);
    }

    TEST_F(SlangCompilerTest, RejectsInvalidSources) {
        for (const auto& [name, field] : std::initializer_list<std::pair<const char*, const char*>>{
                 {"RejectsInvalidParams/int", "[LfsParam(0.0, 1.0, 0.0)] int x;"},
                 {"RejectsInvalidParams/uint", "[LfsParam(0.0, 1.0, 0.0)] uint x;"},
                 {"RejectsInvalidParams/bool", "[LfsParam(0.0, 1.0, 0.0)] bool x;"},
                 {"RejectsInvalidParams/half", "[LfsParam(0.0, 1.0, 0.0)] half x;"},
                 {"RejectsInvalidParams/matrix", "[LfsParam(0.0, 1.0, 0.0)] float2x2 x;"},
                 {"RejectsInvalidParams/array", "[LfsParam(0.0, 1.0, 0.0)] float x[2];"},
                 {"RejectsInvalidParams/reversed range", "[LfsParam(2.0, 1.0, 1.5)] float x;"},
                 {"RejectsInvalidParams/above range", "[LfsParam(0.0, 1.0, 2.0)] float x;"},
                 {"RejectsInvalidParams/below range", "[LfsParam(0.0, 1.0, -1.0)] float x;"},
                 {"RejectsInvalidParams/infinity", "[LfsParam(0.0, 1e39, 0.0)] float x;"},
                 {"RejectsInvalidParams/nan", "[LfsParam(0.0, 0.0 / 0.0, 0.0)] float x;"},
                 {"RejectsInvalidParams/attribute arity", "[LfsParam(0.0, 1.0)] float x;"},
                 {"RejectsInvalidParams/misspelled attribute", "[LfsParm(0.0, 1.0, 0.0)] float x;"},
                 {"RejectsInvalidParams/missing attribute", "float x;"}}) {

            SCOPED_TRACE(name);
            expect_rejected(std::string("import lfs;\nstruct Params { ") + field +
                            " };\nLfsElement effect(LfsElement e, LfsContext ctx, Params p) { return e; }\n");
        }
        for (const auto& [name, source] : std::initializer_list<std::pair<const char*, const char*>>{
                 {"RejectsMissingOrWrongEffect/empty", ""},
                 {"RejectsMissingOrWrongEffect/missing entry", "import lfs;"},
                 {"RejectsMissingOrWrongEffect/wrong return", "import lfs;\nfloat effect(float x) { return x; }"},
                 {"RejectsMissingOrWrongEffect/missing context", "import lfs;\nLfsElement effect(LfsElement e) { return e; }"},
                 {"RejectsMissingOrWrongEffect/wrong params", "import lfs;\nLfsElement effect(LfsElement e, LfsContext ctx, float p) { return e; }"},
                 {"RejectsMissingOrWrongEffect/ambiguous overload", "import lfs;\nstruct Params { [LfsParam(0.0, 1.0, 0.0)] float a; };\n"
                                                                    "LfsElement effect(LfsElement e, LfsContext ctx) { return e; }\n"
                                                                    "LfsElement effect(LfsElement e, LfsContext ctx, Params p) { return e; }"},
                 {"RejectsMissingOrWrongEffect/generic entry", "import lfs;\nLfsElement effect<T>(LfsElement e, LfsContext ctx) { return e; }"},
                 {"RejectsMissingOrWrongEffect/module declaration", "module mine;\nimport lfs;\nLfsElement effect(LfsElement e, LfsContext ctx) { return e; }"},
                 {"RejectsMissingOrWrongEffect/missing import", "LfsElement effect(LfsElement e, LfsContext ctx) { return e; }"}}) {

            SCOPED_TRACE(name);
            expect_rejected(source);
        }
        for (const auto& [name, declaration] : std::initializer_list<std::pair<const char*, const char*>>{
                 {"RejectsResourcesAndForeignImports/storage buffer", "RWStructuredBuffer<float> injected;"},
                 {"RejectsResourcesAndForeignImports/binding collision", "[[vk::binding(0, 0)]] RWStructuredBuffer<float4> output;"},
                 {"RejectsResourcesAndForeignImports/texture", "Texture2D<float4> injected;"},
                 {"RejectsResourcesAndForeignImports/sampler", "SamplerState injected;"},
                 {"RejectsResourcesAndForeignImports/constant buffer", "ConstantBuffer<LfsElement> injected;"},
                 {"RejectsResourcesAndForeignImports/uniform", "uniform float injected;"},
                 {"RejectsResourcesAndForeignImports/host import", "import lfs_test_host;"},
                 {"RejectsResourcesAndForeignImports/missing import", "import missing_dependency;"},
                 {"RejectsResourcesAndForeignImports/adapter include", "#include \"lfs_adapter.slang\""},
                 {"RejectsResourcesAndForeignImports/numbered adapter include", "#include \"lfs_adapter0.slang\""},
                 {"RejectsResourcesAndForeignImports/other effect import", "import lfs_effect1;"},
                 {"RejectsResourcesAndForeignImports/dispatcher import", "import lfs_effects;"},
                 {"RejectsResourcesAndForeignImports/utility include", "#include \"utils.slang\""}}) {

            SCOPED_TRACE(name);
            expect_rejected(std::string("import lfs;\n") + declaration +
                            "\nLfsElement effect(LfsElement e, LfsContext ctx) { return e; }\n");
        }
    }

    TEST_F(SlangCompilerTest, ReportsUserLocationsUnshiftedByTheAdapter) {
        request.effects[0].name = "C:/effects/wave.slang";
        // Line 6 holds the error after a two-byte character and a tab.
        expect_rejected("import lfs;\n"
                        "struct Params { [LfsParam(0.0, 1.0, 0.5)] float amount; };\n"
                        "\n"
                        "LfsElement effect(LfsElement e, LfsContext ctx, Params p) {\n"
                        "    e.opacity = p.amount;\n"
                        "\t/* \xc3\xa9 */ return missing_identifier;\n"
                        "}\n");
        const auto* error = find_error("missing_identifier");
        ASSERT_NE(error, nullptr) << diagnostics.raw;
        EXPECT_EQ(error->file, request.effects[0].name);
        EXPECT_EQ(error->line, 6u);
        EXPECT_EQ(error->column, 17u); // code points, not bytes
        EXPECT_FALSE(error->code.empty());
        for (const auto& d : diagnostics.entries)
            EXPECT_EQ(d.file.find("lfs_adapter"), std::string::npos) << d.message;
    }

    TEST_F(SlangCompilerTest, HandlesLongLinesAndRejectsNonUtf8) {
        // The compiler echoes the whole offending line in its diagnostics.
        expect_rejected("import lfs;\nLfsElement effect(LfsElement e, LfsContext ctx) { " +
                        std::string(400000, ' ') + "return missing_identifier; }\n");
        const auto* error = find_error("missing_identifier");
        ASSERT_NE(error, nullptr) << diagnostics.raw.substr(0, 2000);
        EXPECT_EQ(error->line, 2u);
        // A truncated sequence in a trailing comment would swallow the line
        // break before the generated adapter include.
        for (const char byte : {'\xe9', '\xff'}) {
            expect_rejected(std::string(kIdentity) + "// " + byte + "\n");
            EXPECT_NE(find_error("UTF-8"), nullptr);
        }
        request.effects[0].source = kIdentity;
        request.effects[0].name = "effect\xff.slang";
        EXPECT_FALSE(compile());
    }

    TEST_F(SlangCompilerTest, RejectsMismatchedPackages) {
        ASSERT_TRUE(compile()) << diagnostics.raw;
        for (const auto& [name, change] : std::initializer_list<std::pair<const char*, void (*)(ShaderPackage&)>>{
                 {"ABI", [](ShaderPackage& p) { p.abi_hash = std::string(64, '0'); }},
                 {"compiler", [](ShaderPackage& p) { p.compiler_build += "-other"; }},
                 {"version", [](ShaderPackage& p) { p.version = 2; }},
                 {"profile", [](ShaderPackage& p) { p.profile = "no_such_profile"; }},
                 {"host hash", [](ShaderPackage& p) { p.hosts[0].module.sha256 = std::string(64, '0'); }},
                 {"missing API", [](ShaderPackage& p) { p.api.erase(p.api.begin()); }},
                 {"foreign IR", [](ShaderPackage& p) {
                      p.api.back().bytes = p.hosts[0].module.bytes;
                      p.api.back().sha256 = p.hosts[0].module.sha256;
                  }},
                 {"entry", [](ShaderPackage& p) { p.hosts[0].entry = "absent"; }},
                 {"binding index", [](ShaderPackage& p) { p.hosts[0].bindings[0].binding = 1; }},
                 {"binding count", [](ShaderPackage& p) { p.hosts[0].bindings[0].count = 2; }},
                 {"extra binding", [](ShaderPackage& p) { p.hosts[0].bindings.push_back({"extra", 0, 1, 1}); }},
                 {"push constants", [](ShaderPackage& p) { p.hosts[0].push_constant_bytes = 16; }},
                 {"workgroup", [](ShaderPackage& p) { p.hosts[0].workgroup[0] = 64; }},
                 {"subgroup", [](ShaderPackage& p) { p.hosts[0].subgroup_size *= 2; }},
                 {"import index", [](ShaderPackage& p) { p.hosts[0].imports.push_back(p.modules.size()); }}}) {
            SCOPED_TRACE(name);
            with_package(change);
            const auto result = compile();
            ASSERT_FALSE(result);
            EXPECT_NE(result.error().code(), lfs::ErrorCode::InvalidArgument) << result.error().detail();
            request.package = package;
        }
        request.host = "absent";
        EXPECT_FALSE(compile());
        request.host = "lfs_test_host";
        // Bytes that no longer match their hash are caught before they are compiled.
        with_package([](ShaderPackage& p) { p.api.back().bytes[64] ^= std::byte{1}; });
        request.effects[0].source += "// not cached\n";
        const auto corrupt = compile();
        ASSERT_FALSE(corrupt);
        EXPECT_EQ(corrupt.error().code(), lfs::ErrorCode::DataLoss);
    }

    TEST_F(SlangCompilerTest, CacheHitsOnlyIdenticalInputs) {
        const auto first = compile();
        ASSERT_TRUE(first) << diagnostics.raw;
        const auto again = compile();
        ASSERT_TRUE(again);
        EXPECT_TRUE(again->cache_hit);
        EXPECT_EQ(again->spirv, first->spirv);
        EXPECT_EQ(again->cache_key, first->cache_key);

        const auto original = request;
        const auto expect_miss = [&](const char* what) {
            SCOPED_TRACE(what);
            const auto result = compile();
            ASSERT_TRUE(result) << diagnostics.raw << lfs::format_for_developer(result.error());
            EXPECT_FALSE(result->cache_hit);
            EXPECT_NE(result->cache_key, first->cache_key);
            request = original;
        };
        request.effects[0].source += "// edited\n";
        expect_miss("source");
        request.effects[0].name = "renamed.slang";
        expect_miss("source name");
        with_package([](ShaderPackage& p) { p.optimization = 0; });
        expect_miss("optimization");
        with_package([](ShaderPackage& p) { p.fast_math = false; });
        expect_miss("floating point mode");
        with_package([](ShaderPackage& p) { p.column_major = false; });
        expect_miss("matrix layout");
        with_package([](ShaderPackage& p) { p.profile = "spirv_1_6"; });
        expect_miss("profile");
        with_package([](ShaderPackage& p) { p.hosts[0].defines = {"VARIANT=1"}; });
        expect_miss("host variant");
        const auto hit = compile();
        ASSERT_TRUE(hit);
        EXPECT_TRUE(hit->cache_hit);
    }

    // Each effect is a module of its own, with its own Params, and the
    // dispatcher runs one by its index.
    TEST_F(SlangCompilerTest, ComposesEffectsWithTheirOwnParams) {
        const EffectSource gain{lfs::test::kGain, "gain.slang"};
        const EffectSource tint{lfs::test::kTint, "tint.slang"};
        request.effects = {gain, {kIdentity, "identity.slang"}, tint};
        const auto composed = compile();
        ASSERT_TRUE(composed) << diagnostics.raw << lfs::format_for_developer(composed.error());
        ASSERT_EQ(composed->parameters.size(), 3u);
        EXPECT_EQ(composed->parameters[0], (std::vector<Parameter>{{"gain", 0, 1, 0.0f, 2.0f, 1.0f}}));
        EXPECT_TRUE(composed->parameters[1].empty());
        EXPECT_EQ(composed->parameters[2], (std::vector<Parameter>{{"tint", 0, 3, 0.0f, 1.0f, 0.5f}}));
        // The order is the dispatch order, so it is part of the cache key.
        request.effects = {tint, {kIdentity, "identity.slang"}, gain};
        const auto reordered = compile();
        ASSERT_TRUE(reordered) << diagnostics.raw;
        EXPECT_FALSE(reordered->cache_hit);
        EXPECT_NE(reordered->cache_key, composed->cache_key);
        // One file name per effect.
        request.effects = {gain, gain};
        EXPECT_FALSE(compile());
        EXPECT_NE(find_error("repeated"), nullptr);
        request.effects.clear();
        EXPECT_FALSE(compile());
    }

    TEST_F(SlangCompilerTest, ReportsEachEffectsDiagnosticsInItsOwnFile) {
        request.effects = {{kIdentity, "first.slang"},
                           {"import lfs;\n\nLfsElement effect(LfsElement e, LfsContext ctx) { return missing_identifier; }\n",
                            "second.slang"}};
        EXPECT_FALSE(compile());
        const auto* error = find_error("missing_identifier");
        ASSERT_NE(error, nullptr) << diagnostics.raw;
        EXPECT_EQ(error->file, "second.slang");
        EXPECT_EQ(error->line, 3u);
        // Contract checks name the effect's file too.
        diagnostics = {};
        request.effects[1].source = "import lfs;\nfloat effect(float x) { return x; }\n";
        EXPECT_FALSE(compile());
        error = find_error("Expected exactly one");
        ASSERT_NE(error, nullptr) << diagnostics.raw;
        EXPECT_EQ(error->file, "second.slang");
        EXPECT_EQ(error->code, "LFS");
        // Every failing effect reports, not only the first.
        diagnostics = {};
        request.effects[0].source =
            "import lfs;\n\nLfsElement effect(LfsElement e, LfsContext ctx) { return missing_identifier; }\n";
        EXPECT_FALSE(compile());
        error = find_error("missing_identifier");
        ASSERT_NE(error, nullptr) << diagnostics.raw;
        EXPECT_EQ(error->file, "first.slang");
        error = find_error("Expected exactly one");
        ASSERT_NE(error, nullptr) << diagnostics.raw;
        EXPECT_EQ(error->file, "second.slang");
    }

    TEST_F(SlangCompilerTest, LoadsRelocatedPackageWithoutSources) {
        const auto temp = fs::temp_directory_path() /
                          ("lfs_shader_package_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        struct Cleanup {
            fs::path path;
            ~Cleanup() {
                std::error_code ignored;
                fs::remove_all(path, ignored);
            }
        } cleanup{temp};
        fs::create_directories(temp);
        for (const auto& entry : fs::directory_iterator(package_dir()))
            if (entry.path().extension() == ".slang-module" || entry.path().filename() == "manifest.json")
                fs::copy_file(entry.path(), temp / entry.path().filename());
        auto relocated = SlangCompiler::load_package(temp / "manifest.json");
        ASSERT_TRUE(relocated) << lfs::format_for_developer(relocated.error());
        request.package = *relocated;
        // The compiler's file system serves no source files, so every API
        // symbol below comes from the relocated IR.
        request.effects[0].source = R"(import lfs;
struct Params { [LfsParam(0.0, 1.0, 1.0)] float3 tint; };
LfsElement effect(LfsElement e, LfsContext ctx, Params p) {
    e.color = lfs_hsv_to_rgb(lfs_rgb_to_hsv(p.tint) + lfs_noise(e.position) + lfs_hash(e.position));
    e.opacity *= lfs_reveal_mask(e.position, ctx) * lfs_ease(ctx.progress);
    return e;
})";
        const auto result = compile();
        ASSERT_TRUE(result) << diagnostics.raw << lfs::format_for_developer(result.error());
        EXPECT_EQ(result->parameters[0].size(), 1u);
        {
            std::fstream file(temp / "lfs.slang-module", std::ios::in | std::ios::out | std::ios::binary);
            file.seekp(100);
            file.put('\x7f');
        }
        EXPECT_FALSE(SlangCompiler::load_package(temp / "manifest.json"));
        fs::remove(temp / "lfs.slang-module");
        EXPECT_FALSE(SlangCompiler::load_package(temp / "manifest.json"));
        std::ofstream(temp / "manifest.json") << "{\"version\": 1}";
        EXPECT_FALSE(SlangCompiler::load_package(temp / "manifest.json"));
        EXPECT_FALSE(SlangCompiler::load_package(temp / "absent.json"));
    }

    // The shipped package holds exactly the projection hosts, the only
    // programs that call the effect, and they link against their imports;
    // the program set tests build them.
    TEST_F(SlangCompilerTest, ShipsTheCustomEffectHosts) {
        auto shipped = SlangCompiler::load_package(fs::path(LFS_SHADER_PACKAGE_DIR) / "manifest.json");
        ASSERT_TRUE(shipped) << lfs::format_for_developer(shipped.error());
        const auto& hosts = (*shipped)->hosts;
        const auto& modules = (*shipped)->modules;
        EXPECT_EQ(hosts.size(), 20u);
        // utils, then utils built for banded export.
        ASSERT_EQ(modules.size(), 4u);
        EXPECT_EQ(modules[0].name, "lfs_reveal");
        EXPECT_EQ(modules[1].name, "spherical_harmonics");
        EXPECT_EQ(modules[2].name, "utils");
        EXPECT_EQ(modules[3].name, "utils");
        for (const auto& host : hosts) {
            const auto& name = host.module.name;
            SCOPED_TRACE(name);
            const bool banded = name.ends_with("_custom_banded");
            EXPECT_TRUE(name.starts_with("projection_forward") && (banded || name.ends_with("_custom")));
            // The viewport bands only survivor and 3DGUT chains.
            if (banded)
                EXPECT_TRUE(name.contains("_survivors_") || name.contains("_3dgut_"));
            EXPECT_EQ(host.imports, (std::vector<size_t>{0, 1, banded ? 3u : 2u}));
            EXPECT_EQ(host.subgroup_size, uint32_t{SUBGROUP_SIZE});
            EXPECT_EQ(host.push_constant_bytes, sizeof(VulkanGSRendererUniforms));
        }
        EXPECT_FALSE(fs::exists(fs::path(LFS_SHADER_PACKAGE_DIR) / "lfs_test_host.slang-module"));
        request.package = *shipped;
        request.host = "projection_forward_survivors_custom";
        const auto result = compile();
        ASSERT_TRUE(result) << diagnostics.raw << lfs::format_for_developer(result.error());
        EXPECT_EQ(result->spirv[0], 0x07230203u);
    }

    TEST_F(SlangCompilerTest, RejectsCallsFromAnotherThread) {
        lfs::ErrorCode code = lfs::ErrorCode::Internal;
        bool compiled = true;
        std::thread([&] {
            CompileDiagnostics other;
            const auto result = compiler.compile(request, other);
            compiled = result.has_value();
            if (!result)
                code = result.error().code();
        }).join();
        EXPECT_FALSE(compiled);
        EXPECT_EQ(code, lfs::ErrorCode::FailedPrecondition);
    }
} // namespace
