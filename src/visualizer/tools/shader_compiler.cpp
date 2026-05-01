/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {
    struct Args {
        std::filesystem::path input;
        std::filesystem::path output;
        std::string symbol;
    };

    [[nodiscard]] bool endsWith(const std::string_view value, const std::string_view suffix) {
        return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
    }

    [[nodiscard]] std::optional<Args> parseArgs(const int argc, char** argv) {
        Args args;
        for (int i = 1; i < argc; ++i) {
            const std::string_view key = argv[i];
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << key << '\n';
                return std::nullopt;
            }
            const char* const value = argv[++i];
            if (key == "--input") {
                args.input = value;
            } else if (key == "--output") {
                args.output = value;
            } else if (key == "--symbol") {
                args.symbol = value;
            } else {
                std::cerr << "Unknown argument " << key << '\n';
                return std::nullopt;
            }
        }

        if (args.input.empty() || args.output.empty() || args.symbol.empty()) {
            std::cerr << "Usage: lfs_shader_compiler --input <shader> --output <header> --symbol <name>\n";
            return std::nullopt;
        }
        return args;
    }

    [[nodiscard]] bool isIdentifier(const std::string_view symbol) {
        if (symbol.empty() || (!std::isalpha(static_cast<unsigned char>(symbol.front())) && symbol.front() != '_')) {
            return false;
        }
        return std::ranges::all_of(symbol, [](const char c) {
            return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
        });
    }

    [[nodiscard]] std::optional<EShLanguage> inferStage(const std::filesystem::path& path) {
        const std::string filename = path.filename().string();
        if (endsWith(filename, ".vert") || endsWith(filename, "VS.glsl")) {
            return EShLangVertex;
        }
        if (endsWith(filename, ".frag") || endsWith(filename, "FS.glsl")) {
            return EShLangFragment;
        }
        if (endsWith(filename, ".geom") || endsWith(filename, "GS.glsl")) {
            return EShLangGeometry;
        }
        if (endsWith(filename, ".comp") || endsWith(filename, "CS.glsl")) {
            return EShLangCompute;
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::string> readTextFile(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open shader source: " << path << '\n';
            return std::nullopt;
        }
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }

    [[nodiscard]] std::optional<std::vector<std::uint32_t>> compileGlsl(
        const std::filesystem::path& input,
        const std::string& source,
        const EShLanguage stage) {
        const char* source_ptr = source.c_str();
        const std::string input_name = input.string();
        const char* source_name = input_name.c_str();

        glslang::TShader shader(stage);
        shader.setStringsWithLengthsAndNames(&source_ptr, nullptr, &source_name, 1);
        shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
        shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
        shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);

        constexpr auto messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
        if (!shader.parse(GetDefaultResources(), 450, false, messages)) {
            std::cerr << "Failed to compile " << input << ":\n" << shader.getInfoLog() << '\n'
                      << shader.getInfoDebugLog() << '\n';
            return std::nullopt;
        }

        glslang::TProgram program;
        program.addShader(&shader);
        if (!program.link(messages)) {
            std::cerr << "Failed to link " << input << ":\n" << program.getInfoLog() << '\n'
                      << program.getInfoDebugLog() << '\n';
            return std::nullopt;
        }

        std::vector<std::uint32_t> spirv;
        glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);
        return spirv;
    }

    [[nodiscard]] bool writeHeader(
        const std::filesystem::path& output,
        const std::string_view symbol,
        const std::vector<std::uint32_t>& spirv) {
        if (const auto parent = output.parent_path(); !parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        std::ofstream file(output, std::ios::binary | std::ios::trunc);
        if (!file) {
            std::cerr << "Failed to open generated header: " << output << '\n';
            return false;
        }

        file << "/* Generated by lfs_shader_compiler. Do not edit. */\n"
             << "#pragma once\n\n"
             << "#include <cstdint>\n\n"
             << "namespace lfs::vis::viewport_shaders {\n\n"
             << "static constexpr std::uint32_t " << symbol << "[] = {\n";

        for (std::size_t i = 0; i < spirv.size(); ++i) {
            if (i % 6 == 0) {
                file << "    ";
            }
            file << "0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0')
                 << spirv[i] << std::dec << std::nouppercase << "u";
            if (i + 1 != spirv.size()) {
                file << ", ";
            }
            if (i % 6 == 5 || i + 1 == spirv.size()) {
                file << '\n';
            }
        }

        file << "};\n\n"
             << "} // namespace lfs::vis::viewport_shaders\n";
        return true;
    }
}

int main(const int argc, char** argv) {
    const auto args = parseArgs(argc, argv);
    if (!args) {
        return 2;
    }
    if (!isIdentifier(args->symbol)) {
        std::cerr << "Invalid C++ symbol name: " << args->symbol << '\n';
        return 2;
    }

    const auto stage = inferStage(args->input);
    if (!stage) {
        std::cerr << "Unable to infer shader stage from " << args->input << '\n';
        return 2;
    }

    const auto source = readTextFile(args->input);
    if (!source) {
        return 1;
    }

    if (!glslang::InitializeProcess()) {
        std::cerr << "Failed to initialize glslang\n";
        return 1;
    }

    const auto spirv = compileGlsl(args->input, *source, *stage);
    glslang::FinalizeProcess();

    if (!spirv) {
        return 1;
    }
    return writeHeader(args->output, args->symbol, *spirv) ? 0 : 1;
}
