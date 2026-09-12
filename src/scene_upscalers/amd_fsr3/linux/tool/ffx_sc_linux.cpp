// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: MIT
//
// Linux uses the SDK's GLSL compiler sources with a small driver because the
// v1.1.4 SDK ships only a Windows shader-compiler executable. HLSL/DXC is not
// part of this plugin-side Linux build.

#include "glsl_compiler.h"
#include "utils.h"

#include <filesystem>
#include <map>
#include <stdexcept>

namespace {

    struct PermutationOption {
        std::string definition;
        std::vector<std::string> values;
        unsigned numBits = 0;
    };

    struct Parameters {
        std::vector<PermutationOption> options;
        std::vector<std::string> compilerArgs;
        std::string output;
        std::string input;
        std::string name;
        std::string compiler;
        std::string glslang;
        std::string deps;
        bool reflection = false;
    };

    static bool startsWith(const std::string& value, const char* prefix) {
        return value.rfind(prefix, 0) == 0;
    }

    static void parsePermutation(const std::string& argument, Parameters& params) {
        const size_t equal = argument.find('=');
        const size_t open = argument.find('{', equal);
        const size_t close = argument.rfind('}');
        if (equal == std::string::npos || open == std::string::npos || close <= open)
            throw std::runtime_error("malformed permutation argument: " + argument);

        PermutationOption option;
        option.definition = argument.substr(2, equal - 2);
        size_t begin = open + 1;
        while (begin < close) {
            size_t end = argument.find(',', begin);
            if (end == std::string::npos || end > close)
                end = close;
            option.values.push_back(argument.substr(begin, end - begin));
            begin = end + 1;
        }
        if (option.values.empty())
            throw std::runtime_error("empty permutation argument: " + argument);
        option.numBits = 0;
        while ((1u << option.numBits) < option.values.size())
            ++option.numBits;
        params.options.push_back(std::move(option));
    }

    static Parameters parse(int argc, char** argv) {
        Parameters params;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (startsWith(argument, "-D") && argument.find('{') != std::string::npos)
                parsePermutation(argument, params);
            else if (startsWith(argument, "-D")) {
                params.compilerArgs.push_back("-D");
                params.compilerArgs.push_back(argument.substr(2));
            } else if (argument == "-reflection")
                params.reflection = true;
            else if (startsWith(argument, "-output="))
                params.output = argument.substr(8);
            else if (startsWith(argument, "-name="))
                params.name = argument.substr(6);
            else if (startsWith(argument, "-compiler="))
                params.compiler = argument.substr(10);
            else if (startsWith(argument, "-glslangexe="))
                params.glslang = argument.substr(12);
            else if (startsWith(argument, "-deps="))
                params.deps = argument.substr(6);
            else if (!argument.empty() && argument[0] == '-') {
                params.compilerArgs.push_back(argument);
                while (i + 1 < argc && argv[i + 1][0] != '-')
                    params.compilerArgs.push_back(argv[++i]);
            } else
                params.input = argument;
        }

        if (params.compiler != "glslang" || params.output.empty() || params.input.empty() || params.name.empty())
            throw std::runtime_error("Linux driver requires -compiler=glslang, -name, -output, and an input shader");
        if (params.options.empty())
            throw std::runtime_error("no permutation options supplied");
        return params;
    }

    static void generate(const Parameters& params,
                         size_t optionIndex,
                         unsigned key,
                         std::vector<std::string>& defines,
                         std::vector<Permutation>& out) {
        if (optionIndex == params.options.size()) {
            Permutation permutation;
            permutation.key = key;
            permutation.sourcePath = params.input;
            for (const std::string& define : defines) {
                permutation.defines.push_back(L"-D");
                permutation.defines.push_back(UTF8ToWChar(define));
            }
            out.push_back(std::move(permutation));
            return;
        }

        const auto& option = params.options[optionIndex];
        unsigned bit = 0;
        for (const auto& previous : params.options) {
            if (&previous == &option)
                break;
            bit += previous.numBits;
        }
        for (unsigned value = 0; value < option.values.size(); ++value) {
            defines.push_back(option.definition + "=" + option.values[value]);
            generate(params, optionIndex + 1, key | (value << bit), defines, out);
            defines.pop_back();
        }
    }

    static void writeBinary(const Parameters& params, GLSLCompiler& compiler, const Permutation& permutation, std::mutex& writeMutex) {
        const std::string base = params.name + "_" + permutation.hashDigest;
        const std::filesystem::path path = std::filesystem::path(params.output) / (base + ".h");
        FILE* file = fopen(path.c_str(), "wb");
        if (!file)
            throw std::runtime_error("cannot create generated header: " + path.string());

        fprintf(file, "// %s.h.\n// Auto generated by FidelityFX-SC.\n\n", base.c_str());
        if (params.reflection)
            compiler.WriteBinaryHeaderReflectionData(file, permutation, writeMutex);

        const int size = static_cast<int>(permutation.shaderBinary->BufferSize());
        const uint8_t* data = permutation.shaderBinary->BufferPointer();
        fprintf(file, "static const uint32_t g_%s_size = %d;\n\n", base.c_str(), size);
        fprintf(file, "static const unsigned char g_%s_data[] = {\n", base.c_str());
        for (int i = 0; i < size; ++i)
            fprintf(file, "0x%02x%s", data[i], i == size - 1 ? "" : ((i + 1) % 16 == 0 ? ",\n" : ","));
        fprintf(file, "\n};\n\n");
        fclose(file);
    }

    static void writePermutationHeader(const Parameters& params, GLSLCompiler& compiler,
                                       const std::vector<Permutation>& unique,
                                       const std::map<unsigned, unsigned>& indices) {
        const std::filesystem::path path = std::filesystem::path(params.output) / (params.name + "_permutations.h");
        FILE* file = fopen(path.c_str(), "wb");
        if (!file)
            throw std::runtime_error("cannot create permutation header: " + path.string());

        for (const auto& permutation : unique)
            fprintf(file, "#include \"%s_%s.h\"\n", params.name.c_str(), permutation.hashDigest.c_str());
        fprintf(file, "\n");

        fprintf(file, "typedef union %s_PermutationKey {\n    struct {\n", params.name.c_str());
        for (const auto& option : params.options)
            fprintf(file, "        uint32_t %s : %u;\n", option.definition.c_str(), option.numBits);
        fprintf(file, "    };\n    uint32_t index;\n} %s_PermutationKey;\n\n", params.name.c_str());

        fprintf(file, "typedef struct %s_PermutationInfo {\n    const uint32_t       blobSize;\n    const unsigned char* blobData;\n", params.name.c_str());
        if (params.reflection)
            compiler.WritePermutationHeaderReflectionStructMembers(file);
        fprintf(file, "} %s_PermutationInfo;\n\n", params.name.c_str());

        unsigned usedBits = 0;
        for (const auto& option : params.options)
            usedBits += option.numBits;
        const unsigned possible = 1u << usedBits;
        fprintf(file, "static const uint32_t g_%s_IndirectionTable[] = {\n", params.name.c_str());
        for (unsigned key = 0; key < possible; ++key) {
            const auto it = indices.find(key);
            fprintf(file, "    %u,\n", it == indices.end() ? 0u : it->second);
        }
        fprintf(file, "};\n\n");

        fprintf(file, "static const %s_PermutationInfo g_%s_PermutationInfo[] = {\n", params.name.c_str(), params.name.c_str());
        for (const auto& permutation : unique) {
            const std::string base = params.name + "_" + permutation.hashDigest;
            fprintf(file, "    { g_%s_size, g_%s_data, ", base.c_str(), base.c_str());
            if (params.reflection)
                compiler.WritePermutationHeaderReflectionData(file, permutation);
            fprintf(file, "},\n");
        }
        fprintf(file, "};\n\n");
        fclose(file);
    }

} // namespace

int main(int argc, char** argv) {
    try {
        const Parameters params = parse(argc, argv);
        std::filesystem::create_directories(params.output);
        GLSLCompiler compiler(params.glslang, params.input, params.name,
                              std::filesystem::path(params.input).filename().string(),
                              params.output, false, false);

        std::vector<Permutation> permutations;
        std::vector<std::string> defines;
        generate(params, 0, 0, defines, permutations);

        std::vector<Permutation> unique;
        std::map<std::string, unsigned> hashes;
        std::map<unsigned, unsigned> indices;
        std::mutex writeMutex;
        for (auto& permutation : permutations) {
            std::vector<std::string> args;
            for (const auto& define : permutation.defines)
                args.push_back(WCharToUTF8(define));
            args.insert(args.end(), params.compilerArgs.begin(), params.compilerArgs.end());
            if (!compiler.Compile(permutation, args, writeMutex))
                throw std::runtime_error("failed to compile shader permutation key " + std::to_string(permutation.key));
            if (params.reflection)
                compiler.ExtractReflectionData(permutation);

            auto [it, inserted] = hashes.emplace(permutation.hashDigest, static_cast<unsigned>(unique.size()));
            indices[permutation.key] = it->second;
            if (inserted) {
                writeBinary(params, compiler, permutation, writeMutex);
                permutation.shaderBinary.reset();
                unique.push_back(std::move(permutation));
            } else
                permutation.shaderBinary.reset();
        }
        writePermutationHeader(params, compiler, unique, indices);
        if (params.deps == "gcc") {
            const auto dep = std::filesystem::path(params.output) / (params.name + "_permutations.h.d");
            FILE* file = fopen(dep.c_str(), "wb");
            fprintf(file, "%s:", std::filesystem::absolute(std::filesystem::path(params.output) / (params.name + "_permutations.h")).c_str());
            fclose(file);
        }
        printf("%s: Processed %zu shader permutations, generated %zu unique permutations.\n",
               params.input.c_str(), permutations.size(), unique.size());
        return 0;
    } catch (const std::exception& error) {
        // LFS-CENSUS-OK(empty-catch): standalone build tool without the LichtFeld logger; the failure is printed to stderr and returned as the exit status
        fprintf(stderr, "ffx_sc failed: %s\n", error.what());
        return 1;
    }
}
