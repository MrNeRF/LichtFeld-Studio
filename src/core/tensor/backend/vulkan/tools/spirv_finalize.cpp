/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Build-time post-processing for the compiled tensor compute modules.
//
//   spirv_finalize <input.spv> <output.spv> <manifest.json>
//   spirv_finalize embed <table.cpp> <name> <finalized.spv> [<name> <finalized.spv> ...]
//
// The backend contract (plan D12) requires every module to declare the
// SignedZeroInfNanPreserve execution mode for each float width it uses; the
// pinned Slang release has no switch for it, so it is written here. The
// manifest the runtime loader checks against is read from the finished
// binary (capabilities, extensions, entry point, workgroup size, push
// constant block size, float controls) rather than from the source text.

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp>

#ifdef LFS_SPIRV_FINALIZE_HAS_TOOLS
#include <spirv-tools/libspirv.hpp>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

    struct Instruction {
        spv::Op opcode;
        std::vector<uint32_t> operands;
    };

    struct Module {
        std::array<uint32_t, 5> header{};
        std::vector<Instruction> instructions;
    };

    [[noreturn]] void fail(const std::string& message) {
        throw std::runtime_error(message);
    }

    std::vector<uint32_t> read_words(const char* path) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) {
            fail(std::string("cannot open ") + path);
        }
        const std::streamoff bytes = stream.tellg();
        if (bytes < 20 || bytes % 4 != 0) {
            fail(std::string("not a SPIR-V module: ") + path);
        }
        std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(words.data()), bytes);
        if (!stream || words[0] != spv::MagicNumber) {
            fail(std::string("not a SPIR-V module: ") + path);
        }
        return words;
    }

    Module parse(const std::vector<uint32_t>& words) {
        Module module;
        std::copy_n(words.begin(), 5, module.header.begin());
        size_t index = 5;
        while (index < words.size()) {
            const uint32_t word_count = words[index] >> 16;
            if (word_count == 0 || index + word_count > words.size()) {
                fail("malformed instruction stream");
            }
            module.instructions.push_back(Instruction{
                static_cast<spv::Op>(words[index] & 0xFFFFu),
                std::vector<uint32_t>(words.begin() + static_cast<std::ptrdiff_t>(index) + 1,
                                      words.begin() + static_cast<std::ptrdiff_t>(index + word_count)),
            });
            index += word_count;
        }
        return module;
    }

    std::vector<uint32_t> serialize(const Module& module) {
        std::vector<uint32_t> words(module.header.begin(), module.header.end());
        for (const Instruction& instruction : module.instructions) {
            words.push_back((static_cast<uint32_t>(instruction.operands.size() + 1) << 16) |
                            static_cast<uint32_t>(instruction.opcode));
            words.insert(words.end(), instruction.operands.begin(), instruction.operands.end());
        }
        return words;
    }

    std::string literal_string(const std::vector<uint32_t>& operands, const size_t first) {
        std::string text;
        for (size_t index = first; index < operands.size(); ++index) {
            char bytes[4];
            std::memcpy(bytes, &operands[index], 4);
            for (const char byte : bytes) {
                if (byte == '\0') {
                    return text;
                }
                text.push_back(byte);
            }
        }
        return text;
    }

    std::vector<uint32_t> encode_string(const std::string_view text) {
        std::vector<uint32_t> words((text.size() + 4) / 4, 0u);
        std::memcpy(words.data(), text.data(), text.size());
        return words;
    }

    struct Analysis {
        std::vector<spv::Capability> capabilities;
        std::vector<std::string> extensions;
        uint32_t entry_id = 0;
        std::string entry_name;
        spv::ExecutionModel execution_model = spv::ExecutionModelMax;
        std::array<uint32_t, 3> local_size{};
        std::set<uint32_t> float_widths;
        std::set<uint32_t> preserve_widths;
        uint32_t push_constant_size = 0;
    };

    class Layout {
    public:
        explicit Layout(const Module& module) {
            for (const Instruction& instruction : module.instructions) {
                switch (instruction.opcode) {
                case spv::OpTypeBool:
                case spv::OpTypeInt:
                case spv::OpTypeFloat:
                case spv::OpTypeVector:
                case spv::OpTypeMatrix:
                case spv::OpTypeArray:
                case spv::OpTypeRuntimeArray:
                case spv::OpTypeStruct:
                case spv::OpTypePointer:
                    types_[instruction.operands[0]] = &instruction;
                    break;
                case spv::OpConstant:
                    constants_[instruction.operands[1]] = instruction.operands[2];
                    break;
                case spv::OpDecorate:
                    if (instruction.operands[1] == spv::DecorationArrayStride) {
                        array_strides_[instruction.operands[0]] = instruction.operands[2];
                    }
                    break;
                case spv::OpMemberDecorate:
                    if (instruction.operands[2] == spv::DecorationOffset) {
                        member_offsets_[{instruction.operands[0], instruction.operands[1]}] =
                            instruction.operands[3];
                    }
                    break;
                default:
                    break;
                }
            }
        }

        uint32_t pointee_size(const uint32_t pointer_type) const {
            const Instruction& pointer = type(pointer_type);
            if (pointer.opcode != spv::OpTypePointer) {
                fail("push constant variable type is not a pointer");
            }
            return size(pointer.operands[2]);
        }

    private:
        const Instruction& type(const uint32_t id) const {
            const auto iterator = types_.find(id);
            if (iterator == types_.end()) {
                fail("unknown type id " + std::to_string(id));
            }
            return *iterator->second;
        }

        uint32_t size(const uint32_t id) const {
            const Instruction& instruction = type(id);
            switch (instruction.opcode) {
            case spv::OpTypeBool:
                return 4;
            case spv::OpTypeInt:
            case spv::OpTypeFloat:
                return instruction.operands[1] / 8;
            case spv::OpTypeVector:
                return size(instruction.operands[1]) * instruction.operands[2];
            case spv::OpTypeArray: {
                const auto stride = array_strides_.find(id);
                const auto length = constants_.find(instruction.operands[2]);
                if (stride == array_strides_.end() || length == constants_.end()) {
                    fail("array in a push constant block without a stride or a constant length");
                }
                return stride->second * length->second;
            }
            case spv::OpTypeStruct: {
                uint32_t extent = 0;
                for (uint32_t member = 0; member + 1 < instruction.operands.size(); ++member) {
                    const auto offset = member_offsets_.find({id, member});
                    if (offset == member_offsets_.end()) {
                        fail("struct member without an Offset decoration in a push constant block");
                    }
                    extent = std::max(extent, offset->second + size(instruction.operands[member + 1]));
                }
                return extent;
            }
            default:
                fail("unsupported type in a push constant block");
            }
        }

        std::map<uint32_t, const Instruction*> types_;
        std::map<uint32_t, uint32_t> constants_;
        std::map<uint32_t, uint32_t> array_strides_;
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> member_offsets_;
    };

    Analysis analyze(const Module& module) {
        Analysis analysis;
        const Layout layout(module);
        std::map<uint32_t, uint32_t> constants;
        std::optional<uint32_t> push_constant_variable;
        for (const Instruction& instruction : module.instructions) {
            switch (instruction.opcode) {
            case spv::OpCapability:
                analysis.capabilities.push_back(static_cast<spv::Capability>(instruction.operands[0]));
                break;
            case spv::OpExtension:
                analysis.extensions.push_back(literal_string(instruction.operands, 0));
                break;
            case spv::OpEntryPoint:
                if (analysis.entry_id != 0) {
                    fail("module has more than one entry point");
                }
                analysis.execution_model = static_cast<spv::ExecutionModel>(instruction.operands[0]);
                analysis.entry_id = instruction.operands[1];
                analysis.entry_name = literal_string(instruction.operands, 2);
                break;
            case spv::OpConstant:
                constants[instruction.operands[1]] = instruction.operands[2];
                break;
            case spv::OpTypeFloat:
                analysis.float_widths.insert(instruction.operands[1]);
                break;
            case spv::OpVariable:
                if (instruction.operands[2] == spv::StorageClassPushConstant) {
                    if (push_constant_variable) {
                        fail("module has more than one push constant block");
                    }
                    push_constant_variable = instruction.operands[0];
                }
                break;
            default:
                break;
            }
        }
        for (const Instruction& instruction : module.instructions) {
            if (instruction.opcode != spv::OpExecutionMode && instruction.opcode != spv::OpExecutionModeId) {
                continue;
            }
            const auto mode = static_cast<spv::ExecutionMode>(instruction.operands[1]);
            if (mode == spv::ExecutionModeLocalSize) {
                std::copy_n(instruction.operands.begin() + 2, 3, analysis.local_size.begin());
            } else if (mode == spv::ExecutionModeLocalSizeId) {
                for (size_t axis = 0; axis < 3; ++axis) {
                    const auto constant = constants.find(instruction.operands[2 + axis]);
                    if (constant == constants.end()) {
                        fail("LocalSizeId refers to a non-constant");
                    }
                    analysis.local_size[axis] = constant->second;
                }
            } else if (mode == spv::ExecutionModeSignedZeroInfNanPreserve) {
                analysis.preserve_widths.insert(instruction.operands[2]);
            }
        }
        if (analysis.entry_id == 0) {
            fail("module has no entry point");
        }
        if (analysis.execution_model != spv::ExecutionModelGLCompute) {
            fail("entry point is not a compute shader");
        }
        if (analysis.local_size[0] == 0) {
            fail("compute entry point without a LocalSize execution mode");
        }
        if (push_constant_variable) {
            analysis.push_constant_size = layout.pointee_size(*push_constant_variable);
        }
        return analysis;
    }

    // Inserts the float-controls capability, extension and execution modes the
    // module still lacks, keeping the logical layout order of the header
    // sections.
    void declare_float_controls(Module& module, const Analysis& analysis) {
        std::set<uint32_t> missing;
        for (const uint32_t width : analysis.float_widths) {
            if (!analysis.preserve_widths.contains(width)) {
                missing.insert(width);
            }
        }
        if (missing.empty()) {
            return;
        }
        auto& instructions = module.instructions;
        const auto last_of = [&](auto predicate) {
            auto iterator = instructions.end();
            for (auto current = instructions.begin(); current != instructions.end(); ++current) {
                if (predicate(*current)) {
                    iterator = current;
                }
            }
            return iterator;
        };
        const bool has_capability = std::ranges::find(
                                        analysis.capabilities,
                                        spv::CapabilitySignedZeroInfNanPreserve) != analysis.capabilities.end();
        if (!has_capability) {
            const auto position = last_of([](const Instruction& i) { return i.opcode == spv::OpCapability; });
            instructions.insert(position + 1, Instruction{
                                                  spv::OpCapability,
                                                  {static_cast<uint32_t>(spv::CapabilitySignedZeroInfNanPreserve)}});
        }
        const uint32_t version = module.header[1];
        const bool needs_extension =
            version < 0x00010400u &&
            std::ranges::find(analysis.extensions, "SPV_KHR_float_controls") == analysis.extensions.end();
        if (needs_extension) {
            auto position = last_of([](const Instruction& i) { return i.opcode == spv::OpExtension; });
            if (position == instructions.end()) {
                position = last_of([](const Instruction& i) { return i.opcode == spv::OpCapability; });
            }
            instructions.insert(position + 1, Instruction{spv::OpExtension, encode_string("SPV_KHR_float_controls")});
        }
        auto position = last_of([](const Instruction& i) {
            return i.opcode == spv::OpExecutionMode || i.opcode == spv::OpExecutionModeId;
        });
        if (position == instructions.end()) {
            position = last_of([](const Instruction& i) { return i.opcode == spv::OpEntryPoint; });
        }
        for (const uint32_t width : missing) {
            position = instructions.insert(
                position + 1,
                Instruction{spv::OpExecutionMode,
                            {analysis.entry_id,
                             static_cast<uint32_t>(spv::ExecutionModeSignedZeroInfNanPreserve), width}});
        }
    }

    void validate(const std::vector<uint32_t>& words) {
#ifdef LFS_SPIRV_FINALIZE_HAS_TOOLS
        spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
        std::string diagnostics;
        tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& position,
                                     const char* message) {
            diagnostics += "word " + std::to_string(position.index) + ": " + message + "\n";
        });
        if (!tools.Validate(words)) {
            fail("finalized module fails validation:\n" + diagnostics);
        }
#else
        (void)words;
#endif
    }

    std::string capability_name(const spv::Capability capability) {
        const std::string name = spv::CapabilityToString(capability);
        return name == "Unknown" ? "Capability" + std::to_string(static_cast<uint32_t>(capability)) : name;
    }

    std::string json_string_list(const std::vector<std::string>& items) {
        std::string text = "[";
        for (size_t index = 0; index < items.size(); ++index) {
            text += (index == 0 ? "\"" : ", \"") + items[index] + "\"";
        }
        return text + "]";
    }

    template <typename Range>
    std::string json_number_list(const Range& items) {
        std::string text = "[";
        bool first = true;
        for (const auto item : items) {
            text += (first ? "" : ", ") + std::to_string(item);
            first = false;
        }
        return text + "]";
    }

    void write_manifest(const char* path, const Module& module, const Analysis& analysis) {
        std::vector<std::string> capabilities;
        for (const spv::Capability capability : analysis.capabilities) {
            capabilities.push_back(capability_name(capability));
        }
        const uint32_t version = module.header[1];
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream << "{\n"
               << "  \"entry_point\": \"" << analysis.entry_name << "\",\n"
               << "  \"execution_model\": \"GLCompute\",\n"
               << "  \"local_size\": " << json_number_list(analysis.local_size) << ",\n"
               << "  \"push_constant_size\": " << analysis.push_constant_size << ",\n"
               << "  \"capabilities\": " << json_string_list(capabilities) << ",\n"
               << "  \"extensions\": " << json_string_list(analysis.extensions) << ",\n"
               << "  \"float_widths\": " << json_number_list(analysis.float_widths) << ",\n"
               << "  \"signed_zero_inf_nan_preserve\": " << json_number_list(analysis.preserve_widths) << ",\n"
               << "  \"spirv_version\": \"" << ((version >> 16) & 0xFFu) << "." << ((version >> 8) & 0xFFu)
               << "\"\n"
               << "}\n";
        if (!stream) {
            fail(std::string("cannot write ") + path);
        }
    }

    std::string identifier(const std::string_view name) {
        std::string text(name);
        for (char& character : text) {
            if (!std::isalnum(static_cast<unsigned char>(character))) {
                character = '_';
            }
        }
        return text;
    }

    // Writes one translation unit holding every module and its loader facts, so
    // the library carries its shaders instead of reading them from a build-tree
    // path at run time.
    void write_table(const char* path, const std::vector<std::pair<std::string, std::string>>& modules) {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream << "// Generated by lfs_tensor_spirv_finalize from the finalized SPIR-V modules; do not edit.\n"
               << "#include \"core/tensor/backend/vulkan/vk_shader_table.hpp\"\n\n"
               << "#include <algorithm>\n\n"
               << "namespace lfs::core::internal {\n    namespace {\n";
        std::vector<std::string> names;
        for (const auto& [name, file] : modules) {
            const std::vector<uint32_t> words = read_words(file.c_str());
            const Module module = parse(words);
            const Analysis analysis = analyze(module);
            for (const uint32_t width : analysis.float_widths) {
                if (!analysis.preserve_widths.contains(width)) {
                    fail(name + " is not finalized: SignedZeroInfNanPreserve " + std::to_string(width) + " missing");
                }
            }
            const std::string id = identifier(name);
            names.push_back(name);
            stream << "        const std::array<uint32_t, " << words.size() << "> kWords_" << id << "{";
            for (size_t index = 0; index < words.size(); ++index) {
                if (index % 8 == 0) {
                    stream << "\n           ";
                }
                char text[16];
                std::snprintf(text, sizeof(text), " 0x%08xu,", words[index]);
                stream << text;
            }
            stream << "\n        };\n";
            stream << "        constexpr std::array<std::string_view, " << analysis.capabilities.size()
                   << "> kCapabilities_" << id << "{";
            for (size_t index = 0; index < analysis.capabilities.size(); ++index) {
                stream << (index == 0 ? "\"" : ", \"") << capability_name(analysis.capabilities[index]) << "\"";
            }
            stream << "};\n";
            const auto widths = [&](const char* label, const std::set<uint32_t>& values) {
                stream << "        constexpr std::array<uint32_t, " << values.size() << "> k" << label << "_" << id
                       << "{";
                bool first = true;
                for (const uint32_t value : values) {
                    stream << (first ? "" : ", ") << value << "u";
                    first = false;
                }
                stream << "};\n";
            };
            widths("FloatWidths", analysis.float_widths);
            widths("Preserve", analysis.preserve_widths);
            stream << "        const EmbeddedShader kShader_" << id << "{\n"
                   << "            \"" << name << "\", kWords_" << id << ", \"" << analysis.entry_name << "\",\n"
                   << "            {" << analysis.local_size[0] << "u, " << analysis.local_size[1] << "u, "
                   << analysis.local_size[2] << "u}, " << analysis.push_constant_size << "u,\n"
                   << "            kCapabilities_" << id << ", kFloatWidths_" << id << ", kPreserve_" << id << "};\n\n";
        }
        stream << "        const std::array<EmbeddedShader, " << names.size() << "> kShaders{";
        for (size_t index = 0; index < names.size(); ++index) {
            stream << (index == 0 ? "" : ",") << "\n            kShader_" << identifier(names[index]);
        }
        stream << "};\n    } // namespace\n\n"
               << "    std::span<const EmbeddedShader> embedded_shaders() {\n"
               << "        return kShaders;\n    }\n\n"
               << "    const EmbeddedShader* find_embedded_shader(const std::string_view name) {\n"
               << "        const auto iterator = std::ranges::find(kShaders, name, &EmbeddedShader::name);\n"
               << "        return iterator == kShaders.end() ? nullptr : &*iterator;\n    }\n\n"
               << "} // namespace lfs::core::internal\n";
        if (!stream) {
            fail(std::string("cannot write ") + path);
        }
    }

    int run(const int argc, char** const argv) {
        if (argc >= 5 && std::string_view(argv[1]) == "embed") {
            if ((argc - 3) % 2 != 0) {
                std::fprintf(stderr, "usage: %s embed <table.cpp> <name> <finalized.spv> ...\n", argv[0]);
                return 2;
            }
            std::vector<std::pair<std::string, std::string>> modules;
            for (int index = 3; index + 1 < argc; index += 2) {
                modules.emplace_back(argv[index], argv[index + 1]);
            }
            write_table(argv[2], modules);
            return 0;
        }
        if (argc != 4) {
            std::fprintf(stderr, "usage: %s <input.spv> <output.spv> <manifest.json>\n", argv[0]);
            return 2;
        }
        Module module = parse(read_words(argv[1]));
        declare_float_controls(module, analyze(module));
        const std::vector<uint32_t> words = serialize(module);
        const Analysis finished = analyze(module);
        for (const uint32_t width : finished.float_widths) {
            if (!finished.preserve_widths.contains(width)) {
                fail("SignedZeroInfNanPreserve " + std::to_string(width) + " is still missing");
            }
        }
        validate(words);
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(words.data()),
                     static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
        if (!output) {
            fail(std::string("cannot write ") + argv[2]);
        }
        write_manifest(argv[3], module, finished);
        return 0;
    }

} // namespace

int main(const int argc, char** const argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        // LFS-CENSUS-OK(empty-catch): build-time host tool; the failure goes to
        // stderr and the non-zero exit fails the build.
        std::fprintf(stderr, "spirv_finalize: %s: %s\n", argc > 1 ? argv[1] : "", error.what());
        return 1;
    }
}
