/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "shaders/slang_compiler.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "lfs_shader_abi.hpp"

#include <slang-com-ptr.h>
#include <slang-tag-version.h>
#include <slang.h>

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <list>
#include <optional>
#include <source_location>
#include <thread>
#include <unordered_set>
#include <utility>

namespace lfs::vis::shaders {

    // nlohmann adapter; must be in the namespace of the type.
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Binding, name, set, binding, count)

    namespace {
        using Slang::ComPtr;
        using Json = nlohmann::json;

        constexpr uint32_t kPackageVersion = 1;
        constexpr size_t kMaxParameters = 16;
        constexpr size_t kMaxModuleBytes = size_t{64} << 20;
        constexpr size_t kCacheEntries = 32;
        constexpr size_t kCacheBytes = size_t{64} << 20;
        // Effect i compiles as module lfs_effect<i> with the generated file
        // lfs_adapter<i>.slang; module lfs_effects dispatches to them.
        constexpr char kEffectModule[] = "lfs_effect";
        constexpr char kAdapterFile[] = "lfs_adapter";
        constexpr char kDispatchModule[] = "lfs_effects";

        Error compile_error(std::string detail, const ErrorCode code = ErrorCode::InvalidArgument,
                            const std::source_location location = std::source_location::current()) {
            return make_error(ErrorInit{
                .code = code,
                .domain = ErrorDomain::Rendering,
                .user_message = "Shader compilation failed.",
                .detail = std::move(detail),
                .detection = core::source_site_from_location(location),
            });
        }

        Error package_error(std::string detail, const ErrorCode code = ErrorCode::DataLoss,
                            const std::source_location location = std::source_location::current()) {
            return make_error(ErrorInit{
                .code = code,
                .domain = ErrorDomain::Rendering,
                .user_message = "Cannot load the shader package.",
                .detail = std::move(detail),
                .detection = core::source_site_from_location(location),
            });
        }

        // A contract failure, recorded as a diagnostic of `file` as well as returned.
        Error reject(CompileDiagnostics& diagnostics, std::string detail,
                     const ErrorCode code = ErrorCode::InvalidArgument, std::string file = {},
                     const std::source_location location = std::source_location::current()) {
            diagnostics.entries.push_back({DiagnosticSeverity::Error, "LFS", std::move(file), 0, 0, detail});
            return compile_error(std::move(detail), code, location);
        }

        std::string sha256(const void* data, const size_t size) {
            unsigned char hash[EVP_MAX_MD_SIZE];
            unsigned int length = 0;
            if (EVP_Digest(data, size, hash, &length, EVP_sha256(), nullptr) != 1)
                return {};
            constexpr char hex[] = "0123456789abcdef";
            std::string result;
            for (unsigned int i = 0; i < length; ++i) {
                result += hex[hash[i] >> 4];
                result += hex[hash[i] & 15];
            }
            return result;
        }

        bool verified(const ModuleIR& module) {
            return !module.bytes.empty() && sha256(module.bytes.data(), module.bytes.size()) == module.sha256;
        }

        bool same_guid(const SlangUUID& a, const SlangUUID& b) {
            return std::memcmp(&a, &b, sizeof(a)) == 0;
        }

        // Last path component as plain bytes: names in user source and
        // requests are arbitrary, and a path conversion may throw.
        std::string_view file_name(const std::string_view path) {
            return path.substr(path.find_last_of("/\\") + 1);
        }

        std::string effect_module(const size_t index) { return kEffectModule + std::to_string(index); }
        std::string adapter_file(const size_t index) { return kAdapterFile + std::to_string(index) + ".slang"; }

        // Serves only the generated adapters, by file name. Every other include
        // or import fails, so effects resolve nothing beyond the loaded package IR.
        class AdapterFileSystem final : public ISlangFileSystem {
        public:
            explicit AdapterFileSystem(std::vector<std::pair<std::string, std::string>> adapters)
                : adapters_(std::move(adapters)) {}

            SlangResult SLANG_MCALL queryInterface(const SlangUUID& id, void** out) noexcept override {
                *out = castAs(id);
                if (!*out)
                    return SLANG_E_NO_INTERFACE;
                addRef();
                return SLANG_OK;
            }
            uint32_t SLANG_MCALL addRef() noexcept override { return ++references_; }
            uint32_t SLANG_MCALL release() noexcept override {
                const uint32_t remaining = --references_;
                if (remaining == 0)
                    delete this;
                return remaining;
            }
            void* SLANG_MCALL castAs(const SlangUUID& id) noexcept override {
                const bool known = same_guid(id, ISlangUnknown::getTypeGuid()) ||
                                   same_guid(id, ISlangCastable::getTypeGuid()) ||
                                   same_guid(id, ISlangFileSystem::getTypeGuid());
                return known ? static_cast<ISlangFileSystem*>(this) : nullptr;
            }
            SlangResult SLANG_MCALL loadFile(const char* path, ISlangBlob** out) noexcept override {
                *out = nullptr;
                const auto name = path ? file_name(path) : std::string_view{};
                const auto found = std::ranges::find(adapters_, name, [](const auto& adapter) {
                    return std::string_view(adapter.first);
                });
                if (found == adapters_.end())
                    return SLANG_E_NOT_FOUND;
                *out = slang_createBlob(found->second.data(), found->second.size());
                return *out ? SLANG_OK : SLANG_E_OUT_OF_MEMORY;
            }

        private:
            std::vector<std::pair<std::string, std::string>> adapters_; // file name, text
            std::atomic<uint32_t> references_{1};
        };

        constexpr auto npos = std::string_view::npos;

        std::string_view trimmed(const std::string_view text) {
            const auto begin = text.find_first_not_of(" \t\r");
            return begin == npos ? std::string_view{}
                                 : text.substr(begin, text.find_last_not_of(" \t\r") - begin + 1);
        }

        bool to_number(const std::string_view text, uint32_t& value) {
            const auto* const end = text.data() + text.size();
            const auto [last, error] = std::from_chars(text.data(), end, value);
            return !text.empty() && error == std::errc{} && last == end;
        }

        // "error[E30015]: undefined identifier"; the code is optional.
        std::optional<Diagnostic> parse_header(std::string_view line) {
            static constexpr std::pair<std::string_view, DiagnosticSeverity> kSeverities[] = {
                {"fatal error", DiagnosticSeverity::Error},
                {"internal error", DiagnosticSeverity::Error},
                {"error", DiagnosticSeverity::Error},
                {"warning", DiagnosticSeverity::Warning},
                {"note", DiagnosticSeverity::Note}};
            for (const auto& [word, severity] : kSeverities) {
                if (!line.starts_with(word))
                    continue;
                line.remove_prefix(word.size());
                std::string_view code;
                if (line.starts_with('[')) {
                    const auto close = line.find(']');
                    if (close == npos)
                        return std::nullopt;
                    code = line.substr(1, close - 1);
                    line.remove_prefix(close + 1);
                }
                if (!line.starts_with(':'))
                    return std::nullopt;
                return Diagnostic{severity, std::string(code), {}, 0, 0, std::string(trimmed(line.substr(1)))};
            }
            return std::nullopt;
        }

        // " --> file:3:9"; the file name itself may contain ':'.
        bool parse_location(std::string_view line, Diagnostic& entry) {
            line = trimmed(line);
            if (!line.starts_with("-->"))
                return false;
            line = trimmed(line.substr(3));
            const auto second = line.rfind(':');
            const auto first = second == npos || second == 0 ? npos : line.rfind(':', second - 1);
            uint32_t row = 0;
            uint32_t column = 0;
            if (first == npos || first == 0 || !to_number(line.substr(first + 1, second - first - 1), row) ||
                !to_number(line.substr(second + 1), column))
                return false;
            entry.file = line.substr(0, first);
            entry.line = row;
            entry.column = column;
            return true;
        }

        // "  |        ^^^ undefined identifier 'x'." refines the header message.
        std::optional<std::string_view> parse_caret(std::string_view line) {
            line = trimmed(line);
            if (!line.starts_with('|'))
                return std::nullopt;
            line = trimmed(line.substr(1));
            const auto end = line.starts_with('^') ? line.find_first_not_of("^~") : npos;
            if (end == npos || (line[end] != ' ' && line[end] != '\t'))
                return std::nullopt;
            return trimmed(line.substr(end));
        }

        // Parses Slang's multi-line format line by line, without regular
        // expressions: echoed user source lines can be arbitrarily long.
        void collect(ComPtr<slang::IBlob>& blob, CompileDiagnostics& out) {
            if (!blob)
                return;
            const std::string_view raw(static_cast<const char*>(blob->getBufferPointer()), blob->getBufferSize());
            out.raw += raw;
            const size_t first = out.entries.size();
            for (size_t begin = 0; begin < raw.size();) {
                const auto end = std::min(raw.find('\n', begin), raw.size());
                const auto line = raw.substr(begin, end - begin);
                begin = end + 1;
                if (auto header = parse_header(line)) {
                    out.entries.push_back(std::move(*header));
                    continue;
                }
                if (out.entries.size() == first || parse_location(line, out.entries.back()))
                    continue;
                if (const auto caret = parse_caret(line))
                    out.entries.back().message = *caret;
            }
            if (out.entries.size() == first && raw.find_first_not_of(" \t\r\n") != npos)
                out.entries.push_back({DiagnosticSeverity::Note, {}, {}, 0, 0, std::string(raw)});
            blob.setNull();
        }

        bool named(slang::TypeReflection* type, const char* name) {
            return type && type->getName() && std::strcmp(type->getName(), name) == 0;
        }

        std::optional<float> attribute_number(slang::Attribute* attribute, const uint32_t index) {
            float value = 0.0f;
            if (SLANG_SUCCEEDED(attribute->getArgumentValueFloat(index, &value)))
                return value;
            int integer = 0;
            if (SLANG_SUCCEEDED(attribute->getArgumentValueInt(index, &integer)))
                return static_cast<float>(integer);
            return std::nullopt;
        }

        Result<std::vector<Parameter>> reflect_parameters(slang::ProgramLayout* layout,
                                                          slang::IGlobalSession* global) {
            std::vector<Parameter> parameters;
            auto* type = layout->findTypeByName("Params");
            if (!type)
                return parameters;
            if (type->getKind() != slang::TypeReflection::Kind::Struct)
                return compile_error("Params must be a struct.");
            if (type->getFieldCount() > kMaxParameters)
                return compile_error("Params has more than 16 fields.");
            for (unsigned i = 0; i < type->getFieldCount(); ++i) {
                auto* field = type->getFieldByIndex(i);
                auto* field_type = field->getType();
                const std::string name = field->getName() ? field->getName() : "";
                const auto kind = field_type->getKind();
                if ((kind != slang::TypeReflection::Kind::Scalar && kind != slang::TypeReflection::Kind::Vector) ||
                    field_type->getScalarType() != slang::TypeReflection::ScalarType::Float32)
                    return compile_error("Params field '" + name + "' must be float, float2, float3 or float4.");
                const size_t components = kind == slang::TypeReflection::Kind::Vector ? field_type->getElementCount() : 1;
                if (components < 1 || components > 4)
                    return compile_error("Params field '" + name + "' must have one to four components.");
                auto* attribute = field->findUserAttributeByName(global, "LfsParam");
                if (!attribute || attribute->getArgumentCount() != 3)
                    return compile_error("Params field '" + name + "' needs [LfsParam(minimum, maximum, initial)].");
                const auto minimum = attribute_number(attribute, 0);
                const auto maximum = attribute_number(attribute, 1);
                const auto initial = attribute_number(attribute, 2);
                if (!minimum || !maximum || !initial)
                    return compile_error("Params field '" + name + "': LfsParam arguments must be number literals.");
                if (!std::isfinite(*minimum) || !std::isfinite(*maximum) || !std::isfinite(*initial) ||
                    *minimum > *maximum || *initial < *minimum || *initial > *maximum)
                    return compile_error("Params field '" + name +
                                         "': LfsParam needs finite values with minimum <= initial <= maximum.");
                parameters.push_back({name, i, static_cast<uint32_t>(components), *minimum, *maximum, *initial});
            }
            return parameters;
        }

        // A second file of effect `index`'s module, so it reaches internal
        // declarations. It builds `Params` from the slots for the dispatcher.
        std::string make_adapter(const size_t index, const std::vector<Parameter>& parameters,
                                 const bool takes_params) {
            std::string text = "implementing " + effect_module(index) + ";\nimport lfs;\n\npublic LfsElement _" +
                               effect_module(index) + "(LfsElement e, LfsContext ctx) {\n";
            if (takes_params) {
                text += "    Params p;\n";
                for (const auto& p : parameters)
                    text += "    p." + p.name + " = ctx.param(" + std::to_string(p.slot) + ")." +
                            std::string("xyzw", p.components) + ";\n";
                text += "    return effect(e, ctx, p);\n";
            } else {
                text += "    return effect(e, ctx);\n";
            }
            return text + "}\n";
        }

        // Exports the type the host's `extern struct LfsBoundEffect` links to:
        // a static call of each effect by its index.
        std::string make_dispatcher(const size_t count) {
            std::string imports = "import lfs;\n";
            std::string cases;
            for (size_t i = 0; i < count; ++i) {
                imports += "import " + effect_module(i) + ";\n";
                cases += "        case " + std::to_string(i) + ": return _" + effect_module(i) + "(e, ctx);\n";
            }
            return imports + "\nstruct _LfsDispatch : _LfsEffects {\n"
                             "    LfsElement apply(uint index, LfsElement e, LfsContext ctx) {\n"
                             "        switch (index) {\n" +
                   cases +
                   "        default: return e;\n"
                   "        }\n"
                   "    }\n"
                   "};\n\n"
                   "export struct LfsBoundEffect : _LfsEffects = _LfsDispatch;\n";
        }

        Result<std::vector<std::byte>> read_file(const std::filesystem::path& path) {
            const auto name = path.filename().string();
            std::ifstream input(lfs::core::native_fs::native_path(path), std::ios::binary | std::ios::ate);
            if (!input)
                return package_error("Cannot open " + name + ".", ErrorCode::NotFound);
            const std::streamoff size = input.tellg();
            if (size <= 0 || static_cast<uint64_t>(size) > kMaxModuleBytes)
                return package_error("Invalid size of " + name + ".");
            std::vector<std::byte> bytes(static_cast<size_t>(size));
            input.seekg(0);
            if (!input.read(reinterpret_cast<char*>(bytes.data()), size))
                return package_error("Cannot read " + name + ".");
            return bytes;
        }

        bool is_identifier(const std::string_view name) {
            const auto letter = [](const char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; };
            return !name.empty() && letter(name[0]) &&
                   std::ranges::all_of(name, [&](const char c) { return letter(c) || (c >= '0' && c <= '9'); });
        }

        // API and host modules are stored as <name>.slang-module; a host
        // import variant may use another identifier for its file.
        Result<ModuleIR> load_module(const Json& entry, const std::filesystem::path& directory,
                                     const bool named_file = true) {
            ModuleIR module{entry.at("name").get<std::string>(), entry.at("sha256").get<std::string>(), {}};
            if (!is_identifier(module.name) || module.name.starts_with(kEffectModule))
                return package_error("Invalid IR module name '" + module.name + "'.");
            // Plain identifiers keep every module beside the manifest.
            const auto file = entry.at("file").get<std::string>();
            constexpr std::string_view kExtension = ".slang-module";
            if (!file.ends_with(kExtension) || !is_identifier(std::string_view(file).substr(0, file.size() - kExtension.size())) ||
                (named_file && file != module.name + std::string(kExtension)))
                return package_error("IR module '" + module.name + "' has an invalid file name.");
            auto bytes = read_file(directory / file);
            if (!bytes)
                return std::move(bytes).error();
            module.bytes = std::move(*bytes);
            if (!verified(module))
                return package_error("IR module '" + module.name + "' does not match its hash.");
            return module;
        }
    } // namespace

    struct SlangCompiler::Impl {
        struct Entry {
            CompileOutput output;
            CompileDiagnostics diagnostics;
        };

        // Pass 1 of the latest effects: the hosts of one composition link one
        // after another and reuse it.
        struct Reflection {
            std::string key;
            std::vector<std::vector<Parameter>> parameters;
            std::vector<std::pair<std::string, std::string>> adapters;
            CompileDiagnostics diagnostics;
        };

        std::thread::id owner = std::this_thread::get_id();
        ComPtr<slang::IGlobalSession> global;
        std::list<Entry> cache; // most recently used first
        size_t cache_bytes = 0;
        Reflection reflection;

        Result<CompileOutput> build(const CompileRequest& request, const HostStage& host, const std::string& effects_key,
                                    CompileDiagnostics& diagnostics);
    };

    SlangCompiler::SlangCompiler() : impl_(std::make_unique<Impl>()) {}
    SlangCompiler::~SlangCompiler() = default;

    std::string_view SlangCompiler::compiler_build() { return SLANG_TAG_VERSION; }

    std::string_view SlangCompiler::abi_hash() { return kAbiHash; }

    Result<std::shared_ptr<const ShaderPackage>> SlangCompiler::load_package(const std::filesystem::path& manifest) {
        try {
            std::ifstream input(lfs::core::native_fs::native_path(manifest), std::ios::binary);
            if (!input)
                return package_error("Cannot open the shader package manifest.", ErrorCode::NotFound);
            const auto json = Json::parse(input);
            auto package = std::make_shared<ShaderPackage>();
            json.at("version").get_to(package->version);
            if (package->version != kPackageVersion)
                return package_error("Unsupported shader package version.", ErrorCode::Unsupported);
            json.at("abi_hash").get_to(package->abi_hash);
            json.at("compiler_build").get_to(package->compiler_build);
            json.at("profile").get_to(package->profile);
            json.at("optimization").get_to(package->optimization);
            json.at("fast_math").get_to(package->fast_math);
            json.at("column_major").get_to(package->column_major);
            const auto directory = manifest.parent_path();
            std::unordered_set<std::string> names;
            for (const auto& entry : json.at("api")) {
                auto module = load_module(entry, directory);
                if (!module)
                    return std::move(module).error();
                if (!names.insert(module->name).second)
                    return package_error("Duplicate IR module '" + module->name + "'.");
                package->api.push_back(std::move(*module));
            }
            // Variants of one host import share its name, never an API or host name.
            std::vector<std::string> files;
            for (const auto& entry : json.at("modules")) {
                auto module = load_module(entry, directory, false);
                if (!module)
                    return std::move(module).error();
                auto file = entry.at("file").get<std::string>();
                if (names.contains(module->name) || std::ranges::find(files, file) != files.end())
                    return package_error("Duplicate IR module '" + module->name + "'.");
                files.push_back(std::move(file));
                package->modules.push_back(std::move(*module));
            }
            for (const auto& module : package->modules)
                names.insert(module.name);
            for (const auto& entry : json.at("hosts")) {
                auto module = load_module(entry.at("module"), directory);
                if (!module)
                    return std::move(module).error();
                if (!names.insert(module->name).second)
                    return package_error("Duplicate IR module '" + module->name + "'.");
                HostStage host;
                host.module = std::move(*module);
                for (const auto& file : entry.at("imports")) {
                    const auto found = std::ranges::find(files, file.get<std::string>());
                    if (found == files.end())
                        return package_error("Host '" + host.module.name + "' imports an unknown IR module.");
                    host.imports.push_back(static_cast<size_t>(found - files.begin()));
                }
                entry.at("stage").get_to(host.stage);
                entry.at("entry").get_to(host.entry);
                entry.at("defines").get_to(host.defines);
                entry.at("bindings").get_to(host.bindings);
                entry.at("push_constant_bytes").get_to(host.push_constant_bytes);
                entry.at("workgroup").get_to(host.workgroup);
                entry.at("subgroup_size").get_to(host.subgroup_size);
                if (host.stage != "compute" || host.entry.empty())
                    return package_error("Unsupported host stage in '" + host.module.name + "'.",
                                         ErrorCode::Unsupported);
                package->hosts.push_back(std::move(host));
            }
            if (!names.contains("lfs") || package->hosts.empty())
                return package_error("The shader package lacks the lfs API or a host.");
            return std::shared_ptr<const ShaderPackage>(std::move(package));
        } catch (const Json::exception& e) {
            return package_error(std::string("Invalid shader package manifest: ") + e.what());
        }
    }

    Result<CompileOutput> SlangCompiler::compile(const CompileRequest& request, CompileDiagnostics& diagnostics) {
        diagnostics = {};
        if (impl_->owner != std::this_thread::get_id())
            return reject(diagnostics, "The shader compiler is used outside its worker thread.", ErrorCode::FailedPrecondition);
        if (!request.package)
            return reject(diagnostics, "No shader package.", ErrorCode::FailedPrecondition);
        const auto& package = *request.package;
        const auto host = std::ranges::find(package.hosts, request.host,
                                            [](const HostStage& stage) { return stage.module.name; });
        if (host == package.hosts.end())
            return reject(diagnostics, "Unknown host stage '" + request.host + "'.");
        if (std::ranges::any_of(host->imports, [&](const size_t index) { return index >= package.modules.size(); }))
            return reject(diagnostics, "Host '" + request.host + "' imports an unknown IR module.",
                          ErrorCode::FailedPrecondition);
        if (!impl_->global && SLANG_FAILED(slang::createGlobalSession(impl_->global.writeRef())))
            return reject(diagnostics, "Cannot initialize the shader compiler.", ErrorCode::Unavailable);
        if (package.version != kPackageVersion || package.abi_hash != abi_hash() ||
            package.compiler_build != impl_->global->getBuildTagString())
            return reject(diagnostics, "The shader package was built for a different shader API or compiler.",
                          ErrorCode::FailedPrecondition);
        if (request.effects.empty())
            return reject(diagnostics, "No shader to compile.");
        std::unordered_set<std::string_view> names;
        for (const auto& effect : request.effects) {
            // A truncated multi-byte sequence in a trailing comment would
            // swallow the appended adapter include, so sources must be UTF-8.
            if (effect.source.find('\0') != std::string::npos || !core::detail::is_valid_utf8(effect.source))
                return reject(diagnostics, "Shader source must be UTF-8 text without NUL characters.");
            if (effect.name.empty() || !core::detail::is_valid_utf8(effect.name) ||
                effect.name.find_first_of(std::string("\r\n\0", 3)) != std::string::npos ||
                file_name(effect.name).starts_with(kAdapterFile) || file_name(effect.name).starts_with(kDispatchModule) ||
                !names.insert(effect.name).second)
                return reject(diagnostics, "Invalid or repeated shader file name.");
        }

        // Everything that changes the generated code, the effects in order.
        // Parameter values and time are runtime data, not compiler inputs.
        const auto module_key = [](const ModuleIR& module) { return Json::array({module.name, module.sha256}); };
        Json api = Json::array();
        for (const auto& module : package.api)
            api.push_back(module_key(module));
        Json imports = Json::array();
        for (const size_t index : host->imports)
            imports.push_back(module_key(package.modules[index]));
        Json effects = Json::array();
        for (const auto& effect : request.effects)
            effects.push_back({effect.name, effect.source});
        // What pass 1 reads, then the host as well; effect strings are validated UTF-8 above.
        const Json effects_key = {
            {"compiler", package.compiler_build},
            {"abi", package.abi_hash},
            {"options", {package.profile, package.optimization, package.fast_math, package.column_major}},
            {"api", api},
            {"effects", effects},
        };
        const Json key = {
            effects_key,
            {module_key(host->module), imports, host->stage, host->entry, host->defines, host->bindings, host->push_constant_bytes, host->workgroup, host->subgroup_size},
        };
        const auto serialized = key.dump();
        const auto cache_key = sha256(serialized.data(), serialized.size());
        if (cache_key.empty())
            return reject(diagnostics, "Cannot hash the shader request.", ErrorCode::Internal);

        auto& cache = impl_->cache;
        const auto hit = std::ranges::find(cache, cache_key, [](const Impl::Entry& e) { return e.output.cache_key; });
        if (hit != cache.end()) {
            cache.splice(cache.begin(), cache, hit);
            diagnostics = hit->diagnostics;
            auto output = hit->output;
            output.cache_hit = true;
            return output;
        }

        auto output = impl_->build(request, *host, effects_key.dump(), diagnostics);
        if (!output)
            return output;
        output->cache_key = cache_key;
        const size_t bytes = output->spirv.size() * sizeof(uint32_t);
        if (bytes > kCacheBytes)
            return output;
        while (!cache.empty() && (cache.size() >= kCacheEntries || impl_->cache_bytes + bytes > kCacheBytes)) {
            impl_->cache_bytes -= cache.back().output.spirv.size() * sizeof(uint32_t);
            cache.pop_back();
        }
        cache.push_front({*output, diagnostics});
        impl_->cache_bytes += bytes;
        return output;
    }

    Result<CompileOutput> SlangCompiler::Impl::build(const CompileRequest& request, const HostStage& host,
                                                     const std::string& effects_key, CompileDiagnostics& diagnostics) {
        const auto& package = *request.package;
        std::vector<const ModuleIR*> modules;
        for (const auto& module : package.api)
            modules.push_back(&module);
        for (const size_t index : host.imports)
            modules.push_back(&package.modules[index]);
        modules.push_back(&host.module);
        for (const auto* module : modules)
            if (!verified(*module))
                return reject(diagnostics, "IR module '" + module->name + "' does not match its hash.", ErrorCode::DataLoss);

        // Pinned to the legacy recipes: -O -fp-mode fast -line-directive-mode none, column-major.
        // Optimization runs spirv-opt from the slang-glslang backend, which the
        // compiler loads from its own directory; installs ship it beside it.
        slang::CompilerOptionEntry options[2] = {};
        options[0].name = slang::CompilerOptionName::Optimization;
        options[0].value.kind = slang::CompilerOptionValueKind::Int;
        options[0].value.intValue0 = static_cast<int32_t>(package.optimization);
        options[1].name = slang::CompilerOptionName::EmitSpirvDirectly;
        options[1].value.kind = slang::CompilerOptionValueKind::Int;
        options[1].value.intValue0 = 1;
        slang::TargetDesc target = {};
        target.format = SLANG_SPIRV;
        target.profile = global->findProfile(package.profile.c_str());
        target.floatingPointMode = package.fast_math ? SLANG_FLOATING_POINT_MODE_FAST : SLANG_FLOATING_POINT_MODE_PRECISE;
        target.lineDirectiveMode = SLANG_LINE_DIRECTIVE_MODE_NONE;
        target.compilerOptionEntries = options;
        target.compilerOptionEntryCount = static_cast<uint32_t>(std::size(options));
        if (target.profile == SLANG_PROFILE_UNKNOWN)
            return reject(diagnostics, "Unknown shader profile '" + package.profile + "'.", ErrorCode::Unsupported);

        // Each pass gets a fresh session, so no module of an earlier request
        // or pass can satisfy a lookup. `count` limits the loaded IR modules.
        ComPtr<slang::IBlob> log;
        const auto open_session = [&](std::vector<std::pair<std::string, std::string>> adapters, const size_t count,
                                      ComPtr<slang::ISession>& session) {
            ComPtr<ISlangFileSystem> files;
            files.attach(new AdapterFileSystem(std::move(adapters)));
            slang::SessionDesc desc = {};
            desc.targets = &target;
            desc.targetCount = 1;
            desc.defaultMatrixLayoutMode =
                package.column_major ? SLANG_MATRIX_LAYOUT_COLUMN_MAJOR : SLANG_MATRIX_LAYOUT_ROW_MAJOR;
            desc.fileSystem = files;
            slang::IModule* last = nullptr;
            if (SLANG_FAILED(global->createSession(desc, session.writeRef())))
                return last;
            for (size_t i = 0; i < count; ++i) {
                ComPtr<ISlangBlob> blob;
                blob.attach(slang_createBlob(modules[i]->bytes.data(), modules[i]->bytes.size()));
                const auto file = modules[i]->name + ".slang-module";
                last = session->loadModuleFromIRBlob(modules[i]->name.c_str(), file.c_str(), blob, log.writeRef());
                collect(log, diagnostics);
                if (!last)
                    break;
            }
            return last;
        };

        // Pass 1, per effect: its file with only the API importable, in a
        // session of its own. Its reflection decides which adapter to generate.
        // Every effect reflects, so each failing one reports.
        if (reflection.key != effects_key) {
            Reflection reflected{.key = effects_key};
            std::optional<Error> failure;
            for (size_t i = 0; i < request.effects.size(); ++i) {
                const auto& source = request.effects[i];
                const auto reflect = [&]() -> std::optional<Error> {
                    const auto fail = [&](std::string detail, const ErrorCode code = ErrorCode::InvalidArgument) {
                        return reject(diagnostics, std::move(detail), code, source.name);
                    };
                    ComPtr<slang::ISession> session;
                    if (!open_session({}, package.api.size(), session))
                        return reject(diagnostics, "Cannot load the shader API IR.", ErrorCode::DataLoss);
                    const auto name = effect_module(i);
                    auto* user = session->loadModuleFromSourceString(name.c_str(), source.name.c_str(),
                                                                     source.source.c_str(), log.writeRef());
                    collect(log, diagnostics);
                    if (!user)
                        return fail("The shader source does not compile.");
                    auto* declaration = user->getModuleReflection();
                    if (!declaration || !declaration->getName() || declaration->getName() != name)
                        return fail("Effect sources cannot declare a module name.");
                    auto* layout = user->getLayout(0, log.writeRef());
                    collect(log, diagnostics);
                    if (!layout)
                        return fail("Cannot reflect the shader source.", ErrorCode::Internal);
                    if (layout->getParameterCount() != 0)
                        return fail("Effects cannot declare resources or uniforms; use Params.");
                    auto parameters = reflect_parameters(layout, global);
                    if (!parameters)
                        return fail(std::string(parameters.error().detail()));
                    auto* effect = layout->findFunctionByName("effect");
                    if (!effect)
                        return fail("Missing LfsElement effect(LfsElement e, LfsContext ctx[, Params p]).");
                    const unsigned arguments = effect->isOverloaded() ? 0 : effect->getParameterCount();
                    if (arguments < 2 || arguments > 3 || effect->getGenericContainer() ||
                        !named(effect->getReturnType(), "LfsElement") ||
                        !named(effect->getParameterByIndex(0)->getType(), "LfsElement") ||
                        !named(effect->getParameterByIndex(1)->getType(), "LfsContext") ||
                        (arguments == 3 && !named(effect->getParameterByIndex(2)->getType(), "Params")))
                        return fail("Expected exactly one LfsElement effect(LfsElement e, LfsContext ctx[, Params p]).");
                    reflected.adapters.emplace_back(adapter_file(i), make_adapter(i, *parameters, arguments == 3));
                    reflected.parameters.push_back(std::move(*parameters));
                    return std::nullopt;
                };
                if (auto error = reflect(); error && !failure)
                    failure = std::move(error);
            }
            if (failure) {
                reflection = {};
                return std::move(*failure);
            }
            reflected.diagnostics = diagnostics;
            reflection = std::move(reflected);
        } else {
            diagnostics = reflection.diagnostics;
        }

        // Pass 2: each file plus its adapter as a second file of its module,
        // then the dispatcher. The include line is appended, so user lines
        // keep their numbers.
        ComPtr<slang::ISession> session;
        auto* host_module = open_session(reflection.adapters, modules.size(), session);
        if (!host_module)
            return reject(diagnostics, "Cannot load host IR '" + host.module.name + "'.", ErrorCode::DataLoss);
        for (size_t i = 0; i < request.effects.size(); ++i) {
            const auto& source = request.effects[i];
            const auto bound_source = source.source + "\n\n__include \"" + adapter_file(i) + "\";\n";
            if (!session->loadModuleFromSourceString(effect_module(i).c_str(), source.name.c_str(),
                                                     bound_source.c_str(), log.writeRef())) {
                collect(log, diagnostics);
                return reject(diagnostics, "Cannot bind the effect to the host.", ErrorCode::InvalidArgument,
                              source.name);
            }
        }
        const auto dispatcher_source = make_dispatcher(request.effects.size());
        const auto dispatcher_file = std::string(kDispatchModule) + ".slang";
        auto* dispatcher = session->loadModuleFromSourceString(kDispatchModule, dispatcher_file.c_str(),
                                                               dispatcher_source.c_str(), log.writeRef());
        if (!dispatcher) {
            collect(log, diagnostics);
            return reject(diagnostics, "Cannot bind the effects to the host.");
        }
        log.setNull(); // pass 1 already reported the user's warnings
        ComPtr<slang::IEntryPoint> entry;
        if (SLANG_FAILED(host_module->findEntryPointByName(host.entry.c_str(), entry.writeRef())))
            return reject(diagnostics, "Host '" + host.module.name + "' has no entry point '" + host.entry + "'.",
                          ErrorCode::FailedPrecondition);
        slang::IComponentType* parts[] = {host_module, dispatcher, entry};
        ComPtr<slang::IComponentType> composite;
        auto result = session->createCompositeComponentType(parts, std::size(parts), composite.writeRef(),
                                                            log.writeRef());
        collect(log, diagnostics);
        if (SLANG_FAILED(result) || !composite)
            return reject(diagnostics, "Cannot compose the effect with its host.");
        if (composite->getSpecializationParamCount() != 0)
            return reject(diagnostics, "Host '" + host.module.name + "' has specialization parameters.",
                          ErrorCode::FailedPrecondition);
        ComPtr<slang::IComponentType> linked;
        result = composite->link(linked.writeRef(), log.writeRef());
        collect(log, diagnostics);
        if (SLANG_FAILED(result) || !linked)
            return reject(diagnostics, "Cannot link the effect with its host.");

        // The linked program must expose exactly the host contract.
        auto* program = linked->getLayout(0, log.writeRef());
        collect(log, diagnostics);
        if (!program || program->getEntryPointCount() != 1)
            return reject(diagnostics, "Unexpected entry point layout.", ErrorCode::Internal);
        auto* entry_layout = program->getEntryPointByIndex(0);
        SlangUInt workgroup[3] = {};
        SlangUInt wave = 0;
        entry_layout->getComputeThreadGroupSize(3, workgroup);
        entry_layout->getComputeWaveSize(&wave);
        if (entry_layout->getStage() != SLANG_STAGE_COMPUTE || (wave != 0 && wave != host.subgroup_size) ||
            !std::equal(std::begin(workgroup), std::end(workgroup), host.workgroup.begin()))
            return reject(diagnostics, "Host stage, workgroup or subgroup size differs from the package contract.",
                          ErrorCode::FailedPrecondition);
        std::vector<Binding> bindings;
        // Entry-point uniforms become push constants, as do global ones.
        size_t push_constant_bytes = 0;
        for (unsigned i = 0; i < entry_layout->getParameterCount(); ++i)
            push_constant_bytes +=
                entry_layout->getParameterByIndex(i)->getTypeLayout()->getSize(slang::ParameterCategory::Uniform);
        for (unsigned i = 0; i < program->getParameterCount(); ++i) {
            auto* parameter = program->getParameterByIndex(i);
            auto* type = parameter->getTypeLayout();
            const std::string name = parameter->getName() ? parameter->getName() : "";
            const auto category = parameter->getCategory();
            if (category == slang::ParameterCategory::PushConstantBuffer) {
                push_constant_bytes += type->getElementTypeLayout()->getSize();
                continue;
            }
            const size_t count = type->getSize(slang::ParameterCategory::DescriptorTableSlot);
            if (category != slang::ParameterCategory::DescriptorTableSlot || count == 0 || count > UINT32_MAX)
                return reject(diagnostics, "Unsupported host parameter '" + name + "'.", ErrorCode::FailedPrecondition);
            bindings.push_back({name, parameter->getBindingSpace(), parameter->getBindingIndex(),
                                static_cast<uint32_t>(count)});
        }
        if (bindings != host.bindings || push_constant_bytes != host.push_constant_bytes)
            return reject(diagnostics, "Descriptors or push constants differ from the host contract.",
                          ErrorCode::FailedPrecondition);

        // Without the backend Slang still returns code, unoptimized, and reports
        // an error; an incomplete install must fail instead of differing silently.
        const auto is_error = [](const Diagnostic& entry) { return entry.severity == DiagnosticSeverity::Error; };
        const auto errors = std::ranges::count_if(diagnostics.entries, is_error);
        ComPtr<slang::IBlob> code;
        result = linked->getEntryPointCode(0, 0, code.writeRef(), log.writeRef());
        collect(log, diagnostics);
        if (SLANG_FAILED(result) || !code || std::ranges::count_if(diagnostics.entries, is_error) != errors)
            return reject(diagnostics, "Cannot generate SPIR-V.");
        const size_t size = code->getBufferSize();
        if (size < 5 * sizeof(uint32_t) || size % sizeof(uint32_t) != 0)
            return reject(diagnostics, "Generated SPIR-V has an invalid size.", ErrorCode::Internal);
        CompileOutput output;
        output.spirv.resize(size / sizeof(uint32_t));
        std::memcpy(output.spirv.data(), code->getBufferPointer(), size);
        if (output.spirv[0] != 0x07230203u)
            return reject(diagnostics, "Generated SPIR-V lacks the SPIR-V magic number.", ErrorCode::Internal);
        output.parameters = reflection.parameters;
        return output;
    }

} // namespace lfs::vis::shaders
