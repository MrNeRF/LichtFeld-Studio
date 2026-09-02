/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "usda_read.hpp"
#include "core/path_utils.hpp"
#include "half.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

#include <ascii-parser.hh>
#include <external/fast_float/include/fast_float/fast_float.h>
#include <prim-types.hh>
#include <stage.hh>
#include <stream-reader.hh>
#include <tinyusdz.hh>
#include <usda-reader.hh>
#include <xform.hh>

namespace lfs::io::usd_flat {

    namespace {

        constexpr const char* composition_error =
            "USD composition is unsupported: flat Gaussian interchange does not support references, payloads, subLayers, variants, inherits, specializes, or clips";

        struct FastAttribute {
            std::string path;
            std::string name;
            std::string type_name;
            std::vector<float> values;
            int components = 1;
        };

        struct ArraySpan {
            std::size_t begin = 0;
            std::size_t end = 0;
        };

        struct FastScan {
            std::vector<FastAttribute> attributes;
            std::vector<ArraySpan> spans;
        };

        bool is_word_character(const char character) {
            return std::isalnum(static_cast<unsigned char>(character)) || character == '_' || character == ':' ||
                   character == '!' || character == '.' || character == '-';
        }

        bool is_fast_attribute(const std::string_view name) {
            return name == "positions" || name == "positionsh" || name == "orientations" || name == "orientationsh" ||
                   name == "scales" || name == "scalesh" || name == "opacities" || name == "opacitiesh" ||
                   name == "radiance:sphericalHarmonicsCoefficients" ||
                   name == "radiance:sphericalHarmonicsCoefficientsh" || name == "extent";
        }

        bool read_quoted_name(const std::string_view text, std::size_t& position, std::string& name) {
            if (position >= text.size() || text[position] != '"') {
                return false;
            }
            if (position + 2 < text.size() && text.compare(position, 3, "\"\"\"") == 0) {
                position += 3;
                while (position + 2 < text.size() && text.compare(position, 3, "\"\"\"") != 0) {
                    ++position;
                }
                if (position + 2 >= text.size()) {
                    return false;
                }
                position += 3;
                name.clear();
                return true;
            }
            ++position;
            const auto begin = position;
            while (position < text.size() && text[position] != '"') {
                if (text[position] == '\\') {
                    return false;
                }
                ++position;
            }
            if (position == text.size()) {
                return false;
            }
            name.assign(text.substr(begin, position - begin));
            ++position;
            return true;
        }

        bool skip_space_and_comments(const std::string_view text, std::size_t& position, bool* saw_newline = nullptr) {
            bool newline = false;
            for (;;) {
                while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) {
                    newline = newline || text[position] == '\n';
                    ++position;
                }
                if (position < text.size() && text[position] == '#') {
                    while (position < text.size() && text[position] != '\n') {
                        ++position;
                    }
                    continue;
                }
                break;
            }
            if (saw_newline) {
                *saw_newline = newline;
            }
            return position <= text.size();
        }

        bool parse_fast_array(const std::string_view text,
                              const std::size_t open,
                              const bool half_type,
                              const bool double_type,
                              std::vector<float>& values,
                              std::size_t& close) {
            std::size_t position = open + 1;
            values.clear();
            for (;;) {
                skip_space_and_comments(text, position);
                if (position >= text.size()) {
                    return false;
                }
                if (text[position] == ']') {
                    close = position;
                    return true;
                }
                if (text[position] == ',' || text[position] == '(' || text[position] == ')') {
                    ++position;
                    continue;
                }
                if (text[position] == '"' || text[position] == '[') {
                    return false;
                }
                if (!is_word_character(text[position]) && text[position] != '+') {
                    return false;
                }
                if (double_type) {
                    double value = 0.0;
                    const auto parsed = fast_float::from_chars(text.data() + position, text.data() + text.size(), value);
                    if (parsed.ec != std::errc{} || parsed.ptr == text.data() + position) {
                        return false;
                    }
                    position = static_cast<std::size_t>(parsed.ptr - text.data());
                    values.push_back(static_cast<float>(value));
                } else {
                    float value = 0.0f;
                    const auto parsed = fast_float::from_chars(text.data() + position, text.data() + text.size(), value);
                    if (parsed.ec != std::errc{} || parsed.ptr == text.data() + position) {
                        return false;
                    }
                    position = static_cast<std::size_t>(parsed.ptr - text.data());
                    values.push_back(half_type ? half_to_float(float_to_half(value)) : value);
                }
                if (position < text.size() && !std::isspace(static_cast<unsigned char>(text[position])) &&
                    text[position] != ',' && text[position] != '(' && text[position] != ')' && text[position] != ']') {
                    return false;
                }
            }
        }

        bool fast_attribute_layout(const std::string_view name,
                                   const std::string_view base_type,
                                   int& components,
                                   bool& half_type,
                                   bool& double_type) {
            components = 1;
            half_type = false;
            double_type = false;
            if (name == "orientations" || name == "orientationsh") {
                if (base_type != "quatf" && base_type != "quath" && base_type != "quatd") {
                    return false;
                }
                components = 4;
                half_type = base_type == "quath";
                double_type = base_type == "quatd";
                return true;
            }
            if (name == "opacities" || name == "opacitiesh") {
                if (base_type != "float" && base_type != "half" && base_type != "double") {
                    return false;
                }
                half_type = base_type == "half";
                double_type = base_type == "double";
                return true;
            }
            if (base_type != "point3f" && base_type != "float3" && base_type != "double3" && base_type != "point3h" &&
                base_type != "half3" && base_type != "vector3h" && base_type != "normal3h" && base_type != "color3h") {
                return false;
            }
            components = 3;
            half_type = base_type == "point3h" || base_type == "half3" || base_type == "vector3h" || base_type == "normal3h" ||
                        base_type == "color3h";
            double_type = base_type == "double3";
            return true;
        }

        std::optional<FastScan> scan_flat_arrays(const std::string_view text) {
            FastScan result;
            std::vector<std::string> prim_stack;
            std::string last_word;
            std::size_t last_word_end = 0;
            std::size_t position = 0;
            while (position < text.size()) {
                bool saw_newline = false;
                skip_space_and_comments(text, position, &saw_newline);
                if (saw_newline) {
                    last_word.clear();
                }
                if (position >= text.size()) {
                    break;
                }
                if (text[position] == '"') {
                    std::string ignored;
                    if (!read_quoted_name(text, position, ignored)) {
                        return std::nullopt;
                    }
                    continue;
                }
                if (text[position] == '}') {
                    if (prim_stack.empty()) {
                        return std::nullopt;
                    }
                    prim_stack.pop_back();
                    ++position;
                    last_word.clear();
                    continue;
                }
                if (text[position] == '{') {
                    prim_stack.emplace_back();
                    ++position;
                    last_word.clear();
                    continue;
                }
                if (!is_word_character(text[position])) {
                    ++position;
                    continue;
                }
                const std::size_t word_begin = position;
                while (position < text.size() && is_word_character(text[position])) {
                    ++position;
                }
                const std::string word(text.substr(word_begin, position - word_begin));
                if (word == "def" || word == "over" || word == "class") {
                    std::size_t header = position;
                    std::string name;
                    while (header < text.size()) {
                        bool header_newline = false;
                        skip_space_and_comments(text, header, &header_newline);
                        if (header >= text.size() || text[header] == '}') {
                            return std::nullopt;
                        }
                        if (text[header] == '"') {
                            if (!read_quoted_name(text, header, name)) {
                                return std::nullopt;
                            }
                            continue;
                        }
                        if (text[header] == '{') {
                            std::string parent;
                            for (auto it = prim_stack.rbegin(); it != prim_stack.rend(); ++it) {
                                if (!it->empty()) {
                                    parent = *it;
                                    break;
                                }
                            }
                            const std::string path = parent.empty() ? "/" + name : parent + "/" + name;
                            prim_stack.push_back(path);
                            position = header + 1;
                            last_word.clear();
                            break;
                        }
                        ++header;
                    }
                    if (header >= text.size()) {
                        return std::nullopt;
                    }
                    continue;
                }
                if (!is_fast_attribute(word) || prim_stack.empty() || last_word.empty()) {
                    last_word = word;
                    last_word_end = position;
                    continue;
                }
                std::size_t type_end = last_word_end;
                skip_space_and_comments(text, type_end);
                if (type_end >= text.size() || text[type_end] != '[') {
                    last_word = word;
                    last_word_end = position;
                    continue;
                }
                ++type_end;
                skip_space_and_comments(text, type_end);
                if (type_end >= text.size() || text[type_end] != ']' || type_end >= word_begin) {
                    return std::nullopt;
                }
                std::size_t equals = position;
                skip_space_and_comments(text, equals);
                if (equals >= text.size() || text[equals] != '=') {
                    last_word = word;
                    last_word_end = position;
                    continue;
                }
                ++equals;
                skip_space_and_comments(text, equals);
                if (equals >= text.size() || text[equals] != '[') {
                    return std::nullopt;
                }
                int components = 1;
                bool half_type = false;
                bool double_type = false;
                if (!fast_attribute_layout(word, last_word, components, half_type, double_type)) {
                    return std::nullopt;
                }
                FastAttribute attribute;
                for (auto it = prim_stack.rbegin(); it != prim_stack.rend(); ++it) {
                    if (!it->empty()) {
                        attribute.path = *it;
                        break;
                    }
                }
                attribute.name = word;
                attribute.type_name = last_word + "[]";
                attribute.components = components;
                std::size_t array_close = equals;
                if (!parse_fast_array(text, equals, half_type, double_type, attribute.values, array_close) ||
                    attribute.values.size() % static_cast<std::size_t>(components) != 0) {
                    return std::nullopt;
                }
                result.spans.push_back({equals + 1, array_close});
                result.attributes.push_back(std::move(attribute));
                position = array_close + 1;
                last_word.clear();
            }
            return result;
        }

        bool has_composition_metadata(const tinyusdz::Prim& prim) {
            const auto& metas = prim.metas();
            return metas.references.has_value() || metas.payload.has_value() || metas.inherits.has_value() ||
                   metas.variantSets.has_value() || metas.variants.has_value() || metas.specializes.has_value() ||
                   metas.clips.has_value();
        }

        bool has_composition_metadata(const tinyusdz::Stage& stage) {
            if (!stage.metas().subLayers.empty()) {
                return true;
            }
            const auto contains = [](const auto& self, const tinyusdz::Prim& prim) -> bool {
                if (has_composition_metadata(prim)) {
                    return true;
                }
                for (const auto& child : prim.children()) {
                    if (self(self, child)) {
                        return true;
                    }
                }
                return false;
            };
            for (const auto& prim : stage.root_prims()) {
                if (contains(contains, prim)) {
                    return true;
                }
            }
            return false;
        }

        template <typename T>
        void append_vec3(const std::vector<T>& values, FlatAttribute& output) {
            output.components = 3;
            output.values.resize(values.size() * 3);
            for (std::size_t index = 0; index < values.size(); ++index) {
                for (std::size_t component = 0; component < 3; ++component) {
                    output.values[index * 3 + component] = static_cast<float>(values[index][component]);
                }
            }
            output.authored = true;
        }

        void append_vec3_half(const std::vector<tinyusdz::value::point3h>& values, FlatAttribute& output) {
            output.components = 3;
            output.values.resize(values.size() * 3);
            for (std::size_t index = 0; index < values.size(); ++index) {
                for (std::size_t component = 0; component < 3; ++component) {
                    output.values[index * 3 + component] = half_to_float(values[index][component].value);
                }
            }
            output.authored = true;
        }

        void append_quat(const std::vector<tinyusdz::value::quatf>& values, FlatAttribute& output) {
            output.components = 4;
            output.values.resize(values.size() * 4);
            for (std::size_t index = 0; index < values.size(); ++index) {
                output.values[index * 4 + 0] = values[index].real;
                output.values[index * 4 + 1] = values[index].imag[0];
                output.values[index * 4 + 2] = values[index].imag[1];
                output.values[index * 4 + 3] = values[index].imag[2];
            }
            output.authored = true;
        }

        void append_quat_half(const std::vector<tinyusdz::value::quath>& values, FlatAttribute& output) {
            output.components = 4;
            output.values.resize(values.size() * 4);
            for (std::size_t index = 0; index < values.size(); ++index) {
                output.values[index * 4 + 0] = half_to_float(values[index].real.value);
                output.values[index * 4 + 1] = half_to_float(values[index].imag[0].value);
                output.values[index * 4 + 2] = half_to_float(values[index].imag[1].value);
                output.values[index * 4 + 3] = half_to_float(values[index].imag[2].value);
            }
            output.authored = true;
        }

        void append_scalar(const std::vector<float>& values, FlatAttribute& output) {
            output.components = 1;
            output.values.assign(values.begin(), values.end());
            output.authored = true;
        }

        void append_scalar_half(const std::vector<tinyusdz::value::half>& values, FlatAttribute& output) {
            output.components = 1;
            output.values.resize(values.size());
            for (std::size_t index = 0; index < values.size(); ++index) {
                output.values[index] = half_to_float(values[index].value);
            }
            output.authored = true;
        }

        void extract_attribute(const std::string& name, const tinyusdz::Property& property, FlatPrim& prim) {
            if (!property.is_attribute() || !property.get_attribute().has_value()) {
                return;
            }
            const auto& attribute = property.get_attribute();
            const auto& raw = attribute.get_var().value_raw();
            FlatAttribute output;
            output.type_name = property.value_type_name();

            if (name == "positions" || name == "positionsh" || name == "scales" || name == "scalesh" ||
                name == "radiance:sphericalHarmonicsCoefficients" || name == "radiance:sphericalHarmonicsCoefficientsh") {
                if (const auto* values = raw.as<std::vector<tinyusdz::value::point3f>>()) {
                    append_vec3(*values, output);
                } else if (const auto* values = raw.as<std::vector<tinyusdz::value::float3>>()) {
                    append_vec3(*values, output);
                } else if (const auto* values = raw.as<std::vector<tinyusdz::value::point3h>>()) {
                    append_vec3_half(*values, output);
                } else if (const auto* values = raw.as<std::vector<tinyusdz::value::half3>>()) {
                    output.components = 3;
                    output.values.resize(values->size() * 3);
                    for (std::size_t index = 0; index < values->size(); ++index) {
                        for (std::size_t component = 0; component < 3; ++component) {
                            output.values[index * 3 + component] = half_to_float((*values)[index][component].value);
                        }
                    }
                    output.authored = true;
                }
            } else if (name == "orientations" || name == "orientationsh") {
                if (const auto* values = raw.as<std::vector<tinyusdz::value::quatf>>()) {
                    append_quat(*values, output);
                } else if (const auto* values = raw.as<std::vector<tinyusdz::value::quath>>()) {
                    append_quat_half(*values, output);
                }
            } else if (name == "opacities" || name == "opacitiesh") {
                if (const auto* values = raw.as<std::vector<float>>()) {
                    append_scalar(*values, output);
                } else if (const auto* values = raw.as<std::vector<tinyusdz::value::half>>()) {
                    append_scalar_half(*values, output);
                }
            } else if (name == "radiance:sphericalHarmonicsDegree") {
                if (const auto* value = raw.as<std::int32_t>()) {
                    output.values = {static_cast<float>(*value)};
                    output.authored = true;
                } else if (const auto* value = raw.as<int64_t>()) {
                    output.values = {static_cast<float>(*value)};
                    output.authored = true;
                }
            } else if (name == "extent") {
                if (const auto* values = raw.as<tinyusdz::Extent>()) {
                    output.components = 3;
                    output.values = {values->lower[0], values->lower[1], values->lower[2],
                                     values->upper[0], values->upper[1], values->upper[2]};
                    output.authored = true;
                } else if (const auto* values = raw.as<std::vector<tinyusdz::value::float3>>()) {
                    append_vec3(*values, output);
                } else if (const auto* values = raw.as<std::vector<tinyusdz::value::point3f>>()) {
                    append_vec3(*values, output);
                }
            }
            if (output.authored) {
                prim.attributes[name] = std::move(output);
            }
        }

        std::string prim_path(const tinyusdz::Prim& prim, const std::string& parent) {
            if (prim.absolute_path().is_valid()) {
                return prim.absolute_path().full_path_name();
            }
            return parent + "/" + prim.element_name();
        }

        void extract_prim(const tinyusdz::Prim& source, const std::string& parent, FlatStage& output) {
            FlatPrim prim;
            prim.path = prim_path(source, parent);
            prim.type_name = source.prim_type_name();
            prim.local_transform = identity_matrix();

            if (const auto* model = source.data().as<tinyusdz::Model>()) {
                for (const auto& property : model->props) {
                    extract_attribute(property.first, property.second, prim);
                }
            } else if (const auto* gprim = source.data().as<tinyusdz::GPrim>()) {
                for (const auto& property : gprim->props) {
                    extract_attribute(property.first, property.second, prim);
                }
            }

            const tinyusdz::Xformable* xform = nullptr;
            if (tinyusdz::CastToXformable(source, &xform) && xform) {
                bool reset = false;
                const auto matrix = tinyusdz::GetLocalTransform(source, &reset);
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        prim.local_transform[static_cast<std::size_t>(row * 4 + col)] = matrix.m[row][col];
                    }
                }
                prim.reset_xform_stack = reset;
            }
            output.prims.push_back(std::move(prim));
            const std::string path = output.prims.back().path;
            for (const auto& child : source.children()) {
                extract_prim(child, path, output);
            }
        }

    } // namespace

    lfs::Result<FlatStage> read_usda_text(std::string& text) {
        const auto fast_scan = scan_flat_arrays(text);
        const bool use_fast_path = fast_scan.has_value() && !fast_scan->attributes.empty();
        if (use_fast_path) {
            for (const auto& span : fast_scan->spans) {
                std::fill(text.begin() + static_cast<std::ptrdiff_t>(span.begin),
                          text.begin() + static_cast<std::ptrdiff_t>(span.end),
                          ' ');
            }
        }
        tinyusdz::StreamReader stream(reinterpret_cast<const std::uint8_t*>(text.data()), text.size(), false);
        tinyusdz::usda::USDAReader reader(&stream);
        tinyusdz::usda::USDAReaderConfig config;
        config.allow_unknown_prims = true;
        config.allow_unknown_apiSchema = true;
        reader.set_reader_config(config);
        if (!reader.read(static_cast<std::uint32_t>(tinyusdz::LoadState::Toplevel), false) || !reader.reconstruct_stage()) {
            const auto error = reader.get_error();
            return make_flat_error(error.empty() ? "Failed to parse USDA file" : error);
        }

        if (has_composition_metadata(reader.get_stage())) {
            return make_flat_error(composition_error);
        }

        FlatStage output;
        const auto& metas = reader.get_stage().metas();
        output.default_prim = metas.defaultPrim.str().empty() ? std::string{} : "/" + metas.defaultPrim.str();
        output.meters_per_unit = metas.metersPerUnit.get_value();
        output.up_axis = metas.upAxis.get_value() == tinyusdz::Axis::Y ? "Y" : metas.upAxis.get_value() == tinyusdz::Axis::Z ? "Z"
                                                                                                                             : "X";
        for (const auto& item : metas.customLayerData) {
            if (const auto value = item.second.get_value<std::string>()) {
                output.custom_layer_data[item.first] = *value;
            } else if (const auto value = item.second.get_value<tinyusdz::value::StringData>()) {
                output.custom_layer_data[item.first] = value->value;
            }
        }
        for (const auto& prim : reader.get_stage().root_prims()) {
            extract_prim(prim, "", output);
        }
        if (use_fast_path) {
            for (auto& attribute : fast_scan->attributes) {
                const auto prim = std::find_if(output.prims.begin(), output.prims.end(), [&](const FlatPrim& candidate) {
                    return candidate.path == attribute.path;
                });
                if (prim == output.prims.end()) {
                    return make_flat_error("USDA fast path could not resolve attribute prim " + attribute.path);
                }
                prim->attributes[attribute.name] = FlatAttribute{std::move(attribute.type_name),
                                                                 std::move(attribute.values),
                                                                 attribute.components,
                                                                 true};
            }
        }
        return output;
    }

    lfs::Result<FlatStage> read_usda(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return make_flat_error("Failed to open USD file: " + lfs::core::path_to_utf8(path));
        }
        std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        return read_usda_text(text);
    }

    lfs::Result<FlatStage> read_usda_bytes(const std::uint8_t* data, const std::size_t size) {
        if (!data && size != 0) {
            return make_flat_error("Invalid USDA data");
        }
        std::string text(reinterpret_cast<const char*>(data), size);
        return read_usda_text(text);
    }

} // namespace lfs::io::usd_flat
