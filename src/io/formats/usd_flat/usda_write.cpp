/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "usda_write.hpp"
#include "core/path_utils.hpp"
#include <charconv>
#include <cmath>
#include <fstream>
#include <functional>
#include <iterator>
#include <system_error>

namespace {

    void append_float(std::string& output, const float value) {
        char buffer[64];
        const auto result = std::to_chars(std::begin(buffer), std::end(buffer), value);
        if (result.ec == std::errc{}) {
            output.append(buffer, result.ptr);
        } else {
            output += "0";
        }
    }

    void append_double(std::string& output, const double value) {
        char buffer[64];
        const auto result = std::to_chars(std::begin(buffer), std::end(buffer), value);
        if (result.ec == std::errc{}) {
            output.append(buffer, result.ptr);
        } else {
            output += "0";
        }
    }

    void append_escaped(std::string& output, const std::string& value) {
        output.push_back('"');
        for (const char character : value) {
            if (character == '\\' || character == '"') {
                output.push_back('\\');
            }
            output.push_back(character);
        }
        output.push_back('"');
    }

    bool is_usda_identifier(const std::string& value) {
        if (value.empty()) {
            return false;
        }
        const auto is_alpha = [](const unsigned char character) {
            return (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z');
        };
        const auto is_alphanumeric = [&is_alpha](const unsigned char character) {
            return is_alpha(character) || (character >= '0' && character <= '9');
        };
        if (value.front() != '_' && !is_alpha(static_cast<unsigned char>(value.front()))) {
            return false;
        }
        for (std::size_t index = 1; index < value.size(); ++index) {
            const auto character = static_cast<unsigned char>(value[index]);
            if (character != '_' && !is_alphanumeric(character)) {
                return false;
            }
        }
        return true;
    }

    void append_attribute(std::string& output, const std::string& indent, const std::string& name,
                          const lfs::io::usd_flat::FlatAttribute& attribute) {
        output += indent;
        output += attribute.type_name;
        output += " ";
        output += name;
        output += " = [";
        const std::size_t count = attribute.components == 0 ? 0 : attribute.values.size() / static_cast<std::size_t>(attribute.components);
        for (std::size_t index = 0; index < count; ++index) {
            if (index != 0) {
                output += ", ";
            }
            if (attribute.components == 1) {
                append_float(output, attribute.values[index]);
            } else {
                output.push_back('(');
                for (int component = 0; component < attribute.components; ++component) {
                    if (component != 0) {
                        output += ", ";
                    }
                    append_float(output, attribute.values[index * static_cast<std::size_t>(attribute.components) + static_cast<std::size_t>(component)]);
                }
                output.push_back(')');
            }
        }
        output += "]\n";
    }

    bool is_identity(const std::array<double, 16>& matrix) {
        const auto identity = lfs::io::usd_flat::identity_matrix();
        for (std::size_t index = 0; index < matrix.size(); ++index) {
            if (std::abs(matrix[index] - identity[index]) > 1e-12) {
                return false;
            }
        }
        return true;
    }

} // namespace

namespace lfs::io::usd_flat {

    lfs::Status write_usda(const FlatStage& stage, const std::filesystem::path& path) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            return lfs::Status::failure(make_flat_error("Failed to open USD file for writing: " + lfs::core::path_to_utf8(path)));
        }

        std::string text;
        text.reserve(4096);
        text += "#usda 1.0\n\n(\n";
        if (!stage.default_prim.empty()) {
            text += "    defaultPrim = ";
            const auto slash = stage.default_prim.find_last_of('/');
            append_escaped(text, slash == std::string::npos ? stage.default_prim : stage.default_prim.substr(slash + 1));
            text += "\n";
        }
        text += "    upAxis = \"" + stage.up_axis + "\"\n";
        text += "    metersPerUnit = ";
        append_double(text, stage.meters_per_unit);
        text += "\n";
        if (!stage.custom_layer_data.empty()) {
            text += "    customLayerData = {\n";
            for (const auto& item : stage.custom_layer_data) {
                text += "        string ";
                if (is_usda_identifier(item.first)) {
                    text += item.first;
                } else {
                    append_escaped(text, item.first);
                }
                text += " = ";
                append_escaped(text, item.second);
                text += "\n";
            }
            text += "    }\n";
        }
        text += ")\n\n";

        const auto append_prim = [&](const auto& self, const lfs::io::usd_flat::FlatPrim& prim, const std::string& indent) -> void {
            const auto slash = prim.path.find_last_of('/');
            const std::string name = slash == std::string::npos ? prim.path : prim.path.substr(slash + 1);
            const std::string child_indent = indent + "    ";
            text += indent;
            text += "def ";
            text += prim.type_name.empty() ? "ParticleField" : prim.type_name;
            text += " \"" + name + "\" {\n";
            if (!is_identity(prim.local_transform)) {
                text += child_indent + "matrix4d xformOp:transform = (";
                for (int row = 0; row < 4; ++row) {
                    if (row != 0) {
                        text += ", ";
                    }
                    text.push_back('(');
                    for (int column = 0; column < 4; ++column) {
                        if (column != 0) {
                            text += ", ";
                        }
                        append_double(text, prim.local_transform[static_cast<std::size_t>(row * 4 + column)]);
                    }
                    text.push_back(')');
                }
                text += ")\n" + child_indent + "uniform token[] xformOpOrder = [\"xformOp:transform\"]\n";
            }
            for (const auto& attribute : prim.attributes) {
                if (attribute.second.authored) {
                    if (attribute.first == "radiance:sphericalHarmonicsDegree") {
                        text += child_indent + attribute.second.type_name + " " + attribute.first + " = ";
                        append_float(text, attribute.second.values.front());
                        text += "\n";
                    } else {
                        append_attribute(text, child_indent, attribute.first, attribute.second);
                    }
                }
            }
            for (const auto& child : stage.prims) {
                const auto child_slash = child.path.find_last_of('/');
                const auto child_parent = child_slash == 0 ? std::string{"/"} : child.path.substr(0, child_slash);
                if (child_parent == prim.path) {
                    self(self, child, "\n" + child_indent);
                }
            }
            text += indent + "}\n\n";
        };
        for (const auto& prim : stage.prims) {
            const auto slash = prim.path.find_last_of('/');
            const auto parent = slash == 0 ? std::string{"/"} : prim.path.substr(0, slash);
            if (parent == "/") {
                append_prim(append_prim, prim, "");
            }
        }
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!output) {
            return lfs::Status::failure(make_flat_error("Failed to write USD file: " + lfs::core::path_to_utf8(path)));
        }
        return {};
    }

} // namespace lfs::io::usd_flat
