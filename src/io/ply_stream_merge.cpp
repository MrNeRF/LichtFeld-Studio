/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/ply_stream_merge.hpp"
#include "core/native_fs.hpp"
#include "core/provenance.hpp"
#include "io/atomic_output.hpp"
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>

namespace lfs::io {
    namespace {
        namespace fs = std::filesystem;
        constexpr std::size_t MAX_HEADER_BYTES = 1024 * 1024;
        constexpr std::size_t COPY_BYTES = 8 * 1024 * 1024;

        struct PlyPart {
            fs::path path;
            std::vector<std::string> properties;
            std::uint64_t vertices = 0;
            std::uint64_t body_offset = 0;
            std::uint64_t body_bytes = 0;
        };

        std::size_t scalar_size(const std::string_view type) {
            if (type == "char" || type == "uchar" || type == "int8" || type == "uint8") return 1;
            if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") return 2;
            if (type == "int" || type == "uint" || type == "float" || type == "int32" ||
                type == "uint32" || type == "float32") return 4;
            if (type == "double" || type == "float64" || type == "int64" || type == "uint64") return 8;
            return 0;
        }

        bool read_header_line(std::ifstream& input, std::string& line, std::size_t& bytes) {
            line.clear();
            char ch = 0;
            while (input.get(ch)) {
                if (++bytes > MAX_HEADER_BYTES) return false;
                if (ch == '\n') return true;
                line.push_back(ch);
            }
            return false;
        }

        Result<PlyPart> inspect_part(const fs::path& path) {
            std::ifstream input(core::native_fs::native_path(path), std::ios::binary);
            if (!input) return make_error(ErrorCode::READ_FAILURE, "Cannot open tile PLY", path);
            PlyPart part;
            part.path = path;
            std::string line;
            std::size_t header_bytes = 0;
            bool saw_magic = false, saw_format = false, saw_vertex = false, saw_end = false;
            std::uint64_t stride = 0;
            while (read_header_line(input, line, header_bytes)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!saw_magic) {
                    if (line != "ply") return make_error(ErrorCode::INVALID_HEADER, "Missing PLY magic", path);
                    saw_magic = true;
                    continue;
                }
                if (line == "format binary_little_endian 1.0" && !saw_format && !saw_vertex) {
                    saw_format = true;
                } else if (line.starts_with("comment ") || line.starts_with("obj_info ")) {
                    // A fresh merged provenance stamp is written below.
                } else if (line.starts_with("element vertex ") && saw_format && !saw_vertex) {
                    const auto token = std::string_view(line).substr(15);
                    const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), part.vertices);
                    if (error != std::errc{} || end != token.data() + token.size())
                        return make_error(ErrorCode::INVALID_HEADER, "Invalid PLY vertex count", path);
                    saw_vertex = true;
                } else if (line.starts_with("property ") && saw_vertex) {
                    const auto rest = std::string_view(line).substr(9);
                    const auto separator = rest.find(' ');
                    if (separator == std::string_view::npos || separator + 1 == rest.size() ||
                        rest.find(' ', separator + 1) != std::string_view::npos)
                        return make_error(ErrorCode::UNSUPPORTED_FORMAT, "PLY requires scalar vertex properties", path);
                    const auto bytes = scalar_size(rest.substr(0, separator));
                    if (bytes == 0 || stride > std::numeric_limits<std::uint64_t>::max() - bytes)
                        return make_error(ErrorCode::UNSUPPORTED_FORMAT, "Unsupported PLY property type", path);
                    stride += bytes;
                    part.properties.push_back(line);
                } else if (line == "end_header" && saw_vertex && !part.properties.empty()) {
                    saw_end = true;
                    part.body_offset = header_bytes;
                    break;
                } else {
                    return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                                      "Tile PLY must contain one binary vertex element", path);
                }
            }
            if (!saw_end || stride == 0 || part.vertices > std::numeric_limits<std::uint64_t>::max() / stride)
                return make_error(ErrorCode::INVALID_HEADER, "Incomplete or overflowing PLY header", path);
            part.body_bytes = part.vertices * stride;
            std::error_code ec;
            const auto file_bytes = core::native_fs::file_size(path, ec);
            if (ec || file_bytes < part.body_offset || file_bytes - part.body_offset != part.body_bytes)
                return make_error(ErrorCode::CORRUPTED_DATA, "PLY body length differs from its header", path);
            return part;
        }
    } // namespace

    Result<void> merge_binary_ply_streaming(const std::vector<fs::path>& inputs,
                                             const fs::path& output,
                                             const std::function<bool()>& cancelled) {
        if (inputs.empty()) return make_error(ErrorCode::EMPTY_DATASET, "No tile PLYs to merge");
        std::vector<PlyPart> parts;
        parts.reserve(inputs.size());
        std::uint64_t total_vertices = 0;
        std::uint64_t total_body_bytes = 0;
        for (const auto& path : inputs) {
            if (cancelled && cancelled()) return make_error(ErrorCode::CANCELLED, "PLY merge cancelled");
            std::error_code ec;
            if (core::native_fs::equivalent(path, output, ec) && !ec)
                return make_error(ErrorCode::INVALID_DATASET, "Merged PLY cannot replace a tile input", output);
            auto inspected = inspect_part(path);
            if (!inspected) return std::unexpected(inspected.error());
            if (!parts.empty() && inspected->properties != parts.front().properties)
                return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                                  "Tile PLY property schemas differ; all tiles need the same SH degree and export settings", path);
            if (inspected->vertices > std::numeric_limits<std::uint64_t>::max() - total_vertices ||
                inspected->body_bytes > std::numeric_limits<std::uint64_t>::max() - total_body_bytes)
                return make_error(ErrorCode::RESOURCE_EXHAUSTED, "Merged PLY size overflows", output);
            total_vertices += inspected->vertices;
            total_body_bytes += inspected->body_bytes;
            if (!parts.empty()) inspected->properties.clear();
            parts.push_back(std::move(*inspected));
        }

        std::string header = "ply\nformat binary_little_endian 1.0\n";
        header += "comment lichtfeld_provenance " +
                  core::provenance_to_json(core::make_minimal_provenance_stamp()) + "\n";
        header += "element vertex " + std::to_string(total_vertices) + "\n";
        for (const auto& property : parts.front().properties) header += property + '\n';
        header += "end_header\n";

        if (total_body_bytes > std::numeric_limits<std::uint64_t>::max() - header.size())
            return make_error(ErrorCode::RESOURCE_EXHAUSTED, "Merged PLY size overflows", output);
        if (auto prepared = ensure_output_parent_directory(output); !prepared) return prepared;
        if (auto space = check_disk_space(output, total_body_bytes + header.size()); !space)
            return std::unexpected(space.error());
        ScopedAtomicOutputFile atomic(output);
        std::ofstream merged(core::native_fs::native_path(atomic.temp_path()), std::ios::binary | std::ios::trunc);
        if (!merged) return make_error(ErrorCode::WRITE_FAILURE, "Cannot create temporary merged PLY", output);
        merged.write(header.data(), static_cast<std::streamsize>(header.size()));
        if (!merged) return make_error(ErrorCode::WRITE_FAILURE, "Cannot write merged PLY header", output);
        std::vector<char> buffer(COPY_BYTES);
        for (const auto& part : parts) {
            if (cancelled && cancelled()) return make_error(ErrorCode::CANCELLED, "PLY merge cancelled");
            std::ifstream input(core::native_fs::native_path(part.path), std::ios::binary);
            if (!input || part.body_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
                return make_error(ErrorCode::READ_FAILURE, "Cannot reopen tile PLY", part.path);
            input.seekg(static_cast<std::streamoff>(part.body_offset));
            std::uint64_t remaining = part.body_bytes;
            while (remaining > 0) {
                if (cancelled && cancelled()) return make_error(ErrorCode::CANCELLED, "PLY merge cancelled");
                const auto bytes = static_cast<std::streamsize>(std::min<std::uint64_t>(remaining, buffer.size()));
                input.read(buffer.data(), bytes);
                if (input.gcount() != bytes)
                    return make_error(ErrorCode::CORRUPTED_DATA, "Tile PLY changed or became truncated during merge", part.path);
                merged.write(buffer.data(), bytes);
                if (!merged) return make_error(ErrorCode::WRITE_FAILURE, "Cannot write merged PLY body", output);
                remaining -= static_cast<std::uint64_t>(bytes);
            }
        }
        merged.flush();
        if (!merged) return make_error(ErrorCode::WRITE_FAILURE, "Cannot flush merged PLY", output);
        merged.close();
        if (!merged) return make_error(ErrorCode::WRITE_FAILURE, "Cannot close merged PLY", output);
        return atomic.commit();
    }
} // namespace lfs::io
