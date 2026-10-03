/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// 3D Tiles 1.0/1.1 tilesets with KHR_gaussian_splatting (SPZ) GLB content.
// Implements the OGC 3D Tiles specification (tile tree, transforms, bounding
// volumes, refinement); tile content is decoded by load_spz.

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "formats/spz.hpp"
#include "io/splat_tile_source.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <format>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <nlohmann/json.hpp>
#include <tbb/parallel_for.h>

namespace lfs::io {

    namespace {
        using nlohmann::json;
        namespace fs = std::filesystem;

        constexpr std::size_t kMaxTiles = 1'000'000;
        constexpr int kMaxExternalDepth = 8;
        constexpr std::uint32_t kGlbMagic = 0x46546C67;     // "glTF"
        constexpr std::uint32_t kGlbChunkJson = 0x4E4F534A; // "JSON"
        constexpr std::uint32_t kMaxGlbJsonBytes = 64u << 20;
        constexpr std::uintmax_t kMaxTilesetJsonBytes = 512u << 20;

        struct ParsedTile {
            SplatTile tile;
            fs::path content;
            std::vector<ParsedTile> children;
        };

        json read_json(const fs::path& path) {
            std::ifstream in;
            if (!core::open_file_for_read(path, std::ios::binary, in))
                throw std::runtime_error(std::format("Cannot open '{}'", core::path_to_utf8(path)));
            return json::parse(in);
        }

        glm::dmat4 read_matrix(const json& tile) {
            if (!tile.contains("transform"))
                return glm::dmat4(1.0);
            const auto m = tile.at("transform").get<std::vector<double>>();
            if (m.size() != 16)
                throw std::runtime_error("tile transform must have 16 values");
            return glm::make_mat4(m.data()); // column-major, like glTF
        }

        double uniform_scale(const glm::dmat4& m) {
            return std::cbrt(std::abs(glm::determinant(glm::dmat3(m))));
        }

        glm::dvec3 wgs84_to_ecef(const double lon, const double lat, const double height) {
            constexpr double a = 6378137.0;
            constexpr double e2 = 6.69437999014e-3;
            const double n = a / std::sqrt(1.0 - e2 * std::sin(lat) * std::sin(lat));
            return {(n + height) * std::cos(lat) * std::cos(lon),
                    (n + height) * std::cos(lat) * std::sin(lon),
                    (n * (1.0 - e2) + height) * std::sin(lat)};
        }

        // Bounding volume -> oriented box in the source-local frame. `to_local` maps the
        // tile frame; regions are always EPSG:4979 and map through `ecef_to_local`.
        void read_bounds(const json& volume, const glm::dmat4& to_local,
                         const glm::dmat4& ecef_to_local, SplatTile& tile) {
            if (volume.contains("box")) {
                const auto b = volume.at("box").get<std::vector<double>>();
                if (b.size() != 12)
                    throw std::runtime_error("bounding box must have 12 values");
                tile.center = glm::vec3(to_local * glm::dvec4(b[0], b[1], b[2], 1.0));
                const glm::dmat3 axes(b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11]);
                tile.half_axes = glm::mat3(glm::dmat3(to_local) * axes);
            } else if (volume.contains("sphere")) {
                const auto s = volume.at("sphere").get<std::vector<double>>();
                if (s.size() != 4)
                    throw std::runtime_error("bounding sphere must have 4 values");
                tile.center = glm::vec3(to_local * glm::dvec4(s[0], s[1], s[2], 1.0));
                tile.half_axes = glm::mat3(static_cast<float>(s[3] * uniform_scale(to_local)));
            } else if (volume.contains("region")) {
                const auto r = volume.at("region").get<std::vector<double>>();
                if (r.size() != 6)
                    throw std::runtime_error("bounding region must have 6 values");
                glm::dvec3 lo(std::numeric_limits<double>::max()), hi(std::numeric_limits<double>::lowest());
                for (int i = 0; i <= 2; ++i)
                    for (int j = 0; j <= 2; ++j)
                        for (int k = 0; k <= 1; ++k) {
                            const auto p = glm::dvec3(ecef_to_local * glm::dvec4(wgs84_to_ecef(
                                                                                     r[0] + (r[2] - r[0]) * i / 2.0,
                                                                                     r[1] + (r[3] - r[1]) * j / 2.0,
                                                                                     k ? r[5] : r[4]),
                                                                                 1.0));
                            lo = glm::min(lo, p);
                            hi = glm::max(hi, p);
                        }
                tile.center = glm::vec3((lo + hi) * 0.5);
                const glm::vec3 half((hi - lo) * 0.5);
                tile.half_axes = glm::mat3(half.x, 0, 0, 0, half.y, 0, 0, 0, half.z);
            } else {
                throw std::runtime_error("tile has no supported bounding volume");
            }
        }

        std::string content_uri(const json& tile) {
            if (tile.contains("content"))
                return tile.at("content").at("uri").get<std::string>();
            if (tile.contains("contents") && !tile.at("contents").empty()) {
                if (tile.at("contents").size() > 1)
                    LOG_WARN("3D Tiles: only the first of {} tile contents is used", tile.at("contents").size());
                return tile.at("contents").at(0).at("uri").get<std::string>();
            }
            return {};
        }

        struct ParseContext {
            glm::dmat4 ecef_to_local{1.0};
            std::size_t tiles = 0;
        };

        ParsedTile parse_tile(const json& node, const fs::path& base, const glm::dmat4& parent_world,
                              const bool parent_additive, ParseContext& context, const int depth) {
            if (++context.tiles > kMaxTiles)
                throw std::runtime_error(std::format("tileset exceeds {} tiles", kMaxTiles));
            ParsedTile parsed;
            const glm::dmat4 world = parent_world * read_matrix(node);
            const glm::dmat4 to_local = context.ecef_to_local * world;
            auto& tile = parsed.tile;
            tile.transform = glm::mat4(to_local);
            tile.geometric_error = static_cast<float>(node.at("geometricError").get<double>() *
                                                      uniform_scale(context.ecef_to_local));
            tile.additive = node.contains("refine") ? node.at("refine").get<std::string>() == "ADD" : parent_additive;
            read_bounds(node.at("boundingVolume"), to_local, context.ecef_to_local, tile);

            if (const auto uri = content_uri(node); !uri.empty()) {
                const auto path = (base / core::utf8_to_path(uri)).lexically_normal();
                if (path.extension() == ".json") {
                    // External tileset: its root becomes this tile's only child.
                    if (depth >= kMaxExternalDepth)
                        throw std::runtime_error("external tilesets nest too deeply");
                    const auto external = read_json(path);
                    parsed.children.push_back(parse_tile(external.at("root"), path.parent_path(), world,
                                                         tile.additive, context, depth + 1));
                } else {
                    parsed.content = path;
                }
            }
            if (node.contains("children"))
                for (const auto& child : node.at("children"))
                    parsed.children.push_back(parse_tile(child, base, world, tile.additive, context, depth));
            return parsed;
        }

        // POSITION count and SH degree from the GLB JSON chunk, without reading the payload.
        void probe_glb(const fs::path& path, SplatTile& tile) {
            std::ifstream in;
            std::uint32_t header[5] = {};
            if (!core::open_file_for_read(path, std::ios::binary, in) ||
                !in.read(reinterpret_cast<char*>(header), sizeof(header)) ||
                header[0] != kGlbMagic || header[4] != kGlbChunkJson || header[3] > kMaxGlbJsonBytes)
                throw std::runtime_error(std::format("'{}' is not a glTF binary", core::path_to_utf8(path)));
            std::string text(header[3], '\0');
            if (!in.read(text.data(), static_cast<std::streamsize>(text.size())))
                throw std::runtime_error(std::format("'{}' has a truncated JSON chunk", core::path_to_utf8(path)));
            const auto doc = json::parse(text);
            for (const auto& mesh : doc.value("meshes", json::array()))
                for (const auto& primitive : mesh.value("primitives", json::array())) {
                    const auto& attributes = primitive.value("attributes", json::object());
                    if (!attributes.contains("POSITION"))
                        continue;
                    tile.splat_count = doc.at("accessors").at(attributes.at("POSITION").get<std::size_t>()).at("count").get<std::uint64_t>();
                    for (int degree = 3; degree > 0; --degree)
                        if (attributes.contains(std::format("KHR_gaussian_splatting:SH_DEGREE_{}_COEF_0", degree))) {
                            tile.sh_degree = degree;
                            break;
                        }
                    return;
                }
            throw std::runtime_error(std::format("'{}' has no splat primitive", core::path_to_utf8(path)));
        }

        class Tiles3dSource final : public SplatTileSource {
        public:
            std::vector<SplatTile> tiles_;
            std::vector<fs::path> contents_;

            std::span<const SplatTile> tiles() const override { return tiles_; }

            std::expected<core::SplatData, std::string> load_tile(const std::uint32_t tile) const override {
                if (tile >= tiles_.size() || contents_[tile].empty())
                    return std::unexpected(std::format("tile {} has no content", tile));
                return load_spz(contents_[tile]);
            }
        };

        // Breadth-first so every tile's children are contiguous.
        void flatten(ParsedTile& root, Tiles3dSource& source) {
            std::deque<std::pair<ParsedTile*, std::uint32_t>> queue{{&root, 0u}};
            source.tiles_.push_back(root.tile);
            source.contents_.push_back(std::move(root.content));
            while (!queue.empty()) {
                auto [parsed, self] = queue.front();
                queue.pop_front();
                source.tiles_[self].first_child = static_cast<std::uint32_t>(source.tiles_.size());
                source.tiles_[self].child_count = static_cast<std::uint32_t>(parsed->children.size());
                for (auto& child : parsed->children) {
                    child.tile.parent = self;
                    const auto child_index = static_cast<std::uint32_t>(source.tiles_.size());
                    source.tiles_.push_back(child.tile);
                    source.contents_.push_back(std::move(child.content));
                    queue.emplace_back(&child, child_index);
                }
            }
        }
    } // namespace

    bool is_tiles3d_path(const std::filesystem::path& path) {
        // Recognized by the tileset's required properties, not its name: asset.version,
        // geometricError and a root tile with boundingVolume and geometricError.
        std::error_code ec;
        if (path.extension() != ".json" || !fs::is_regular_file(path, ec) ||
            fs::file_size(path, ec) > kMaxTilesetJsonBytes)
            return false;
        try {
            const auto doc = read_json(path);
            const auto has_volume = [](const json& volume) {
                return volume.is_object() && (volume.contains("box") || volume.contains("region") ||
                                              volume.contains("sphere"));
            };
            return doc.is_object() && doc.contains("asset") && doc["asset"].is_object() &&
                   doc["asset"].contains("version") && doc["asset"]["version"].is_string() &&
                   doc.contains("geometricError") && doc["geometricError"].is_number() &&
                   doc.contains("root") && doc["root"].is_object() &&
                   doc["root"].contains("geometricError") && doc["root"]["geometricError"].is_number() &&
                   doc["root"].contains("boundingVolume") && has_volume(doc["root"]["boundingVolume"]);
        } catch (const std::exception&) {
            return false;
        }
    }

    std::expected<std::shared_ptr<SplatTileSource>, std::string> open_tiles3d(const std::filesystem::path& path) {
        try {
            const auto doc = read_json(path);
            const auto& root = doc.at("root");
            // Keep float geometry near the origin: the root frame (often ECEF) becomes local.
            auto source = std::make_shared<Tiles3dSource>();
            source->local_to_world = read_matrix(root);
            ParseContext context{.ecef_to_local = glm::inverse(source->local_to_world)};
            auto parsed = parse_tile(root, path.parent_path(), glm::dmat4(1.0), false, context, 0);
            flatten(parsed, *source);

            std::vector<std::string> errors(source->tiles_.size());
            tbb::parallel_for(std::size_t{0}, source->tiles_.size(), [&](const std::size_t i) {
                if (source->contents_[i].empty())
                    return;
                try {
                    probe_glb(source->contents_[i], source->tiles_[i]);
                } catch (const std::exception& e) {
                    errors[i] = e.what();
                }
            });
            for (const auto& error : errors)
                if (!error.empty())
                    return std::unexpected(error);

            std::uint64_t splats = 0;
            for (const auto& tile : source->tiles_)
                splats += tile.splat_count;
            LOG_INFO("3D Tiles '{}': {} tiles, {} splats", core::path_to_utf8(path), source->tiles_.size(), splats);
            return source;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Invalid 3D Tiles tileset '{}': {}", core::path_to_utf8(path), e.what()));
        }
    }

} // namespace lfs::io
