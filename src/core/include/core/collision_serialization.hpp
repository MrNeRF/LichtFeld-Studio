/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace lfs::core {
    inline void to_json(nlohmann::json& j, const CollisionDisplay& d) {
        j = {{"preview", d.preview}, {"surface_ready", d.surface_ready}, {"opacity", d.opacity}, {"slice", d.slice}, {"surface_stride", d.surface_stride}, {"surface_shape", d.surface_shape}};
    }
    inline void from_json(const nlohmann::json& j, CollisionDisplay& d) {
        d.preview = j.value("preview", true);
        d.surface_ready = j.value("surface_ready", true);
        d.opacity = j.value("opacity", 1.0f);
        d.slice = j.value("slice", 1.0f);
        d.surface_stride = j.value("surface_stride", 4);
        d.surface_shape = j.value("surface_shape", CollisionShape::Smooth);
        if (!std::isfinite(d.opacity) || d.opacity < 0 || d.opacity > 1 ||
            !std::isfinite(d.slice) || d.slice < 0 || d.slice > 1 ||
            (d.surface_shape != CollisionShape::Smooth && d.surface_shape != CollisionShape::Faces) ||
            d.surface_stride < 1 || d.surface_stride > 16 || (d.surface_stride & (d.surface_stride - 1)))
            throw std::invalid_argument("Invalid collision display settings");
    }

    inline void to_json(nlohmann::json& j, const CollisionSettings& s) {
        j = {{"units_per_meter", s.units_per_meter}, {"preset", s.preset}, {"cluster", s.cluster}, {"cluster_resolution", s.cluster_resolution}, {"cluster_opacity", s.cluster_opacity}, {"cluster_min_contribution", s.cluster_min_contribution}, {"voxel_size", s.voxel_size}, {"opacity", s.opacity}, {"external_fill", s.external_fill}, {"external_size", s.external_size}, {"floor_below_seed", s.floor_below_seed}, {"floor_fill", s.floor_fill}, {"floor_size", s.floor_size}, {"carve_height", s.carve_height}, {"carve_radius", s.carve_radius}, {"shape", s.shape}, {"seed", {s.seed.x, s.seed.y, s.seed.z}}};
    }
    inline void from_json(const nlohmann::json& j, CollisionSettings& s) {
        s.units_per_meter = j.value("units_per_meter", 1.0f);
        j.at("preset").get_to(s.preset);
        j.at("cluster").get_to(s.cluster);
        j.at("cluster_resolution").get_to(s.cluster_resolution);
        j.at("cluster_opacity").get_to(s.cluster_opacity);
        j.at("cluster_min_contribution").get_to(s.cluster_min_contribution);
        j.at("voxel_size").get_to(s.voxel_size);
        j.at("opacity").get_to(s.opacity);
        j.at("external_fill").get_to(s.external_fill);
        j.at("external_size").get_to(s.external_size);
        j.at("floor_fill").get_to(s.floor_fill);
        s.floor_below_seed = j.value("floor_below_seed", false);
        j.at("floor_size").get_to(s.floor_size);
        j.at("carve_height").get_to(s.carve_height);
        j.at("carve_radius").get_to(s.carve_radius);
        j.at("shape").get_to(s.shape);
        auto seed = j.at("seed").get<std::array<float, 3>>();
        s.seed = {seed[0], seed[1], seed[2]};
        s.validate();
    }
    inline void to_json(nlohmann::json& j, const CollisionVolume& v) {
        v.validate();
        if (!v.tree.empty()) {
            std::string tags((v.tree.size() + 1) / 2, '0');
            constexpr char hex[] = "0123456789abcdef";
            for (size_t i = 0; i < tags.size(); ++i) {
                auto first = std::min(v.tree[i * 2], uint32_t(2));
                auto second = i * 2 + 1 < v.tree.size() ? std::min(v.tree[i * 2 + 1], uint32_t(2)) : 0;
                tags[i] = hex[first | (second << 2)];
            }
            j = {{"version", 2}, {"settings", v.settings}, {"origin", {v.origin.x, v.origin.y, v.origin.z}}, {"dimensions", {v.dimensions.x, v.dimensions.y, v.dimensions.z}}, {"tree_size", v.tree_size}, {"tree_nodes", v.tree.size()}, {"occupancy_tree", tags}, {"show_voxels", v.show_voxels}};
            return;
        }
        std::vector<uint32_t> runs;
        uint8_t value = 0;
        uint32_t count = 0;
        for (auto cell : v.solid) {
            if (cell != value) {
                runs.push_back(count);
                count = 0;
                value = cell;
            }
            ++count;
        }
        runs.push_back(count);
        j = {{"version", 1}, {"settings", v.settings}, {"origin", {v.origin.x, v.origin.y, v.origin.z}}, {"dimensions", {v.dimensions.x, v.dimensions.y, v.dimensions.z}}, {"occupancy_rle", runs}, {"show_voxels", v.show_voxels}};
    }
    inline void from_json(const nlohmann::json& j, CollisionVolume& v) {
        const auto version = j.at("version").get<int>();
        if (version != 1 && version != 2)
            throw std::invalid_argument("Unsupported collision payload version");
        j.at("settings").get_to(v.settings);
        auto origin = j.at("origin").get<std::array<float, 3>>();
        auto dimensions = j.at("dimensions").get<std::array<int, 3>>();
        v.origin = {origin[0], origin[1], origin[2]};
        v.dimensions = {dimensions[0], dimensions[1], dimensions[2]};
        v.tree.clear();
        v.tree_size = 0;
        v.solid.clear();
        if (version == 2) {
            const auto count = j.at("tree_nodes").get<int64_t>();
            if (count <= 0 || uint64_t(count) > CollisionVolume::max_tree_words)
                throw std::invalid_argument("Sparse collision payload exceeds node memory limit");
            const auto& tags = j.at("occupancy_tree").get_ref<const std::string&>();
            if (tags.size() != (size_t(count) + 1) / 2)
                throw std::invalid_argument("Invalid sparse collision topology length");
            v.tree.resize(size_t(count));
            for (size_t i = 0; i < tags.size(); ++i) {
                auto position = std::string_view("0123456789abcdef").find(tags[i]);
                if (position == std::string_view::npos || (position & 3) == 3 || (position >> 2) == 3)
                    throw std::invalid_argument("Invalid sparse collision topology tag");
                v.tree[i * 2] = position & 3;
                if (i * 2 + 1 < v.tree.size())
                    v.tree[i * 2 + 1] = position >> 2;
                else if (position >> 2)
                    throw std::invalid_argument("Invalid sparse collision topology padding");
            }
            j.at("tree_size").get_to(v.tree_size);
            size_t next = 1;
            std::function<void(uint32_t, int)> restore = [&](uint32_t i, int edge) {
                if (v.tree[i] != 2)
                    return;
                if (edge <= 1 || next + 8 > v.tree.size())
                    throw std::invalid_argument("Invalid sparse collision topology");
                const auto base = uint32_t(next);
                next += 8;
                v.tree[i] = base + 2;
                for (int child = 0; child < 8; ++child)
                    restore(base + child, edge / 2);
            };
            if (v.tree_size <= 0 || v.tree_size > (1 << 21) || (v.tree_size & (v.tree_size - 1)))
                throw std::invalid_argument("Invalid sparse collision root size");
            restore(0, v.tree_size);
            if (next != v.tree.size())
                throw std::invalid_argument("Unreachable sparse collision topology");
            j.at("show_voxels").get_to(v.show_voxels);
            v.validate();
            return;
        }
        uint64_t total = 1;
        for (auto n : dimensions) {
            if (n <= 0 || uint64_t(n) > CollisionVolume::max_cells / total)
                throw std::invalid_argument("Collision payload exceeds grid resource limit");
            total *= n;
        }
        v.solid.clear();
        v.solid.reserve(total);
        const auto& runs = j.at("occupancy_rle");
        if (!runs.is_array() || runs.size() > total + 1)
            throw std::invalid_argument("Invalid collision occupancy RLE");
        uint8_t value = 0;
        for (const auto& run : runs) {
            if (!run.is_number_unsigned() && !run.is_number_integer())
                throw std::invalid_argument("Invalid collision occupancy run");
            auto count = run.get<int64_t>();
            if (count < 0 || uint64_t(count) > total - v.solid.size())
                throw std::invalid_argument("Collision occupancy run exceeds payload dimensions");
            v.solid.insert(v.solid.end(), size_t(count), value);
            value = !value;
        }
        j.at("show_voxels").get_to(v.show_voxels);
        v.validate();
    }
} // namespace lfs::core
