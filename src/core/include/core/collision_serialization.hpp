/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace lfs::core {
    inline void to_json(nlohmann::json& j, const CollisionSettings& s) {
        j = {{"preset", s.preset}, {"cluster", s.cluster}, {"cluster_resolution", s.cluster_resolution}, {"cluster_opacity", s.cluster_opacity}, {"cluster_min_contribution", s.cluster_min_contribution}, {"voxel_size", s.voxel_size}, {"opacity", s.opacity}, {"external_fill", s.external_fill}, {"external_size", s.external_size}, {"floor_fill", s.floor_fill}, {"floor_size", s.floor_size}, {"carve_height", s.carve_height}, {"carve_radius", s.carve_radius}, {"shape", s.shape}, {"seed", {s.seed.x, s.seed.y, s.seed.z}}};
    }
    inline void from_json(const nlohmann::json& j, CollisionSettings& s) {
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
        if (j.at("version").get<int>() != 1)
            throw std::invalid_argument("Unsupported collision payload version");
        j.at("settings").get_to(v.settings);
        auto origin = j.at("origin").get<std::array<float, 3>>();
        auto dimensions = j.at("dimensions").get<std::array<int, 3>>();
        v.origin = {origin[0], origin[1], origin[2]};
        v.dimensions = {dimensions[0], dimensions[1], dimensions[2]};
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
