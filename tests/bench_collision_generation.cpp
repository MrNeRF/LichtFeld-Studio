/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/collision_serialization.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace lfs::core;
// Native snapshot rows: centre xyz, three column-major one-sigma axes, alpha.
// This deliberately times generation only, not PLY parsing, training or meshing.
int main(int argc, char** argv) {
    if (argc != 4 && argc != 7) {
        std::cerr << "Usage: bench_collision_generation snapshot.f32 voxel_size none|cleanup|connected|strict|auto [seed_x seed_y seed_z]\n";
        return 2;
    }
    try {
        static_assert(sizeof(CollisionGaussian) == 13 * sizeof(float));
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        if (!input || input.tellg() <= 0 || uint64_t(input.tellg()) % sizeof(CollisionGaussian) ||
            uint64_t(input.tellg()) / sizeof(CollisionGaussian) > CollisionGaussian::max_count)
            throw std::runtime_error("Invalid native Gaussian snapshot");
        std::vector<CollisionGaussian> source(size_t(input.tellg()) / sizeof(CollisionGaussian));
        input.seekg(0);
        if (!input.read(reinterpret_cast<char*>(source.data()), source.size() * sizeof(CollisionGaussian)))
            throw std::runtime_error("Truncated native Gaussian snapshot");
        CollisionSettings settings;
        settings.voxel_size = std::stof(argv[2]);
        const std::string mode = argv[3];
        if (mode != "none" && mode != "cleanup" && mode != "connected" && mode != "strict" && mode != "auto")
            throw std::runtime_error("Unknown filter mode");
        settings.adaptive_filter = mode == "cleanup" || mode == "connected" || mode == "auto";
        settings.auto_resolution = mode == "auto";
        settings.cluster = mode == "connected" || mode == "strict";
        if (argc == 7)
            settings.seed = {std::stof(argv[4]), std::stof(argv[5]), std::stof(argv[6])};
        const auto start = std::chrono::steady_clock::now();
        auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
        try {
            auto volume = generateCollision(source, settings, {}, [&](float value, const auto& stage) { std::cout << nlohmann::json{{"seconds", elapsed()}, {"progress", value}, {"stage", stage}}.dump() << std::endl; }, voxelizeCollisionGpu);
            std::cout << nlohmann::json{{"generation_s", elapsed()}, {"source", source.size()}, {"cell_size", volume.settings.voxel_size}, {"occupied", volume.occupiedCount()}, {"cpu_tree_bytes", volume.tree.size() * 4}, {"dimensions", {volume.dimensions.x, volume.dimensions.y, volume.dimensions.z}}, {"note", volume.warning}}.dump() << std::endl;
            // Export is deliberately outside the reported generation interval.
            if (const auto* path = std::getenv("LFS_COLLISION_BENCHMARK_VOLUME")) {
                std::ofstream output(path);
                output << nlohmann::json(volume);
                if (!output)
                    throw std::runtime_error("Could not write collision benchmark volume");
            }
        } catch (const std::exception& e) {
            std::cout << nlohmann::json{{"failure_s", elapsed()}, {"error", e.what()}}.dump() << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
