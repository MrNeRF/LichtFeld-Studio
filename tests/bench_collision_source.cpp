/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_source.hpp"
#include "core/splat_data.hpp"
#include "io/loader.hpp"
#include <chrono>
#include <fstream>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace lfs::core;
// Original scalar authoring loop, retained only as an independent timing oracle.
static std::vector<CollisionGaussian> reference(std::span<const Scene::SplatGeometrySnapshot> snapshots,
                                                const glm::mat4& world_to_grid) {
    std::vector<CollisionGaussian> result;
    for (const auto& source : snapshots) {
        const auto* m = source.means.ptr<float>();
        const auto* s = source.scaling.ptr<float>();
        const auto* r = source.rotation.ptr<float>();
        const auto* o = source.opacity.ptr<float>();
        const auto transform = world_to_grid * source.world_transform;
        for (size_t i = 0; i < source.size; ++i) {
            if (source.deleted.is_valid() && source.deleted.ptr<bool>()[i])
                continue;
            auto q = glm::normalize(glm::quat(r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]));
            auto axes = glm::mat3_cast(q);
            for (int axis = 0; axis < 3; ++axis)
                axes[axis] *= std::exp(s[i * 3 + axis]);
            result.push_back({glm::vec3(transform * glm::vec4(m[i * 3], m[i * 3 + 1], m[i * 3 + 2], 1)),
                              glm::mat3(transform) * axes, 1.0f / (1.0f + std::exp(-o[i]))});
        }
    }
    return result;
}
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: bench_collision_source model.ply snapshot.f32\n";
        return 2;
    }
    try {
        using Clock = std::chrono::steady_clock;
        const auto loaded = Clock::now();
        auto result = lfs::io::Loader::create()->load(argv[1]);
        if (!result || !std::holds_alternative<std::shared_ptr<SplatData>>(result->data))
            throw std::runtime_error("Could not load the source splat model");
        const auto model = std::get<std::shared_ptr<SplatData>>(result->data);
        if (size_t(model->size()) > CollisionGaussian::max_count)
            throw std::runtime_error("Collision source exceeds 32 million Gaussians");
        const auto snapshot_start = Clock::now();
        Scene::SplatGeometrySnapshot source{model->means_raw().to(Device::CPU).contiguous(),
                                            model->scaling_raw().to(Device::CPU).contiguous(),
                                            model->rotation_raw().to(Device::CPU).contiguous(),
                                            model->opacity_raw().to(Device::CPU).contiguous(),
                                            model->deleted().is_valid() ? model->deleted().to(Device::CPU).contiguous() : Tensor{},
                                            glm::mat4(1), size_t(model->size())};
        const auto captured = Clock::now();
        const glm::mat4 frame{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1};
        const std::span inputs(&source, 1);
        for (int round = 0; round < 3; ++round) {
            const auto start = Clock::now();
            const auto old = reference(inputs, frame);
            const auto middle = Clock::now();
            const auto current = extractCollisionSource(inputs, frame);
            const auto end = Clock::now();
            if (old.size() != current.size())
                throw std::runtime_error("Extraction source count mismatch");
            double largest_error = 0;
            for (size_t i = 0; i < old.size(); ++i) {
                largest_error = std::max(largest_error, double(glm::length(old[i].center - current[i].center)));
                largest_error = std::max(largest_error, double(std::abs(old[i].opacity - current[i].opacity)));
                for (int axis = 0; axis < 3; ++axis)
                    largest_error = std::max(largest_error, double(glm::length(old[i].axes[axis] - current[i].axes[axis])));
            }
            std::cout << nlohmann::json{{"round", round}, {"source", current.size()}, {"sh_degree", model->get_active_sh_degree()}, {"load_s", std::chrono::duration<double>(snapshot_start - loaded).count()}, {"snapshot_s", std::chrono::duration<double>(captured - snapshot_start).count()}, {"reference_s", std::chrono::duration<double>(middle - start).count()}, {"parallel_s", std::chrono::duration<double>(end - middle).count()}, {"max_record_error", largest_error}, {"reference_capacity_bytes", old.capacity() * sizeof(CollisionGaussian)}, {"parallel_capacity_bytes", current.capacity() * sizeof(CollisionGaussian)}}.dump() << std::endl;
            if (largest_error > 1e-5)
                throw std::runtime_error("Extraction transform mismatch");
            if (!round) {
                std::ofstream file(argv[2], std::ios::binary);
                file.write(reinterpret_cast<const char*>(old.data()), old.size() * sizeof(CollisionGaussian));
                if (!file)
                    throw std::runtime_error("Could not write native source snapshot");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
