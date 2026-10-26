/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision_walk.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace lfs::core {
    namespace {
        bool finite(glm::vec3 p) {
            return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        }
        bool segmentBox(glm::vec3 a, glm::vec3 b, glm::vec3 lo, glm::vec3 hi, float radius) {
            const auto delta = b - a;
            const auto separation = glm::max(glm::max(lo - glm::max(a, b), glm::min(a, b) - hi), glm::vec3(0));
            if (glm::dot(separation, separation) >= radius * radius)
                return false;
            std::array<float, 8> times{0, 1};
            int count = 2;
            for (int axis = 0; axis < 3; ++axis)
                if (std::abs(delta[axis]) > 1e-12f)
                    for (const float plane : {lo[axis], hi[axis]}) {
                        const float t = (plane - a[axis]) / delta[axis];
                        if (t > 0 && t < 1)
                            times[count++] = t;
                    }
            std::sort(times.begin(), times.begin() + count);
            // Squared segment/box distance is quadratic between box-plane crossings.
            for (int i = 1; i < count; ++i) {
                const float left = times[i - 1], right = times[i], mid = (left + right) * .5f;
                const auto point = a + delta * mid;
                float numerator = 0, denominator = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    if (point[axis] >= lo[axis] && point[axis] <= hi[axis])
                        continue;
                    const float bound = point[axis] < lo[axis] ? lo[axis] : hi[axis];
                    numerator += delta[axis] * (a[axis] - bound);
                    denominator += delta[axis] * delta[axis];
                }
                const float t = denominator > 0 ? glm::clamp(-numerator / denominator, left, right) : mid;
                const auto p = a + delta * t;
                const auto distance = p - glm::clamp(p, lo, hi);
                if (glm::dot(distance, distance) < radius * radius)
                    return true;
            }
            return false;
        }
        struct Sweep {
            glm::vec3 position;
            bool hit = false;
        };
        template <typename Hit>
        Sweep sweep(glm::vec3 start, glm::vec3 delta, float spacing, const Hit& hit) {
            const float length = glm::length(delta);
            if (length == 0)
                return {start};
            const double required = std::ceil(double(length) / spacing);
            if (required > 4096)
                throw std::runtime_error("Walk movement exceeds the sweep budget");
            const int count = std::max(1, static_cast<int>(required));
            float clear = 0;
            auto clear_position = start;
            for (int i = 1; i <= count; ++i) {
                float contact = float(i) / count;
                const auto candidate = start + delta * contact;
                if (hit(candidate)) {
                    for (int j = 0; j < 16; ++j) {
                        const float mid = (clear + contact) * .5f;
                        const auto point = start + delta * mid;
                        if (hit(point))
                            contact = mid;
                        else {
                            clear = mid;
                            clear_position = point;
                        }
                    }
                    return {clear_position, true};
                }
                clear = contact;
                clear_position = candidate;
            }
            return {clear_position};
        }
    } // namespace
    void WalkSettings::validate() const {
        for (float value : {height, radius, speed})
            if (!std::isfinite(value) || value <= 0)
                throw std::invalid_argument("Walk dimensions and speed must be finite and positive");
        if (!std::isfinite(step) || step < 0 || step > height || height < 2 * radius)
            throw std::invalid_argument("Invalid Walk capsule or step height");
    }

    bool collisionCapsuleIntersects(const CollisionVolume& volume, const glm::mat4& transform,
                                    glm::vec3 feet, glm::vec3 up, float height, float radius) {
        if (!finite(feet) || !finite(up) || !std::isfinite(height) || !std::isfinite(radius) ||
            radius <= 0 || height < 2 * radius || glm::length(up) < 1e-6f)
            throw std::invalid_argument("Invalid collision capsule");
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (!std::isfinite(transform[i][j]))
                    throw std::invalid_argument("Non-finite collision transform");
        const double determinant = glm::determinant(glm::dmat3(transform));
        if (!std::isfinite(determinant) || determinant == 0 ||
            transform[0][3] != 0 || transform[1][3] != 0 || transform[2][3] != 0 || transform[3][3] != 1)
            throw std::invalid_argument("Invalid collision transform");
        up = glm::normalize(up);
        const auto inverse = glm::inverse(transform);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (!std::isfinite(inverse[i][j]))
                    throw std::invalid_argument("Collision inverse transform is not representable");
        const auto linear = glm::mat3(inverse);
        float norm1 = 0, norm_inf = 0, max_scale = 0;
        bool orthogonal = true;
        for (int i = 0; i < 3; ++i) {
            float row = 0, column = 0;
            max_scale = std::max(max_scale, glm::length(linear[i]));
            for (int j = 0; j < 3; ++j) {
                column += std::abs(linear[i][j]);
                row += std::abs(linear[j][i]);
                if (i != j && std::abs(glm::dot(linear[i], linear[j])) > 1e-5f * glm::length(linear[i]) * glm::length(linear[j]))
                    orthogonal = false;
            }
            norm1 = std::max(norm1, column);
            norm_inf = std::max(norm_inf, row);
        }
        // Exact for rigid/uniform transforms; conservative for nonuniform scale/shear.
        const float world_radius = radius;
        radius *= orthogonal ? max_scale : std::sqrt(norm1 * norm_inf);
        const auto a = glm::vec3(inverse * glm::vec4(feet + up * world_radius, 1));
        const auto b = glm::vec3(inverse * glm::vec4(feet + up * (height - world_radius), 1));
        size_t visited = 0;
        auto visit = [&](auto&& self, uint32_t index, glm::ivec3 lo, int edge) -> bool {
            const auto value = volume.words[index];
            if (!value || glm::any(glm::greaterThanEqual(lo, volume.dimensions)))
                return false;
            if (++visited > 2'000'000)
                throw std::runtime_error("Walk query exceeds the tree budget");
            const auto hi = glm::min(lo + edge, volume.dimensions);
            if (!segmentBox(a, b, volume.origin + glm::vec3(lo) * volume.settings.voxel_size,
                            volume.origin + glm::vec3(hi) * volume.settings.voxel_size, radius))
                return false;
            if (value == 1)
                return true;
            if (value & CollisionVolume::brick_tag) {
                for (int bit = 0; bit < 64; ++bit) {
                    if (!(volume.words[(value & ~CollisionVolume::brick_tag) + bit / 32] & (1u << (bit % 32))))
                        continue;
                    const auto p = lo + glm::ivec3(bit % 4, (bit / 4) % 4, bit / 16);
                    if (glm::any(glm::greaterThanEqual(p, volume.dimensions)))
                        continue;
                    if (segmentBox(a, b, volume.origin + glm::vec3(p) * volume.settings.voxel_size,
                                   volume.origin + glm::vec3(p + 1) * volume.settings.voxel_size, radius))
                        return true;
                }
                return false;
            }
            for (int child = 0; child < 8; ++child)
                if (self(self, value - 2 + child, lo + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * (edge / 2), edge / 2))
                    return true;
            return false;
        };
        return visit(visit, 0, glm::ivec3(0), volume.edge);
    }

    void settleWalk(WalkState& state, const WalkSettings& settings, glm::vec3 up, const CapsuleQuery& blocked) {
        settings.validate();
        if (!finite(state.eye) || !finite(up) || glm::length(up) < 1e-6f)
            throw std::invalid_argument("Invalid Walk spawn");
        up = glm::normalize(up);
        const auto feet = state.eye - up * settings.height;
        auto hit = [&](glm::vec3 p) { return blocked(p, up, settings.height, settings.radius); };
        if (hit(feet))
            throw std::runtime_error("Walk capsule does not fit at the camera");
        const auto support = sweep(feet, -up * (2 * settings.height), settings.radius / 3, hit);
        if (!support.hit)
            throw std::runtime_error("No Walk support below the camera");
        state.feet = support.position;
        state.eye = state.feet + up * settings.height;
        state.presentation = state.eye;
        state.grounded = true;
    }

    void advanceWalk(WalkState& state, const WalkSettings& settings, glm::vec3 up, glm::vec3 velocity,
                     float dt, const CapsuleQuery& blocked) {
        settings.validate();
        if (!finite(up) || !finite(velocity) || !finite(state.eye) || !finite(state.presentation) || !finite(state.feet) || !state.grounded ||
            !std::isfinite(dt) || dt < 0 || glm::length(up) < 1e-6f)
            throw std::invalid_argument("Invalid Walk input");
        up = glm::normalize(up);
        dt = std::min(dt, .1f);
        velocity -= up * glm::dot(velocity, up);
        auto feet = state.feet;
        const float spacing = settings.radius / 3;
        auto hit = [&](glm::vec3 p) { return blocked(p, up, settings.height, settings.radius); };
        if (hit(feet))
            throw std::runtime_error("Walk capsule overlaps collision geometry at its current position");
        const auto displacement = velocity * dt;
        const auto travel = sweep(feet, displacement, spacing, hit);
        bool stepped = false;
        if (travel.hit && settings.step > 0) {
            auto foot_hit = [&](float lift) {
                return blocked(feet + displacement + up * lift, up, 2 * settings.radius, settings.radius);
            };
            const double required = std::ceil(double(settings.step) / spacing);
            if (required > 4096)
                throw std::runtime_error("Walk step exceeds the sweep budget");
            const int count = std::max(1, static_cast<int>(required));
            float low = 0;
            for (int i = 1; i <= count && !stepped; ++i) {
                float high = settings.step * i / count;
                if (foot_hit(high)) {
                    low = high;
                    continue;
                }
                for (int j = 0; j < 16; ++j) {
                    const float mid = (low + high) * .5f;
                    if (foot_hit(mid))
                        low = mid;
                    else
                        high = mid;
                }
                const auto lift = sweep(feet, up * high, spacing, hit);
                const auto across = sweep(lift.position, displacement, spacing, hit);
                const auto down = sweep(across.position, -up * (high + settings.radius * .001f), spacing, hit);
                if (!lift.hit && !across.hit && down.hit) {
                    feet = down.position;
                    stepped = true;
                }
            }
        }
        if (!stepped) {
            feet = travel.position;
            if (travel.hit)
                for (int axis = 0; axis < 3; ++axis) {
                    glm::vec3 delta(0);
                    delta[axis] = (displacement - (travel.position - state.feet))[axis];
                    delta -= up * glm::dot(delta, up);
                    feet = sweep(feet, delta, spacing, hit).position;
                }
        }
        const auto down = sweep(feet, -up * std::max(settings.step, settings.radius * .001f), spacing, hit);
        if (!down.hit) {
            // Refuse an unsupported move while retaining Walk and its last safe body.
            // Actual removal of the supporting geometry remains an explicit failure.
            if (!sweep(state.feet, -up * std::max(settings.step, settings.radius * .001f), spacing, hit).hit)
                throw std::runtime_error("Walk support is no longer present");
            return;
        }
        state.feet = down.position;
        state.eye = state.feet + up * settings.height;
        const float offset = glm::clamp(glm::dot(state.presentation - state.eye, up) * std::exp(-20 * dt),
                                        -settings.step, settings.step);
        auto head_hit = [&](glm::vec3 eye) { return blocked(eye - up * (2 * settings.radius), up, 2 * settings.radius, settings.radius); };
        state.presentation = sweep(state.eye, up * offset, spacing, head_hit).position;
    }
} // namespace lfs::core
