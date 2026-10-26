/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include <algorithm>
#include <array>
#include <cmath>

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
                const auto bound = glm::clamp(point, lo, hi);
                const auto slope = delta * glm::vec3(glm::notEqual(point, bound));
                const float numerator = glm::dot(slope, a - bound), denominator = glm::dot(slope, slope);
                const float t = denominator > 0 ? glm::clamp(-numerator / denominator, left, right) : mid;
                const auto p = a + delta * t;
                const auto distance = p - glm::clamp(p, lo, hi);
                if (glm::dot(distance, distance) < radius * radius)
                    return true;
            }
            return false;
        }
        template <typename Hit, typename Contact>
        bool scanSweep(float length, float extent, float spacing, const Hit& hit, const Contact& contact) {
            const double required = std::ceil(double(length) / spacing);
            if (required > 4096)
                throw collisionError("collision.walk_budget", "Walk movement exceeds the sweep budget", ErrorCode::ResourceExhausted);
            const int count = std::max(1, static_cast<int>(required));
            float clear = 0;
            for (int i = 1; i <= count; ++i) {
                float boundary = extent * i / count;
                if (hit(boundary)) {
                    for (int j = 0; j < 16; ++j) {
                        const float mid = (clear + boundary) * .5f;
                        if (hit(mid))
                            boundary = mid;
                        else
                            clear = mid;
                    }
                    if (contact(boundary))
                        return true;
                } else
                    clear = boundary;
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
            Sweep result{start};
            const auto probe = [&](float t) {
                const auto point = start + delta * t;
                if (hit(point))
                    return true;
                result.position = point;
                return false;
            };
            result.hit = scanSweep(length, 1, spacing, probe, [](float) { return true; });
            return result;
        }
    } // namespace
    void WalkSettings::validate() const {
        for (float value : {height, radius, speed})
            if (!std::isfinite(value) || value <= 0)
                throw collisionError("preferences.viewport_defaults_invalid", "Walk dimensions and speed must be finite and positive", ErrorCode::InvalidArgument);
        if (!std::isfinite(step) || step < 0 || step > height || height < 2 * radius)
            throw collisionError("preferences.viewport_defaults_invalid", "Invalid Walk capsule or step height", ErrorCode::InvalidArgument);
    }

    bool collisionCapsuleIntersects(const CollisionVolume& volume, const glm::mat4& transform,
                                    glm::vec3 feet, glm::vec3 up, float height, float radius) {
        if (!finite(feet) || !finite(up) || !std::isfinite(height) || !std::isfinite(radius) ||
            radius <= 0 || height < 2 * radius || glm::length(up) < 1e-6f)
            throw collisionError("preferences.viewport_defaults_invalid", "Invalid collision capsule", ErrorCode::InvalidArgument);
        const auto inverse = collisionInverse(transform);
        up = glm::normalize(up);
        const auto linear = glm::mat3(inverse);
        const auto transposed = glm::transpose(linear);
        float norm1 = 0, norm_inf = 0, max_scale = 0;
        bool orthogonal = true;
        for (int i = 0; i < 3; ++i) {
            max_scale = std::max(max_scale, glm::length(linear[i]));
            for (int j = 0; j < 3; ++j)
                if (i != j && std::abs(glm::dot(linear[i], linear[j])) > 1e-5f * glm::length(linear[i]) * glm::length(linear[j]))
                    orthogonal = false;
            norm1 = std::max(norm1, glm::dot(glm::abs(linear[i]), glm::vec3(1)));
            norm_inf = std::max(norm_inf, glm::dot(glm::abs(transposed[i]), glm::vec3(1)));
        }
        // Exact for rigid/uniform transforms; conservative for nonuniform scale/shear.
        const float world_radius = radius;
        radius *= orthogonal ? max_scale : std::sqrt(norm1 * norm_inf);
        const auto a = glm::vec3(inverse * glm::vec4(feet + up * world_radius, 1));
        const auto b = glm::vec3(inverse * glm::vec4(feet + up * (height - world_radius), 1));
        const auto overlaps = [&](glm::ivec3 lo, glm::ivec3 hi) {
            return segmentBox(a, b, volume.origin + glm::vec3(lo) * volume.settings.voxel_size,
                              volume.origin + glm::vec3(hi) * volume.settings.voxel_size, radius);
        };
        size_t visited = 0;
        auto visit = [&](auto&& self, uint32_t index, glm::ivec3 lo, int edge) -> bool {
            const auto value = volume.words[index];
            if (!value || glm::any(glm::greaterThanEqual(lo, volume.dimensions)))
                return false;
            if (++visited > 2'000'000)
                throw collisionError("collision.walk_budget", "Walk query exceeds the tree budget", ErrorCode::ResourceExhausted);
            if (!overlaps(lo, glm::min(lo + edge, volume.dimensions)))
                return false;
            if (value == 1)
                return true;
            if (value & CollisionVolume::brick_tag) {
                for (int bit = 0; bit < 64; ++bit) {
                    if (!(volume.words[(value & ~CollisionVolume::brick_tag) + bit / 32] & (1u << (bit % 32))))
                        continue;
                    const auto p = lo + glm::ivec3(bit % 4, (bit / 4) % 4, bit / 16);
                    if (glm::all(glm::lessThan(p, volume.dimensions)) && overlaps(p, p + 1))
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
            throw collisionError("preferences.viewport_defaults_invalid", "Invalid Walk spawn", ErrorCode::InvalidArgument);
        up = glm::normalize(up);
        const auto feet = state.eye - up * settings.height;
        auto hit = [&](glm::vec3 p) { return blocked(p, up, settings.height, settings.radius); };
        if (hit(feet))
            throw collisionError("collision.walk_blocked", "Walk capsule does not fit at the camera");
        const auto support = sweep(feet, -up * (2 * settings.height), settings.radius / 3, hit);
        if (!support.hit)
            throw collisionError("collision.walk_no_support", "No Walk support below the camera");
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
            throw collisionError("preferences.viewport_defaults_invalid", "Invalid Walk input", ErrorCode::InvalidArgument);
        up = glm::normalize(up);
        dt = std::min(dt, .1f);
        velocity -= up * glm::dot(velocity, up);
        auto feet = state.feet;
        const float spacing = settings.radius / 3;
        auto hit = [&](glm::vec3 p) { return blocked(p, up, settings.height, settings.radius); };
        if (hit(feet))
            throw collisionError("collision.walk_blocked", "Walk capsule overlaps collision geometry at its current position");
        const auto displacement = velocity * dt;
        const auto travel = sweep(feet, displacement, spacing, hit);
        const auto foot_clear = [&](float lift) {
            return !blocked(feet + displacement + up * lift, up, 2 * settings.radius, settings.radius);
        };
        const auto try_step = [&](float high) {
            const auto lift = sweep(feet, up * high, spacing, hit);
            const auto across = sweep(lift.position, displacement, spacing, hit);
            const auto down = sweep(across.position, -up * (high + settings.radius * .001f), spacing, hit);
            if (lift.hit || across.hit || !down.hit)
                return false;
            feet = down.position;
            return true;
        };
        const bool stepped = travel.hit && settings.step > 0 &&
                             scanSweep(settings.step, settings.step, spacing, foot_clear, try_step);
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
        const auto support_delta = -up * std::max(settings.step, settings.radius * .001f);
        const auto down = sweep(feet, support_delta, spacing, hit);
        if (!down.hit) {
            // Refuse an unsupported move while retaining Walk and its last safe body.
            // Actual removal of the supporting geometry remains an explicit failure.
            if (!sweep(state.feet, support_delta, spacing, hit).hit)
                throw collisionError("collision.walk_no_support", "Walk support is no longer present");
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
