/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "interpolation.hpp"
#include <algorithm>
#include <cmath>
#include <glm/ext/quaternion_exponential.hpp>
#include <limits>
#include <stdexcept>

namespace lfs::sequencer {

    namespace {
        [[nodiscard]] float easeIn(const float t) {
            return t * t * t;
        }

        [[nodiscard]] float easeOut(const float t) {
            const float u = 1.0f - t;
            return 1.0f - u * u * u;
        }

        [[nodiscard]] float easeInOut(const float t) {
            if (t < 0.5f) {
                return 4.0f * t * t * t;
            }
            const float u = -2.0f * t + 2.0f;
            return 1.0f - 0.5f * u * u * u;
        }

        [[nodiscard]] float monotoneTangent(const float left, const float right) {
            if ((left <= 0.0f && right >= 0.0f) || (left >= 0.0f && right <= 0.0f))
                return 0.0f;
            const float small = std::min(std::abs(left), std::abs(right));
            const float large = std::max(std::abs(left), std::abs(right));
            return std::copysign(small / (0.5f + 0.5f * small / large), left);
        }

        [[nodiscard]] float monotoneCubic(
            const float p0, const float p1, const float p2, const float p3, const float t) {
            const float t2 = t * t;
            const float t3 = t2 * t;
            const float start_tangent = monotoneTangent(p1 - p0, p2 - p1);
            const float end_tangent = monotoneTangent(p2 - p1, p3 - p2);
            const float value = (2.0f * t3 - 3.0f * t2 + 1.0f) * p1 +
                                (t3 - 2.0f * t2 + t) * start_tangent +
                                (-2.0f * t3 + 3.0f * t2) * p2 +
                                (t3 - t2) * end_tangent;
            return std::clamp(value, std::min(p1, p2), std::max(p1, p2));
        }

        struct SegmentKeyframes {
            const Keyframe& previous;
            const Keyframe& start;
            const Keyframe& end;
            const Keyframe& next;
        };

        [[nodiscard]] SegmentKeyframes segmentKeyframes(
            const std::span<const Keyframe> keyframes, const size_t segment) {
            const bool is_loop = keyframes.back().is_loop_point;
            if (is_loop) {
                // The controller appends a copy of the first keyframe to close a loop.
                // Use the real keys cyclically around that derived endpoint so the
                // first and last segment share a tangent.
                const size_t real_count = keyframes.size() - 1;
                return {
                    keyframes[(segment + real_count - 1) % real_count],
                    keyframes[segment],
                    keyframes[segment + 1],
                    keyframes[(segment + 2) % real_count]};
            }

            return {
                keyframes[segment > 0 ? segment - 1 : segment],
                keyframes[segment],
                keyframes[segment + 1],
                keyframes[segment + 2 < keyframes.size() ? segment + 2 : segment + 1]};
        }

        [[nodiscard]] glm::quat alignHemisphere(const glm::quat& reference, glm::quat value) {
            if (glm::dot(reference, value) < 0.0f)
                value = -value;
            return value;
        }

        [[nodiscard]] glm::quat squadControl(
            const glm::quat& previous, const glm::quat& current, const glm::quat& next) {
            const glm::quat previous_aligned = alignHemisphere(current, previous);
            const glm::quat next_aligned = alignHemisphere(current, next);
            const glm::quat tangent = glm::log(glm::inverse(current) * previous_aligned) +
                                      glm::log(glm::inverse(current) * next_aligned);
            const glm::quat offset = -0.25f * tangent;
            const glm::vec3 vector(offset.x, offset.y, offset.z);
            constexpr float epsilon = std::numeric_limits<float>::epsilon();
            if (glm::dot(vector, vector) < epsilon * epsilon)
                return current;
            return glm::normalize(current * glm::exp(offset));
        }

        [[nodiscard]] glm::quat squadShortestPath(
            const glm::quat& q0, const glm::quat& q1,
            const glm::quat& q2, const glm::quat& q3, const float t) {
            const glm::quat start = glm::normalize(q1);
            const glm::quat previous = alignHemisphere(start, glm::normalize(q0));
            const glm::quat end = alignHemisphere(start, glm::normalize(q2));
            const glm::quat next = alignHemisphere(end, glm::normalize(q3));
            const glm::quat start_control = squadControl(previous, start, end);
            const glm::quat end_control = squadControl(start, end, next);
            const glm::quat direct = glm::slerp(start, end, t);
            const glm::quat hemisphere = glm::normalize(start + end);
            if (glm::dot(hemisphere, start_control) <= 1.0e-3f ||
                glm::dot(hemisphere, end_control) <= 1.0e-3f)
                return glm::normalize(direct);
            // Preserve the control path's sign when blending the two SQUAD curves.
            const glm::quat controls = glm::mix(start_control, end_control, t);
            return glm::normalize(glm::mix(direct, controls, 2.0f * t * (1.0f - t)));
        }
    } // namespace

    float applyEasing(const float t, const EasingType easing) {
        const float clamped = std::clamp(t, 0.0f, 1.0f);
        switch (easing) {
        case EasingType::LINEAR:
            return clamped;
        case EasingType::EASE_IN:
            return easeIn(clamped);
        case EasingType::EASE_OUT:
            return easeOut(clamped);
        case EasingType::EASE_IN_OUT:
            return easeInOut(clamped);
        }
        return clamped;
    }

    glm::vec3 catmullRom(
        const glm::vec3& p0, const glm::vec3& p1,
        const glm::vec3& p2, const glm::vec3& p3,
        const float t) {
        const float t2 = t * t;
        const float t3 = t2 * t;
        return 0.5f * ((2.0f * p1) +
                       (-p0 + p2) * t +
                       (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                       (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
    }

    CameraState interpolateSpline(std::span<const Keyframe> keyframes, const float time) {
        if (keyframes.empty()) {
            return {};
        }
        if (keyframes.size() == 1) {
            return {keyframes[0].position, keyframes[0].rotation, keyframes[0].focal_length_mm};
        }

        const float clamped_time = std::clamp(time, keyframes.front().time, keyframes.back().time);

        // Find segment containing time
        size_t i = 0;
        for (; i < keyframes.size() - 1; ++i) {
            if (clamped_time <= keyframes[i + 1].time)
                break;
        }
        if (i >= keyframes.size() - 1) {
            i = keyframes.size() - 2;
        }

        const Keyframe& k1 = keyframes[i];
        const Keyframe& k2 = keyframes[i + 1];

        // Local parameter t in [0,1]
        const float segment_duration = k2.time - k1.time;
        const float t = segment_duration > 0.0f ? (clamped_time - k1.time) / segment_duration : 0.0f;
        const float eased_t = applyEasing(t, k1.easing);

        const auto [k0, spline_start, spline_end, k3] = segmentKeyframes(keyframes, i);

        return {
            catmullRom(k0.position, spline_start.position, spline_end.position, k3.position, eased_t),
            squadShortestPath(k0.rotation, spline_start.rotation, spline_end.rotation, k3.rotation, eased_t),
            monotoneCubic(k0.focal_length_mm, spline_start.focal_length_mm,
                          spline_end.focal_length_mm, k3.focal_length_mm, eased_t)};
    }

    std::vector<glm::vec3> generatePathPoints(
        std::span<const Keyframe> keyframes, const int samples_per_segment) {
        if (keyframes.size() < 2) {
            return keyframes.empty() ? std::vector<glm::vec3>{} : std::vector<glm::vec3>{keyframes[0].position};
        }
        if (samples_per_segment <= 0)
            throw std::invalid_argument("Path samples per segment must be positive");

        const size_t segments = keyframes.size() - 1;
        if (segments > (MAX_GENERATED_PATH_SAMPLES - 1) / static_cast<size_t>(samples_per_segment))
            throw std::length_error("Generated path exceeds the sample budget");

        std::vector<glm::vec3> points;
        points.reserve(segments * static_cast<size_t>(samples_per_segment) + 1);

        // Generate points per-segment to avoid redundant segment lookups
        for (size_t seg = 0; seg < keyframes.size() - 1; ++seg) {
            const auto [k0, k1, k2, k3] = segmentKeyframes(keyframes, seg);

            const int samples = (seg == keyframes.size() - 2) ? samples_per_segment + 1 : samples_per_segment;
            for (int i = 0; i < samples; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(samples_per_segment);
                const float eased_t = applyEasing(t, k1.easing);
                points.push_back(catmullRom(k0.position, k1.position, k2.position, k3.position, eased_t));
            }
        }
        return points;
    }

} // namespace lfs::sequencer
