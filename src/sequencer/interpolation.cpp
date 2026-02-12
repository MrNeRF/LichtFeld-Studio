/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "interpolation.hpp"
#include <algorithm>

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

        // Neighboring keyframes for spline (clamped at boundaries)
        const Keyframe& k0 = keyframes[i > 0 ? i - 1 : i];
        const Keyframe& k3 = keyframes[i + 2 < keyframes.size() ? i + 2 : i + 1];

        return {
            catmullRom(k0.position, k1.position, k2.position, k3.position, eased_t),
            glm::slerp(k1.rotation, k2.rotation, eased_t),
            glm::mix(k1.focal_length_mm, k2.focal_length_mm, eased_t)};
    }

    std::vector<glm::vec3> generatePathPoints(
        std::span<const Keyframe> keyframes, const int samples_per_segment) {
        if (keyframes.size() < 2) {
            return keyframes.empty() ? std::vector<glm::vec3>{} : std::vector<glm::vec3>{keyframes[0].position};
        }

        std::vector<glm::vec3> points;
        points.reserve((keyframes.size() - 1) * static_cast<size_t>(samples_per_segment) + 1);

        // Generate points per-segment to avoid redundant segment lookups
        for (size_t seg = 0; seg < keyframes.size() - 1; ++seg) {
            const Keyframe& k0 = keyframes[seg > 0 ? seg - 1 : seg];
            const Keyframe& k1 = keyframes[seg];
            const Keyframe& k2 = keyframes[seg + 1];
            const Keyframe& k3 = keyframes[seg + 2 < keyframes.size() ? seg + 2 : seg + 1];

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
