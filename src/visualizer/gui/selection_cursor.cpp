/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/selection_cursor.hpp"
#include <algorithm>
#include <cmath>

namespace lfs::vis::gui {

    namespace {

        [[nodiscard]] uint8_t alphaFromCoverage(const uint8_t alpha, const float coverage) {
            return static_cast<uint8_t>(std::lround(static_cast<float>(alpha) *
                                                    std::clamp(coverage, 0.0f, 1.0f)));
        }

        void setPixel(SelectionCursorImage& image, const int x, const int y,
                      const SelectionCursorColor color, const uint8_t alpha) {
            if (x < 0 || x >= image.size || y < 0 || y >= image.size || alpha == 0) {
                return;
            }
            const size_t offset = (static_cast<size_t>(y) * image.size + x) * 4;
            image.rgba[offset] = color.r;
            image.rgba[offset + 1] = color.g;
            image.rgba[offset + 2] = color.b;
            image.rgba[offset + 3] = std::max(image.rgba[offset + 3], alpha);
        }

        void compositePixel(SelectionCursorImage& image, const int x, const int y,
                            const uint8_t* const source) {
            if (x < 0 || x >= image.size || y < 0 || y >= image.size || source[3] == 0) {
                return;
            }

            const size_t offset = (static_cast<size_t>(y) * image.size + x) * 4;
            const uint32_t source_alpha = source[3];
            const uint32_t destination_alpha = image.rgba[offset + 3];
            const uint32_t output_alpha = source_alpha +
                                          (destination_alpha * (255u - source_alpha) + 127u) / 255u;
            if (output_alpha == 0) {
                return;
            }

            image.rgba[offset] = static_cast<uint8_t>(
                (source[0] * source_alpha * 255u +
                 image.rgba[offset] * destination_alpha * (255u - source_alpha) +
                 output_alpha * 127u) /
                (output_alpha * 255u));
            image.rgba[offset + 1] = static_cast<uint8_t>(
                (source[1] * source_alpha * 255u +
                 image.rgba[offset + 1] * destination_alpha * (255u - source_alpha) +
                 output_alpha * 127u) /
                (output_alpha * 255u));
            image.rgba[offset + 2] = static_cast<uint8_t>(
                (source[2] * source_alpha * 255u +
                 image.rgba[offset + 2] * destination_alpha * (255u - source_alpha) +
                 output_alpha * 127u) /
                (output_alpha * 255u));
            image.rgba[offset + 3] = static_cast<uint8_t>(output_alpha);
        }

    } // namespace

    bool useHardwareSelectionRing(const bool preview_active,
                                  const SelectionPreviewMode mode,
                                  const int radius_px) {
        if (!preview_active || mode != SelectionPreviewMode::Centers || radius_px <= 0) {
            return false;
        }
        const int size = 2 * (radius_px + selectionCursorPadding());
        return size <= selectionCursorMaxSize();
    }

    SelectionCursorImage makeSelectionCursorImage(const int radius_px,
                                                  const SelectionCursorColor color,
                                                  const std::span<const uint8_t> badge_rgba,
                                                  const int badge_width,
                                                  const int badge_height) {
        SelectionCursorImage image;
        if (radius_px <= 0) {
            return image;
        }

        image.hotspot = radius_px + selectionCursorPadding();
        image.size = image.hotspot * 2;
        if (!useHardwareSelectionRing(true, SelectionPreviewMode::Centers, radius_px)) {
            image = {};
            return image;
        }
        image.rgba.resize(static_cast<size_t>(image.size) * image.size * 4, 0);

        constexpr float kRingRadius = 1.5f;
        constexpr float kDotRadius = 3.0f;
        for (int y = 0; y < image.size; ++y) {
            for (int x = 0; x < image.size; ++x) {
                const float dx = static_cast<float>(x - image.hotspot);
                const float dy = static_cast<float>(y - image.hotspot);
                const float distance = std::sqrt(dx * dx + dy * dy);
                const float ring_coverage = std::max(0.0f, kRingRadius - std::abs(distance - radius_px));
                const float dot_coverage = std::max(0.0f, kDotRadius + 0.5f - distance);
                const uint8_t alpha = std::max(alphaFromCoverage(color.a, ring_coverage),
                                               alphaFromCoverage(color.a, dot_coverage));
                setPixel(image, x, y, color, alpha);
            }
        }

        if (badge_width > 0 && badge_height > 0 &&
            badge_rgba.size() == static_cast<size_t>(badge_width) * badge_height * 4) {
            image.badge_x = image.hotspot + radius_px - badge_width;
            image.badge_y = image.hotspot + radius_px - badge_height;
            for (int y = 0; y < badge_height; ++y) {
                for (int x = 0; x < badge_width; ++x) {
                    const auto* const source = badge_rgba.data() +
                                               (static_cast<size_t>(y) * badge_width + x) * 4;
                    compositePixel(image, image.badge_x + x, image.badge_y + y, source);
                }
            }
        }
        return image;
    }

} // namespace lfs::vis::gui
