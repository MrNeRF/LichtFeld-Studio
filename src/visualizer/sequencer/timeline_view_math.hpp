/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "sequencer/timeline.hpp"

#include <algorithm>
#include <cmath>

namespace lfs::vis::sequencer_ui {

    [[nodiscard]] inline float unzoomedEndTime(const lfs::sequencer::Timeline& timeline) {
        return timeline.clipDuration();
    }

    [[nodiscard]] inline float displayEndTime(const lfs::sequencer::Timeline& timeline, const float zoom_level) {
        const float safe_zoom = std::max(zoom_level, 0.001f);
        return unzoomedEndTime(timeline) / safe_zoom;
    }

    [[nodiscard]] inline float maxPanOffset(const lfs::sequencer::Timeline& timeline, const float zoom_level) {
        const float visible_range = displayEndTime(timeline, zoom_level);
        return std::max(0.0f, unzoomedEndTime(timeline) - visible_range);
    }

    [[nodiscard]] inline float timeToScreenX(const float time, const float timeline_x, const float timeline_width,
                                             const float display_end_time, const float pan_offset) {
        if (timeline_width <= 0.0f || display_end_time <= 0.0f)
            return timeline_x;
        return timeline_x + ((time - pan_offset) / display_end_time) * timeline_width;
    }

    [[nodiscard]] inline float screenXToTime(const float x, const float timeline_x, const float timeline_width,
                                             const float display_end_time, const float pan_offset) {
        if (timeline_width <= 0.0f || display_end_time <= 0.0f)
            return pan_offset;
        return ((x - timeline_x) / timeline_width) * display_end_time + pan_offset;
    }

    [[nodiscard]] inline float thumbnailDensityScale(const float zoom_level) {
        return std::clamp(std::sqrt(std::max(zoom_level, 0.001f)), 0.75f, 2.0f);
    }

    [[nodiscard]] inline float targetThumbnailWidth(const float base_thumbnail_width, const float zoom_level) {
        return std::max(1.0f, base_thumbnail_width / thumbnailDensityScale(zoom_level));
    }

    [[nodiscard]] inline int thumbnailCount(const float timeline_width, const float base_thumbnail_width,
                                            const float zoom_level) {
        if (timeline_width <= 0.0f || base_thumbnail_width <= 0.0f)
            return 0;
        return std::max(1, static_cast<int>(std::ceil(
                               timeline_width / targetThumbnailWidth(base_thumbnail_width, zoom_level))));
    }

    struct ThumbnailSlot {
        float screen_x = 0.0f;
        float screen_width = 0.0f;
        float screen_center_x = 0.0f;
        float interval_start_time = 0.0f;
        float interval_end_time = 0.0f;
        float sample_time = 0.0f;
    };

    [[nodiscard]] inline ThumbnailSlot thumbnailSlotAt(const int index, const int num_thumbnails,
                                                       const float timeline_x, const float timeline_width,
                                                       const float display_end_time, const float pan_offset) {
        ThumbnailSlot slot;
        if (num_thumbnails <= 0 || timeline_width <= 0.0f)
            return slot;

        slot.screen_width = timeline_width / static_cast<float>(num_thumbnails);
        slot.screen_x = timeline_x + slot.screen_width * static_cast<float>(index);
        slot.screen_center_x = slot.screen_x + slot.screen_width * 0.5f;
        slot.interval_start_time = screenXToTime(slot.screen_x, timeline_x, timeline_width, display_end_time, pan_offset);
        slot.interval_end_time = screenXToTime(slot.screen_x + slot.screen_width, timeline_x, timeline_width, display_end_time, pan_offset);
        slot.sample_time = screenXToTime(slot.screen_center_x, timeline_x, timeline_width, display_end_time, pan_offset);
        return slot;
    }

    [[nodiscard]] inline float resolvedThumbnailSampleTime(const float default_sample_time,
                                                           const float interval_start_time,
                                                           const float interval_end_time,
                                                           const float content_start_time,
                                                           const float content_end_time) {
        const float clamped_sample = std::clamp(default_sample_time, content_start_time, content_end_time);
        if (interval_start_time <= content_start_time && interval_end_time >= content_start_time)
            return content_start_time;
        if (interval_start_time <= content_end_time && interval_end_time >= content_end_time)
            return content_end_time;
        return clamped_sample;
    }

} // namespace lfs::vis::sequencer_ui
