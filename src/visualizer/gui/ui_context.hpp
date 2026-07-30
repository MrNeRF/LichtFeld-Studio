/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

namespace lfs::vis {
    // Forward declarations
    class VisualizerImpl;
    class EditorContext;
    class SequencerController;

    namespace gui {
        class RmlUIManager;
        class RmlViewportOverlay;

        // Font set for typography hierarchy
        struct FontSet {
            using FontHandle = void*;

            FontHandle regular = nullptr;
            FontHandle bold = nullptr;
            FontHandle heading = nullptr;
            FontHandle small_font = nullptr; // Avoid Windows macro collision
            FontHandle section = nullptr;
            FontHandle monospace = nullptr; // For code editor

            static constexpr int MONO_SIZE_COUNT = 5;
            FontHandle monospace_sized[MONO_SIZE_COUNT] = {};
            float monospace_sizes[MONO_SIZE_COUNT] = {};

            FontHandle monoForScale(float scale) const {
                if (MONO_SIZE_COUNT == 0 || !monospace_sized[0])
                    return monospace;
                int best = 0;
                float best_diff = std::numeric_limits<float>::max();
                for (int i = 0; i < MONO_SIZE_COUNT; ++i) {
                    if (!monospace_sized[i])
                        break;
                    const float diff = monospace_sizes[i] - scale;
                    const float dist = diff * diff;
                    if (dist < best_diff) {
                        best_diff = dist;
                        best = i;
                    }
                }
                return monospace_sized[best];
            }
        };

        struct UIContext {
            VisualizerImpl* viewer = nullptr;
            std::unordered_map<std::string, bool>* window_states = nullptr;
            EditorContext* editor = nullptr;
            SequencerController* sequencer_controller = nullptr;
            RmlUIManager* rml_manager = nullptr;
            RmlViewportOverlay* viewport_overlay = nullptr;
            FontSet fonts;
        };

    } // namespace gui
} // namespace lfs::vis
