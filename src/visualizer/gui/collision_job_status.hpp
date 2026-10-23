/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include <cctype>
#include <charconv>
#include <string>
#include <string_view>

namespace lfs::vis::gui {
    inline std::string collisionTaskLabel(std::string_view text) {
        if (text.starts_with("Clustering coarse "))
            return "Connected-region filter: " + collisionTaskLabel(text.substr(18));
        if (text.starts_with("Voxelizing GPU: ")) {
            int percent = 0;
            const auto number = text.substr(16);
            if (std::from_chars(number.data(), number.data() + number.size(), percent).ec == std::errc{}) {
                if (percent < 15)
                    return "Preparing Gaussian bounds on CPU";
                if (percent < 97)
                    return "Evaluating sparse collision regions and cells on GPU";
                return "Assembling sparse occupancy on CPU";
            }
        }
        if (text == "Clustering source cleanup")
            return "Finding reconstruction outliers on CPU";
        if (text.starts_with("Clustering source connectivity: "))
            return "Connecting observed source cells on CPU: " + std::string(text.substr(32));
        if (text == "Clustering Gaussians")
            return "Preparing the connected-region filter";
        if (text == "Clustering connected component")
            return "Finding the occupied region nearest the seed";
        if (text.starts_with("Clustering splats: "))
            return "Selecting contributing splats: " + std::string(text.substr(19));
        if (text == "Reading Gaussian snapshot")
            return "Reading source positions, scales, rotations and opacity";
        if (text == "Voxelizing acceleration structure (CPU)")
            return "Building the Gaussian search structure on CPU";
        if (text == "Voxelizing rotated Gaussian ellipsoids")
            return "Preparing full-resolution collision cells";
        return std::string(text);
    }

    struct CollisionFailureInfo {
        std::string title;
        std::string advice;
        bool cancelled = false;
    };
    inline CollisionFailureInfo collisionFailureInfo(std::string_view message) {
        auto lower = std::string(message);
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (lower.find("cancelled") != std::string::npos)
            return {"Collision operation cancelled", "No generated result was applied. You can change the settings and try again.", true};
        if (lower.find("budget") != std::string::npos || lower.find("coordinate range") != std::string::npos)
            return {"Collision resource limit reached",
                    "Enable Auto resolution or choose larger cells in Custom. Try Surface cleanup for reconstruction outliers. Large outlier splats can create enormous collision regions. The exact limit and the failed step are below."};
        if (lower.find("out of memory") != std::string::npos || lower.find("bad_alloc") != std::string::npos)
            return {"Not enough memory for this collision operation",
                    "Try larger cells, a coarser optional triangle surface, or fewer visible source splats. Free GPU memory if the details name CUDA."};
        if (lower.find("excluded all") != std::string::npos)
            return {"All source splats were filtered out", "Increase Maximum splat area or choose Keep all splats, then generate again. The original splats and existing collider are unchanged."};
        if (lower.find("no occupied") != std::string::npos || lower.find("retained no") != std::string::npos)
            return {"No collision cells found",
                    "Lower the opacity threshold. If filtering is enabled, check its threshold and place the seed near the intended surface."};
        if (lower.find("invalid number") != std::string::npos || lower.find("must be") != std::string::npos)
            return {"Check the collision settings", "Correct the value named in the details, then retry the action."};
        return {"Collision action could not finish", "Read the details below, correct the cause, and retry. No failed result replaces the existing collider."};
    }
} // namespace lfs::vis::gui
