/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <filesystem>
#include <optional>
#include <span>
#include <vector>

namespace lfs::core {

    /// Photographic EV = log2(t / N^2 * ISO) from JPEG APP1 Exif, or nullopt.
    [[nodiscard]] LFS_CORE_API std::optional<double>
    exif_exposure_ev(const std::filesystem::path& image);

    /// Like exif_exposure_ev, then the same filename under dataset_root/images/.
    /// COLMAP images_N/ copies strip APP1; the full-resolution images/ folder keeps it.
    [[nodiscard]] LFS_CORE_API std::optional<double>
    exif_exposure_ev_for_training_image(const std::filesystem::path& image_path,
                                        const std::filesystem::path& dataset_root);

    /// exif_exposure_ev_for_training_image for many images, read in parallel; results follow the input order.
    [[nodiscard]] LFS_CORE_API std::vector<std::optional<double>>
    exif_exposure_ev_for_training_images(std::span<const std::filesystem::path> image_paths,
                                         const std::filesystem::path& dataset_root);

} // namespace lfs::core
