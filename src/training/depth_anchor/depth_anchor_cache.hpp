/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "training/kernels/depth_loss.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::core {
    class Camera;
    class Tensor;
} // namespace lfs::core

namespace lfs::training {

    using RawDepthAnchorMap = std::unordered_map<std::string, kernels::DepthAnchor>;

    // Identifies the inputs a cached anchor set was fitted against. A mismatch
    // (re-run SfM, regenerated depth, changed resize) forces a recompute.
    struct DepthAnchorFingerprint {
        std::uint64_t hash = 0;
        int version = 0;

        [[nodiscard]] bool operator==(const DepthAnchorFingerprint&) const = default;
    };

    // Robust per-camera affine fit of each depth prior against the point cloud.
    // The GPU projection runs serially on the calling thread; the CPU RANSAC
    // fits drain on a worker pool and overlap the next camera's depth decode.
    // Keyed by camera image_name; returns raw pre-aggregation anchors (both the
    // disparity and depth candidates), so the caller resolves the dataset prior.
    // The optional progress callback fires once per depth camera as it is
    // projected, with (done, total_depth_cameras).
    using DepthAnchorProgress = std::function<void(std::size_t done, std::size_t total)>;
    [[nodiscard]] RawDepthAnchorMap computeRawDepthAnchors(
        const lfs::core::Tensor& means,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        int resize_factor,
        int max_width,
        const DepthAnchorProgress& progress = {});

    // Fingerprints only the inputs that both producers (trainer, preprocess) see
    // identically and that meaningfully change the fit: per-camera intrinsics,
    // pose, and depth-file identity (size+mtime). Cloud subsample (max_cap) and
    // prior resolution are deliberately excluded — the robust fit is invariant to
    // them, and they differ between the two producers.
    [[nodiscard]] DepthAnchorFingerprint computeAnchorFingerprint(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras);

    // Sidecar lives next to the depth priors it aligns: derived from the shared
    // parent of the cameras' depth paths (handles depth/ vs depths/). Empty when
    // no camera carries a depth path.
    [[nodiscard]] std::filesystem::path depthAnchorSidecarPath(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras);

    bool writeDepthAnchorSidecar(
        const std::filesystem::path& path,
        const RawDepthAnchorMap& anchors,
        const DepthAnchorFingerprint& fingerprint);

    [[nodiscard]] std::optional<RawDepthAnchorMap> readDepthAnchorSidecar(
        const std::filesystem::path& path,
        const DepthAnchorFingerprint& expected);

} // namespace lfs::training
