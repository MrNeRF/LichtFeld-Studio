/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "training/kernels/depth_loss.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::core {
    class Camera;
    class Tensor;
} // namespace lfs::core

namespace lfs::training {

    using RawDepthAnchorMap = std::unordered_map<std::string, kernels::DepthAnchor>;

    // Identifies the geometry, camera data and depth files behind a cached fit.
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
    // The camera's lens as the GPU projection models it; pinhole once undistortion is prepared.
    [[nodiscard]] kernels::DepthCameraProjection depth_camera_projection(const lfs::core::Camera& cam);
    [[nodiscard]] kernels::DepthCameraProjection photograph_projection(const lfs::core::Camera& cam);

    [[nodiscard]] std::optional<kernels::DepthRayGrid> depth_prior_ray_grid(
        const lfs::core::Camera& cam, int tile_x, int tile_y);

    using DepthAnchorProgress = std::function<void(std::size_t done, std::size_t total)>;
    [[nodiscard]] RawDepthAnchorMap computeRawDepthAnchors(
        const lfs::core::Tensor& means,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        int resize_factor,
        int max_width,
        const DepthAnchorProgress& progress = {});

    // Fingerprints the fit's seed geometry, camera observations, intrinsics,
    // pose, and depth-file metadata and content. Startup only: the cloud is
    // copied to the host once. Different cloud subsets conservatively miss.
    [[nodiscard]] std::uint64_t computeAnchorSeedFingerprint(const lfs::core::Tensor& means);
    [[nodiscard]] DepthAnchorFingerprint computeAnchorFingerprint(
        std::uint64_t seed_fingerprint,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        const DepthAnchorProgress& progress = {});

    [[nodiscard]] inline DepthAnchorFingerprint computeAnchorFingerprint(
        const lfs::core::Tensor& means,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        return computeAnchorFingerprint(computeAnchorSeedFingerprint(means), cameras);
    }

    // Sidecar lives next to the depth priors it aligns: derived from the shared
    // parent of the cameras' depth paths (handles depth/ vs depths/). Empty when
    // no camera carries a depth path.
    [[nodiscard]] std::filesystem::path depthAnchorSidecarPath(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras);

    bool writeDepthAnchorSidecar(
        const std::filesystem::path& path,
        const RawDepthAnchorMap& anchors,
        const DepthAnchorFingerprint& fingerprint);

    [[nodiscard]] std::expected<RawDepthAnchorMap, std::string_view> readDepthAnchorSidecar(
        const std::filesystem::path& path,
        const DepthAnchorFingerprint& expected);

} // namespace lfs::training
