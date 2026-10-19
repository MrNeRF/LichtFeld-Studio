/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "kernels/blob_seeding.hpp"

#include <cstddef>
#include <stop_token>
#include <unordered_set>
#include <vector>

namespace lfs::core {
    class Camera;
}

namespace lfs::training {

    struct BlobSeeds {
        std::vector<float> means;      // [n, 3]
        std::vector<float> colors;     // [n, 3] in [0, 1]
        std::vector<float> log_scales; // [n], isotropic
        [[nodiscard]] size_t size() const { return log_scales.size(); }
    };

    // Small bright or dark blobs (lamps, sockets, distant lights) often have no SfM point and
    // growth only splits the surface behind them, so they never get a Gaussian of their own.
    // The seeder keeps the peak maps of the training views as they are first trained on, then
    // places a seed wherever several views agree on a blob depth that the initial point cloud
    // does not already cover.
    class BlobSeeder {
    public:
        // Detection runs on block averages of images above this pixel count: its buffers scale with it.
        static constexpr size_t DEFAULT_MAX_DETECTION_PIXELS = 3'000'000;

        explicit BlobSeeder(lfs::core::Tensor reference_points,
                            size_t max_detection_pixels = DEFAULT_MAX_DETECTION_PIXELS);

        void capture(const lfs::core::Camera& camera, const lfs::core::Tensor& image);
        [[nodiscard]] size_t captured_views() const { return _views.size(); }
        [[nodiscard]] bool budget_exhausted() const;
        // Seeds ordered by the number of views that triangulated them, most supported first.
        // Returns no seeds once stop is requested.
        [[nodiscard]] BlobSeeds triangulate(std::stop_token stop = {}) const;

    private:
        struct CapturedView {
            kernels::blob_seeding::SweepView geometry{};
            kernels::blob_seeding::ViewPeaks peaks;
        };

        lfs::core::Tensor _reference_points; // CPU [N, 3]
        std::vector<CapturedView> _views;
        std::unordered_set<int> _captured_uids;
        size_t _captured_bytes = 0;
        size_t _max_detection_pixels;
        kernels::blob_seeding::DetectionWorkspace _workspace;
    };

} // namespace lfs::training
