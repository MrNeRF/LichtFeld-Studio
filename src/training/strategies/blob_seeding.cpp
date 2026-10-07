/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "blob_seeding.hpp"
#include "core/camera.hpp"
#include "core/logger.hpp"
#include "nanoflann.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace lfs::training {
    namespace {
        namespace bs = kernels::blob_seeding;
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        constexpr size_t CAPTURE_BUDGET_BYTES = size_t{256} << 20;
        constexpr size_t FRUSTUM_SAMPLE_POINTS = 65536;
        constexpr size_t MIN_FRUSTUM_POINTS = 16;
        constexpr int NEIGHBOR_VIEWS = 64;
        constexpr float NEAR_DEPTH_PERCENTILE = 0.01f;
        constexpr float FAR_DEPTH_PERCENTILE = 0.99f;
        constexpr float NEAR_DEPTH_MARGIN = 0.3f;
        constexpr float FAR_DEPTH_MARGIN = 3.0f;
        constexpr float EXPLAINED_RADIUS_PX = 3.0f;
        constexpr float MERGE_RADIUS_DEPTH_FRACTION = 0.012f;
        constexpr int MIN_VIEWS_PER_SEED = 3;
        constexpr float SEED_RADIUS_PX = 1.5f;
        constexpr size_t SWEEP_CHUNK_PEAKS = size_t{1} << 17;

        struct PointCloudAdaptor {
            const float* points;
            size_t count;
            [[nodiscard]] size_t kdtree_get_point_count() const { return count; }
            [[nodiscard]] float kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx * 3 + dim]; }
            template <class BBox>
            bool kdtree_get_bbox(BBox&) const { return false; }
        };
        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
                                                           PointCloudAdaptor, 3>;

        [[nodiscard]] bool has_pinhole_image(const lfs::core::Camera& camera) {
            return camera.is_undistort_prepared() ||
                   (camera.camera_model_type() == lfs::core::CameraModelType::PINHOLE && !camera.has_distortion());
        }

        struct Projection {
            float u, v, z;
        };

        [[nodiscard]] Projection project(const bs::SweepView& view, const float* p) {
            const float* R = view.R;
            const float X = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + view.t[0];
            const float Y = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + view.t[1];
            const float Z = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + view.t[2];
            return {view.fx * X / Z + view.cx, view.fy * Y / Z + view.cy, Z};
        }

        [[nodiscard]] float percentile(std::vector<float>& values, const float q) {
            const auto k = static_cast<size_t>(q * static_cast<float>(values.size() - 1));
            std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k), values.end());
            return values[k];
        }
    } // namespace

    BlobSeeder::BlobSeeder(Tensor reference_points, const size_t max_detection_pixels)
        : _reference_points(reference_points.cpu().contiguous()),
          _max_detection_pixels(max_detection_pixels) {
        assert(_reference_points.ndim() == 2 && _reference_points.shape()[1] == 3);
        assert(_reference_points.dtype() == DataType::Float32);
    }

    bool BlobSeeder::budget_exhausted() const {
        return _captured_bytes >= CAPTURE_BUDGET_BYTES;
    }

    void BlobSeeder::capture(const lfs::core::Camera& camera, const Tensor& image) {
        if (budget_exhausted() || _captured_uids.contains(camera.uid()) || !has_pinhole_image(camera))
            return;
        if (!image.is_valid() || image.device() != Device::GPU || image.ndim() != 3 || image.shape()[0] < 3 ||
            (image.dtype() != DataType::UInt8 && image.dtype() != DataType::Float32))
            return;
        const int source_height = static_cast<int>(image.shape()[1]);
        const int source_width = static_cast<int>(image.shape()[2]);
        if (source_width != camera.image_width() || source_height != camera.image_height())
            return;
        const int factor = std::max(1, static_cast<int>(std::ceil(std::sqrt(
                                           static_cast<double>(source_width) * source_height / _max_detection_pixels))));
        const Tensor rgb = factor == 1 && image.dtype() == DataType::Float32 ? image : bs::downsample_rgb(image, factor, _workspace);
        const int height = static_cast<int>(rgb.shape()[1]);
        const int width = static_cast<int>(rgb.shape()[2]);

        CapturedView view;
        const auto R = camera.R().cpu().contiguous();
        const auto T = camera.T().cpu().contiguous();
        assert(R.numel() == 9 && T.numel() == 3);
        std::copy_n(R.ptr<float>(), 9, view.geometry.R);
        std::copy_n(T.ptr<float>(), 3, view.geometry.t);
        const auto [fx, fy, cx, cy] = camera.get_intrinsics();
        const float inverse_factor = 1.0f / static_cast<float>(factor);
        view.geometry.fx = fx * inverse_factor;
        view.geometry.fy = fy * inverse_factor;
        view.geometry.cx = cx * inverse_factor;
        view.geometry.cy = cy * inverse_factor;
        view.geometry.width = width;
        view.geometry.height = height;
        view.peaks = bs::detect_peaks(rgb, static_cast<int>(_views.size()), _workspace);
        view.geometry.density[0] = view.peaks.density[0];
        view.geometry.density[1] = view.peaks.density[1];
        // Results wait on the host until triangulation, so capturing allocates nothing on the device per view.
        if (view.peaks.peaks.is_valid())
            view.peaks.peaks = view.peaks.peaks.cpu();
        view.peaks.bitmap = view.peaks.bitmap.cpu();
        _captured_bytes += view.peaks.bitmap.bytes() + view.peaks.peaks.bytes();
        _captured_uids.insert(camera.uid());
        _views.push_back(std::move(view));
    }

    BlobSeeds BlobSeeder::triangulate(const std::stop_token stop) const {
        BlobSeeds seeds;
        const size_t view_count = _views.size();
        const size_t reference_count = _reference_points.shape()[0];
        if (view_count < 2 || reference_count < MIN_FRUSTUM_POINTS)
            return seeds;
        const float* reference = _reference_points.ptr<float>();

        // Neighbours share the most reference points inside both frusta and look within 90 degrees.
        const size_t stride = std::max<size_t>(1, reference_count / FRUSTUM_SAMPLE_POINTS);
        const size_t sampled = (reference_count + stride - 1) / stride;
        const size_t words = (sampled + 63) / 64;
        std::vector<uint64_t> visible(view_count * words, 0);
        std::vector<bs::SweepView> geometry(view_count);
        std::vector<size_t> bitmap_offsets(view_count + 1, 0);
        for (size_t vi = 0; vi < view_count; ++vi)
            bitmap_offsets[vi + 1] = bitmap_offsets[vi] + _views[vi].peaks.bitmap.numel();
        std::vector<uint32_t> bitmaps_host(bitmap_offsets.back());
        for (size_t vi = 0; vi < view_count; ++vi)
            std::copy_n(_views[vi].peaks.bitmap.ptr<uint32_t>(), _views[vi].peaks.bitmap.numel(),
                        bitmaps_host.begin() + static_cast<std::ptrdiff_t>(bitmap_offsets[vi]));
        const auto bitmaps = Tensor::from_blob(bitmaps_host.data(), {bitmaps_host.size()}, Device::CPU, DataType::UInt32)
                                 .to(Device::GPU);
        for (size_t vi = 0; vi < view_count; ++vi) {
            auto g = _views[vi].geometry;
            std::vector<float> depths;
            for (size_t s = 0; s < sampled; ++s) {
                const auto p = project(g, reference + s * stride * 3);
                if (p.z > 0.0f && p.u >= 0.0f && p.u < static_cast<float>(g.width) && p.v >= 0.0f &&
                    p.v < static_cast<float>(g.height)) {
                    visible[vi * words + s / 64] |= uint64_t{1} << (s % 64);
                    depths.push_back(p.z);
                }
            }
            if (depths.size() >= MIN_FRUSTUM_POINTS) {
                g.z_min = NEAR_DEPTH_MARGIN * percentile(depths, NEAR_DEPTH_PERCENTILE);
                g.z_max = FAR_DEPTH_MARGIN * percentile(depths, FAR_DEPTH_PERCENTILE);
            }
            g.bitmap_offset = bitmap_offsets[vi];
            geometry[vi] = g;
        }

        const int neighbor_count = static_cast<int>(std::min<size_t>(NEIGHBOR_VIEWS, view_count - 1));
        std::vector<int> neighbors(view_count * neighbor_count, -1);
        for (size_t i = 0; i < view_count; ++i) {
            const float* axis_i = geometry[i].R + 6;
            std::vector<std::pair<int, int>> shared;
            for (size_t j = 0; j < view_count; ++j) {
                const float* axis_j = geometry[j].R + 6;
                if (j == i || axis_i[0] * axis_j[0] + axis_i[1] * axis_j[1] + axis_i[2] * axis_j[2] < 0.0f)
                    continue;
                int overlap = 0;
                for (size_t w = 0; w < words; ++w)
                    overlap += std::popcount(visible[i * words + w] & visible[j * words + w]);
                if (overlap > 0)
                    shared.emplace_back(overlap, static_cast<int>(j));
            }
            const size_t keep = std::min<size_t>(shared.size(), neighbor_count);
            std::partial_sort(shared.begin(), shared.begin() + static_cast<std::ptrdiff_t>(keep), shared.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
            for (size_t k = 0; k < keep; ++k)
                neighbors[i * neighbor_count + k] = shared[k].second;
        }

        std::vector<Tensor> per_view_peaks;
        per_view_peaks.reserve(view_count);
        for (const auto& view : _views)
            if (view.peaks.peaks.numel() > 0)
                per_view_peaks.push_back(view.peaks.peaks);
        if (per_view_peaks.empty())
            return seeds;
        const Tensor peaks_host = Tensor::cat(per_view_peaks, 0);
        const size_t peak_count = peaks_host.shape()[0];
        const auto geometry_device = Tensor::from_blob(geometry.data(),
                                                       {view_count * sizeof(bs::SweepView)}, Device::CPU, DataType::UInt8)
                                         .gpu();
        const auto neighbors_device = Tensor::from_vector(
            neighbors, {view_count, static_cast<size_t>(neighbor_count)}, Device::GPU);
        // Chunks bound the device copies of peaks and sweep results on captures with millions of peaks.
        std::vector<float> sweep_values(peak_count * bs::SweepFieldCount);
        for (size_t first = 0; first < peak_count; first += SWEEP_CHUNK_PEAKS) {
            if (stop.stop_requested())
                return seeds;
            const size_t count = std::min(SWEEP_CHUNK_PEAKS, peak_count - first);
            const auto chunk = peaks_host.slice(0, first, first + count).contiguous().gpu();
            const auto sweep = bs::sweep_peaks(
                                   chunk, geometry_device, neighbors_device, bitmaps)
                                   .cpu();
            std::copy_n(sweep.ptr<float>(), count * bs::SweepFieldCount,
                        sweep_values.begin() + static_cast<std::ptrdiff_t>(first * bs::SweepFieldCount));
        }
        const float* sweep_rows = sweep_values.data();
        const float* peak_rows = peaks_host.ptr<float>();

        struct Detection {
            float position[3];
            float depth;
            float fx;
            size_t peak;
        };
        std::vector<Detection> detections;
        for (size_t p = 0; p < peak_count; ++p) {
            const float* s = sweep_rows + p * bs::SweepFieldCount;
            if (s[bs::SweepAccepted] == 0.0f)
                continue;
            const float* row = peak_rows + p * bs::PeakFieldCount;
            const auto& g = geometry[static_cast<size_t>(row[bs::PeakView])];
            const float z = s[bs::SweepDepth];
            const float dx = (row[bs::PeakX] + 0.5f - g.cx) / g.fx * z;
            const float dy = (row[bs::PeakY] + 0.5f - g.cy) / g.fy * z;
            const float c[3] = {dx - g.t[0], dy - g.t[1], z - g.t[2]};
            Detection d{};
            for (int a = 0; a < 3; ++a)
                d.position[a] = g.R[a] * c[0] + g.R[3 + a] * c[1] + g.R[6 + a] * c[2];
            d.depth = z;
            d.fx = g.fx;
            d.peak = p;
            detections.push_back(d);
        }

        if (stop.stop_requested())
            return seeds;
        PointCloudAdaptor reference_cloud{reference, reference_count};
        KDTree reference_tree(3, reference_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        std::vector<uint8_t> explained(detections.size(), 0);
        for (size_t i = 0; i < detections.size(); ++i) {
            size_t nearest = 0;
            float distance_sq = 0.0f;
            nanoflann::KNNResultSet<float> result(1);
            result.init(&nearest, &distance_sq);
            reference_tree.findNeighbors(result, detections[i].position, nanoflann::SearchParameters(0));
            const float radius = EXPLAINED_RADIUS_PX * detections[i].depth / detections[i].fx;
            explained[i] = distance_sq < radius * radius;
        }
        size_t kept = 0;
        for (size_t i = 0; i < detections.size(); ++i)
            if (!explained[i])
                detections[kept++] = detections[i];
        detections.resize(kept);
        if (detections.empty())
            return seeds;

        std::vector<float> positions(detections.size() * 3);
        for (size_t i = 0; i < detections.size(); ++i)
            std::copy_n(detections[i].position, 3, positions.data() + i * 3);
        PointCloudAdaptor detection_cloud{positions.data(), detections.size()};
        KDTree detection_tree(3, detection_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        std::vector<int> cluster(detections.size(), -1);
        std::vector<std::pair<int, size_t>> representatives; // (members, detection)
        std::vector<nanoflann::ResultItem<uint32_t, float>> nearby;
        for (size_t i = 0; i < detections.size(); ++i) {
            if (cluster[i] >= 0)
                continue;
            const float radius = MERGE_RADIUS_DEPTH_FRACTION * detections[i].depth;
            detection_tree.radiusSearch(detections[i].position, radius * radius, nearby);
            int members = 0;
            for (const auto& item : nearby)
                if (cluster[item.first] < 0) {
                    cluster[item.first] = static_cast<int>(representatives.size());
                    ++members;
                }
            representatives.emplace_back(members, i);
        }
        std::erase_if(representatives, [](const auto& r) { return r.first < MIN_VIEWS_PER_SEED; });
        std::stable_sort(representatives.begin(), representatives.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        seeds.means.reserve(representatives.size() * 3);
        seeds.colors.reserve(representatives.size() * 3);
        seeds.log_scales.reserve(representatives.size());
        for (const auto& [members, index] : representatives) {
            const auto& d = detections[index];
            const float* row = peak_rows + d.peak * bs::PeakFieldCount;
            seeds.means.insert(seeds.means.end(), d.position, d.position + 3);
            seeds.colors.insert(seeds.colors.end(), {row[bs::PeakR], row[bs::PeakG], row[bs::PeakB]});
            seeds.log_scales.push_back(std::log(SEED_RADIUS_PX * d.depth / d.fx));
        }
        LOG_INFO("Blob seeding: {} views, {} peaks, {} triangulated, {} seeds", view_count, peak_count,
                 detections.size(), seeds.size());
        return seeds;
    }

} // namespace lfs::training
