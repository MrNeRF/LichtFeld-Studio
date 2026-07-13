/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "depth_anchor_cache.hpp"

#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "io/atomic_output.hpp"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>

namespace lfs::training {
    namespace {

        constexpr int kSchemaVersion = 1;

        [[nodiscard]] bool anchor_diag_enabled() {
            const char* v = std::getenv("LFS_DEPTH_LOSS_DIAG");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }

        void hash_bytes(std::uint64_t& hash, const void* data, const std::size_t size) {
            const auto* bytes = static_cast<const unsigned char*>(data);
            for (std::size_t i = 0; i < size; ++i) {
                hash ^= bytes[i];
                hash *= 0x100000001b3ull;
            }
        }

        template <typename T>
        void hash_scalar(std::uint64_t& hash, const T& value) {
            hash_bytes(hash, &value, sizeof(T));
        }

        nlohmann::json candidate_to_json(const kernels::DepthAnchorCandidate& c) {
            return {{"valid", c.valid}, {"scale", c.scale}, {"shift", c.shift}, {"corr", c.corr}, {"samples", c.samples}};
        }

        kernels::DepthAnchorCandidate candidate_from_json(const nlohmann::json& j) {
            kernels::DepthAnchorCandidate c;
            c.valid = j.value("valid", false);
            c.scale = j.value("scale", 0.0f);
            c.shift = j.value("shift", 0.0f);
            c.corr = j.value("corr", 0.0f);
            c.samples = j.value("samples", 0);
            return c;
        }

        nlohmann::json anchor_to_json(const kernels::DepthAnchor& a) {
            nlohmann::json j;
            j["valid"] = a.valid;
            j["model"] = a.model;
            j["scale"] = a.scale;
            j["shift"] = a.shift;
            j["floor"] = a.floor;
            j["corr"] = a.corr;
            j["samples"] = a.samples;
            j["disparity"] = candidate_to_json(a.disparity);
            j["depth"] = candidate_to_json(a.depth);
            return j;
        }

        kernels::DepthAnchor anchor_from_json(const nlohmann::json& j) {
            kernels::DepthAnchor a;
            a.valid = j.value("valid", false);
            a.model = j.value("model", 0);
            a.scale = j.value("scale", 0.0f);
            a.shift = j.value("shift", 0.0f);
            a.floor = j.value("floor", 0.0f);
            a.corr = j.value("corr", 0.0f);
            a.samples = j.value("samples", 0);
            if (j.contains("disparity")) {
                a.disparity = candidate_from_json(j["disparity"]);
            }
            if (j.contains("depth")) {
                a.depth = candidate_from_json(j["depth"]);
            }
            return a;
        }

    } // namespace

    RawDepthAnchorMap computeRawDepthAnchors(
        const lfs::core::Tensor& means_in,
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        const int resize_factor,
        const int max_width,
        const DepthAnchorProgress& progress) {

        RawDepthAnchorMap anchors;
        if (!means_in.is_valid() || means_in.ndim() != 2 || means_in.shape()[0] == 0) {
            return anchors;
        }
        const auto means = means_in.device() == lfs::core::Device::CUDA
                               ? means_in
                               : means_in.to(lfs::core::Device::CUDA);
        const auto num_points = static_cast<std::size_t>(means.shape()[0]);

        // Robust world-space bounds of the anchor cloud: sparse reconstructions
        // carry extreme outliers that poison the per-camera fits.
        float aabb_lo[3] = {-std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()};
        float aabb_hi[3] = {std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity()};
        {
            const std::size_t bbox_stride = std::max<std::size_t>(1, num_points / 100000);
            const auto means_cpu = means.cpu().contiguous();
            const float* mp = means_cpu.ptr<float>();
            std::array<std::vector<float>, 3> axis_vals;
            for (auto& v : axis_vals) {
                v.reserve(num_points / bbox_stride + 1);
            }
            for (std::size_t i = 0; i < num_points; i += bbox_stride) {
                for (int k = 0; k < 3; ++k) {
                    const float val = mp[i * 3 + k];
                    if (std::isfinite(val)) {
                        axis_vals[k].push_back(val);
                    }
                }
            }
            for (int k = 0; k < 3; ++k) {
                auto& v = axis_vals[k];
                if (v.size() < 100) {
                    continue;
                }
                const std::size_t lo_idx = v.size() / 50;
                const std::size_t hi_idx = v.size() - 1 - v.size() / 50;
                std::nth_element(v.begin(), v.begin() + lo_idx, v.end());
                const float lo = v[lo_idx];
                std::nth_element(v.begin(), v.begin() + hi_idx, v.end());
                const float hi = v[hi_idx];
                const float margin = 0.5f * std::max(hi - lo, 1e-3f);
                aabb_lo[k] = lo - margin;
                aabb_hi[k] = hi + margin;
            }
        }

        // The per-camera GPU projection (collect) runs serially on this thread to
        // keep all CUDA on one thread; the expensive robust affine fits are pure
        // host work, so they drain on a worker pool in parallel and overlap the
        // next camera's depth decode.
        struct AnchorJob {
            std::string image_name;
            std::vector<float2> samples;
        };

        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        std::deque<AnchorJob> job_queue;
        bool producing = true;

        std::mutex results_mutex;
        const bool anchor_diag = anchor_diag_enabled();

        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const std::size_t worker_count = std::clamp<std::size_t>(
            hw > 1 ? hw - 1 : 1, 1, std::max<std::size_t>(cameras.size(), 1));
        const std::size_t queue_capacity = std::max<std::size_t>(worker_count * 2, 2);

        const auto worker = [&]() {
            while (true) {
                AnchorJob job;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [&] { return !job_queue.empty() || !producing; });
                    if (job_queue.empty()) {
                        return;
                    }
                    job = std::move(job_queue.front());
                    job_queue.pop_front();
                }
                queue_cv.notify_all();

                const auto anchor = lfs::training::kernels::fit_depth_anchor_from_samples(job.samples);
                if (anchor_diag) {
                    LOG_INFO("[DEPTH_ANCHOR] camera='{}' valid={} model={} scale={:.6f} shift={:.6f} floor={:.4f} corr={:.4f} samples={} disp_valid={} disp_corr={:.4f} depth_valid={} depth_corr={:.4f}",
                             job.image_name, anchor.valid ? 1 : 0, anchor.model,
                             anchor.scale, anchor.shift, anchor.floor, anchor.corr, anchor.samples,
                             anchor.disparity.valid ? 1 : 0, anchor.disparity.corr,
                             anchor.depth.valid ? 1 : 0, anchor.depth.corr);
                }
                if (anchor.disparity.valid || anchor.depth.valid) {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    anchors[job.image_name] = anchor;
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t i = 0; i < worker_count; ++i) {
            workers.emplace_back(worker);
        }

        std::size_t total_depth_cameras = 0;
        for (const auto& cam : cameras) {
            if (cam && cam->has_depth()) {
                ++total_depth_cameras;
            }
        }
        std::size_t processed_depth_cameras = 0;

        for (const auto& cam : cameras) {
            if (!cam || !cam->has_depth()) {
                continue;
            }
            if (progress) {
                progress(++processed_depth_cameras, total_depth_cameras);
            }
            auto prior = cam->load_and_get_depth(resize_factor, max_width);
            if (!prior.is_valid() || prior.numel() == 0) {
                continue;
            }
            if (prior.ndim() == 3 && prior.shape()[0] == 1) {
                prior = prior.squeeze(0);
            }
            if (!prior.is_contiguous()) {
                prior = prior.contiguous();
            }
            if (prior.ndim() != 2) {
                cam->release_depth_cache();
                continue;
            }

            const int prior_w = static_cast<int>(prior.shape()[1]);
            const int prior_h = static_cast<int>(prior.shape()[0]);
            const float sx = static_cast<float>(prior_w) / static_cast<float>(cam->camera_width());
            const float sy = static_cast<float>(prior_h) / static_cast<float>(cam->camera_height());

            // The prior's lazy ops materialize on their own stream; the collect
            // kernel reads raw pointers, so settle the device first (startup only).
            prior.ptr<float>();
            cudaDeviceSynchronize();

            auto samples = lfs::training::kernels::collect_depth_anchor_samples(
                means.ptr<float>(),
                num_points,
                cam->world_view_transform_ptr(),
                cam->focal_x() * sx,
                cam->focal_y() * sy,
                cam->center_x() * sx,
                cam->center_y() * sy,
                prior.ptr<float>(),
                prior_w,
                prior_h,
                0.01f,
                aabb_lo,
                aabb_hi);
            cam->release_depth_cache();

            if (samples.empty()) {
                continue;
            }
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [&] { return job_queue.size() < queue_capacity; });
                job_queue.push_back({cam->image_name(), std::move(samples)});
            }
            queue_cv.notify_one();
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            producing = false;
        }
        queue_cv.notify_all();
        for (auto& t : workers) {
            t.join();
        }

        return anchors;
    }

    DepthAnchorFingerprint computeAnchorFingerprint(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {

        DepthAnchorFingerprint fingerprint;
        fingerprint.version = kSchemaVersion;

        std::uint64_t hash = 0xcbf29ce484222325ull;
        hash_scalar(hash, kSchemaVersion);

        std::vector<const lfs::core::Camera*> depth_cameras;
        depth_cameras.reserve(cameras.size());
        for (const auto& cam : cameras) {
            if (cam && cam->has_depth()) {
                depth_cameras.push_back(cam.get());
            }
        }
        std::sort(depth_cameras.begin(), depth_cameras.end(),
                  [](const lfs::core::Camera* a, const lfs::core::Camera* b) {
                      return a->image_name() < b->image_name();
                  });

        for (const auto* cam : depth_cameras) {
            const std::string& name = cam->image_name();
            hash_bytes(hash, name.data(), name.size());
            hash_scalar(hash, cam->focal_x());
            hash_scalar(hash, cam->focal_y());
            hash_scalar(hash, cam->center_x());
            hash_scalar(hash, cam->center_y());
            hash_scalar(hash, cam->camera_width());
            hash_scalar(hash, cam->camera_height());

            const auto pose = cam->world_view_transform().cpu().contiguous();
            if (pose.is_valid() && pose.numel() >= 16) {
                const float* p = pose.ptr<float>();
                for (int i = 0; i < 16; ++i) {
                    hash_scalar(hash, p[i]);
                }
            }

            std::error_code ec;
            const auto size = std::filesystem::file_size(cam->depth_path(), ec);
            hash_scalar(hash, ec ? std::uint64_t{0} : static_cast<std::uint64_t>(size));
            ec.clear();
            const auto mtime = std::filesystem::last_write_time(cam->depth_path(), ec);
            const long long ticks = ec ? 0 : static_cast<long long>(mtime.time_since_epoch().count());
            hash_scalar(hash, ticks);
        }

        fingerprint.hash = hash;
        return fingerprint;
    }

    std::filesystem::path depthAnchorSidecarPath(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        for (const auto& cam : cameras) {
            if (cam && cam->has_depth()) {
                return cam->depth_path().parent_path() / "depth_anchors.json";
            }
        }
        return {};
    }

    bool writeDepthAnchorSidecar(
        const std::filesystem::path& path,
        const RawDepthAnchorMap& anchors,
        const DepthAnchorFingerprint& fingerprint) {
        if (path.empty()) {
            return false;
        }

        nlohmann::json root;
        root["schema_version"] = fingerprint.version;
        root["fingerprint"] = std::format("{:016x}", fingerprint.hash);
        nlohmann::json entries = nlohmann::json::object();
        for (const auto& [name, anchor] : anchors) {
            entries[name] = anchor_to_json(anchor);
        }
        root["anchors"] = std::move(entries);
        const std::string text = root.dump(2);

        if (!lfs::io::ensure_output_parent_directory(path)) {
            return false;
        }
        lfs::io::ScopedAtomicOutputFile output(path);
        {
            std::ofstream file(output.temp_path(), std::ios::binary | std::ios::trunc);
            if (!file) {
                return false;
            }
            file.write(text.data(), static_cast<std::streamsize>(text.size()));
            if (!file) {
                return false;
            }
        }
        return output.commit().has_value();
    }

    std::optional<RawDepthAnchorMap> readDepthAnchorSidecar(
        const std::filesystem::path& path,
        const DepthAnchorFingerprint& expected) {
        std::error_code ec;
        if (path.empty() || !std::filesystem::exists(path, ec)) {
            return std::nullopt;
        }
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return std::nullopt;
        }
        try {
            const nlohmann::json root = nlohmann::json::parse(file);
            if (root.value("schema_version", -1) != expected.version) {
                return std::nullopt;
            }
            if (root.value("fingerprint", std::string{}) != std::format("{:016x}", expected.hash)) {
                return std::nullopt;
            }
            RawDepthAnchorMap anchors;
            if (root.contains("anchors")) {
                for (const auto& [name, value] : root["anchors"].items()) {
                    anchors[name] = anchor_from_json(value);
                }
            }
            return anchors;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

} // namespace lfs::training
