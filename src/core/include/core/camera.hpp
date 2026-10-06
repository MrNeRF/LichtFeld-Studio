/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_types.h"
#include "core/cuda_types.hpp"
#include "core/export.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"
#include "core/uuid.hpp"
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace lfs::core {

    struct CameraCalibration {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        int width = 0;
        int height = 0;
    };

    enum class CameraSplit : uint8_t {
        Train,
        Eval
    };

    class LFS_CORE_API Camera {
    public:
        Camera() = default;

        Camera(const Tensor& R,
               const Tensor& T,
               float focal_x, float focal_y,
               float center_x, float center_y,
               Tensor radial_distortion,
               Tensor tangential_distortion,
               CameraModelType camera_model_type,
               const std::string& image_name,
               const std::filesystem::path& image_path,
               const std::filesystem::path& mask_path,
               int camera_width, int camera_height,
               int uid,
               int camera_id = 0,
               const std::filesystem::path& depth_path = {},
               const std::filesystem::path& normal_path = {});
        Camera(const Camera&, const Tensor& transform);

        // Destructor to clean up CUDA stream
        ~Camera();

        // Delete copy, define proper move semantics
        Camera(const Camera&) = delete;
        Camera& operator=(const Camera&) = delete;
        Camera(Camera&& other) noexcept;
        Camera& operator=(Camera&& other) noexcept;

        // Initialize GPU tensors on demand
        void initialize_cuda_tensors();

        // Load image from disk and return it
        Tensor load_and_get_image(int resize_factor = -1, int max_width = 0, bool output_uint8 = false,
                                  bool update_dimensions = true);

        // Load mask from disk, process it, and return it (cached)
        Tensor load_and_get_mask(int resize_factor = -1, int max_width = 0,
                                 bool invert_mask = false, float mask_threshold = 0.5f, bool binarize = true,
                                 bool apply_undistortion = true);

        // Same processing as load_and_get_mask for an explicit mask file, without caching.
        Tensor load_mask_file(const std::filesystem::path& path, int resize_factor, int max_width,
                              bool invert_mask, float mask_threshold, bool binarize, bool apply_undistortion) const;

        // Load depth map from disk, convert to [H,W] float32 [0,1], and return it (cached)
        Tensor load_and_get_depth(int resize_factor = -1, int max_width = 0);

        // Dataset-level normal-prior decode settings resolved by the trainer's
        // startup probe. Must be consistent across calls (the cache stores the
        // converted map).
        struct NormalPriorDecode {
            bool srgb = false;    // invert the sRGB display transform before decode
            bool flip_yz = false; // OpenGL -> OpenCV camera convention
            bool world_space = false;
            // Prior-world -> reconstruction-world rotation (row-major); the
            // camera's world-to-camera rotation is applied on top.
            std::array<float, 9> world_rotation{1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
        };

        // Load normal map from disk, decode unit normals as [3,H,W]
        // float32 in [-1,1] (file encoding v = n*0.5+0.5), and return it (cached).
        Tensor load_and_get_normal(int resize_factor, int max_width,
                                   const NormalPriorDecode& decode,
                                   bool apply_undistortion = true);

        // Quantization step of the depth prior's file encoding in target units
        // (1/255 for 8-bit, 1/65535 for 16-bit, 0 for float). Header probe on
        // first call, cached.
        float depth_prior_quantization_step();

        void release_depth_cache() {
            _cached_depth = Tensor();
            _depth_loaded = false;
        }

        void release_normal_cache() {
            _cached_normal = Tensor();
            _normal_loaded = false;
        }

        // Attach an in-memory mask (skips file load). Expected (H,W) or
        // (1,H,W) at the image's on-disk resolution; dtype uint8 [0,255] or
        // float32 [0,1]. Lazily processed on the next load_and_get_mask call.
        void set_mask_tensor(Tensor mask);

        // Load image from disk just to populate _image_width/_image_height
        void load_image_size(int resize_factor = -1, int max_width = 0);
        bool image_size_loaded() const noexcept { return _image_size_loaded; }

        // Get number of bytes in the image file
        size_t get_num_bytes_from_file(int resize_factor = -1, int max_width = 0) const;
        size_t get_num_bytes_from_file() const;

        // Accessors - now return const references to avoid copies
        const Tensor& world_view_transform() const {
            return _world_view_transform;
        }
        const Tensor& cam_position() const {
            return _cam_position;
        }

        // Direct GPU pointer access (tensors are already contiguous on CUDA)
        const float* world_view_transform_ptr() const {
            return _world_view_transform.ptr<float>();
        }
        const float* cam_position_ptr() const {
            return _cam_position.ptr<float>();
        }

        const Tensor& R() const { return _R; }
        const Tensor& T() const { return _T; }

        void to_backend(GpuBackend backend);

        Tensor K() const;

        std::tuple<float, float, float, float> get_intrinsics() const;

        int image_height() const noexcept { return _image_height; }
        int image_width() const noexcept { return _image_width; }
        void set_image_dimensions(int width, int height) noexcept {
            _image_width = width;
            _image_height = height;
            _image_size_loaded = true;
        }
        void restore_image_dimensions(int width, int height, bool size_loaded) noexcept {
            _image_width = width;
            _image_height = height;
            _image_size_loaded = size_loaded;
        }
        int camera_height() const noexcept { return _camera_height; }
        int camera_width() const noexcept { return _camera_width; }
        float focal_x() const noexcept { return _focal_x; }
        float focal_y() const noexcept { return _focal_y; }
        float center_x() const noexcept { return _center_x; }
        float center_y() const noexcept { return _center_y; }
        Tensor radial_distortion() const noexcept { return _radial_distortion; }
        Tensor tangential_distortion() const noexcept { return _tangential_distortion; }
        CameraModelType camera_model_type() const noexcept { return _camera_model_type; }
        const std::string& image_name() const noexcept { return _image_name; }
        const std::filesystem::path& image_path() const noexcept { return _image_path; }
        const std::filesystem::path& mask_path() const noexcept { return _mask_path; }
        const std::filesystem::path& depth_path() const noexcept { return _depth_path; }
        const std::filesystem::path& normal_path() const noexcept { return _normal_path; }
        void set_normal_path(std::filesystem::path path);

        // Sparse SfM observations for this image (COLMAP 2D points with a 3D id).
        // Pixel coordinates are in the camera's native width/height at load
        // (images_N scale already applied). World xyz is reconstruction space
        // and follows Camera::translate.
        struct SfmObservation {
            float u = 0.0f;
            float v = 0.0f;
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
        };

        // Observations stored in a project and decoded on first use.
        class SfmObservationSource {
        public:
            virtual ~SfmObservationSource() = default;
            [[nodiscard]] virtual const std::vector<SfmObservation>& observations(
                const Uuid& camera_node) const = 0;
        };

        void set_sfm_observations(std::vector<SfmObservation> observations) {
            _sfm_observations = std::move(observations);
            _sfm_observation_source.reset();
        }
        void set_sfm_observation_source(std::shared_ptr<const SfmObservationSource> source,
                                        const Uuid& camera_node, std::size_t count) {
            _sfm_observations.clear();
            _sfm_observation_source = std::move(source);
            _sfm_observation_node = camera_node;
            _sfm_observation_count = count;
        }
        [[nodiscard]] const std::vector<SfmObservation>& sfm_observations() const {
            return _sfm_observation_source ? _sfm_observation_source->observations(_sfm_observation_node)
                                           : _sfm_observations;
        }
        [[nodiscard]] std::size_t sfm_observation_count() const noexcept {
            return _sfm_observation_source ? _sfm_observation_count : _sfm_observations.size();
        }
        [[nodiscard]] const SfmObservationSource* sfm_observation_source() const noexcept {
            return _sfm_observation_source.get();
        }

        // Rewrites image/mask/depth/normal paths that live under old_root to the same
        // relative location under new_root. Paths outside old_root are left unchanged.
        // Sidecar caches and in-memory masks (set_mask_tensor) are not cleared: a
        // dataset relocation keeps the same files, and injected masks do not originate
        // from _mask_path.
        void rebase_asset_paths(const std::filesystem::path& old_root,
                                const std::filesystem::path& new_root);
        bool has_in_memory_mask() const noexcept { return _in_memory_mask_raw.is_valid(); }
        bool has_mask() const noexcept {
            return has_in_memory_mask() ||
                   (!_mask_path.empty() && std::filesystem::exists(_mask_path));
        }
        bool has_depth() const noexcept {
            return !_depth_path.empty() && std::filesystem::exists(_depth_path);
        }
        bool has_normal() const noexcept {
            return !_normal_path.empty() && std::filesystem::exists(_normal_path);
        }
        bool has_alpha() const noexcept { return _has_alpha; }
        void set_has_alpha(bool v) noexcept { _has_alpha = v; }
        // Set at dataset load when the bound image file is absent. Independent of
        // scene-graph training_enabled (user disable). Dummy/test cameras keep the
        // default true even if image_path does not exist.
        bool has_image() const noexcept { return _has_image; }
        void set_has_image(bool v) noexcept { _has_image = v; }
        CameraSplit split() const noexcept { return _split; }
        void set_split(const CameraSplit split) noexcept {
            assert((split == CameraSplit::Train || split == CameraSplit::Eval) && "Camera split must be Train or Eval");
            _split = split;
        }
        int uid() const noexcept { return _uid; }
        int camera_id() const noexcept { return _camera_id; }

        float FoVx() const noexcept { return _FoVx; }
        float FoVy() const noexcept { return _FoVy; }

        // Translate the camera by trans in world space (cam_pos += trans, T updated accordingly)
        void translate(const Tensor& trans);

        void precompute_undistortion(float blank_pixels = 0.0f);
        // Install parameters computed for an equivalent camera record without
        // repeating the distortion solve or its device-to-host copies.
        void adopt_undistortion(const UndistortParams& params) noexcept;
        std::uint64_t calibration_revision() const noexcept { return _calibration_revision.load(); }
        bool is_undistort_precomputed() const noexcept { return _undistort_precomputed; }
        void prepare_undistortion(float blank_pixels = 0.0f);
        void restore_undistortion_state(
            const CameraCalibration& source,
            const CameraCalibration& destination,
            bool prepared,
            bool crop_solve_failed = false);
        bool is_undistort_prepared() const noexcept { return _undistort_prepared; }
        bool has_distortion() const noexcept;
        const UndistortParams& undistort_params() const noexcept { return _undistort_params; }

    private:
        // IDs
        float _FoVx = 0.f;
        float _FoVy = 0.f;
        int _uid = -1;
        int _camera_id = 0;
        float _focal_x = 0.f;
        float _focal_y = 0.f;
        float _center_x = 0.f;
        float _center_y = 0.f;

        // redundancy with _world_view_transform, but save calculation and passing from GPU 2 CPU
        Tensor _R;
        Tensor _T;

        Tensor _radial_distortion;
        Tensor _tangential_distortion;
        CameraModelType _camera_model_type = CameraModelType::PINHOLE;

        // Image info
        std::string _image_name;
        std::filesystem::path _image_path;
        Tensor read_mask_image(const std::filesystem::path& path, int resize_factor, int max_width,
                               bool apply_undistortion) const;
        Tensor finish_mask(Tensor mask, int resize_factor, int max_width, bool invert_mask, float mask_threshold,
                           bool binarize, bool apply_undistortion) const;
        std::filesystem::path _mask_path;
        std::filesystem::path _depth_path;
        std::filesystem::path _normal_path;
        bool _has_alpha = false;
        bool _has_image = true;
        CameraSplit _split = CameraSplit::Train;
        int _camera_width = 0;
        int _camera_height = 0;
        int _image_width = 0;
        int _image_height = 0;
        bool _image_size_loaded = false;

        // GPU tensors (computed on demand)
        Tensor _world_view_transform;
        Tensor _cam_position;

        // Mask caching (processed mask stored on GPU). Keyed on the processing
        // arguments: a binarized 0.5 load must not poison a later SegmentAndIgnore
        // keep-band load (or the reverse).
        Tensor _cached_mask;
        bool _mask_loaded = false;
        int _cached_mask_resize_factor = 0;
        int _cached_mask_max_width = 0;
        bool _cached_mask_invert = false;
        float _cached_mask_threshold = 0.5f;
        bool _cached_mask_binarize = true;
        bool _cached_mask_was_undistorted = false;
        // Raw, pre-supplied in-memory mask (used by direct-scene plugins) —
        // takes precedence over _mask_path when set. Processed on first use.
        Tensor _in_memory_mask_raw;

        // Depth caching (processed depth stored on GPU)
        Tensor _cached_depth;
        bool _depth_loaded = false;
        float _depth_quantization_step = -1.0f;

        // Normal caching (decoded camera-space normals stored on GPU)
        Tensor _cached_normal;
        bool _normal_loaded = false;

        // Undistortion state
        bool _undistort_precomputed = false;
        bool _undistort_prepared = false;
        UndistortParams _undistort_params{};
        std::atomic<std::uint64_t> _calibration_revision{0};

        // CUDA stream for async operations
        cudaStream_t _stream = nullptr;

        std::vector<SfmObservation> _sfm_observations;
        std::shared_ptr<const SfmObservationSource> _sfm_observation_source;
        Uuid _sfm_observation_node{};
        std::size_t _sfm_observation_count = 0;
    };
    inline float focal2fov(float focal, int pixels) {
        return 2.0f * std::atan(pixels / (2.0f * focal));
    }

    inline float fov2focal(float fov, int pixels) {
        float tan_fov = std::tan(fov * 0.5f);
        return pixels / (2.0f * tan_fov);
    }

} // namespace lfs::core
