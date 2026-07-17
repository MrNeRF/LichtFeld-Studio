/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cuda_runtime.h>

namespace lfs::core {
    static Tensor world_to_view(const Tensor& R, const Tensor& t) {
        // Create 4x4 identity matrix
        auto w2c = Tensor::eye(4, R.device());

        // Set rotation and translation parts
        auto w2c_cpu = w2c.cpu();
        auto R_cpu = R.cpu();
        auto t_cpu = t.cpu();

        auto w2c_acc = w2c_cpu.accessor<float, 2>();
        auto R_acc = R_cpu.accessor<float, 2>();
        auto t_acc = t_cpu.accessor<float, 1>();

        // Copy rotation [0:3, 0:3] = R
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                w2c_acc(i, j) = R_acc(i, j);
            }
        }

        // Copy translation [0:3, 3] = t
        for (size_t i = 0; i < 3; ++i) {
            w2c_acc(i, 3) = t_acc(i);
        }

        // Return as [1, 4, 4] on CUDA
        return w2c_cpu.to(Device::CUDA).unsqueeze(0).contiguous();
    }

    static std::array<float, 9> camera_rotation_to_cpu_array(const Tensor& R) {
        std::array<float, 9> result{};
        auto R_cpu = R.cpu().contiguous();
        auto R_acc = R_cpu.accessor<float, 2>();
        for (size_t row = 0; row < 3; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                result[row * 3 + col] = R_acc(row, col);
            }
        }
        return result;
    }

    Camera::Camera(const Tensor& R,
                   const Tensor& T,
                   float focal_x, float focal_y,
                   float center_x, float center_y,
                   const Tensor radial_distortion,
                   const Tensor tangential_distortion,
                   CameraModelType camera_model_type,
                   const std::string& image_name,
                   const std::filesystem::path& image_path,
                   const std::filesystem::path& mask_path,
                   int camera_width, int camera_height,
                   int uid,
                   int camera_id,
                   const std::filesystem::path& depth_path,
                   const std::filesystem::path& normal_path)
        : _uid(uid),
          _camera_id(camera_id),
          _focal_x(focal_x),
          _focal_y(focal_y),
          _center_x(center_x),
          _center_y(center_y),
          _R(R),
          _T(T),
          _radial_distortion(radial_distortion),
          _tangential_distortion(tangential_distortion),
          _camera_model_type(camera_model_type),
          _image_name(image_name),
          _image_path(image_path),
          _mask_path(mask_path),
          _depth_path(depth_path),
          _normal_path(normal_path),
          _camera_width(camera_width),
          _camera_height(camera_height),
          _image_width(camera_width),
          _image_height(camera_height) {

        // Validate inputs
        if (!R.is_valid() || R.numel() == 0) {
            LOG_ERROR("Camera constructor: R tensor is invalid or empty");
            throw std::runtime_error("Camera constructor: R tensor is invalid or empty");
        }
        if (!T.is_valid() || T.numel() == 0) {
            LOG_ERROR("Camera constructor: T tensor is invalid or empty");
            throw std::runtime_error("Camera constructor: T tensor is invalid or empty");
        }

        // Compute world-to-view transform
        _world_view_transform = world_to_view(R, T);

        // Compute camera position: inverse of w2v gives c2w, position is c2w[:3, 3]
        // For transformation matrix [R|t], inverse is [R^T | -R^T*t]
        auto w2v_cpu = _world_view_transform.squeeze(0).cpu();
        auto w2v_acc = w2v_cpu.accessor<float, 2>();

        // Extract 3x3 rotation part
        std::vector<float> rot_data(9);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                rot_data[i * 3 + j] = w2v_acc(i, j);
            }
        }
        auto R_part = Tensor::from_vector(rot_data, TensorShape({3, 3}), Device::CPU);

        // Extract translation part
        std::vector<float> t_data = {w2v_acc(0, 3), w2v_acc(1, 3), w2v_acc(2, 3)};
        auto t_part = Tensor::from_vector(t_data, TensorShape({3}), Device::CPU);

        // Compute camera position: -R^T * t
        auto R_T = R_part.transpose(0, 1);
        auto cam_pos = R_T.matmul(t_part.unsqueeze(1)).squeeze(1).neg();

        _cam_position = cam_pos.to(Device::CUDA).contiguous();

        _FoVx = focal2fov(_focal_x, _camera_width);
        _FoVy = focal2fov(_focal_y, _camera_height);

        // Non-blocking so image loading doesn't serialize with the legacy stream.
        // On failure fall back to the default stream rather than a bad handle.
        // log_cuda_teardown_failure is the frozen no-throw log-and-continue adapter.
        const auto stream_create_site = LFS_SOURCE_SITE_CURRENT();
        ::lfs::core::record_cuda_breadcrumb(
            "cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking)",
            __FILE__, __LINE__, nullptr);
        const auto stream_create_state = ::lfs::core::prepare_cuda_check(
            "cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking)",
            stream_create_site, nullptr);
        const cudaError_t stream_create_status =
            cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
        const auto stream_create_completion =
            ::lfs::core::complete_cuda_check(stream_create_status, stream_create_state);
        if (stream_create_completion.effective_error != cudaSuccess) {
            ::lfs::core::log_cuda_teardown_failure(
                stream_create_status, stream_create_state, stream_create_completion,
                "cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking)",
                "camera stream creation, falling back to default stream",
                stream_create_site);
            _stream = nullptr;
        }
    }

    Camera::~Camera() {
        // Destroy CUDA stream if it was created
        if (_stream) {
            CudaMemoryPool::instance().release_stream(_stream);
            LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(_stream), _stream, "camera stream teardown");
            _stream = nullptr;
        }
    }

    Camera::Camera(Camera&& other) noexcept
        : _FoVx(other._FoVx),
          _FoVy(other._FoVy),
          _uid(other._uid),
          _camera_id(other._camera_id),
          _focal_x(other._focal_x),
          _focal_y(other._focal_y),
          _center_x(other._center_x),
          _center_y(other._center_y),
          _R(std::move(other._R)),
          _T(std::move(other._T)),
          _radial_distortion(std::move(other._radial_distortion)),
          _tangential_distortion(std::move(other._tangential_distortion)),
          _image_path(std::move(other._image_path)),
          _image_name(std::move(other._image_name)),
          _mask_path(std::move(other._mask_path)),
          _depth_path(std::move(other._depth_path)),
          _normal_path(std::move(other._normal_path)),
          _split(other._split),
          _camera_width(other._camera_width),
          _camera_height(other._camera_height),
          _image_width(other._image_width),
          _image_height(other._image_height),
          _image_size_loaded(other._image_size_loaded),
          _world_view_transform(std::move(other._world_view_transform)),
          _cam_position(std::move(other._cam_position)),
          _cached_mask(std::move(other._cached_mask)),
          _mask_loaded(other._mask_loaded),
          _in_memory_mask_raw(std::move(other._in_memory_mask_raw)),
          _cached_depth(std::move(other._cached_depth)),
          _depth_loaded(other._depth_loaded),
          _depth_quantization_step(other._depth_quantization_step),
          _cached_normal(std::move(other._cached_normal)),
          _normal_loaded(other._normal_loaded),
          _undistort_precomputed(other._undistort_precomputed),
          _undistort_prepared(other._undistort_prepared),
          _undistort_params(other._undistort_params),
          _stream(other._stream) {
        // Take ownership of the stream
        other._stream = nullptr;
        other._mask_loaded = false;
        other._depth_loaded = false;
        other._normal_loaded = false;
        other._undistort_precomputed = false;
        other._undistort_prepared = false;
    }

    Camera& Camera::operator=(Camera&& other) noexcept {
        if (this != &other) {
            // Destroy our current stream
            if (_stream) {
                CudaMemoryPool::instance().release_stream(_stream);
                LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(_stream), _stream, "camera stream teardown");
            }

            // Move all members
            _FoVx = other._FoVx;
            _FoVy = other._FoVy;
            _uid = other._uid;
            _camera_id = other._camera_id;
            _focal_x = other._focal_x;
            _focal_y = other._focal_y;
            _center_x = other._center_x;
            _center_y = other._center_y;
            _R = std::move(other._R);
            _T = std::move(other._T);
            _radial_distortion = std::move(other._radial_distortion);
            _tangential_distortion = std::move(other._tangential_distortion);
            _image_path = std::move(other._image_path);
            _image_name = std::move(other._image_name);
            _mask_path = std::move(other._mask_path);
            _depth_path = std::move(other._depth_path);
            _normal_path = std::move(other._normal_path);
            _split = other._split;
            _camera_width = other._camera_width;
            _camera_height = other._camera_height;
            _image_width = other._image_width;
            _image_height = other._image_height;
            _image_size_loaded = other._image_size_loaded;
            _world_view_transform = std::move(other._world_view_transform);
            _cam_position = std::move(other._cam_position);
            _cached_mask = std::move(other._cached_mask);
            _mask_loaded = other._mask_loaded;
            _in_memory_mask_raw = std::move(other._in_memory_mask_raw);
            _cached_depth = std::move(other._cached_depth);
            _depth_loaded = other._depth_loaded;
            _depth_quantization_step = other._depth_quantization_step;
            _cached_normal = std::move(other._cached_normal);
            _normal_loaded = other._normal_loaded;
            _undistort_precomputed = other._undistort_precomputed;
            _undistort_prepared = other._undistort_prepared;
            _undistort_params = other._undistort_params;

            // Take ownership of the stream
            _stream = other._stream;
            other._stream = nullptr;
            other._mask_loaded = false;
            other._depth_loaded = false;
            other._normal_loaded = false;
            other._undistort_precomputed = false;
            other._undistort_prepared = false;
        }
        return *this;
    }

    Camera::Camera(const Camera& other, const Tensor& transform)
        : _uid(other._uid),
          _camera_id(other._camera_id),
          _focal_x(other._focal_x),
          _focal_y(other._focal_y),
          _center_x(other._center_x),
          _center_y(other._center_y),
          _R(other._R),
          _T(other._T),
          _radial_distortion(other._radial_distortion),
          _tangential_distortion(other._tangential_distortion),
          _camera_model_type(other._camera_model_type),
          _image_name(other._image_name),
          _image_path(other._image_path),
          _mask_path(other._mask_path),
          _depth_path(other._depth_path),
          _normal_path(other._normal_path),
          _split(other._split),
          _camera_width(other._camera_width),
          _camera_height(other._camera_height),
          _image_width(other._image_width),
          _image_height(other._image_height),
          _image_size_loaded(other._image_size_loaded),
          _cam_position(other._cam_position),
          _FoVx(other._FoVx),
          _FoVy(other._FoVy) {
        _world_view_transform = transform;

        // Non-blocking so image loading doesn't serialize with the legacy stream.
        // On failure fall back to the default stream rather than a bad handle.
        // log_cuda_teardown_failure is the frozen no-throw log-and-continue adapter.
        const auto stream_create_site = LFS_SOURCE_SITE_CURRENT();
        ::lfs::core::record_cuda_breadcrumb(
            "cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking)",
            __FILE__, __LINE__, nullptr);
        const auto stream_create_state = ::lfs::core::prepare_cuda_check(
            "cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking)",
            stream_create_site, nullptr);
        const cudaError_t stream_create_status =
            cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
        const auto stream_create_completion =
            ::lfs::core::complete_cuda_check(stream_create_status, stream_create_state);
        if (stream_create_completion.effective_error != cudaSuccess) {
            ::lfs::core::log_cuda_teardown_failure(
                stream_create_status, stream_create_state, stream_create_completion,
                "cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking)",
                "camera stream creation, falling back to default stream",
                stream_create_site);
            _stream = nullptr;
        }
    }
    Tensor Camera::K() const {
        // Create [1, 3, 3] zero matrix on same device as world_view_transform
        auto K = Tensor::zeros({1, 3, 3}, _world_view_transform.device());
        auto [fx, fy, cx, cy] = get_intrinsics();

        // Fill in the intrinsic matrix on CPU then move to device
        auto K_cpu = K.cpu();
        auto K_acc = K_cpu.accessor<float, 3>();
        K_acc(0, 0, 0) = fx;
        K_acc(0, 1, 1) = fy;
        K_acc(0, 0, 2) = cx;
        K_acc(0, 1, 2) = cy;
        K_acc(0, 2, 2) = 1.0f;

        return K_cpu.to(_world_view_transform.device()).contiguous();
    }

    std::tuple<float, float, float, float> Camera::get_intrinsics() const {
        const float x_scale = static_cast<float>(_image_width) / static_cast<float>(_camera_width);
        const float y_scale = static_cast<float>(_image_height) / static_cast<float>(_camera_height);
        return {_focal_x * x_scale, _focal_y * y_scale, _center_x * x_scale, _center_y * y_scale};
    }

    Tensor Camera::load_and_get_image(int resize_factor, int max_width, const bool output_uint8,
                                      const bool update_dimensions) {
        const ImageLoadParams params{
            .path = _image_path,
            .resize_factor = resize_factor,
            .max_width = max_width,
            .stream = _stream,
            .output_uint8 = output_uint8};

        auto image = load_image_cached(params);

        if (update_dimensions) {
            const auto shape = image.shape();
            _image_width = shape[2];
            _image_height = shape[1];
            _image_size_loaded = true;
        }

        if (image.device() != Device::CUDA) {
            image = image.to(Device::CUDA, _stream);
            if (_stream) {
                LFS_CUDA_TRY(cudaStreamSynchronize(_stream), _stream, "image upload sync");
            }
        }

        return image;
    }

    void Camera::load_image_size(int resize_factor, int max_width) {
        int w, h;
        if (_undistort_prepared) {
            w = _camera_width;
            h = _camera_height;
        } else {
            auto result = get_image_info(_image_path);
            w = std::get<0>(result);
            h = std::get<1>(result);
        }

        LOG_DEBUG("load_image_size(): Base dimensions: {}x{}, resize_factor={}, max_width={}",
                  w, h, resize_factor, max_width);

        if (resize_factor > 0) {
            if (w % resize_factor || h % resize_factor) {
                LOG_WARN("width or height are not divisible by resize_factor w {} h {} resize_factor {}", w, h, resize_factor);
            }
            _image_width = w / resize_factor;
            _image_height = h / resize_factor;
        } else {
            _image_width = w;
            _image_height = h;
        }

        LOG_DEBUG("load_image_size(): After resize_factor: {}x{}", _image_width, _image_height);

        if (max_width > 0 && (_image_width > max_width || _image_height > max_width)) {
            int old_width = _image_width;
            int old_height = _image_height;

            if (_image_width > _image_height) {
                _image_width = max_width;
                _image_height = (old_height * max_width) / old_width; // Fixed: Use old_width
                LOG_DEBUG("load_image_size(): Resized {}x{} → {}x{} (limited by max_width={})",
                          old_width, old_height, _image_width, _image_height, max_width);
            } else {
                _image_height = max_width;
                _image_width = (old_width * max_width) / old_height; // Fixed: Use old_height
                LOG_DEBUG("load_image_size(): Resized {}x{} → {}x{} (limited by max_width={})",
                          old_width, old_height, _image_width, _image_height, max_width);
            }
        }

        LOG_DEBUG("load_image_size(): Final dimensions: {}x{}", _image_width, _image_height);
        _image_size_loaded = true;
    }

    size_t Camera::get_num_bytes_from_file(int resize_factor, int max_width) const {
        auto result = get_image_info(_image_path);

        int w = std::get<0>(result);
        int h = std::get<1>(result);
        int c = std::get<2>(result);

        if (resize_factor > 0) {
            w = w / resize_factor;
            h = h / resize_factor;
        }

        if (max_width > 0 && (w > max_width || h > max_width)) {
            if (w > h) {
                h = (h * max_width) / w;
                w = max_width;
            } else {
                w = (w * max_width) / h;
                h = max_width;
            }
        }

        size_t num_bytes = w * h * c * sizeof(uint8_t);
        return num_bytes;
    }

    size_t Camera::get_num_bytes_from_file() const {
        auto [w, h, c] = get_image_info(_image_path);
        size_t num_bytes = w * h * c * sizeof(uint8_t);
        return num_bytes;
    }

    void Camera::set_mask_tensor(Tensor mask) {
        _in_memory_mask_raw = std::move(mask);
        // Force reprocessing on the next load_and_get_mask call.
        _cached_mask = Tensor();
        _mask_loaded = false;
    }

    Tensor Camera::load_and_get_mask(const int resize_factor, const int max_width,
                                     const bool invert_mask, const float mask_threshold,
                                     const bool binarize) {
        if (_mask_loaded && _cached_mask.is_valid()) {
            return _cached_mask;
        }

        Tensor mask;
        if (_in_memory_mask_raw.is_valid()) {
            // In-memory mask (set via set_mask_tensor). The plugin supplies it
            // at the image's on-disk resolution; further resize_factor /
            // max_width handling is the trainer's responsibility upstream.
            mask = _in_memory_mask_raw;
            if (mask.device() != Device::CUDA) {
                mask = mask.to(Device::CUDA, _stream);
                if (_stream) {
                    LFS_CUDA_TRY(cudaStreamSynchronize(_stream), _stream, "mask upload sync");
                }
            }
            if (mask.dtype() == DataType::UInt8) {
                mask = mask.to(DataType::Float32).div(255.0f);
            } else if (mask.dtype() != DataType::Float32) {
                mask = mask.to(DataType::Float32);
            }
            // Allow (1, H, W) or (H, W, 1) shapes by squeezing the singleton.
            if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                mask = mask.squeeze(0);
            } else if (mask.ndim() == 3 && mask.shape()[2] == 1) {
                mask = mask.squeeze(2);
            }
        } else if (!_mask_path.empty() && std::filesystem::exists(_mask_path)) {
            const ImageLoadParams params{
                .path = _mask_path,
                .resize_factor = resize_factor,
                .max_width = max_width,
                .stream = _stream};

            mask = load_image_cached(params);

            if (mask.device() != Device::CUDA) {
                mask = mask.to(Device::CUDA, _stream);
                if (_stream) {
                    LFS_CUDA_TRY(cudaStreamSynchronize(_stream), _stream, "mask upload sync");
                }
            }

            // Convert RGB [C,H,W] to grayscale [H,W]
            if (mask.ndim() == 3 && mask.shape()[0] >= 3) {
                const auto r = mask.slice(0, 0, 1).squeeze(0);
                const auto g = mask.slice(0, 1, 2).squeeze(0);
                const auto b = mask.slice(0, 2, 3).squeeze(0);
                mask = (r + g + b) / 3.0f;
            } else if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                mask = mask.squeeze(0);
            }
        } else {
            return Tensor();
        }

        if (invert_mask) {
            mask = Tensor::full(mask.shape(), 1.0f, mask.device()) - mask;
        }

        // Threshold before undistort; final binarization happens after geometric resampling.
        if (binarize && mask_threshold > 0.0f && mask_threshold < 1.0f) {
            mask = mask.ge(mask_threshold).to(DataType::Float32);
        }

        if (_undistort_prepared) {
            const auto scaled = scale_undistort_params(
                _undistort_params,
                static_cast<int>(mask.shape()[1]),
                static_cast<int>(mask.shape()[0]));
            mask = undistort_mask(mask, scaled, _stream);
        }

        if (binarize) {
            mask = mask.ge(0.5f).to(DataType::UInt8).contiguous();
        } else {
            mask = (mask * 255.f).to(DataType::UInt8).contiguous();
        }
        _cached_mask = mask;
        _mask_loaded = true;

        LOG_DEBUG("Loaded mask for {}: [{},{}]", _image_name, mask.shape()[0], mask.shape()[1]);

        return _cached_mask;
    }

    Tensor Camera::load_and_get_depth(const int resize_factor, const int max_width) {
        if (_depth_loaded && _cached_depth.is_valid()) {
            return _cached_depth;
        }

        if (_depth_path.empty() || !std::filesystem::exists(_depth_path)) {
            return Tensor();
        }

        Tensor depth;
        if (auto [gray, native_w, native_h] = load_image_gray_high_bitdepth(_depth_path); gray) {
            auto cpu_depth = Tensor::from_blob(
                gray,
                TensorShape({static_cast<size_t>(native_h), static_cast<size_t>(native_w)}),
                Device::CPU, DataType::Float32);
            depth = cpu_depth.to(Device::CUDA, _stream);
            if (_stream) {
                LFS_CUDA_TRY(cudaStreamSynchronize(_stream), _stream, "depth upload sync");
            }
            free_image_float(gray);

            int target_w = native_w;
            int target_h = native_h;
            if (resize_factor > 1) {
                target_w /= resize_factor;
                target_h /= resize_factor;
            }
            if (max_width > 0 && (target_w > max_width || target_h > max_width)) {
                if (target_w > target_h) {
                    target_h = std::max(1, max_width * target_h / target_w);
                    target_w = max_width;
                } else {
                    target_w = std::max(1, max_width * target_w / target_h);
                    target_h = max_width;
                }
            }
            if (target_w != native_w || target_h != native_h) {
                depth = lanczos_resize_grayscale(depth, target_h, target_w, 2, _stream);
                if (_stream) {
                    LFS_CUDA_TRY(cudaStreamSynchronize(_stream), _stream, "depth resize sync");
                }
            }
        } else {
            const ImageLoadParams params{
                .path = _depth_path,
                .resize_factor = resize_factor,
                .max_width = max_width,
                .stream = _stream};

            depth = load_image_cached(params);

            if (depth.device() != Device::CUDA) {
                depth = depth.to(Device::CUDA, _stream);
                if (_stream) {
                    LFS_CUDA_TRY(cudaStreamSynchronize(_stream), _stream, "depth upload sync");
                }
            }

            if (depth.dtype() == DataType::UInt8) {
                depth = depth.to(DataType::Float32).div(255.0f);
            } else if (depth.dtype() != DataType::Float32) {
                depth = depth.to(DataType::Float32);
            }
        }

        // Convert RGB [C,H,W] to grayscale [H,W].
        if (depth.ndim() == 3 && depth.shape()[0] >= 3) {
            const auto r = depth.slice(0, 0, 1).squeeze(0);
            const auto g = depth.slice(0, 1, 2).squeeze(0);
            const auto b = depth.slice(0, 2, 3).squeeze(0);
            depth = (r + g + b) / 3.0f;
        } else if (depth.ndim() == 3 && depth.shape()[0] == 1) {
            depth = depth.squeeze(0);
        } else if (depth.ndim() == 3 && depth.shape()[2] == 1) {
            depth = depth.squeeze(2);
        }

        if (_undistort_prepared) {
            const auto scaled = scale_undistort_params(
                _undistort_params,
                static_cast<int>(depth.shape()[1]),
                static_cast<int>(depth.shape()[0]));
            depth = undistort_mask(depth, scaled, _stream);
        }

        _cached_depth = depth.contiguous();
        _depth_loaded = true;

        LOG_DEBUG("Loaded depth for {}: [{},{}]", _image_name, _cached_depth.shape()[0], _cached_depth.shape()[1]);

        return _cached_depth;
    }

    Tensor Camera::load_and_get_normal(const int resize_factor, const int max_width,
                                       const NormalPriorDecode& decode) {
        if (_normal_loaded && _cached_normal.is_valid()) {
            return _cached_normal;
        }

        if (_normal_path.empty() || !std::filesystem::exists(_normal_path)) {
            return Tensor();
        }

        Tensor normal;
        if (auto [rgb, native_w, native_h] = load_image_rgb_high_bitdepth(_normal_path); rgb) {
            auto cpu_normal = Tensor::from_blob(
                rgb,
                TensorShape({static_cast<size_t>(native_h), static_cast<size_t>(native_w), 3}),
                Device::CPU, DataType::Float32);
            normal = cpu_normal.to(Device::CUDA, _stream);
            if (_stream) {
                LFS_CUDA_TRY(cudaStreamSynchronize(_stream), _stream, "normal upload sync");
            }
            free_image_float(rgb);
            normal = normal.permute({2, 0, 1}).contiguous();
        } else {
            const ImageLoadParams params{
                .path = _normal_path,
                .stream = _stream};

            normal = load_image_cached(params);

            if (normal.is_valid() && normal.device() != Device::CUDA) {
                normal = normal.to(Device::CUDA, _stream);
                if (_stream) {
                    LFS_CUDA_TRY(cudaStreamSynchronize(_stream), _stream, "normal upload sync");
                }
            }
            if (normal.is_valid()) {
                if (normal.dtype() == DataType::UInt8) {
                    normal = normal.to(DataType::Float32).div(255.0f);
                } else if (normal.dtype() != DataType::Float32) {
                    normal = normal.to(DataType::Float32);
                }
            }
        }

        if (!normal.is_valid() || normal.ndim() != 3) {
            LOG_WARN("Normal map for {} could not be loaded from {}", _image_name,
                     lfs::core::path_to_utf8(_normal_path));
            return Tensor();
        }
        if (normal.shape()[0] != 3) {
            if (normal.shape()[0] > 3) {
                normal = normal.slice(0, 0, 3).contiguous();
            } else if (normal.shape()[2] == 3) {
                normal = normal.permute({2, 0, 1}).contiguous();
            } else {
                LOG_WARN("Normal map for {} has {} channels, expected 3", _image_name, normal.shape()[0]);
                return Tensor();
            }
        }

        // Decode the v = n*0.5 + 0.5 file encoding; the loss re-normalizes per
        // pixel, so quantization/resampling shrinkage is harmless here.
        normal = normal.mul(2.0f).sub(1.0f);

        if (decode.srgb || decode.flip_yz || decode.world_space) {
            auto normal_cpu = normal.cpu().contiguous();
            const size_t pixel_count = normal_cpu.shape()[1] * normal_cpu.shape()[2];
            if (decode.srgb) {
                srgb_normal_prior_to_linear_chw(normal_cpu.ptr<float>(), pixel_count * 3);
            }
            if (decode.flip_yz) {
                flip_normal_prior_yz_chw(normal_cpu.ptr<float>(), pixel_count);
            }
            if (decode.world_space) {
                const std::array<float, 9> r_w2c = camera_rotation_to_cpu_array(_R);
                std::array<float, 9> world_to_camera{};
                for (size_t row = 0; row < 3; ++row) {
                    for (size_t col = 0; col < 3; ++col) {
                        float sum = 0.0f;
                        for (size_t k = 0; k < 3; ++k) {
                            sum += r_w2c[row * 3 + k] * decode.world_rotation[k * 3 + col];
                        }
                        world_to_camera[row * 3 + col] = sum;
                    }
                }
                transform_normal_prior_world_to_camera_chw(
                    normal_cpu.ptr<float>(),
                    pixel_count,
                    world_to_camera);
            }
            normal = normal_cpu.to(Device::CUDA, _stream);
            if (_stream) {
                LFS_CUDA_TRY(cudaStreamSynchronize(_stream), _stream, "normal upload sync");
            }
        }

        const int native_h = static_cast<int>(normal.shape()[1]);
        const int native_w = static_cast<int>(normal.shape()[2]);
        int target_w = native_w;
        int target_h = native_h;
        if (resize_factor > 1) {
            target_w /= resize_factor;
            target_h /= resize_factor;
        }
        if (max_width > 0 && (target_w > max_width || target_h > max_width)) {
            if (target_w > target_h) {
                target_h = std::max(1, max_width * target_h / target_w);
                target_w = max_width;
            } else {
                target_w = std::max(1, max_width * target_w / target_h);
                target_h = max_width;
            }
        }
        if (target_w != native_w || target_h != native_h) {
            normal = lanczos_resize_float_chw(normal, target_h, target_w, 2, _stream);
            if (_stream) {
                LFS_CUDA_TRY(cudaStreamSynchronize(_stream), _stream, "normal resize sync");
            }
        }

        if (_undistort_prepared) {
            const auto scaled = scale_undistort_params(
                _undistort_params,
                static_cast<int>(normal.shape()[2]),
                static_cast<int>(normal.shape()[1]));
            normal = undistort_image(normal, scaled, _stream);
        }

        _cached_normal = normal.contiguous();
        _normal_loaded = true;

        LOG_DEBUG("Loaded normal map for {}: [{},{}]", _image_name,
                  _cached_normal.shape()[1], _cached_normal.shape()[2]);

        return _cached_normal;
    }

    float Camera::depth_prior_quantization_step() {
        if (_depth_quantization_step < 0.0f) {
            _depth_quantization_step = has_depth() ? image_quantization_step(_depth_path) : 0.0f;
        }
        return _depth_quantization_step;
    }

    void Camera::precompute_undistortion(float blank_pixels) {
        if (_undistort_precomputed)
            return;
        if (!has_distortion())
            return;

        _undistort_params = compute_undistort_params(
            _focal_x, _focal_y, _center_x, _center_y,
            _camera_width, _camera_height,
            _radial_distortion, _tangential_distortion,
            _camera_model_type, blank_pixels);

        _undistort_precomputed = true;
    }

    void Camera::prepare_undistortion(float blank_pixels) {
        if (_undistort_prepared)
            return;

        precompute_undistortion(blank_pixels);
        if (!_undistort_precomputed)
            return;

        _focal_x = _undistort_params.dst_fx;
        _focal_y = _undistort_params.dst_fy;
        _center_x = _undistort_params.dst_cx;
        _center_y = _undistort_params.dst_cy;
        _camera_width = _undistort_params.dst_width;
        _camera_height = _undistort_params.dst_height;
        _FoVx = focal2fov(_focal_x, _camera_width);
        _FoVy = focal2fov(_focal_y, _camera_height);
        _undistort_prepared = true;
    }

    void Camera::translate(const Tensor& trans) {
        // Shift the camera's world-space position by trans.
        // For the view transform: new T = T - R * trans
        // (so that p_cam = R*p_world_new + T_new still holds after shifting world by trans)
        auto R_cpu = _R.cpu().contiguous();
        auto T_cpu = _T.cpu().contiguous();
        auto t_cpu = trans.cpu().contiguous();

        auto R_acc = R_cpu.accessor<float, 2>();
        auto T_acc = T_cpu.accessor<float, 1>();
        auto t_acc = t_cpu.accessor<float, 1>();

        std::vector<float> T_new(3);
        for (int i = 0; i < 3; ++i) {
            float Rt_i = 0.f;
            for (int j = 0; j < 3; ++j)
                Rt_i += R_acc(i, j) * t_acc(j);
            T_new[i] = T_acc(i) - Rt_i;
        }

        _T = Tensor::from_vector(T_new, {3}, Device::CPU);
        _world_view_transform = world_to_view(_R, _T);
        _cam_position = _cam_position + trans.to(Device::CUDA).contiguous();
    }

    bool Camera::has_distortion() const noexcept {
        // Equirectangular is a complete projection model handled natively by the
        // rasterizer; there is no pinhole image to "undistort" to, so it must not
        // be routed through the undistortion path.
        if (_camera_model_type == CameraModelType::EQUIRECTANGULAR)
            return false;
        if (_radial_distortion.is_valid() && _radial_distortion.numel() > 0)
            return true;
        if (_tangential_distortion.is_valid() && _tangential_distortion.numel() > 0)
            return true;
        if (_camera_model_type != CameraModelType::PINHOLE)
            return true;
        return false;
    }

} // namespace lfs::core
