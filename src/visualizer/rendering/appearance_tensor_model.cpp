/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: GPL-3.0-or-later AND Apache-2.0
 */

#include "appearance_tensor_model.hpp"
#include "core/path_utils.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_ppisp.hpp"
#include "io/ppisp_file.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>

namespace lfs::vis {
    namespace {
        using namespace lfs::core;

        void require(const bool valid, const char* message) {
            if (!valid)
                throw lfs::Exception(lfs::make_error({
                    .code = lfs::ErrorCode::InvalidArgument,
                    .domain = lfs::ErrorDomain::Rendering,
                    .detail = message,
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
        }

        template <typename T>
        T read(std::istream& input) {
            T value{};
            require(static_cast<bool>(input.read(reinterpret_cast<char*>(&value), sizeof(value))),
                    "Appearance model is truncated");
            return value;
        }

        Tensor readWeights(std::istream& input, TensorArchiveReader& archive,
                           const TensorShape& shape, const bool gpu) {
            // Check the expected allocation size before deserializing the payload.
            const auto position = input.tellg();
            const auto header = read<TensorFileHeader>(input);
            require(header.dtype == static_cast<uint8_t>(DataType::Float32) && header.numel == shape.elements(),
                    "Appearance tensor has an unexpected type or size");
            input.seekg(position);
            require(static_cast<bool>(input), "Cannot seek in appearance model");
            Tensor value;
            archive.read_host_tensor(value, false);
            require(value.shape() == shape, "Appearance tensor has an unexpected shape");
            return gpu ? value.gpu() : value;
        }

        float finite(const float value) { return std::isfinite(value) ? value : 0.0f; }
        float positive(const float raw, const float minimum) {
            const float value = std::min(32.0f, finite(raw));
            return minimum + std::max(value, 0.0f) + std::log1p(std::exp(-std::abs(value)));
        }
        float center(const float raw) {
            const float value = finite(raw);
            const float exponential = std::exp(-std::abs(value));
            const float sigmoid = value >= 0 ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential);
            return std::clamp(sigmoid, 1e-4f, 1.0f - 1e-4f);
        }

        glm::mat3 colorMatrix(const std::array<float, 8>& color) {
            constexpr float blocks[4][4] = {
                {0.0480542f, -0.0043631f, -0.0043631f, 0.0481283f},
                {0.0580570f, -0.0179872f, -0.0179872f, 0.0431061f},
                {0.0433336f, -0.0180537f, -0.0180537f, 0.0580500f},
                {0.0128369f, -0.0034654f, -0.0034654f, 0.0128158f},
            };
            std::array<glm::vec3, 4> target;
            constexpr float x[4] = {0, 1, 0, 1.0f / 3.0f};
            constexpr float y[4] = {0, 0, 1, 1.0f / 3.0f};
            for (int i = 0; i < 4; ++i) {
                const float a = finite(color[i * 2]), b = finite(color[i * 2 + 1]);
                target[i] = {x[i] + blocks[i][0] * a + blocks[i][1] * b,
                             y[i] + blocks[i][2] * a + blocks[i][3] * b, 1.0f};
            }
            const glm::mat3 t{target[0], target[1], target[2]};
            const auto& gray = target[3];
            const glm::mat3 skew{{0, gray.z, -gray.y}, {-gray.z, 0, gray.x}, {gray.y, -gray.x, 0}};
            const glm::mat3 m = skew * t;
            const glm::vec3 r0{m[0][0], m[1][0], m[2][0]}, r1{m[0][1], m[1][1], m[2][1]}, r2{m[0][2], m[1][2], m[2][2]};
            auto lambda = glm::cross(r0, r1);
            if (glm::dot(lambda, lambda) < 1e-20f) {
                lambda = glm::cross(r0, r2);
                if (glm::dot(lambda, lambda) < 1e-20f)
                    lambda = glm::cross(r1, r2);
            }
            const glm::mat3 diagonal{{lambda.x, 0, 0}, {0, lambda.y, 0}, {0, 0, lambda.z}};
            const glm::mat3 inverse{{-1, 1, 0}, {-1, 0, 1}, {1, 0, 0}};
            auto h = t * diagonal * inverse;
            if (std::abs(h[2][2]) > 1e-20f)
                h *= 1.0f / h[2][2];
            return h;
        }

        void requireImage(const Tensor& rgb, const GpuBackend backend) {
            require(rgb.is_valid() && rgb.device() == Device::GPU && gpu_backend_of(rgb) == backend &&
                        rgb.dtype() == DataType::Float32 && rgb.ndim() == 3 && rgb.size(0) == 3 &&
                        rgb.size(1) > 0 && rgb.size(2) > 0 && rgb.size(1) <= std::numeric_limits<int>::max() &&
                        rgb.size(2) <= std::numeric_limits<int>::max(),
                    "Appearance correction requires float RGB [3,H,W] on the model's backend");
        }
    } // namespace

    lfs::Result<std::unique_ptr<AppearanceTensorModel>> AppearanceTensorModel::load(
        const std::filesystem::path& path, const lfs::core::GpuBackend backend) {
        using namespace lfs::core;
        try {
            std::ifstream file;
            if (!open_file_for_read(path, std::ios::binary, file))
                return lfs::make_error({.code = lfs::ErrorCode::Unavailable,
                                        .domain = lfs::ErrorDomain::IO,
                                        .detail = "Cannot open appearance model: " + path_to_utf8(path),
                                        .detection = LFS_SOURCE_SITE_CURRENT()});
            return load(file, backend);
        } catch (const lfs::Exception& error) {
            return error.error();
        } catch (const std::exception& error) {
            return lfs::make_error({.code = lfs::ErrorCode::DataLoss,
                                    .domain = lfs::ErrorDomain::IO,
                                    .detail = error.what(),
                                    .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }

    lfs::Result<std::unique_ptr<AppearanceTensorModel>> AppearanceTensorModel::load(
        std::istream& file, const lfs::core::GpuBackend backend, const int camera_index) {
        using namespace lfs::core;
        try {
            const auto header = read<lfs::training::PPISPFileHeader>(file);
            require(header.magic == lfs::training::PPISP_FILE_MAGIC && header.version > 0 &&
                        header.version <= lfs::training::PPISP_FILE_VERSION && header.num_cameras > 0 &&
                        header.num_cameras <= 100000 && header.num_frames > 0 && header.num_frames <= 10000000,
                    "Unsupported appearance model header");
            require(read<uint32_t>(file) == 0x4C465049 && read<uint32_t>(file) == 1,
                    "Unsupported appearance inference format");
            auto model = std::make_unique<AppearanceTensorModel>();
            model->backend_ = backend;
            model->camera_index_ = camera_index;
            model->num_cameras_ = read<int>(file);
            model->num_frames_ = read<int>(file);
            require(model->num_cameras_ == static_cast<int>(header.num_cameras) &&
                        model->num_frames_ == static_cast<int>(header.num_frames),
                    "Appearance model dimensions disagree with its header");
            require(camera_index >= 0 && camera_index < model->num_cameras_, "Appearance camera is out of range");
            const auto cameras = static_cast<size_t>(model->num_cameras_), frames = static_cast<size_t>(model->num_frames_);
            TensorArchiveReader archive(file);
            model->exposure_ = readWeights(file, archive, {frames}, false).to_vector();
            model->vignetting_ = readWeights(file, archive, {cameras * 15}, false).to_vector();
            model->color_ = readWeights(file, archive, {frames * 8}, false).to_vector();
            model->crf_ = readWeights(file, archive, {cameras * 12}, false).to_vector();
            const GpuBackendScope scope(backend);
            if (lfs::training::has_flag(header.flags, lfs::training::PPISPFileFlags::HAS_CONTROLLER)) {
                require(read<uint32_t>(file) == 0x4C464349 && read<uint32_t>(file) == 1 &&
                            read<int>(file) == model->num_cameras_,
                        "Unsupported appearance controller format");
                const std::array<TensorShape, 6> shapes{{{16, 3}, {16}, {32, 16}, {32}, {64, 32}, {64}}};
                for (size_t i = 0; i < shapes.size(); ++i)
                    model->convolution_[i] = readWeights(file, archive, shapes[i], true);
                model->controller_.resize(cameras);
                const std::array<TensorShape, 8> fc_shapes{{{128, 1601}, {128}, {128, 128}, {128}, {128, 128}, {128}, {9, 128}, {9}}};
                for (auto& camera : model->controller_)
                    for (size_t i = 0; i < fc_shapes.size(); ++i)
                        camera[i] = readWeights(file, archive, fc_shapes[i], true);
            }
            if (header.version >= 2 && lfs::training::has_flag(header.flags, lfs::training::PPISPFileFlags::HAS_METADATA)) {
                const auto bytes = read<uint64_t>(file);
                require(bytes <= 64 * 1024 * 1024, "Appearance metadata exceeds its size limit");
                const auto position = file.tellg();
                file.seekg(0, std::ios::end);
                const auto end = file.tellg();
                file.seekg(position);
                require(position >= 0 && end >= position &&
                            static_cast<uint64_t>(end - position) >= bytes,
                        "Appearance metadata is truncated");
                std::string metadata(static_cast<size_t>(bytes), '\0');
                require(static_cast<bool>(file.read(metadata.data(), metadata.size())),
                        "Appearance metadata is truncated");
                require(nlohmann::json::parse(metadata).is_object(), "Appearance metadata must be an object");
            }
            return model;
        } catch (const lfs::Exception& error) {
            return error.error();
        } catch (const std::exception& error) {
            return lfs::make_error({.code = lfs::ErrorCode::DataLoss,
                                    .domain = lfs::ErrorDomain::IO,
                                    .detail = error.what(),
                                    .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }

    lfs::core::Tensor AppearanceTensorModel::predict(const lfs::core::Tensor& rgb) const {
        using namespace lfs::core;
        requireImage(rgb, backend_);
        require(hasController() && rgb.size(1) >= 3 && rgb.size(2) >= 3,
                "Appearance prediction requires a controller and an image at least 3x3");
        const GpuBackendScope scope(backend_);
        auto features = rgb.unsqueeze(0).conv1x1(convolution_[0], convolution_[1]);
        features = features.max_pool2d(3, 3).relu();
        features = features.conv1x1(convolution_[2], convolution_[3]).relu();
        features = features.conv1x1(convolution_[4], convolution_[5]).adaptive_avg_pool2d(5, 5);
        auto input = Tensor::ones({1, 1601}, Device::GPU);
        input.slice(1, 0, 1600).copy_from(features.reshape({1, 1600}));
        // The companion format maps every frame to camera zero.
        const auto& weights = controller_[camera_index_];
        for (int layer = 0; layer < 4; ++layer) {
            input = input.linear(weights[layer * 2], weights[layer * 2 + 1]);
            if (layer != 3)
                input = input.relu();
        }
        return input;
    }

    lfs::core::Tensor AppearanceTensorModel::apply(const lfs::core::Tensor& rgb,
                                                   const int uid, const PPISPOverrides& ov,
                                                   const bool use_controller) const {
        using namespace lfs::core;
        requireImage(rgb, backend_);
        const GpuBackendScope scope(backend_);
        const auto prediction = use_controller && hasController() ? predict(rgb) : Tensor{};
        return ppisp_apply(rgb, parameters(uid, ov, prediction));
    }

    lfs::core::PpispParams AppearanceTensorModel::parameters(
        int uid, const PPISPOverrides& ov, const lfs::core::Tensor& controller_params) const {
        using namespace lfs::core;
        float exposure = 0;
        std::array<float, 8> color{};
        if (controller_params.is_valid()) {
            require(controller_params.dtype() == DataType::Float32 && controller_params.shape() == TensorShape({1, 9}),
                    "Appearance controller must supply [1,9] parameters");
            const auto values = controller_params.to_vector();
            exposure = values[0];
            std::copy_n(values.begin() + 1, 8, color.begin());
        } else if (uid >= 0 && uid < num_frames_) {
            exposure = exposure_[static_cast<size_t>(uid)];
            std::copy_n(color_.begin() + static_cast<size_t>(uid) * 8, 8, color.begin());
        }
        const float factor = std::exp2(std::clamp(finite(exposure + ov.exposure_offset), -16.0f, 16.0f));
        const std::array<float, 8> offsets{ov.color_blue_x, ov.color_blue_y, ov.color_red_x, ov.color_red_y,
                                           ov.color_green_x, ov.color_green_y, ov.wb_temperature * 2, ov.wb_tint * 2};
        for (size_t i = 0; i < color.size(); ++i)
            color[i] += offsets[i] * 12.0f;
        const auto matrix = colorMatrix(color);
        PpispParams params;
        params.exposure_factor = factor;
        const float strength = ov.vignette_enabled ? ov.vignette_strength : 0.0f;
        const float gamma_offsets[3] = {ov.gamma_red, ov.gamma_green, ov.gamma_blue};
        for (int channel = 0; channel < 3; ++channel) {
            const auto* v = vignetting_.data() + camera_index_ * 15 + channel * 5;
            for (int j = 0; j < 5; ++j)
                params.vignetting[channel * 5 + j] = finite(v[j] * (j >= 2 ? strength : 1.0f));
            for (int j = 0; j < 3; ++j)
                params.color_matrix[channel * 3 + j] = matrix[j][channel];
            const auto* raw = crf_.data() + camera_index_ * 12 + channel * 4;
            auto* crf = params.crf + channel * 5;
            crf[0] = positive(raw[0] + ov.crf_toe, 0.3f);
            crf[1] = positive(raw[1] + ov.crf_shoulder, 0.3f);
            crf[2] = positive(raw[2] + (std::log(ov.gamma_multiplier) + gamma_offsets[channel]), 0.1f);
            crf[3] = center(raw[3]);
            crf[4] = crf[1] * crf[3] / std::fma(crf[1] - crf[0], crf[3], crf[0]);
        }
        return params;
    }
} // namespace lfs::vis
