/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/models/lpips.hpp"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "internal/cuda_stream_context.hpp"
#include "internal/size_bucketed_pool.hpp"
#include "nn_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <utility>
#include <vector>

namespace lfs::core::nn::models {
    namespace {
        constexpr float kNormEps = 1.0e-10f;
        constexpr int kBlocks = 5;
        constexpr std::size_t kTileHalo = 112;
        constexpr std::size_t kExactBytesPerPixel = 1400;
        constexpr std::size_t kFastBytesPerPixel = 560;
        struct Layer {
            int index;
            int cin;
            int cout;
        };
        constexpr std::array<Layer, 13> kLayers{{{0, 3, 64},
                                                 {2, 64, 64},
                                                 {5, 64, 128},
                                                 {7, 128, 128},
                                                 {10, 128, 256},
                                                 {12, 256, 256},
                                                 {14, 256, 256},
                                                 {17, 256, 512},
                                                 {19, 512, 512},
                                                 {21, 512, 512},
                                                 {24, 512, 512},
                                                 {26, 512, 512},
                                                 {28, 512, 512}}};
        constexpr std::array<int, kBlocks> kStageLayers{2, 2, 3, 3, 3};

        TensorShape shape_of(std::initializer_list<std::size_t> dims) {
            return TensorShape(std::vector<std::size_t>(dims));
        }

        std::size_t fast_activation_bytes(const std::size_t height, const std::size_t width,
                                          const bool tiled) {
            const auto pixels = height * width;
            return 4 * SizeBucketedPool::get_bucket_size(64 * pixels * sizeof(uint16_t)) +
                   (tiled ? 2 * SizeBucketedPool::get_bucket_size(3 * pixels * sizeof(float)) : 0) +
                   SizeBucketedPool::get_bucket_size(kBlocks * sizeof(float));
        }

        lfs::Error lpips_error(const lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({
                .code = code,
                .domain = lfs::ErrorDomain::Core,
                .user_message = "LPIPS inference failed",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        Tensor as_batch(const Tensor& image) {
            return image.ndim() == 3 ? image.unsqueeze(0) : image;
        }

        void recapture(Tensor& slot, const Tensor& src) {
            if (!slot.is_valid() || slot.dtype() != src.dtype() || slot.device() != src.device() ||
                slot.shape() != src.shape()) {
                slot = src.clone();
                return;
            }
            slot.set_stream(src.stream());
            if (src.bytes() > 0) {
                LFS_CUDA_CHECK(cudaMemcpyAsync(slot.data_ptr(), src.data_ptr(), src.bytes(),
                                               cudaMemcpyDeviceToDevice, src.stream()));
            }
        }
    } // namespace

    lfs::Result<Lpips> Lpips::load(const std::filesystem::path& weights, Device device,
                                   std::optional<DataType> compute, InputScaling scaling,
                                   const std::size_t activation_budget_bytes) {
        if (device != Device::CUDA) {
            return lpips_error(lfs::ErrorCode::InvalidArgument,
                               "LPIPS requires a CUDA device");
        }
        auto file = WeightFile::open(weights);
        if (!file)
            return std::move(file.error());

        const DataType dtype = compute.value_or(DataType::Float32);
        if (dtype != DataType::Float16 && dtype != DataType::Float32) {
            return lpips_error(lfs::ErrorCode::InvalidArgument,
                               "LPIPS compute dtype must be float16 or float32");
        }
        auto loaded = file->load_all(device, dtype);
        if (!loaded)
            return std::move(loaded.error());

        Lpips model;
        model.device_ = device;
        model.compute_ = dtype;
        model.scaling_ = scaling;
        model.activation_budget_bytes_ = activation_budget_bytes;
        model.weights_ = std::move(*loaded);

        for (const char* name : {"scaling.shift", "scaling.scale"}) {
            if (!model.weights_.contains(name)) {
                return lpips_error(lfs::ErrorCode::NotFound,
                                   std::format("weight file is missing {}", name));
            }
        }
        for (const auto& layer : kLayers) {
            const auto prefix = std::format("vgg.features.{}", layer.index);
            for (const char* suffix : {"weight", "bias"}) {
                if (!model.weights_.contains(prefix + "." + suffix)) {
                    return lpips_error(lfs::ErrorCode::NotFound,
                                       std::format("weight file is missing {}.{}", prefix, suffix));
                }
            }
            if (model.w(prefix + ".weight").shape() != shape_of({static_cast<std::size_t>(layer.cout),
                                                                 static_cast<std::size_t>(layer.cin), 3, 3}) ||
                model.w(prefix + ".bias").shape() != shape_of({static_cast<std::size_t>(layer.cout)}))
                return lpips_error(lfs::ErrorCode::InvalidArgument,
                                   std::format("invalid VGG16 weight shape for {}", prefix));
        }
        for (int i = 0; i < kBlocks; ++i) {
            if (!model.weights_.contains(std::format("lin{}.weight", i))) {
                return lpips_error(lfs::ErrorCode::NotFound,
                                   std::format("weight file is missing lin{}.weight", i));
            }
            constexpr std::array<std::size_t, kBlocks> channels{64, 128, 256, 512, 512};
            if (model.w(std::format("lin{}.weight", i)).shape() != shape_of({1, channels[i], 1, 1}))
                return lpips_error(lfs::ErrorCode::InvalidArgument,
                                   std::format("invalid LPIPS linear weight shape for lin{}", i));
        }
        const auto shift = model.w("scaling.shift").to(DataType::Float32).to(Device::CPU).contiguous().to_vector();
        const auto scale = model.w("scaling.scale").to(DataType::Float32).to(Device::CPU).contiguous().to_vector();
        if (shift.size() != 3 || scale.size() != 3) {
            return lpips_error(lfs::ErrorCode::InvalidArgument,
                               "LPIPS scaling.shift and scaling.scale must hold three values");
        }
        std::copy(shift.begin(), shift.end(), model.scaling_shift_.begin());
        std::copy(scale.begin(), scale.end(), model.scaling_scale_.begin());
        return model;
    }

    std::size_t Lpips::tile_size_for(const int height, const int width) const {
        if (height <= 0 || width <= 0)
            return 0;
        const std::size_t pixels = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
        const std::size_t bytes_per_pixel = compute_ == DataType::Float16
                                                ? kFastBytesPerPixel
                                                : kExactBytesPerPixel;
        const auto untiled_bytes = compute_ == DataType::Float16
                                       ? fast_activation_bytes(height, width, false)
                                       : pixels * bytes_per_pixel;
        if (untiled_bytes <= activation_budget_bytes_)
            return std::max(static_cast<std::size_t>(height), static_cast<std::size_t>(width));
        if (activation_budget_bytes_ <= bytes_per_pixel * kTileHalo * kTileHalo)
            return 16;
        const auto max_crop_pixels = activation_budget_bytes_ / bytes_per_pixel;
        const auto max_crop_edge = static_cast<std::size_t>(std::sqrt(
            static_cast<double>(max_crop_pixels)));
        if (max_crop_edge <= 2 * kTileHalo + 16)
            return 16;
        auto tile = std::max<std::size_t>(16, ((max_crop_edge - 2 * kTileHalo) / 16) * 16);
        if (compute_ == DataType::Float16) {
            while (tile > 16 && fast_activation_bytes(std::min<std::size_t>(height, tile + 2 * kTileHalo),
                                                      std::min<std::size_t>(width, tile + 2 * kTileHalo), true) >
                                    activation_budget_bytes_)
                tile -= 16;
        }
        return tile;
    }

    std::size_t Lpips::estimated_peak_bytes(const int height, const int width) const {
        const auto tile = tile_size_for(height, width);
        if (tile == 0)
            return 0;
        const std::size_t tile_h = std::min<std::size_t>(static_cast<std::size_t>(height), tile);
        const std::size_t tile_w = std::min<std::size_t>(static_cast<std::size_t>(width), tile);
        const std::size_t crop_h = std::min<std::size_t>(static_cast<std::size_t>(height), tile_h + 2 * kTileHalo);
        const std::size_t crop_w = std::min<std::size_t>(static_cast<std::size_t>(width), tile_w + 2 * kTileHalo);
        if (compute_ == DataType::Float32)
            return crop_h * crop_w * kExactBytesPerPixel;

        // Count new allocations, including pool rounding. Existing buffers are
        // already reflected in cudaMemGetInfo; they need no second reservation.
        constexpr std::size_t driver_reserve = 64ULL * 1024 * 1024;
        std::size_t bytes = driver_reserve;
        const auto feature_elems = 64 * crop_h * crop_w;
        for (const auto& buffer : fast_features_) {
            if (!buffer.is_valid() || buffer.numel() < feature_elems)
                bytes += SizeBucketedPool::get_bucket_size(feature_elems * sizeof(uint16_t));
        }
        if (tile < static_cast<std::size_t>(std::max(height, width)))
            bytes += 2 * SizeBucketedPool::get_bucket_size(3 * crop_h * crop_w * sizeof(float));
        if (!fast_scores_.is_valid())
            bytes += SizeBucketedPool::get_bucket_size(kBlocks * sizeof(float));
        if (!fast_weight_taps_.is_valid()) {
            std::size_t taps_bytes = 0;
            for (std::size_t i = 1; i < kLayers.size(); ++i)
                taps_bytes += kernels::conv2d_weight_scratch_bytes(kLayers[i].cout, kLayers[i].cin, compute_);
            if (taps_bytes > 0)
                bytes += SizeBucketedPool::get_bucket_size(taps_bytes);
        }
        return bytes;
    }

    std::size_t Lpips::weights_bytes() const {
        std::size_t bytes = 0;
        for (const auto& [name, tensor] : weights_) {
            (void)name;
            if (tensor.is_valid())
                bytes += tensor.bytes();
        }
        return bytes;
    }

    void Lpips::release_activations() {
        arena_ = ActivationArena{};
        workspace_ = Tensor{};
        feature_x_hold_ = Tensor{};
        feature_y_hold_ = Tensor{};
        normalized_x_hold_ = Tensor{};
        normalized_y_hold_ = Tensor{};
        for (auto& buffer : fast_features_)
            buffer = Tensor{};
        fast_scores_ = Tensor{};
        fast_weight_taps_ = Tensor{};
    }

    const Tensor& Lpips::w(std::string_view name) const {
        const auto it = weights_.find(std::string(name));
        LFS_ASSERT_MSG(it != weights_.end(), std::format("LPIPS weight {} is not loaded", name));
        return it->second;
    }

    Tensor Lpips::ensure_workspace(std::size_t bytes, const Tensor& like) {
        if (bytes == 0)
            return like;
        if (workspace_.is_valid() && workspace_.bytes() >= bytes &&
            workspace_.dtype() == like.dtype() && workspace_.device() == like.device()) {
            workspace_.set_stream(like.stream());
            return workspace_;
        }
        const std::size_t elem = dtype_size(like.dtype());
        workspace_ = Tensor::empty(shape_of({(bytes + elem - 1) / elem}), like.device(), like.dtype());
        workspace_.set_stream(like.stream());
        return workspace_;
    }

    Tensor Lpips::conv(const Tensor& x, std::string_view weight, std::string_view bias) {
        Conv2dParams params;
        if (w(weight).ndim() == 4 && w(weight).shape()[2] == 3 && w(weight).shape()[3] == 3) {
            params.pad_h = 1;
            params.pad_w = 1;
        }
        const auto& ww = w(weight);
        const auto bytes = conv2d_workspace_bytes(x.shape(), ww.shape(), params, x.dtype());
        auto workspace = ensure_workspace(bytes, x);
        const Tensor* bb = bias.empty() ? nullptr : &w(bias);
        return conv2d(x, ww, bb, params, &workspace);
    }

    Tensor Lpips::conv(const Tensor& x, const Tensor& weight) {
        Conv2dParams params;
        if (weight.ndim() == 4 && weight.shape()[2] == 3 && weight.shape()[3] == 3) {
            params.pad_h = 1;
            params.pad_w = 1;
        }
        const auto bytes = conv2d_workspace_bytes(x.shape(), weight.shape(), params, x.dtype());
        auto workspace = ensure_workspace(bytes, x);
        return conv2d(x, weight, nullptr, params, &workspace);
    }

    Tensor Lpips::normalize_feature(const Tensor& x) {
        const auto fp32 = x.to(DataType::Float32);
        const auto norm = fp32.mul(fp32).sum(1, true).add(kNormEps).sqrt();
        return fp32.div(norm).to(x.dtype());
    }

    lfs::Result<float> Lpips::forward(const Tensor& pred, const Tensor& target,
                                      std::optional<InputScaling> scaling) {
        return run(pred, target, scaling.value_or(scaling_), nullptr);
    }

    lfs::Result<LpipsTaps> Lpips::forward_with_taps(const Tensor& pred, const Tensor& target,
                                                    std::optional<InputScaling> scaling) {
        LpipsTaps taps;
        auto value = run(pred, target, scaling.value_or(scaling_), &taps);
        if (!value)
            return std::move(value.error());
        return taps;
    }

    lfs::Result<float> Lpips::run(const Tensor& pred, const Tensor& target,
                                  const InputScaling scaling, LpipsTaps* taps) {
        if (taps == nullptr && pred.is_valid() && target.is_valid() &&
            (pred.ndim() == 3 || pred.ndim() == 4) && pred.shape() == target.shape()) {
            const int height = static_cast<int>(pred.shape()[pred.ndim() - 2]);
            const int width = static_cast<int>(pred.shape()[pred.ndim() - 1]);
            if (compute_ == DataType::Float16)
                return run_fast(pred, target, scaling);
            if (tile_size_for(height, width) < static_cast<std::size_t>(std::max(height, width)))
                return run_tiled(pred, target, scaling);
        }
        return run_untiled(pred, target, scaling, taps);
    }

    std::optional<lfs::Error> Lpips::validate_pair(const Tensor& pred, const Tensor& target) const {
        if (!pred.is_valid() || !target.is_valid())
            return lpips_error(lfs::ErrorCode::InvalidArgument, "LPIPS inputs must be valid");
        if ((pred.ndim() != 3 && pred.ndim() != 4) || pred.shape()[pred.ndim() - 3] != 3 ||
            pred.dtype() != DataType::Float32 || pred.device() != Device::CUDA)
            return lpips_error(lfs::ErrorCode::InvalidArgument,
                               "LPIPS prediction must be CUDA fp32 RGB [3,H,W] or [1,3,H,W]");
        if (target.shape() != pred.shape() || target.dtype() != DataType::Float32 ||
            target.device() != Device::CUDA)
            return lpips_error(lfs::ErrorCode::InvalidArgument,
                               "LPIPS target must match the CUDA fp32 prediction shape");
        if ((pred.ndim() == 4 && pred.shape()[0] != 1) ||
            pred.shape()[pred.ndim() - 2] < 16 || pred.shape()[pred.ndim() - 1] < 16)
            return lpips_error(lfs::ErrorCode::InvalidArgument,
                               "LPIPS requires one image with height and width at least 16");
        return std::nullopt;
    }

    void Lpips::bind_weights_to_stream(const cudaStream_t stream) {
        if (weights_on_stream_)
            return;
        for (auto& [name, tensor] : weights_) {
            (void)name;
            tensor.set_stream(stream);
        }
        weights_on_stream_ = true;
    }

    lfs::Result<float> Lpips::run_fast(const Tensor& pred, const Tensor& target,
                                       const InputScaling scaling) {
        if (auto error = validate_pair(pred, target))
            return std::move(*error);
        Tensor x_in = as_batch(pred).contiguous();
        Tensor y_in = as_batch(target).contiguous();
        const int height = static_cast<int>(x_in.shape()[2]);
        const int width = static_cast<int>(x_in.shape()[3]);
        const std::size_t tile = tile_size_for(height, width);
        if (tile == 0)
            return lpips_error(lfs::ErrorCode::InvalidArgument, "LPIPS tiled input has an empty shape");
        const auto full_dim = [](const int dimension, const int block) {
            return static_cast<int>(static_cast<std::size_t>(dimension) >> block);
        };

        const cudaStream_t stream = x_in.stream();
        lfs::core::CUDAStreamGuard stream_guard(stream);
        if (y_in.stream() != stream) {
            cudaEvent_t ready = nullptr;
            LFS_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            LFS_CUDA_CHECK(cudaEventRecord(ready, y_in.stream()));
            LFS_CUDA_CHECK(cudaStreamWaitEvent(stream, ready, 0));
            LFS_CUDA_CHECK(cudaEventDestroy(ready));
        }
        bind_weights_to_stream(stream);

        std::size_t taps_bytes = 0;
        for (std::size_t i = 1; i < kLayers.size(); ++i) {
            fast_weight_tap_offsets_[i] = taps_bytes;
            taps_bytes += kernels::conv2d_weight_scratch_bytes(kLayers[i].cout, kLayers[i].cin,
                                                               DataType::Float16);
        }
        if (taps_bytes > 0 && !fast_weight_taps_.is_valid()) {
            fast_weight_taps_ = Tensor::empty(shape_of({taps_bytes / sizeof(uint16_t)}), Device::CUDA,
                                              DataType::Float16);
            fast_weight_taps_.set_stream(stream);
            auto* taps = static_cast<unsigned char*>(fast_weight_taps_.data_ptr());
            for (std::size_t i = 1; i < kLayers.size(); ++i) {
                kernels::conv3x3_weight_taps(
                    w(std::format("vgg.features.{}.weight", kLayers[i].index)).data_ptr(),
                    taps + fast_weight_tap_offsets_[i], kLayers[i].cout, kLayers[i].cin, stream);
            }
        }
        if (fast_weight_taps_.is_valid())
            fast_weight_taps_.set_stream(stream);
        const auto* taps_base = taps_bytes > 0
                                    ? static_cast<const unsigned char*>(fast_weight_taps_.data_ptr())
                                    : nullptr;

        const bool tiled = tile < static_cast<std::size_t>(std::max(height, width));
        const auto crop_height = std::min<std::size_t>(height, tile + 2 * kTileHalo);
        const auto crop_width = std::min<std::size_t>(width, tile + 2 * kTileHalo);
        Tensor tile_x, tile_y;
        if (tiled) {
            tile_x = Tensor::empty(shape_of({1, 3, crop_height, crop_width}), Device::CUDA);
            tile_y = Tensor::empty(tile_x.shape(), Device::CUDA);
        }
        const std::size_t max_feature_elems = 64ULL * crop_height * crop_width;
        for (auto& buffer : fast_features_) {
            if (!buffer.is_valid() || buffer.numel() < max_feature_elems)
                buffer = Tensor::empty(shape_of({max_feature_elems}), Device::CUDA, DataType::Float16);
            buffer.set_stream(stream);
        }
        if (!fast_scores_.is_valid())
            fast_scores_ = Tensor::empty(shape_of({kBlocks}), Device::CUDA, DataType::Float32);
        fast_scores_.set_stream(stream);
        auto* scores = fast_scores_.ptr<float>();
        LFS_CUDA_CHECK(cudaMemsetAsync(scores, 0, kBlocks * sizeof(float), stream));

        const auto copy_tile = [&](const Tensor& source, Tensor& destination, const int cy0,
                                   const int cx0, const int crop_h, const int crop_w) {
            const auto* src = static_cast<const float*>(source.data_ptr());
            auto* dst = static_cast<float*>(destination.data_ptr());
            const std::size_t source_plane = static_cast<std::size_t>(height) * width;
            const std::size_t destination_plane = static_cast<std::size_t>(crop_h) * crop_w;
            for (int c = 0; c < 3; ++c) {
                const auto* src_channel = src + static_cast<std::size_t>(c) * source_plane +
                                          static_cast<std::size_t>(cy0) * width + cx0;
                auto* dst_channel = dst + static_cast<std::size_t>(c) * destination_plane;
                LFS_CUDA_CHECK(cudaMemcpy2DAsync(
                    dst_channel, static_cast<std::size_t>(crop_w) * sizeof(float), src_channel,
                    static_cast<std::size_t>(width) * sizeof(float),
                    static_cast<std::size_t>(crop_w) * sizeof(float), static_cast<std::size_t>(crop_h),
                    cudaMemcpyDeviceToDevice, stream));
            }
        };

        for (int y0 = 0; y0 < height; y0 += static_cast<int>(tile)) {
            const int y1 = std::min(height, y0 + static_cast<int>(tile));
            const int cy0 = std::max(0, y0 - static_cast<int>(kTileHalo));
            const int cy1 = std::min(height, y1 + static_cast<int>(kTileHalo));
            for (int x0 = 0; x0 < width; x0 += static_cast<int>(tile)) {
                const int x1 = std::min(width, x0 + static_cast<int>(tile));
                const int cx0 = std::max(0, x0 - static_cast<int>(kTileHalo));
                const int cx1 = std::min(width, x1 + static_cast<int>(kTileHalo));
                const int crop_h = cy1 - cy0;
                const int crop_w = cx1 - cx0;
                if (tiled) {
                    copy_tile(x_in, tile_x, cy0, cx0, crop_h, crop_w);
                    copy_tile(y_in, tile_y, cy0, cx0, crop_h, crop_w);
                }
                void* cur[2] = {tiled ? tile_x.data_ptr() : x_in.data_ptr(),
                                tiled ? tile_y.data_ptr() : y_in.data_ptr()};
                void* next[2] = {fast_features_[0].data_ptr(), fast_features_[1].data_ptr()};
                int cur_h = crop_h;
                int cur_w = crop_w;
                int layer = 0;
                for (int stage = 0; stage < kBlocks; ++stage) {
                    for (int i = 0; i < kStageLayers[static_cast<std::size_t>(stage)]; ++i,
                             ++layer) {
                        const auto& spec = kLayers[static_cast<std::size_t>(layer)];
                        const auto& weight = w(std::format("vgg.features.{}.weight", spec.index));
                        const auto& bias = w(std::format("vgg.features.{}.bias", spec.index));
                        for (int side = 0; side < 2; ++side) {
                            if (layer == 0) {
                                kernels::lpips_rgb_conv3x3(
                                    static_cast<const float*>(cur[side]), weight.data_ptr(),
                                    bias.data_ptr(), next[side], scaling_shift_.data(),
                                    scaling_scale_.data(), scaling == InputScaling::Normalize, 1, cur_h,
                                    cur_w, stream);
                            } else {
                                const void* weight_taps =
                                    taps_base
                                        ? taps_base + fast_weight_tap_offsets_[static_cast<std::size_t>(layer)]
                                        : nullptr;
                                kernels::conv2d_implicit(
                                    cur[side], weight.data_ptr(), weight_taps, bias.data_ptr(), next[side],
                                    nullptr, 1, spec.cin, cur_h, cur_w, spec.cout, 3, 3, cur_h, cur_w,
                                    1, 1, 1, 1, 1, 1, 0, static_cast<int>(Activation::Relu),
                                    DataType::Float16, stream);
                            }
                        }
                        std::swap(cur[0], next[0]);
                        std::swap(cur[1], next[1]);
                        if (layer == 0) {
                            next[0] = fast_features_[2].data_ptr();
                            next[1] = fast_features_[3].data_ptr();
                        }
                    }

                    const int factor = 1 << stage;
                    const int feature_h = full_dim(height, stage);
                    const int feature_w = full_dim(width, stage);
                    const int gy0 = y0 / factor;
                    const int gy1 = std::min(feature_h, y1 / factor);
                    const int gx0 = x0 / factor;
                    const int gx1 = std::min(feature_w, x1 / factor);
                    const int interior_y0 = std::clamp(gy0 - cy0 / factor, 0, cur_h);
                    const int interior_y1 = std::clamp(gy1 - cy0 / factor, 0, cur_h);
                    const int interior_x0 = std::clamp(gx0 - cx0 / factor, 0, cur_w);
                    const int interior_x1 = std::clamp(gx1 - cx0 / factor, 0, cur_w);
                    const bool last = stage + 1 == kBlocks;
                    const int channels = kLayers[static_cast<std::size_t>(layer - 1)].cout;
                    kernels::lpips_pool_reduce(
                        cur[0], cur[1], w(std::format("lin{}.weight", stage)).data_ptr(),
                        scores + stage, last ? nullptr : next[0], last ? nullptr : next[1], 1,
                        channels, cur_h, cur_w, interior_y0, interior_y1, interior_x0, interior_x1,
                        1.0f / (static_cast<float>(feature_h) * static_cast<float>(feature_w)), stream);
                    if (!last) {
                        std::swap(cur[0], next[0]);
                        std::swap(cur[1], next[1]);
                        cur_h /= 2;
                        cur_w /= 2;
                    }
                }
            }
        }

        std::array<float, kBlocks> values{};
        LFS_CUDA_CHECK(cudaMemcpyAsync(values.data(), scores, kBlocks * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        float total = 0.0f;
        for (const float value : values)
            total += value;
        if (!std::isfinite(total))
            return lpips_error(lfs::ErrorCode::Internal, "LPIPS produced a non-finite value");
        return total;
    }

    lfs::Result<float> Lpips::run_tiled(const Tensor& pred, const Tensor& target,
                                        const InputScaling scaling) {
        if (pred.ndim() != 3 && pred.ndim() != 4)
            return lpips_error(lfs::ErrorCode::InvalidArgument, "LPIPS tiled inputs must be rank 3 or 4");
        const int height = static_cast<int>(pred.shape()[pred.ndim() - 2]);
        const int width = static_cast<int>(pred.shape()[pred.ndim() - 1]);
        const std::size_t tile = tile_size_for(height, width);
        if (tile == 0)
            return lpips_error(lfs::ErrorCode::InvalidArgument, "LPIPS tiled input has an empty shape");

        std::array<double, kBlocks> weighted{};
        std::array<std::size_t, kBlocks> counts{};
        const auto full_dim = [](const int dimension, const int block) {
            return static_cast<int>(static_cast<std::size_t>(dimension) >> block);
        };
        const auto crop = [](const Tensor& input, const int y0, const int y1,
                             const int x0, const int x1) {
            const auto batched = as_batch(input);
            return batched.slice(2, static_cast<std::size_t>(y0), static_cast<std::size_t>(y1))
                .slice(3, static_cast<std::size_t>(x0), static_cast<std::size_t>(x1));
        };

        for (int y0 = 0; y0 < height; y0 += static_cast<int>(tile)) {
            const int y1 = std::min(height, y0 + static_cast<int>(tile));
            const int cy0 = std::max(0, y0 - static_cast<int>(kTileHalo));
            const int cy1 = std::min(height, y1 + static_cast<int>(kTileHalo));
            for (int x0 = 0; x0 < width; x0 += static_cast<int>(tile)) {
                const int x1 = std::min(width, x0 + static_cast<int>(tile));
                const int cx0 = std::max(0, x0 - static_cast<int>(kTileHalo));
                const int cx1 = std::min(width, x1 + static_cast<int>(kTileHalo));
                const auto target_tile = crop(target, cy0, cy1, cx0, cx1);
                const auto pred_tile = crop(pred, cy0, cy1, cx0, cx1);

                LpipsTaps target_taps;
                auto target_result = run_untiled(target_tile, target_tile, scaling, &target_taps);
                if (!target_result)
                    return std::move(target_result.error());
                release_activations();
                LpipsTaps pred_taps;
                auto pred_result = run_untiled(pred_tile, pred_tile, scaling, &pred_taps);
                if (!pred_result)
                    return std::move(pred_result.error());
                release_activations();
                for (int block = 0; block < kBlocks; ++block) {
                    const int factor = 1 << block;
                    const int feature_h = full_dim(height, block);
                    const int feature_w = full_dim(width, block);
                    const int gy0 = y0 / factor;
                    const int gy1 = std::min(feature_h, y1 / factor);
                    const int gx0 = x0 / factor;
                    const int gx1 = std::min(feature_w, x1 / factor);
                    const int ly0 = gy0 - cy0 / factor;
                    const int ly1 = gy1 - cy0 / factor;
                    const int lx0 = gx0 - cx0 / factor;
                    const int lx1 = gx1 - cx0 / factor;
                    if (ly1 <= ly0 || lx1 <= lx0)
                        continue;
                    const auto& target_feature = target_taps.normalized_features[block];
                    const auto& pred_feature = pred_taps.normalized_features[block];
                    const auto target_region = target_feature.slice(
                                                                 2, static_cast<std::size_t>(ly0), static_cast<std::size_t>(ly1))
                                                   .slice(3, static_cast<std::size_t>(lx0),
                                                          static_cast<std::size_t>(lx1));
                    const auto pred_region = pred_feature.slice(
                                                             2, static_cast<std::size_t>(ly0), static_cast<std::size_t>(ly1))
                                                 .slice(3, static_cast<std::size_t>(lx0),
                                                        static_cast<std::size_t>(lx1));
                    const auto difference = pred_region.sub(target_region).square();
                    const auto linear_weight = w(std::format("lin{}.weight", block)).to(DataType::Float32);
                    const auto score = conv(difference, linear_weight);
                    const auto region_count = static_cast<std::size_t>(ly1 - ly0) *
                                              static_cast<std::size_t>(lx1 - lx0);
                    weighted[block] += static_cast<double>(score.mean().item<float>()) * region_count;
                    counts[block] += region_count;
                }
                target_taps = LpipsTaps{};
                pred_taps = LpipsTaps{};
            }
        }
        double total = 0.0;
        for (int block = 0; block < kBlocks; ++block) {
            const auto denominator = static_cast<std::size_t>(full_dim(height, block)) *
                                     static_cast<std::size_t>(full_dim(width, block));
            if (counts[block] != denominator)
                return lpips_error(lfs::ErrorCode::Internal, "LPIPS tiled coverage is incomplete");
            total += weighted[block] / static_cast<double>(denominator);
        }
        return static_cast<float>(total);
    }

    lfs::Result<float> Lpips::run_untiled(const Tensor& pred, const Tensor& target,
                                          const InputScaling scaling, LpipsTaps* taps) {
        if (auto error = validate_pair(pred, target))
            return std::move(*error);

        const auto scale_input = [&](const Tensor& input) {
            if (scaling == InputScaling::Normalize) {
                const Tensor scaled = as_batch(input)
                                          .mul(2.0f)
                                          .sub(1.0f)
                                          .sub(w("scaling.shift"))
                                          .div(w("scaling.scale"))
                                          .contiguous();
                return scaled;
            }
            const Tensor scaled = as_batch(input)
                                      .sub(w("scaling.shift"))
                                      .div(w("scaling.scale"))
                                      .contiguous();
            return scaled;
        };
        Tensor x = scale_input(pred);
        Tensor y = scale_input(target);
        if (compute_ != DataType::Float32) {
            x = cast(x, compute_);
            y = cast(y, compute_);
        }
        const cudaStream_t stream = x.stream();
        lfs::core::CUDAStreamGuard stream_guard(stream);
        bind_weights_to_stream(stream);
        ActivationArenaGuard arena_guard(arena_);
        arena_.begin(stream);
        struct ArenaCloser {
            ActivationArena& arena;
            ~ArenaCloser() {
                arena.end();
            }
        } arena_closer{arena_};

        auto persist_feature_pair = [&] {
            ActivationArena::bind(nullptr);
            recapture(feature_x_hold_, x);
            recapture(feature_y_hold_, y);
            x = Tensor{};
            y = Tensor{};
            ActivationArena::bind(&arena_);
            arena_.rewind(0);
            x = feature_x_hold_;
            y = feature_y_hold_;
        };
        persist_feature_pair();

        std::array<float, kBlocks> values{};
        int layer = 0;
        for (int i = 0; i <= 30; ++i) {
            if (i == 1 || i == 3 || i == 6 || i == 8 || i == 11 || i == 13 || i == 15 ||
                i == 18 || i == 20 || i == 22 || i == 25 || i == 27 || i == 29) {
                x = relu(x);
                y = relu(y);
            } else if (i == 4 || i == 9 || i == 16 || i == 23 || i == 30) {
                persist_feature_pair();
                const auto feature_mark = arena_.mark();
                {
                    auto normalized_x = normalize_feature(x);
                    ActivationArena::bind(nullptr);
                    recapture(normalized_x_hold_, normalized_x);
                    Tensor normalized_x_snapshot;
                    if (taps) {
                        normalized_x_snapshot = normalized_x_hold_.to(DataType::Float32).clone();
                    }
                    ActivationArena::bind(&arena_);
                    normalized_x = Tensor{};
                    arena_.rewind(feature_mark);

                    auto normalized_y = normalize_feature(y);
                    ActivationArena::bind(nullptr);
                    recapture(normalized_y_hold_, normalized_y);
                    ActivationArena::bind(&arena_);
                    normalized_y = Tensor{};
                    arena_.rewind(feature_mark);

                    auto diff = normalized_x_hold_.sub(normalized_y_hold_);
                    diff = diff.mul(diff);
                    auto score = conv(diff, std::format("lin{}.weight", layer));
                    values[static_cast<std::size_t>(layer)] =
                        score.mean().to(DataType::Float32).item<float>();
                    if (taps) {
                        taps->normalized_features[static_cast<std::size_t>(layer)] =
                            std::move(normalized_x_snapshot);
                        taps->scalars[static_cast<std::size_t>(layer)] =
                            values[static_cast<std::size_t>(layer)];
                    }
                    normalized_x = Tensor{};
                    normalized_y = Tensor{};
                    diff = Tensor{};
                    arena_.rewind(feature_mark);
                }
                ++layer;
            }
            if (i == 4 || i == 9 || i == 16 || i == 23 || i == 30) {
                if (i != 30) {
                    x = max_pool2d(x, 2, 2, 2, 2, 0, 0);
                    y = max_pool2d(y, 2, 2, 2, 2, 0, 0);
                    persist_feature_pair();
                }
            } else if (i == 0 || i == 2 || i == 5 || i == 7 || i == 10 || i == 12 ||
                       i == 14 || i == 17 || i == 19 || i == 21 || i == 24 || i == 26 || i == 28) {
                x = conv(x, std::format("vgg.features.{}.weight", i),
                         std::format("vgg.features.{}.bias", i));
                y = conv(y, std::format("vgg.features.{}.weight", i),
                         std::format("vgg.features.{}.bias", i));
            }
            if (i == 1 || i == 3 || i == 6 || i == 8 || i == 11 || i == 13 || i == 15 ||
                i == 18 || i == 20 || i == 22 || i == 25 || i == 27 || i == 29) {
                persist_feature_pair();
            }
        }
        float total = 0.0f;
        for (const float value : values)
            total += value;
        if (taps)
            LFS_CUDA_CHECK(cudaDeviceSynchronize());
        if (!std::isfinite(total))
            return lpips_error(lfs::ErrorCode::Internal, "LPIPS produced a non-finite value");
        return total;
    }

} // namespace lfs::core::nn::models
