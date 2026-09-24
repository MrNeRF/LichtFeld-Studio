/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/models/romav1.hpp"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/source_site.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "nn_kernels.hpp"
#include "nn_nvtx.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>
#include <vector>

namespace lfs::core::nn::models {
    namespace {

        constexpr int kDinoDim = 1024;
        constexpr int kDinoHeads = 16;
        constexpr int kDecoderDim = 1024;
        constexpr int kDecoderHeads = 8;
        constexpr int kGpDim = 512;
        constexpr float kLnEps = 1e-6f;
        constexpr float kRefineInit = 4.0f;
        // The trained displacement embedding expects inputs scaled by 40/32.
        constexpr float kDisplacementGain = 40.0f / 32.0f;
        // VGG19-BN: how many convolutions run before each captured scale.
        constexpr std::array<int, 4> kVggStageConvs{2, 4, 8, 12};
        constexpr std::array<int, 5> kRefinerRadius{7, 3, 2, 0, 0};

        const float kImagenetMean[3] = {0.485f, 0.456f, 0.406f};
        const float kImagenetStd[3] = {0.229f, 0.224f, 0.225f};

        TensorShape shape_of(std::initializer_list<std::size_t> dims) {
            return TensorShape(std::vector<std::size_t>(dims));
        }

        lfs::Error roma_error(const lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({
                .code = code,
                .domain = lfs::ErrorDomain::Core,
                .user_message = "RoMa v1 inference failed",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

    } // namespace

    lfs::Result<RomaV1> RomaV1::load(const std::filesystem::path& weights, Device device,
                                     std::optional<DataType> compute, int resolution) {
        if (device != Device::GPU) {
            return roma_error(lfs::ErrorCode::InvalidArgument, "RoMa v1 requires a GPU device");
        }
        auto file = WeightFile::open(weights);
        if (!file) {
            return std::move(file.error());
        }
        const DataType dtype = compute.value_or(DataType::Float16);
        if (dtype != DataType::Float16 && dtype != DataType::Float32) {
            return roma_error(lfs::ErrorCode::InvalidArgument,
                              "RoMa v1 compute dtype must be float16 or float32");
        }
        RomaV1 model;
        model.device_ = device;
        model.compute_ = dtype;
        model.resolution_ = resolution;
        const auto& meta = file->meta();
        if (meta.contains("coarse_classes")) {
            model.classes_side_ = meta["coarse_classes"].get<int>();
        }
        if (meta.contains("kernel_temperature")) {
            model.temperature_ = meta["kernel_temperature"].get<float>();
        }
        if (meta.contains("sigma_noise")) {
            model.sigma_noise_ = meta["sigma_noise"].get<float>();
        }
        if (resolution % (kPatch * 8) != 0) {
            return roma_error(
                lfs::ErrorCode::InvalidArgument,
                std::format("RoMa v1 resolution must be a multiple of {}, got {}", kPatch * 8,
                            resolution));
        }
        auto loaded = file->load_all(device, dtype);
        if (!loaded) {
            return std::move(loaded.error());
        }
        model.weights_ = std::move(*loaded);
        if (!model.weights_.contains(std::format("dino.pos_embed.{}", resolution))) {
            return roma_error(lfs::ErrorCode::NotFound,
                              std::format("the weight file has no position embedding baked for "
                                          "resolution {}",
                                          resolution));
        }
        for (const char* required : {"dino.patch_embed.weight", "dino.cls_token", "vgg.conv0.weight",
                                     "gp.pos_conv.weight", "dec.to_out.weight",
                                     "refiner.1.out_conv.weight"}) {
            if (!model.weights_.contains(required)) {
                return roma_error(lfs::ErrorCode::NotFound,
                                  std::format("weight file is missing {}", required));
            }
        }
        if (default_gpu_backend() == GpuBackend::CUDA && dtype == DataType::Float16 &&
            kernels::conv3x3_mma_available()) {
            for (const auto& [name, tensor] : model.weights_) {
                if (tensor.ndim() != 4 || tensor.shape()[2] != 3 || tensor.shape()[3] != 3 ||
                    tensor.shape()[1] % 8 != 0) {
                    continue;
                }
                auto taps =
                    Tensor::empty(shape_of({9, tensor.shape()[0], tensor.shape()[1]}), device, dtype);
                kernels::conv3x3_weight_taps(tensor.data_ptr(), taps.data_ptr(),
                                             static_cast<int>(tensor.shape()[0]),
                                             static_cast<int>(tensor.shape()[1]), taps.stream());
                model.weight_taps_.emplace(name, std::move(taps));
            }
            if (!model.weight_taps_.empty()) {
                LFS_CUDA_CHECK(cudaStreamSynchronize(model.weight_taps_.begin()->second.stream()));
            }
        }
        return model;
    }

    std::size_t RomaV1::weights_bytes() const {
        std::size_t bytes = 0;
        for (const auto& [name, tensor] : weights_) {
            (void)name;
            bytes += tensor.is_valid() ? tensor.bytes() : 0;
        }
        for (const auto& [name, tensor] : weight_taps_) {
            (void)name;
            bytes += tensor.is_valid() ? tensor.bytes() : 0;
        }
        return bytes;
    }

    const Tensor& RomaV1::w(std::string_view name) const {
        const auto it = weights_.find(std::string(name));
        LFS_ASSERT_MSG(it != weights_.end(), std::format("RoMa v1 weight {} is not loaded", name));
        return it->second;
    }

    Tensor RomaV1::ensure_workspace(std::size_t bytes, const Tensor& like) {
        if (bytes == 0) {
            return like;
        }
        if (workspace_.is_valid() && workspace_.bytes() >= bytes &&
            workspace_.dtype() == like.dtype() && workspace_.device() == like.device()) {
            workspace_.set_stream(like.stream());
            return workspace_;
        }
        const std::size_t elem = dtype_size(like.dtype());
        workspace_ = Tensor::empty(shape_of({(bytes + elem - 1) / elem}), like.device(),
                                   like.dtype());
        workspace_.set_stream(like.stream());
        return workspace_;
    }

    Tensor RomaV1::vit_block(const Tensor& x, const std::string& prefix, int heads,
                             bool layer_scale, bool qkv_bias) {
        NvtxRange nvtx("romav1/vit_block");
        auto n1 = layer_norm(x, w(prefix + ".norm1.weight"), w(prefix + ".norm1.bias"), kLnEps);
        const Tensor* bias = qkv_bias ? &w(prefix + ".attn.qkv.bias") : nullptr;
        auto qkv = gemm(n1, w(prefix + ".attn.qkv.weight"), false, true, bias);
        auto split = split_qkv(qkv, heads);
        auto ctx = merge_heads(attention(split[0], split[1], split[2]));
        const Tensor* ls1 = layer_scale ? &w(prefix + ".ls1.gamma") : nullptr;
        auto y = gemm(ctx, w(prefix + ".attn.proj.weight"), false, true,
                      &w(prefix + ".attn.proj.bias"), Activation::None, &x, ls1);
        auto n2 = layer_norm(y, w(prefix + ".norm2.weight"), w(prefix + ".norm2.bias"), kLnEps);
        auto h = gemm(n2, w(prefix + ".mlp.fc1.weight"), false, true, &w(prefix + ".mlp.fc1.bias"),
                      Activation::GeluErf);
        const Tensor* ls2 = layer_scale ? &w(prefix + ".ls2.gamma") : nullptr;
        return gemm(h, w(prefix + ".mlp.fc2.weight"), false, true, &w(prefix + ".mlp.fc2.bias"),
                    Activation::None, &y, ls2);
    }

    Tensor RomaV1::dino(const Tensor& normalized) {
        NvtxRange nvtx("romav1/dino");
        Conv2dParams p;
        p.stride_h = kPatch;
        p.stride_w = kPatch;
        const auto& pw = w("dino.patch_embed.weight");
        const auto bytes = conv2d_workspace_bytes(normalized.shape(), pw.shape(), p, compute_);
        Tensor ws = ensure_workspace(bytes, normalized);
        auto patches = conv2d(normalized, pw, &w("dino.patch_embed.bias"), p, &ws);

        const int th = static_cast<int>(patches.shape()[2]);
        const int tw = static_cast<int>(patches.shape()[3]);
        const std::size_t tokens = static_cast<std::size_t>(th) * tw;
        auto flat = patches.permute({0, 2, 3, 1}).contiguous().reshape(shape_of({1, tokens, kDinoDim}));

        auto x = Tensor::empty(shape_of({1, tokens + 1, kDinoDim}), device_, compute_);
        x.set_stream(stream_);
        const auto& cls = w("dino.cls_token");
        LFS_CUDA_CHECK(cudaMemcpyAsync(x.data_ptr(), cls.data_ptr(), cls.bytes(),
                                       cudaMemcpyDeviceToDevice, stream_));
        LFS_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(x.data_ptr()) + cls.bytes(),
                                       flat.data_ptr(), flat.bytes(), cudaMemcpyDeviceToDevice,
                                       stream_));
        x = x.add(w(std::format("dino.pos_embed.{}", resolution_))
                      .reshape(shape_of({1, tokens + 1, kDinoDim})));
        for (int i = 0; i < kDinoBlocks; ++i) {
            x = vit_block(x, std::format("dino.blocks.{}", i), kDinoHeads, true, true);
        }
        x = layer_norm(x, w("dino.norm.weight"), w("dino.norm.bias"), kLnEps);
        return x.slice(1, 1, tokens + 1).contiguous().reshape(shape_of({tokens, kDinoDim}));
    }

    std::vector<Tensor> RomaV1::vgg(const Tensor& normalized) {
        NvtxRange nvtx("romav1/vgg");
        Conv2dParams p;
        p.pad_h = 1;
        p.pad_w = 1;
        p.activation = Activation::Relu;
        Tensor x = normalized;
        std::vector<Tensor> out;
        out.reserve(kVggStageConvs.size());
        int conv = 0;
        for (std::size_t stage = 0; stage < kVggStageConvs.size(); ++stage) {
            while (conv < kVggStageConvs[stage]) {
                const std::string name = std::format("vgg.conv{}", conv);
                const auto& weight = w(name + ".weight");
                const auto tap_it = weight_taps_.find(name + ".weight");
                const Tensor* taps = tap_it == weight_taps_.end() ? nullptr : &tap_it->second;
                const auto bytes =
                    taps ? 0 : conv2d_workspace_bytes(x.shape(), weight.shape(), p, x.dtype());
                Tensor ws = ensure_workspace(bytes, x);
                x = conv2d(x, weight, &w(name + ".bias"), p, &ws, taps);
                ++conv;
            }
            out.push_back(x);
            x = max_pool2d(x, 2, 2, 2, 2, 0, 0);
        }
        return out;
    }

    lfs::Result<std::shared_ptr<RomaV1Image>> RomaV1::prepare(const Tensor& input) {
        // A dataset camera hands out [3, H, W], so accept that alongside the
        // batched and interleaved forms. An image that is both three rows tall
        // and three channels wide reads as interleaved.
        const bool hwc = input.ndim() == 3 && input.shape()[2] == 3;
        const bool chw = !hwc && input.ndim() == 3 && input.shape()[0] == 3;
        const bool nchw = input.ndim() == 4 && input.shape()[0] == 1 && input.shape()[1] == 3;
        if (!input.is_valid() || (!nchw && !hwc && !chw)) {
            return roma_error(
                lfs::ErrorCode::InvalidArgument,
                "RoMa v1 image must be [1, 3, H, W], [3, H, W] or [H, W, 3] uint8/float");
        }
        const Tensor image =
            chw ? input.reshape(shape_of({1, 3, input.shape()[1], input.shape()[2]})) : input;
        if (image.device() != Device::GPU) {
            return roma_error(lfs::ErrorCode::InvalidArgument, "RoMa v1 image must be on the GPU");
        }
        NvtxRange nvtx("romav1/prepare");
        GpuBackendScope backend_scope(*gpu_backend_of(image));
        stream_ = image.stream();
        lfs::core::CUDAStreamGuard stream_guard(stream_);
        if (!weights_on_stream_) {
            for (auto& [name, tensor] : weights_) {
                (void)name;
                tensor.set_stream(stream_);
            }
            for (auto& [name, tensor] : weight_taps_) {
                (void)name;
                tensor.set_stream(stream_);
            }
            weights_on_stream_ = true;
        }

        const int in_h = static_cast<int>(hwc ? image.shape()[0] : image.shape()[2]);
        const int in_w = static_cast<int>(hwc ? image.shape()[1] : image.shape()[3]);
        if (in_h <= 0 || in_w <= 0) {
            return roma_error(lfs::ErrorCode::InvalidArgument, "RoMa v1 image is empty");
        }
        auto out = std::make_shared<RomaV1Image>();
        auto resized = Tensor::empty(shape_of({1, 3, static_cast<std::size_t>(resolution_),
                                               static_cast<std::size_t>(resolution_)}),
                                     device_, DataType::Float32);
        resized.set_stream(stream_);
        auto scratch = Tensor::empty(shape_of({3, static_cast<std::size_t>(in_h),
                                               static_cast<std::size_t>(resolution_)}),
                                     device_, DataType::Float32);
        scratch.set_stream(stream_);
        if (hwc) {
            auto src = image.contiguous();
            src.set_stream(stream_);
            kernels::resize_bicubic_aa_hwc(src.data_ptr(), resized.ptr<float>(),
                                           scratch.ptr<float>(), in_h, in_w, 3, resolution_,
                                           resolution_, src.dtype(), stream_);
        } else if (in_h == resolution_ && in_w == resolution_) {
            resized = image.dtype() == DataType::Float32 ? image.contiguous()
                                                         : image.to(DataType::Float32).contiguous();
            resized.set_stream(stream_);
        } else {
            auto src = image.dtype() == DataType::Float32 ? image.contiguous()
                                                          : image.to(DataType::Float32).contiguous();
            src.set_stream(stream_);
            kernels::resize_bicubic_aa(src.ptr<float>(), resized.ptr<float>(), scratch.ptr<float>(),
                                       1, 3, in_h, in_w, resolution_, resolution_, stream_);
        }
        out->image = resized;

        auto normalized = Tensor::empty(resized.shape(), device_, compute_);
        normalized.set_stream(stream_);
        kernels::normalize_image(resized.data_ptr(), normalized.data_ptr(),
                                 resolution_ * resolution_, kImagenetMean, kImagenetStd,
                                 resized.dtype(), compute_, stream_);

        auto fine = vgg(normalized);
        auto coarse = dino(normalized);

        out->levels.resize(kScales.size());
        out->sizes.resize(kScales.size());
        for (std::size_t i = 0; i < kScales.size(); ++i) {
            const int scale = kScales[i];
            Tensor channel_last;
            if (scale == kCoarseStride) {
                channel_last = coarse;
            } else {
                // The VGG pyramid is channel-first; the rest of the model runs
                // channel-last, so transpose once here.
                const auto& feat =
                    fine[static_cast<std::size_t>(std::lround(std::log2(scale)))];
                const int channels = static_cast<int>(feat.shape()[1]);
                const std::size_t pixels = feat.shape()[2] * feat.shape()[3];
                channel_last = Tensor::empty(shape_of({pixels, static_cast<std::size_t>(channels)}),
                                             device_, compute_);
                channel_last.set_stream(stream_);
                kernels::transpose2d(feat.data_ptr(), channel_last.data_ptr(), channels,
                                     static_cast<int>(pixels), compute_, stream_);
            }
            const int extent = level_extent(scale);
            out->sizes[i] = {extent, extent};
            const std::string proj = std::format("proj.{}", scale);
            out->levels[i] = gemm(channel_last, w(proj + ".weight"), false, true,
                                  &w(proj + ".bias"));
        }
        return out;
    }

    Tensor RomaV1::gaussian_process(const Tensor& f_a, const Tensor& f_b, int height, int width) {
        NvtxRange nvtx("romav1/gaussian_process");
        const int pixels = height * width;
        const int dim = static_cast<int>(f_a.shape()[1]);

        // Fourier basis over image B's pixel-centre grid.
        auto grid = Tensor::empty(shape_of({static_cast<std::size_t>(pixels), 2}), device_,
                                  DataType::Float32);
        grid.set_stream(stream_);
        kernels::normalized_grid(grid.ptr<float>(), height, width, stream_);
        auto angles = gemm(grid, w("gp.pos_conv.weight").to(DataType::Float32), false, true,
                           nullptr);
        auto bias = w("gp.pos_conv.bias").to(DataType::Float32);
        angles = angles.add(bias);
        auto basis = Tensor::empty(angles.shape(), device_, DataType::Float32);
        basis.set_stream(stream_);
        kernels::fourier_cos(angles.ptr<float>(), basis.ptr<float>(),
                             static_cast<long long>(pixels) * kGpDim, stream_);

        // Cosine kernels. Normalising the rows first turns the kernel into a
        // plain Gram matrix followed by an elementwise exponential.
        auto norm_a = Tensor::empty(shape_of({static_cast<std::size_t>(pixels),
                                              static_cast<std::size_t>(dim)}),
                                    device_, DataType::Float32);
        auto norm_b = Tensor::empty(norm_a.shape(), device_, DataType::Float32);
        norm_a.set_stream(stream_);
        norm_b.set_stream(stream_);
        kernels::l2_normalize_rows(f_a.data_ptr(), norm_a.ptr<float>(), pixels, dim, compute_,
                                   stream_);
        kernels::l2_normalize_rows(f_b.data_ptr(), norm_b.ptr<float>(), pixels, dim, compute_,
                                   stream_);
        auto k_yy = gemm(norm_b, norm_b, false, true);
        kernels::cosine_kernel_inplace(k_yy.ptr<float>(),
                                       static_cast<long long>(pixels) * pixels, temperature_,
                                       stream_);
        kernels::add_diagonal(k_yy.ptr<float>(), pixels, sigma_noise_, stream_);
        auto k_xy = gemm(norm_a, norm_b, false, true);
        kernels::cosine_kernel_inplace(k_xy.ptr<float>(),
                                       static_cast<long long>(pixels) * pixels, temperature_,
                                       stream_);
        // The iteration only has to reach float16 accuracy, so the matrix-vector
        // product runs on tensor cores while the residual stays in float32.
        auto k_yy_compute = k_yy.to(compute_);
        auto k_xy_compute = k_xy.to(compute_);

        // Conjugate gradients solve (K_yy + sigma I) z = basis. The system is
        // well conditioned, so a fixed iteration count beats factorising it.
        auto z = Tensor::zeros(basis.shape(), device_, DataType::Float32);
        auto r = basis.clone();
        auto p = basis.clone();
        z.set_stream(stream_);
        r.set_stream(stream_);
        p.set_stream(stream_);
        auto rs = Tensor::empty(shape_of({kGpDim}), device_, DataType::Float32);
        auto rs_new = Tensor::empty(rs.shape(), device_, DataType::Float32);
        auto pap = Tensor::empty(rs.shape(), device_, DataType::Float32);
        rs.set_stream(stream_);
        rs_new.set_stream(stream_);
        pap.set_stream(stream_);
        kernels::column_dot(r.ptr<float>(), r.ptr<float>(), rs.ptr<float>(), pixels, kGpDim,
                            stream_);
        for (int i = 0; i < cg_iterations_; ++i) {
            auto ap = gemm(k_yy_compute, p.to(compute_), false, false).to(DataType::Float32);
            kernels::column_dot(p.ptr<float>(), ap.ptr<float>(), pap.ptr<float>(), pixels, kGpDim,
                                stream_);
            kernels::cg_step(z.ptr<float>(), r.ptr<float>(), p.ptr<float>(), ap.ptr<float>(),
                             rs.ptr<float>(), pap.ptr<float>(), pixels, kGpDim, stream_);
            kernels::column_dot(r.ptr<float>(), r.ptr<float>(), rs_new.ptr<float>(), pixels,
                                kGpDim, stream_);
            kernels::cg_direction(p.ptr<float>(), r.ptr<float>(), rs_new.ptr<float>(),
                                  rs.ptr<float>(), pixels, kGpDim, stream_);
            std::swap(rs, rs_new);
        }
        return gemm(k_xy_compute, z.to(compute_), false, false);
    }

    Tensor RomaV1::coarse_decoder(const Tensor& gp_feats, const Tensor& f_a) {
        NvtxRange nvtx("romav1/coarse_decoder");
        const std::size_t pixels = gp_feats.shape()[0];
        const std::size_t elem = dtype_size(compute_);
        auto tokens = Tensor::empty(shape_of({1, pixels, 2 * kGpDim}), device_, compute_);
        tokens.set_stream(stream_);
        const std::size_t row = kGpDim * elem;
        LFS_CUDA_CHECK(cudaMemcpy2DAsync(tokens.data_ptr(), 2 * row, gp_feats.data_ptr(), row, row,
                                         pixels, cudaMemcpyDeviceToDevice, stream_));
        LFS_CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(tokens.data_ptr()) + row, 2 * row,
                                         f_a.data_ptr(), row, row, pixels,
                                         cudaMemcpyDeviceToDevice, stream_));
        for (int i = 0; i < kDecoderBlocks; ++i) {
            tokens = vit_block(tokens, std::format("dec.blocks.{}", i), kDecoderHeads, false,
                               false);
        }
        return gemm(tokens.reshape(shape_of({pixels, kDecoderDim})), w("dec.to_out.weight"), false,
                    true, &w("dec.to_out.bias"));
    }

    Tensor RomaV1::refiner(int scale, const Tensor& f_a, const Tensor& f_b, const Tensor& flow,
                           int height, int width) {
        NvtxRange nvtx("romav1/refiner");
        const std::string prefix = std::format("refiner.{}", scale);
        const int channels = static_cast<int>(f_a.shape()[1]);
        const std::size_t pixels = static_cast<std::size_t>(height) * width;
        const std::size_t elem = dtype_size(compute_);
        const int radius = kRefinerRadius[static_cast<std::size_t>(std::lround(
            std::log2(static_cast<double>(kScales[0]) / scale)))];
        const int corr_channels = radius > 0 ? (2 * radius + 1) * (2 * radius + 1) : 0;
        const int emb_channels = static_cast<int>(w(prefix + ".disp_emb.weight").shape()[0]);
        const int used = 2 * channels + emb_channels + corr_channels;
        // The exporter pads the stack to an even channel count for the
        // channel-pair depthwise kernel; the pad channel stays zero throughout.
        const int hidden = static_cast<int>(w(prefix + ".block0.dw.weight").shape()[1]);

        auto d = Tensor::empty(shape_of({pixels, static_cast<std::size_t>(hidden)}), device_,
                               compute_);
        d.set_stream(stream_);
        char* base = static_cast<char*>(d.data_ptr());
        const std::size_t pitch = static_cast<std::size_t>(hidden) * elem;
        auto place = [&](std::size_t offset, const void* src, std::size_t count) {
            LFS_CUDA_CHECK(cudaMemcpy2DAsync(base + offset * elem, pitch, src, count * elem,
                                             count * elem, pixels, cudaMemcpyDeviceToDevice,
                                             stream_));
        };
        if (hidden > used) {
            LFS_CUDA_CHECK(cudaMemset2DAsync(base + static_cast<std::size_t>(used) * elem, pitch, 0,
                                             static_cast<std::size_t>(hidden - used) * elem, pixels,
                                             stream_));
        }
        place(0, f_a.data_ptr(), static_cast<std::size_t>(channels));

        auto warped = Tensor::empty(shape_of({pixels, static_cast<std::size_t>(channels)}), device_,
                                    compute_);
        warped.set_stream(stream_);
        kernels::grid_sample_bhwc(f_b.data_ptr(), flow.ptr<float>(), warped.data_ptr(), channels,
                                  height, width, height, width, compute_, stream_);
        place(static_cast<std::size_t>(channels), warped.data_ptr(),
              static_cast<std::size_t>(channels));

        auto disp_in = Tensor::empty(shape_of({pixels, 2}), device_, compute_);
        disp_in.set_stream(stream_);
        const float gain =
            kDisplacementGain * static_cast<float>(resolution_) / static_cast<float>(kTrainResolution);
        kernels::displacement_input(flow.ptr<float>(), disp_in.data_ptr(), height, width, gain,
                                    gain, compute_, stream_);
        auto emb = gemm(disp_in, w(prefix + ".disp_emb.weight"), false, true,
                        &w(prefix + ".disp_emb.bias"));
        place(static_cast<std::size_t>(2 * channels), emb.data_ptr(),
              static_cast<std::size_t>(emb_channels));

        if (radius > 0) {
            auto corr = Tensor::empty(shape_of({pixels, static_cast<std::size_t>(corr_channels)}),
                                      device_, compute_);
            corr.set_stream(stream_);
            kernels::local_correlation(f_a.data_ptr(), f_b.data_ptr(), flow.ptr<float>(),
                                       corr.data_ptr(), channels, height, width, radius, compute_,
                                       stream_);
            place(static_cast<std::size_t>(2 * channels + emb_channels), corr.data_ptr(),
                  static_cast<std::size_t>(corr_channels));
        }

        auto z = d;
        auto scratch = Tensor::empty(d.shape(), device_, compute_);
        scratch.set_stream(stream_);
        for (int b = 0; b < kRefinerBlocks; ++b) {
            const std::string block = std::format("{}.block{}", prefix, b);
            kernels::depthwise_conv2d_nhwc(z.data_ptr(), w(block + ".dw.weight").data_ptr(),
                                           w(block + ".dw.bias").data_ptr(), scratch.data_ptr(),
                                           hidden, height, width, 5,
                                           static_cast<int>(Activation::Relu), compute_, stream_);
            z = gemm(scratch, w(block + ".pw.weight"), false, true, &w(block + ".pw.bias"));
        }
        return gemm(z, w(prefix + ".out_conv.weight"), false, true, &w(prefix + ".out_conv.bias"));
    }

    lfs::Result<RomaMatch> RomaV1::match(const RomaV1Image& a, const RomaV1Image& b) {
        NvtxRange nvtx("romav1/match");
        if (a.levels.size() != kScales.size() || b.levels.size() != kScales.size()) {
            return roma_error(lfs::ErrorCode::InvalidArgument, "RoMa v1 images are not prepared");
        }
        GpuBackendScope backend_scope(*gpu_backend_of(a.levels[0]));
        stream_ = a.levels[0].stream();
        lfs::core::CUDAStreamGuard stream_guard(stream_);

        Tensor flow;
        Tensor certainty;
        Tensor coarse_certainty;
        auto resize_channel_last = [&](const Tensor& src, int channels, int in_h, int in_w,
                                       int out_h, int out_w) {
            const std::size_t in_pixels = static_cast<std::size_t>(in_h) * in_w;
            auto planar = Tensor::empty(shape_of({1, static_cast<std::size_t>(channels),
                                                  static_cast<std::size_t>(in_h),
                                                  static_cast<std::size_t>(in_w)}),
                                        device_, DataType::Float32);
            planar.set_stream(stream_);
            kernels::transpose2d(src.data_ptr(), planar.data_ptr(), static_cast<int>(in_pixels),
                                 channels, DataType::Float32, stream_);
            auto scaled = resize2d(planar, out_h, out_w, ResizeMode::Bilinear,
                                   CoordTransform::HalfPixel);
            const std::size_t out_pixels = static_cast<std::size_t>(out_h) * out_w;
            auto packed = Tensor::empty(shape_of({out_pixels, static_cast<std::size_t>(channels)}),
                                        device_, DataType::Float32);
            packed.set_stream(stream_);
            kernels::transpose2d(scaled.data_ptr(), packed.data_ptr(), channels,
                                 static_cast<int>(out_pixels), DataType::Float32, stream_);
            return packed;
        };

        for (std::size_t i = 0; i < kScales.size(); ++i) {
            const int scale = kScales[i];
            const int height = a.sizes[i].first;
            const int width = a.sizes[i].second;
            const std::size_t pixels = static_cast<std::size_t>(height) * width;
            const Tensor& f_a = a.levels[i];
            const Tensor& f_b = b.levels[i];

            if (scale == kCoarseStride) {
                auto gp_feats = gaussian_process(f_a, f_b, height, width);
                auto logits = coarse_decoder(gp_feats, f_a);
                const int classes = classes_side_ * classes_side_;
                flow = Tensor::empty(shape_of({pixels, 2}), device_, DataType::Float32);
                flow.set_stream(stream_);
                // The decoder emits one row per pixel: `classes` class logits
                // followed by the certainty, which the arg-max must not see.
                kernels::cls_to_flow(logits.data_ptr(), flow.ptr<float>(),
                                     static_cast<int>(pixels), classes, classes + 1,
                                     classes_side_, compute_, stream_);
                certainty = logits.slice(1, classes, classes + 1)
                                .contiguous()
                                .to(DataType::Float32);
                certainty.set_stream(stream_);
                coarse_certainty = certainty;
            }

            auto delta = refiner(scale, f_a, f_b, flow, height, width);
            auto new_flow = Tensor::empty(flow.shape(), device_, DataType::Float32);
            auto new_cert = Tensor::empty(certainty.shape(), device_, DataType::Float32);
            new_flow.set_stream(stream_);
            new_cert.set_stream(stream_);
            const float step = static_cast<float>(scale) / (kRefineInit * resolution_);
            kernels::v1_refiner_update(flow.ptr<float>(), certainty.ptr<float>(), delta.data_ptr(),
                                       new_flow.ptr<float>(), new_cert.ptr<float>(),
                                       static_cast<int>(pixels), step, compute_, stream_);
            flow = std::move(new_flow);
            certainty = std::move(new_cert);

            if (scale != 1) {
                const int next_h = a.sizes[i + 1].first;
                const int next_w = a.sizes[i + 1].second;
                flow = resize_channel_last(flow, 2, height, width, next_h, next_w);
                certainty = resize_channel_last(certainty, 1, height, width, next_h, next_w);
            }
        }

        const std::size_t pixels = static_cast<std::size_t>(resolution_) * resolution_;
        auto low = resize_channel_last(coarse_certainty, 1, a.sizes[0].first, a.sizes[0].second,
                                       resolution_, resolution_);
        auto probability = Tensor::empty(shape_of({pixels}), device_, DataType::Float32);
        probability.set_stream(stream_);
        kernels::attenuate_certainty(certainty.ptr<float>(), low.ptr<float>(),
                                     probability.ptr<float>(), static_cast<int>(pixels), stream_);

        RomaMatch result;
        result.warp = flow.reshape(shape_of({static_cast<std::size_t>(resolution_),
                                             static_cast<std::size_t>(resolution_), 2}));
        result.overlap = probability.reshape(shape_of({static_cast<std::size_t>(resolution_),
                                                       static_cast<std::size_t>(resolution_)}));
        return result;
    }

    lfs::Result<RomaMatch> RomaV1::match_with_grid(const RomaV1Image& a, const RomaV1Image& b) {
        auto matched = match(a, b);
        if (!matched) {
            return std::move(matched.error());
        }
        const std::size_t rows =
            static_cast<std::size_t>(resolution_) * static_cast<std::size_t>(resolution_);
        auto packed = Tensor::empty(shape_of({static_cast<std::size_t>(resolution_),
                                              static_cast<std::size_t>(resolution_), 4}),
                                    device_, DataType::Float32);
        packed.set_stream(stream_);
        auto grid = Tensor::empty(shape_of({rows, 2}), device_, DataType::Float32);
        grid.set_stream(stream_);
        kernels::normalized_grid(grid.ptr<float>(), resolution_, resolution_, stream_);

        constexpr std::size_t kPair = 2 * sizeof(float);
        constexpr std::size_t kRow = 4 * sizeof(float);
        char* dst = static_cast<char*>(packed.data_ptr());
        LFS_CUDA_CHECK(cudaMemcpy2DAsync(dst, kRow, grid.data_ptr(), kPair, kPair, rows,
                                         cudaMemcpyDeviceToDevice, stream_));
        LFS_CUDA_CHECK(cudaMemcpy2DAsync(dst + kPair, kRow, matched->warp.data_ptr(), kPair, kPair,
                                         rows, cudaMemcpyDeviceToDevice, stream_));
        matched->warp = std::move(packed);
        return matched;
    }

} // namespace lfs::core::nn::models
