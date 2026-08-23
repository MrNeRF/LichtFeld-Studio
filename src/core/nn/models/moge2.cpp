/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/models/moge2.hpp"

#include "core/assert.hpp"
#include "core/source_site.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>

namespace lfs::core::nn::models {
    namespace {

        constexpr int kPatch = 14;
        constexpr int kDim = 768;
        constexpr int kHeads = 12;
        constexpr int kHeadDim = 64;
        constexpr int kMlp = 3072;
        constexpr int kBlocks = 12;
        constexpr float kLnEps = 1e-6f;
        constexpr float kNormEps = 1e-12f;

        TensorShape shape_of(std::initializer_list<std::size_t> dims) {
            return TensorShape(std::vector<std::size_t>(dims));
        }

        std::string matmul_name(int block, const char* which) {
            const int base = 3480 + 6 * block;
            int id = base;
            if (which[0] == 'p') {
                id = base + 3;
            } else if (which[0] == '1') {
                id = base + 4;
            } else if (which[0] == '2') {
                id = base + 5;
            }
            return std::format("onnx::MatMul_{}", id);
        }

        std::string block_name(int block, std::string_view suffix) {
            return std::format("encoder.backbone.blocks.{}.{}", block, suffix);
        }

        lfs::Error moge_error(const lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({
                .code = code,
                .domain = lfs::ErrorDomain::Core,
                .user_message = "MoGe-2 inference failed",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        Tensor as_compute(Tensor t, DataType dtype) {
            if (t.dtype() != dtype) {
                t = t.to(dtype);
            }
            return t;
        }

        Tensor l2_normalize_last(const Tensor& x) {
            auto nrm = x.square().sum(-1, true).sqrt().clamp(kNormEps, 1e30f);
            return x.div(nrm);
        }

    } // namespace

    void Moge2::token_grid(int image_h, int image_w, std::int64_t num_tokens, int& token_h,
                           int& token_w) {
        const float aspect = static_cast<float>(image_w) / static_cast<float>(std::max(image_h, 1));
        const double nt = static_cast<double>(std::max<std::int64_t>(num_tokens, 1));
        token_h = static_cast<int>(std::round(std::sqrt(nt / static_cast<double>(aspect))));
        token_w = static_cast<int>(std::round(std::sqrt(nt * static_cast<double>(aspect))));
        token_h = std::max(token_h, 1);
        token_w = std::max(token_w, 1);
    }

    lfs::Result<Moge2> Moge2::load(const std::filesystem::path& weights, Device device,
                                   std::optional<DataType> compute) {
        if (device != Device::CUDA) {
            return moge_error(lfs::ErrorCode::InvalidArgument,
                              "MoGe-2 requires a CUDA device");
        }
        auto file = WeightFile::open(weights);
        if (!file) {
            return std::move(file.error());
        }
        DataType dtype = DataType::Float32;
        if (compute) {
            dtype = *compute;
        } else {
            for (const auto& name : file->names()) {
                const auto* info = file->info(name);
                if (info && info->dtype == DataType::Float16) {
                    dtype = DataType::Float16;
                    break;
                }
            }
        }
        if (dtype != DataType::Float16 && dtype != DataType::Float32) {
            return moge_error(lfs::ErrorCode::InvalidArgument,
                              "MoGe-2 compute dtype must be float16 or float32");
        }
        auto loaded = file->load_all(device, dtype);
        if (!loaded) {
            return std::move(loaded.error());
        }
        Moge2 model;
        model.device_ = device;
        model.compute_ = dtype;
        model.weights_ = std::move(*loaded);
        const std::array<const char*, 8> required = {
            "encoder.image_mean",
            "encoder.image_std",
            "encoder.backbone.cls_token",
            "encoder.backbone.patch_embed.proj.weight",
            "onnx::Reshape_3473",
            "onnx::Expand_3477",
            "encoder.backbone.norm.weight",
            "scale_head.4.weight",
        };
        for (const char* name : required) {
            if (!model.weights_.contains(name)) {
                return moge_error(lfs::ErrorCode::NotFound,
                                  std::format("weight file is missing {}", name));
            }
        }
        return model;
    }

    const Tensor& Moge2::w(std::string_view name) const {
        const auto it = weights_.find(std::string(name));
        LFS_ASSERT_MSG(it != weights_.end(),
                       std::format("MoGe-2 weight {} is not loaded", name));
        return it->second;
    }

    Tensor Moge2::ensure_workspace(std::size_t bytes, const Tensor& like) {
        if (workspace_.is_valid() && workspace_.bytes() >= bytes &&
            workspace_.dtype() == like.dtype() && workspace_.device() == like.device()) {
            workspace_.set_stream(like.stream());
            return workspace_;
        }
        const std::size_t elem = dtype_size(like.dtype());
        const std::size_t count = (bytes + elem - 1) / elem;
        workspace_ = Tensor::empty(shape_of({count}), like.device(), like.dtype());
        workspace_.set_stream(like.stream());
        return workspace_;
    }

    Tensor Moge2::conv1x1(const Tensor& x, std::string_view weight, std::string_view bias) {
        Conv2dParams p;
        const auto& ww = w(weight);
        const auto bytes = conv2d_workspace_bytes(x.shape(), ww.shape(), p, x.dtype());
        Tensor ws = ensure_workspace(bytes, x);
        return conv2d(x, ww, &w(bias), p, &ws);
    }

    Tensor Moge2::conv3x3(const Tensor& x, std::string_view weight, std::string_view bias) {
        Conv2dParams p;
        p.pad_h = 1;
        p.pad_w = 1;
        p.pad_mode = ConvPadMode::Replicate;
        const auto& ww = w(weight);
        const auto bytes = conv2d_workspace_bytes(x.shape(), ww.shape(), p, x.dtype());
        Tensor ws = ensure_workspace(bytes, x);
        return conv2d(x, ww, &w(bias), p, &ws);
    }

    Tensor Moge2::conv_transpose2x(const Tensor& x, std::string_view weight,
                                   std::string_view bias) {
        Conv2dParams p;
        p.stride_h = 2;
        p.stride_w = 2;
        const auto& ww = w(weight);
        const auto bytes = conv_transpose2d_workspace_bytes(x.shape(), ww.shape(), p, x.dtype());
        Tensor ws = ensure_workspace(bytes, x);
        return conv_transpose2d(x, ww, &w(bias), p, &ws);
    }

    Tensor Moge2::gemm_nn(const Tensor& x, std::string_view weight, std::string_view bias) {
        return gemm(x, w(weight), false, false, &w(bias));
    }

    Tensor Moge2::res_block(const Tensor& x, std::string_view w1, std::string_view b1,
                            std::string_view w2, std::string_view b2) {
        auto y = relu(x);
        y = conv3x3(y, w1, b1);
        y = relu(y);
        y = conv3x3(y, w2, b2);
        return x.add(y);
    }

    Tensor Moge2::vit_block(const Tensor& x, int index) {
        const auto bsz = x.shape()[0];
        const auto seq = x.shape()[1];
        auto n1 = layer_norm(x, w(block_name(index, "norm1.weight")),
                             w(block_name(index, "norm1.bias")), kLnEps);
        auto qkv = gemm_nn(n1, matmul_name(index, "qkv"), block_name(index, "attn.qkv.bias"));
        auto five = qkv.reshape(shape_of({bsz, seq, 3, static_cast<std::size_t>(kHeads),
                                          static_cast<std::size_t>(kHeadDim)}));
        auto q = five.slice(2, 0, 1)
                     .squeeze(2)
                     .permute({0, 2, 1, 3})
                     .contiguous();
        auto k = five.slice(2, 1, 2)
                     .squeeze(2)
                     .permute({0, 2, 1, 3})
                     .contiguous();
        auto v = five.slice(2, 2, 3)
                     .squeeze(2)
                     .permute({0, 2, 1, 3})
                     .contiguous();
        auto attn = attention(q, k, v)
                        .permute({0, 2, 1, 3})
                        .contiguous()
                        .reshape(shape_of({bsz, seq, static_cast<std::size_t>(kDim)}));
        auto proj = gemm_nn(attn, matmul_name(index, "proj"), block_name(index, "attn.proj.bias"));
        auto gamma1 = w(block_name(index, "ls1.gamma"))
                          .reshape(shape_of({1, 1, static_cast<std::size_t>(kDim)}));
        auto y = x.add(proj.mul(gamma1));

        auto n2 = layer_norm(y, w(block_name(index, "norm2.weight")),
                             w(block_name(index, "norm2.bias")), kLnEps);
        auto h = gemm_nn(n2, matmul_name(index, "1"), block_name(index, "mlp.fc1.bias"));
        h = gelu(h, GELUApprox::Erf);
        h = gemm_nn(h, matmul_name(index, "2"), block_name(index, "mlp.fc2.bias"));
        auto gamma2 = w(block_name(index, "ls2.gamma"))
                          .reshape(shape_of({1, 1, static_cast<std::size_t>(kDim)}));
        return y.add(h.mul(gamma2));
    }

    Tensor Moge2::interpolate_pos_embed(int batch, int token_h, int token_w, DataType dtype,
                                        cudaStream_t stream) {
        auto patches = w("onnx::Reshape_3473");
        const int src = static_cast<int>(
            std::lround(std::sqrt(static_cast<double>(patches.shape()[1]))));
        auto spatial = patches
                           .reshape(shape_of({1, static_cast<std::size_t>(src),
                                              static_cast<std::size_t>(src),
                                              static_cast<std::size_t>(kDim)}))
                           .permute({0, 3, 1, 2})
                           .contiguous();
        spatial.set_stream(stream);
        auto resized = resize2d(spatial, token_h, token_w, ResizeMode::Cubic,
                                CoordTransform::HalfPixel);
        auto flat = resized.permute({0, 2, 3, 1})
                        .contiguous()
                        .reshape(shape_of({1, static_cast<std::size_t>(token_h) * static_cast<std::size_t>(token_w),
                                           static_cast<std::size_t>(kDim)}));
        auto cls = w("onnx::Expand_3477");
        auto pe = Tensor::cat({cls, flat}, 1);
        if (batch > 1) {
            pe = pe.expand(shape_of({static_cast<std::size_t>(batch), pe.shape()[1], pe.shape()[2]}))
                     .contiguous();
        }
        return as_compute(std::move(pe), dtype);
    }

    Tensor Moge2::uv_map(int height, int width, float aspect, DataType dtype, Device device,
                         cudaStream_t stream) const {
        const float span_x = aspect / std::sqrt(1.0f + aspect * aspect);
        const float span_y = 1.0f / std::sqrt(1.0f + aspect * aspect);
        const float u0 = -span_x * static_cast<float>(width - 1) / static_cast<float>(width);
        const float u1 = span_x * static_cast<float>(width - 1) / static_cast<float>(width);
        const float v0 = -span_y * static_cast<float>(height - 1) / static_cast<float>(height);
        const float v1 = span_y * static_cast<float>(height - 1) / static_cast<float>(height);
        std::vector<float> data(static_cast<std::size_t>(2) * height * width);
        for (int y = 0; y < height; ++y) {
            const float vv = (height == 1)
                                 ? v0
                                 : v0 + (v1 - v0) * static_cast<float>(y) /
                                            static_cast<float>(height - 1);
            for (int x = 0; x < width; ++x) {
                const float uu = (width == 1)
                                     ? u0
                                     : u0 + (u1 - u0) * static_cast<float>(x) /
                                                static_cast<float>(width - 1);
                const std::size_t pix = static_cast<std::size_t>(y) * width + x;
                data[pix] = uu;
                data[static_cast<std::size_t>(height) * width + pix] = vv;
            }
        }
        auto t = Tensor::from_vector(
            data,
            shape_of({1, 2, static_cast<std::size_t>(height), static_cast<std::size_t>(width)}),
            Device::CPU);
        t = t.to(device);
        t.set_stream(stream);
        return as_compute(std::move(t), dtype);
    }

    lfs::Result<Moge2Outputs> Moge2::forward(const Tensor& image, std::int64_t num_tokens) {
        return run(image, num_tokens, nullptr);
    }

    lfs::Result<std::pair<Moge2Outputs, Moge2Taps>>
    Moge2::forward_with_taps(const Tensor& image, std::int64_t num_tokens) {
        Moge2Taps taps;
        auto out = run(image, num_tokens, &taps);
        if (!out) {
            return std::move(out.error());
        }
        return std::make_pair(std::move(*out), std::move(taps));
    }

    lfs::Result<Moge2Outputs> Moge2::run(const Tensor& image, std::int64_t num_tokens,
                                         Moge2Taps* taps) {
        if (!image.is_valid() || image.ndim() != 4 || image.shape()[1] != 3) {
            return moge_error(lfs::ErrorCode::InvalidArgument,
                              "MoGe-2 image must be NCHW with 3 channels");
        }
        if (image.device() != Device::CUDA) {
            return moge_error(lfs::ErrorCode::InvalidArgument, "MoGe-2 image must be on CUDA");
        }
        if (num_tokens <= 0) {
            return moge_error(lfs::ErrorCode::InvalidArgument, "num_tokens must be positive");
        }

        const int batch = static_cast<int>(image.shape()[0]);
        const int img_h = static_cast<int>(image.shape()[2]);
        const int img_w = static_cast<int>(image.shape()[3]);
        int token_h = 0;
        int token_w = 0;
        token_grid(img_h, img_w, num_tokens, token_h, token_w);
        const int enc_h = token_h * kPatch;
        const int enc_w = token_w * kPatch;
        const float aspect = static_cast<float>(img_w) / static_cast<float>(std::max(img_h, 1));

        Tensor img = image.contiguous();
        if (img.dtype() != DataType::Float32) {
            img = img.to(DataType::Float32);
        }
        img = resize2d(img, enc_h, enc_w, ResizeMode::Bilinear, CoordTransform::HalfPixel);
        auto mean = w("encoder.image_mean").to(DataType::Float32);
        auto stdv = w("encoder.image_std").to(DataType::Float32);
        img = img.sub(mean).div(stdv);
        img = as_compute(std::move(img), compute_);

        Conv2dParams patch;
        patch.stride_h = kPatch;
        patch.stride_w = kPatch;
        const auto& pw = w("encoder.backbone.patch_embed.proj.weight");
        const auto pbytes = conv2d_workspace_bytes(img.shape(), pw.shape(), patch, img.dtype());
        Tensor pws = ensure_workspace(pbytes, img);
        auto conv = conv2d(img, pw, &w("encoder.backbone.patch_embed.proj.bias"), patch, &pws);
        auto tokens = conv.reshape(shape_of({static_cast<std::size_t>(batch),
                                             static_cast<std::size_t>(kDim),
                                             static_cast<std::size_t>(token_h * token_w)}))
                          .permute({0, 2, 1})
                          .contiguous();
        if (taps) {
            taps->patch_embed = tokens.to(DataType::Float32);
        }

        auto cls = w("encoder.backbone.cls_token");
        if (batch > 1) {
            cls = cls.expand(shape_of({static_cast<std::size_t>(batch), 1,
                                       static_cast<std::size_t>(kDim)}))
                      .contiguous();
        }
        auto x = Tensor::cat({cls, tokens}, 1);
        auto pe = interpolate_pos_embed(batch, token_h, token_w, compute_, img.stream());
        x = x.add(pe);

        std::array<Tensor, 12> block_out{};
        for (int i = 0; i < kBlocks; ++i) {
            x = vit_block(x, i);
            block_out[static_cast<std::size_t>(i)] = x;
            if (taps) {
                taps->blocks[static_cast<std::size_t>(i)] = x.to(DataType::Float32);
            }
        }

        const auto& ln_w = w("encoder.backbone.norm.weight");
        const auto& ln_b = w("encoder.backbone.norm.bias");
        auto n5 = layer_norm(block_out[5], ln_w, ln_b, kLnEps);
        auto n11 = layer_norm(block_out[11], ln_w, ln_b, kLnEps);
        auto cls_tok = n11.slice(1, 0, 1).squeeze(1).contiguous();
        auto tok5 = n5.slice(1, 1, n5.shape()[1])
                        .permute({0, 2, 1})
                        .contiguous()
                        .reshape(shape_of({static_cast<std::size_t>(batch),
                                           static_cast<std::size_t>(kDim),
                                           static_cast<std::size_t>(token_h),
                                           static_cast<std::size_t>(token_w)}));
        auto tok11 = n11.slice(1, 1, n11.shape()[1])
                         .permute({0, 2, 1})
                         .contiguous()
                         .reshape(shape_of({static_cast<std::size_t>(batch),
                                            static_cast<std::size_t>(kDim),
                                            static_cast<std::size_t>(token_h),
                                            static_cast<std::size_t>(token_w)}));
        auto feat = conv1x1(tok5, "encoder.output_projections.0.weight",
                            "encoder.output_projections.0.bias")
                        .add(conv1x1(tok11, "encoder.output_projections.1.weight",
                                     "encoder.output_projections.1.bias"));
        if (taps) {
            taps->encoder_feat = feat.to(DataType::Float32);
        }

        std::array<Tensor, 5> pyramid{};
        pyramid[0] = Tensor::cat({feat, uv_map(token_h, token_w, aspect, compute_, device_,
                                               img.stream())},
                                 1);
        for (int level = 1; level < 5; ++level) {
            const int hh = token_h << level;
            const int ww = token_w << level;
            pyramid[static_cast<std::size_t>(level)] =
                uv_map(hh, ww, aspect, compute_, device_, img.stream());
        }

        const std::array<const char*, 5> neck_in_w = {
            "neck.input_blocks.0.weight", "neck.input_blocks.1.weight",
            "neck.input_blocks.2.weight", "neck.input_blocks.3.weight",
            "neck.input_blocks.4.weight"};
        const std::array<const char*, 5> neck_in_b = {
            "neck.input_blocks.0.bias", "neck.input_blocks.1.bias", "neck.input_blocks.2.bias",
            "neck.input_blocks.3.bias", "neck.input_blocks.4.bias"};
        Tensor neck_x;
        std::array<Tensor, 5> neck_out{};
        for (int i = 0; i < 5; ++i) {
            auto fin = conv1x1(pyramid[static_cast<std::size_t>(i)], neck_in_w[static_cast<std::size_t>(i)],
                               neck_in_b[static_cast<std::size_t>(i)]);
            neck_x = (i == 0) ? std::move(fin) : neck_x.add(fin);
            if (i >= 1 && i <= 3) {
                neck_x = res_block(neck_x,
                                   std::format("neck.res_blocks.{}.0.layers.2.weight", i),
                                   std::format("neck.res_blocks.{}.0.layers.2.bias", i),
                                   std::format("neck.res_blocks.{}.0.layers.5.weight", i),
                                   std::format("neck.res_blocks.{}.0.layers.5.bias", i));
            }
            neck_out[static_cast<std::size_t>(i)] = neck_x;
            if (taps) {
                taps->neck[static_cast<std::size_t>(i)] = neck_x.to(DataType::Float32);
            }
            if (i < 3) {
                neck_x = conv_transpose2x(neck_x,
                                          std::format("neck.resamplers.{}.0.weight", i),
                                          std::format("neck.resamplers.{}.0.bias", i));
                neck_x = conv3x3(neck_x, std::format("neck.resamplers.{}.1.weight", i),
                                 std::format("neck.resamplers.{}.1.bias", i));
            } else if (i == 3) {
                const int oh = static_cast<int>(neck_x.shape()[2]) * 2;
                const int ow = static_cast<int>(neck_x.shape()[3]) * 2;
                neck_x = resize2d(neck_x, oh, ow, ResizeMode::Bilinear, CoordTransform::HalfPixel);
                neck_x = conv3x3(neck_x, "neck.resamplers.3.1.weight", "neck.resamplers.3.1.bias");
            }
        }

        auto run_head = [&](const char* prefix, int out_ch, Tensor* tap) {
            Tensor hx;
            for (int i = 0; i < 5; ++i) {
                auto fin = conv1x1(neck_out[static_cast<std::size_t>(i)],
                                   std::format("{}.input_blocks.{}.weight", prefix, i),
                                   std::format("{}.input_blocks.{}.bias", prefix, i));
                hx = (i == 0) ? std::move(fin) : hx.add(fin);
                if (i >= 1 && i <= 3) {
                    hx = res_block(hx,
                                   std::format("{}.res_blocks.{}.0.layers.2.weight", prefix, i),
                                   std::format("{}.res_blocks.{}.0.layers.2.bias", prefix, i),
                                   std::format("{}.res_blocks.{}.0.layers.5.weight", prefix, i),
                                   std::format("{}.res_blocks.{}.0.layers.5.bias", prefix, i));
                }
                if (i < 3) {
                    hx = conv_transpose2x(hx,
                                          std::format("{}.resamplers.{}.0.weight", prefix, i),
                                          std::format("{}.resamplers.{}.0.bias", prefix, i));
                    hx = conv3x3(hx, std::format("{}.resamplers.{}.1.weight", prefix, i),
                                 std::format("{}.resamplers.{}.1.bias", prefix, i));
                } else if (i == 3) {
                    const int oh = static_cast<int>(hx.shape()[2]) * 2;
                    const int ow = static_cast<int>(hx.shape()[3]) * 2;
                    hx = resize2d(hx, oh, ow, ResizeMode::Bilinear, CoordTransform::HalfPixel);
                    hx = conv3x3(hx, std::format("{}.resamplers.3.1.weight", prefix),
                                 std::format("{}.resamplers.3.1.bias", prefix));
                }
            }
            hx = conv1x1(hx, std::format("{}.output_blocks.4.weight", prefix),
                         std::format("{}.output_blocks.4.bias", prefix));
            (void)out_ch;
            if (tap) {
                *tap = hx.to(DataType::Float32);
            }
            return hx;
        };

        auto points_nchw = run_head("points_head", 3, taps ? &taps->points_head : nullptr);
        auto normal_nchw = run_head("normal_head", 3, taps ? &taps->normal_head : nullptr);
        auto mask_nchw = run_head("mask_head", 1, taps ? &taps->mask_head : nullptr);

        points_nchw = resize2d(points_nchw, img_h, img_w, ResizeMode::Bilinear,
                               CoordTransform::HalfPixel)
                          .to(DataType::Float32);
        normal_nchw = resize2d(normal_nchw, img_h, img_w, ResizeMode::Bilinear,
                               CoordTransform::HalfPixel)
                          .to(DataType::Float32);
        mask_nchw = resize2d(mask_nchw, img_h, img_w, ResizeMode::Bilinear,
                             CoordTransform::HalfPixel)
                        .to(DataType::Float32);

        auto points = points_nchw.permute({0, 2, 3, 1}).contiguous();
        auto xy = points.slice(3, 0, 2);
        auto z = points.slice(3, 2, 3).exp();
        points = Tensor::cat({xy.mul(z), z}, 3);

        auto normal = l2_normalize_last(normal_nchw.permute({0, 2, 3, 1}).contiguous());

        Tensor mask = mask_nchw;
        if (mask.ndim() == 4 && mask.shape()[1] == 1) {
            mask = mask.squeeze(1);
        }
        mask = mask.sigmoid();

        auto scale = linear(cls_tok.to(compute_), w("scale_head.0.weight"),
                            &w("scale_head.0.bias"));
        scale = relu(scale);
        scale = linear(scale, w("scale_head.2.weight"), &w("scale_head.2.bias"));
        scale = relu(scale);
        scale = linear(scale, w("scale_head.4.weight"), &w("scale_head.4.bias"));
        scale = scale.to(DataType::Float32);
        if (scale.ndim() >= 2 && scale.shape()[scale.ndim() - 1] == 1) {
            scale = scale.squeeze(static_cast<int>(scale.ndim() - 1));
        }
        scale = scale.exp();

        return Moge2Outputs{
            .points = std::move(points),
            .normal = std::move(normal),
            .mask = std::move(mask),
            .metric_scale = std::move(scale),
        };
    }

} // namespace lfs::core::nn::models
