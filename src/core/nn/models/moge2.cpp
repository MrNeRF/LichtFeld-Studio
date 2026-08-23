/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/models/moge2.hpp"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/source_site.hpp"
#include "internal/cuda_stream_context.hpp"
#include "internal/memory_pool.hpp"
#include "nn_nvtx.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <utility>
#include <vector>

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

        void configure_nn_mempool() {
#if CUDART_VERSION >= 11020
            int device = 0;
            LFS_CUDA_CHECK(cudaGetDevice(&device));
            cudaMemPool_t pool = nullptr;
            LFS_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));
            std::uint64_t zero = 0;
            LFS_CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &zero));
#endif
        }

        // Tensor::cat's middle-dim CUDA path always launches a float kernel, so
        // fp16 concatenations overrun the output. For N=1 the two layouts we
        // need (sequence and NCHW-channel) are contiguous, so a pair of
        // device copies is enough and dtype-correct.
        void recapture(Tensor& slot, const Tensor& src) {
            if (!slot.is_valid() || slot.dtype() != src.dtype() || slot.device() != src.device() ||
                slot.numel() != src.numel()) {
                slot = src.clone();
                return;
            }
            slot.set_stream(src.stream());
            if (src.bytes() > 0) {
                LFS_CUDA_CHECK(cudaMemcpyAsync(slot.data_ptr(), src.data_ptr(), src.bytes(),
                                               cudaMemcpyDeviceToDevice, src.stream()));
            }
        }

        Tensor concat_contiguous(const Tensor& a, const Tensor& b) {
            LFS_ASSERT_MSG(a.is_valid() && b.is_valid(), "concat requires valid tensors");
            LFS_ASSERT_MSG(a.dtype() == b.dtype() && a.device() == b.device(),
                           "concat dtype/device mismatch");
            LFS_ASSERT_MSG(a.ndim() == b.ndim() && a.ndim() >= 2, "concat rank mismatch");
            int dim = -1;
            std::vector<std::size_t> out_dims;
            out_dims.reserve(a.ndim());
            for (std::size_t i = 0; i < a.ndim(); ++i) {
                if (a.shape()[i] == b.shape()[i]) {
                    out_dims.push_back(a.shape()[i]);
                    continue;
                }
                LFS_ASSERT_MSG(dim < 0, "concat tensors differ on more than one dim");
                dim = static_cast<int>(i);
                out_dims.push_back(a.shape()[i] + b.shape()[i]);
            }
            LFS_ASSERT_MSG(dim >= 0, "concat tensors have identical shapes");
            auto a_c = a.contiguous();
            auto b_c = b.contiguous();
            auto out = Tensor::empty(TensorShape(out_dims), a_c.device(), a_c.dtype());
            out.set_stream(a_c.stream());
            const cudaStream_t stream = out.stream();
            LFS_CUDA_CHECK(cudaMemcpyAsync(out.data_ptr(), a_c.data_ptr(), a_c.bytes(),
                                           cudaMemcpyDeviceToDevice, stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(out.data_ptr()) + a_c.bytes(),
                                           b_c.data_ptr(), b_c.bytes(), cudaMemcpyDeviceToDevice,
                                           stream));
            return out;
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
        configure_nn_mempool();
        for (auto& [name, tensor] : model.weights_) {
            if (name.rfind("onnx::MatMul_", 0) == 0 && tensor.ndim() == 2) {
                tensor = tensor.t().contiguous();
            }
        }
        if (model.weights_.contains("encoder.image_mean")) {
            model.weights_["encoder.image_mean"] =
                model.weights_["encoder.image_mean"].to(DataType::Float32).contiguous();
        }
        if (model.weights_.contains("encoder.image_std")) {
            model.weights_["encoder.image_std"] =
                model.weights_["encoder.image_std"].to(DataType::Float32).contiguous();
        }
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

    std::size_t Moge2::weights_bytes() const {
        std::size_t bytes = 0;
        for (const auto& [name, tensor] : weights_) {
            (void)name;
            if (tensor.is_valid()) {
                bytes += tensor.bytes();
            }
        }
        return bytes;
    }

    const Tensor& Moge2::w(std::string_view name) const {
        const auto it = weights_.find(std::string(name));
        LFS_ASSERT_MSG(it != weights_.end(),
                       std::format("MoGe-2 weight {} is not loaded", name));
        return it->second;
    }

    Tensor Moge2::ensure_workspace(std::size_t bytes, const Tensor& like) {
        if (bytes == 0) {
            return like;
        }
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
        NvtxRange nvtx("moge2/conv1x1");
        Conv2dParams p;
        const auto& ww = w(weight);
        const auto bytes = conv2d_workspace_bytes(x.shape(), ww.shape(), p, x.dtype());
        Tensor ws = ensure_workspace(bytes, x);
        return conv2d(x, ww, &w(bias), p, &ws);
    }

    Tensor Moge2::conv3x3(const Tensor& x, std::string_view weight, std::string_view bias) {
        NvtxRange nvtx("moge2/conv3x3");
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
        NvtxRange nvtx("moge2/conv_transpose2x");
        Conv2dParams p;
        p.stride_h = 2;
        p.stride_w = 2;
        const auto& ww = w(weight);
        const auto bytes = conv_transpose2d_workspace_bytes(x.shape(), ww.shape(), p, x.dtype());
        Tensor ws = ensure_workspace(bytes, x);
        return conv_transpose2d(x, ww, &w(bias), p, &ws);
    }

    Tensor Moge2::gemm_nn(const Tensor& x, std::string_view weight, std::string_view bias) {
        NvtxRange nvtx("moge2/gemm");
        return gemm(x, w(weight), false, true, &w(bias));
    }

    Tensor Moge2::res_block(const Tensor& x, std::string_view w1, std::string_view b1,
                            std::string_view w2, std::string_view b2) {
        NvtxRange nvtx("moge2/res_block");
        auto y = relu(x);
        y = conv3x3(y, w1, b1);
        y = relu(y);
        y = conv3x3(y, w2, b2);
        return x.add(y);
    }

    Tensor Moge2::vit_block(const Tensor& x, int index) {
        char tag[32];
        std::snprintf(tag, sizeof(tag), "moge2/block%d", index);
        NvtxRange block_nvtx(tag);
        Tensor n1;
        Tensor qkv;
        {
            NvtxRange nvtx("moge2/qkv");
            n1 = layer_norm(x, w(block_name(index, "norm1.weight")),
                            w(block_name(index, "norm1.bias")), kLnEps);
            qkv = gemm_nn(n1, matmul_name(index, "qkv"), block_name(index, "attn.qkv.bias"));
        }
        Tensor q;
        Tensor k;
        Tensor v;
        {
            NvtxRange nvtx("moge2/qkv_split");
            auto split = split_qkv(qkv, kHeads);
            q = std::move(split[0]);
            k = std::move(split[1]);
            v = std::move(split[2]);
        }
        Tensor attn;
        {
            NvtxRange nvtx("moge2/attention");
            attn = merge_heads(attention(q, k, v));
        }
        Tensor y;
        {
            NvtxRange nvtx("moge2/proj");
            y = gemm(attn, w(matmul_name(index, "proj")), false, true,
                     &w(block_name(index, "attn.proj.bias")), Activation::None, &x,
                     &w(block_name(index, "ls1.gamma")));
        }
        {
            NvtxRange nvtx("moge2/mlp");
            auto n2 = layer_norm(y, w(block_name(index, "norm2.weight")),
                                 w(block_name(index, "norm2.bias")), kLnEps);
            auto h = gemm(n2, w(matmul_name(index, "1")), false, true,
                          &w(block_name(index, "mlp.fc1.bias")), Activation::GeluErf);
            return gemm(h, w(matmul_name(index, "2")), false, true,
                        &w(block_name(index, "mlp.fc2.bias")), Activation::None, &y,
                        &w(block_name(index, "ls2.gamma")));
        }
    }

    Tensor Moge2::interpolate_pos_embed(int batch, int token_h, int token_w, DataType dtype,
                                        cudaStream_t stream) {
        NvtxRange nvtx("moge2/pos_embed");
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
        auto pe = concat_contiguous(cls, flat);
        if (batch > 1) {
            pe = pe.expand(shape_of({static_cast<std::size_t>(batch), pe.shape()[1], pe.shape()[2]}))
                     .contiguous();
        }
        return as_compute(std::move(pe), dtype);
    }

    Tensor Moge2::uv_map(int height, int width, float aspect, DataType dtype, Device device,
                         cudaStream_t stream) const {
        NvtxRange nvtx("moge2/uv_map");
        return uv_grid(height, width, aspect, dtype, device, stream);
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

        NvtxRange forward_nvtx("moge2/forward");
        const cudaStream_t fwd_stream = image.stream();
        lfs::core::CUDAStreamGuard stream_guard(fwd_stream);
        if (!weights_on_stream_) {
            for (auto& [name, tensor] : weights_) {
                (void)name;
                tensor.set_stream(fwd_stream);
            }
            weights_on_stream_ = true;
        }
        ActivationArenaGuard arena_guard(arena_);
        arena_.begin(fwd_stream);
        struct ArenaCloser {
            ActivationArena& arena;
            bool& trimmed;
            ~ArenaCloser() {
                arena.end();
                if (!trimmed && arena.capacity() > 0) {
                    configure_nn_mempool();
                    trimmed = true;
                }
            }
        } arena_closer{arena_, mempool_trimmed_};
        StageProfile profile(fwd_stream);

        Tensor img;
        {
            NvtxRange nvtx("moge2/preprocess");
            profile.mark("preprocess");
            img = image.contiguous();
            if (img.dtype() != DataType::Float32) {
                img = img.to(DataType::Float32);
            }
            img = resize2d(img, enc_h, enc_w, ResizeMode::Bilinear, CoordTransform::HalfPixel);
            const auto& mean = w("encoder.image_mean");
            const auto& stdv = w("encoder.image_std");
            img = img.sub(mean).div(stdv);
            img = as_compute(std::move(img), compute_);
        }

        Tensor tokens;
        {
            NvtxRange nvtx("moge2/patch_embed");
            profile.mark("patch_embed");
            Conv2dParams patch;
            patch.stride_h = kPatch;
            patch.stride_w = kPatch;
            const auto& pw = w("encoder.backbone.patch_embed.proj.weight");
            const auto pbytes = conv2d_workspace_bytes(img.shape(), pw.shape(), patch, img.dtype());
            Tensor pws = ensure_workspace(pbytes, img);
            auto conv = conv2d(img, pw, &w("encoder.backbone.patch_embed.proj.bias"), patch, &pws);
            tokens = conv.reshape(shape_of({static_cast<std::size_t>(batch),
                                            static_cast<std::size_t>(kDim),
                                            static_cast<std::size_t>(token_h * token_w)}))
                         .permute({0, 2, 1})
                         .contiguous();
        }
        if (taps) {
            ActivationArena::bind(nullptr);
            taps->patch_embed = tokens.to(DataType::Float32);
            ActivationArena::bind(&arena_);
        }

        Tensor x;
        {
            NvtxRange nvtx("moge2/pos_embed_add");
            profile.mark("pos_embed");
            auto cls = w("encoder.backbone.cls_token");
            if (batch > 1) {
                cls = cls.expand(shape_of({static_cast<std::size_t>(batch), 1,
                                           static_cast<std::size_t>(kDim)}))
                          .contiguous();
            }
            x = concat_contiguous(cls, tokens);
            auto pe = interpolate_pos_embed(batch, token_h, token_w, compute_, img.stream());
            x = x.add(pe);
        }

        std::array<Tensor, 12> block_out{};
        for (int i = 0; i < kBlocks; ++i) {
            char mark_name[16];
            std::snprintf(mark_name, sizeof(mark_name), "block%d", i);
            profile.mark(mark_name);
            x = vit_block(x, i);
            block_out[static_cast<std::size_t>(i)] = x;
            if (taps) {
                ActivationArena::bind(nullptr);
                taps->blocks[static_cast<std::size_t>(i)] = x.to(DataType::Float32);
                ActivationArena::bind(&arena_);
            }
        }

        Tensor feat;
        Tensor cls_tok;
        {
            NvtxRange nvtx("moge2/encoder_feat");
            profile.mark("encoder_feat");
            const auto& ln_w = w("encoder.backbone.norm.weight");
            const auto& ln_b = w("encoder.backbone.norm.bias");
            auto n5 = layer_norm(block_out[5], ln_w, ln_b, kLnEps);
            auto n11 = layer_norm(block_out[11], ln_w, ln_b, kLnEps);
            cls_tok = n11.slice(1, 0, 1).squeeze(1).contiguous();
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
            feat = conv1x1(tok5, "encoder.output_projections.0.weight",
                           "encoder.output_projections.0.bias")
                       .add(conv1x1(tok11, "encoder.output_projections.1.weight",
                                    "encoder.output_projections.1.bias"));
        }
        if (taps) {
            ActivationArena::bind(nullptr);
            taps->encoder_feat = feat.to(DataType::Float32);
            ActivationArena::bind(&arena_);
        }
        block_out = {};
        {
            ActivationArena::bind(nullptr);
            recapture(feat_hold_, feat);
            recapture(cls_hold_, cls_tok);
            feat = feat_hold_;
            cls_tok = cls_hold_;
            ActivationArena::bind(&arena_);
            arena_.rewind(0);
        }

        std::array<Tensor, 5> pyramid{};
        {
            NvtxRange nvtx("moge2/uv_maps");
            profile.mark("uv_maps");
            pyramid[0] = concat_contiguous(
                feat, uv_map(token_h, token_w, aspect, compute_, device_, img.stream()));
            for (int level = 1; level < 5; ++level) {
                const int hh = token_h << level;
                const int ww = token_w << level;
                pyramid[static_cast<std::size_t>(level)] =
                    uv_map(hh, ww, aspect, compute_, device_, img.stream());
            }
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
        {
            NvtxRange neck_nvtx("moge2/neck");
            profile.mark("neck");
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
                    ActivationArena::bind(nullptr);
                    taps->neck[static_cast<std::size_t>(i)] = neck_x.to(DataType::Float32);
                    ActivationArena::bind(&arena_);
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
        }

        const std::size_t after_neck = arena_.mark();
        int persist_i = 0;
        auto persist_and_rewind = [&](Tensor t) {
            ActivationArena::bind(nullptr);
            Tensor& slot = head_hold_[persist_i++];
            recapture(slot, t);
            ActivationArena::bind(&arena_);
            arena_.rewind(after_neck);
            return slot;
        };

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
                ActivationArena::bind(nullptr);
                *tap = hx.to(DataType::Float32);
                ActivationArena::bind(&arena_);
            }
            return hx;
        };

        Tensor points_nchw;
        Tensor normal_nchw;
        Tensor mask_nchw;
        {
            NvtxRange nvtx("moge2/points_head");
            profile.mark("points_head");
            points_nchw = persist_and_rewind(
                run_head("points_head", 3, taps ? &taps->points_head : nullptr));
        }
        {
            NvtxRange nvtx("moge2/normal_head");
            profile.mark("normal_head");
            normal_nchw = persist_and_rewind(
                run_head("normal_head", 3, taps ? &taps->normal_head : nullptr));
        }
        {
            NvtxRange nvtx("moge2/mask_head");
            profile.mark("mask_head");
            mask_nchw = persist_and_rewind(
                run_head("mask_head", 1, taps ? &taps->mask_head : nullptr));
        }

        Tensor points;
        Tensor normal;
        Tensor mask;
        Tensor scale;
        {
            NvtxRange nvtx("moge2/postprocess");
            profile.mark("postprocess");
            points_nchw = resize2d(points_nchw, img_h, img_w, ResizeMode::Bilinear,
                                   CoordTransform::HalfPixel)
                              .to(DataType::Float32);
            normal_nchw = resize2d(normal_nchw, img_h, img_w, ResizeMode::Bilinear,
                                   CoordTransform::HalfPixel)
                              .to(DataType::Float32);
            mask_nchw = resize2d(mask_nchw, img_h, img_w, ResizeMode::Bilinear,
                                 CoordTransform::HalfPixel)
                            .to(DataType::Float32);

            points = points_nchw.permute({0, 2, 3, 1}).contiguous();
            auto xy = points.slice(3, 0, 2);
            auto z = points.slice(3, 2, 3).exp();
            points = Tensor::cat({xy.mul(z), z}, 3);

            normal = l2_normalize_last(normal_nchw.permute({0, 2, 3, 1}).contiguous());

            mask = mask_nchw;
            if (mask.ndim() == 4 && mask.shape()[1] == 1) {
                mask = mask.squeeze(1);
            }
            mask = mask.sigmoid();

            scale = linear(cls_tok.to(compute_), w("scale_head.0.weight"),
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
        }

        profile.mark("end");
        profile.dump();
        if (profile.enabled()) {
            std::size_t free_b = 0;
            std::size_t total_b = 0;
            LFS_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            const std::size_t used = total_b >= free_b ? total_b - free_b : 0;
            std::fprintf(stderr,
                         "[nn.profile] vram used=%.2f MiB  weights=%.2f MiB  arena=%.2f MiB  "
                         "workspace=%.2f MiB  arena_hw=%.2f MiB\n",
                         static_cast<double>(used) / (1024.0 * 1024.0),
                         static_cast<double>(weights_bytes()) / (1024.0 * 1024.0),
                         static_cast<double>(arena_.capacity()) / (1024.0 * 1024.0),
                         static_cast<double>(workspace_bytes()) / (1024.0 * 1024.0),
                         static_cast<double>(arena_.high_water()) / (1024.0 * 1024.0));
        }
        return Moge2Outputs{
            .points = std::move(points),
            .normal = std::move(normal),
            .mask = std::move(mask),
            .metric_scale = std::move(scale),
        };
    }

} // namespace lfs::core::nn::models
