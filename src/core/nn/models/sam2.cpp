/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nn/models/sam2.hpp"

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

        constexpr int kImage = 1024;
        constexpr int kEmbed = 64;
        constexpr int kDim = 256;
        constexpr int kDepth = 24;
        constexpr int kDecDepth = 2;
        constexpr int kDecHeads = 8;
        constexpr float kHieraEps = 1e-6f;
        constexpr float kDecLnEps = 1e-5f;
        constexpr float kLn2dEps = 1e-6f;
        constexpr std::array<int, kDepth> kWindow = {
            8, 8, 8, 4, 4, 4, 14, 14, 14, 14, 14, 14, 0, 14, 14, 14, 0, 14, 14, 14, 0, 14, 7, 7};
        constexpr std::array<int, 3> kQPool = {2, 5, 21};
        constexpr std::array<int, 4> kStageEnds = {1, 4, 20, 23};

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

        Tensor concat_contiguous(const Tensor& a, const Tensor& b, int dim) {
            LFS_ASSERT_MSG(a.is_valid() && b.is_valid(), "concat requires valid tensors");
            LFS_ASSERT_MSG(a.dtype() == b.dtype() && a.device() == b.device(),
                           "concat dtype/device mismatch");
            LFS_ASSERT_MSG(a.ndim() == b.ndim() && a.ndim() >= 1, "concat rank mismatch");
            if (dim < 0) {
                dim += static_cast<int>(a.ndim());
            }
            LFS_ASSERT_MSG(dim >= 0 && dim < static_cast<int>(a.ndim()), "concat dim out of range");
            std::size_t leading = 1;
            std::vector<std::size_t> out_dims;
            out_dims.reserve(a.ndim());
            for (std::size_t i = 0; i < a.ndim(); ++i) {
                if (static_cast<int>(i) == dim) {
                    out_dims.push_back(a.shape()[i] + b.shape()[i]);
                    continue;
                }
                LFS_ASSERT_MSG(a.shape()[i] == b.shape()[i], "concat tensors differ on a non-cat dim");
                out_dims.push_back(a.shape()[i]);
                if (static_cast<int>(i) < dim) {
                    leading *= a.shape()[i];
                }
            }
            LFS_ASSERT_MSG(leading == 1,
                           "concat_contiguous requires unit leading dims (batch=1)");
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

        lfs::Error sam_error(const lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({
                .code = code,
                .domain = lfs::ErrorDomain::Core,
                .user_message = "SAM2 inference failed",
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

        bool is_q_pool(int index) {
            return index == kQPool[0] || index == kQPool[1] || index == kQPool[2];
        }

        int hiera_heads(int dim_out) {
            LFS_ASSERT_MSG(dim_out % 56 == 0, "unexpected Hiera dim_out");
            return dim_out / 56;
        }

        std::string blk(int index, std::string_view suffix) {
            return std::format("image_encoder.trunk.blocks.{}.{}", index, suffix);
        }

        std::string dec_layer(int layer, std::string_view suffix) {
            return std::format("sam_mask_decoder.transformer.layers.{}.{}", layer, suffix);
        }

    } // namespace

    lfs::Result<Sam2> Sam2::load(const std::filesystem::path& weights, Device device,
                                 std::optional<DataType> compute) {
        if (device != Device::CUDA) {
            return sam_error(lfs::ErrorCode::InvalidArgument, "SAM2 requires a CUDA device");
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
            return sam_error(lfs::ErrorCode::InvalidArgument,
                             "SAM2 compute dtype must be float16 or float32");
        }
        auto loaded = file->load_all(device, dtype);
        if (!loaded) {
            return std::move(loaded.error());
        }
        Sam2 model;
        model.device_ = device;
        model.compute_ = dtype;
        model.weights_ = std::move(*loaded);
        configure_nn_mempool();
        const std::array<const char*, 8> required = {
            "image_encoder.trunk.patch_embed.proj.weight",
            "image_encoder.trunk.pos_embed",
            "image_encoder.neck.convs.0.conv.weight",
            "no_mem_embed",
            "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
            "sam_mask_decoder.transformer.norm_final_attn.weight",
            "sam_mask_decoder.conv_s0.weight",
            "sam_mask_decoder.mask_tokens.weight",
        };
        for (const char* name : required) {
            if (!model.weights_.contains(name)) {
                return sam_error(lfs::ErrorCode::NotFound,
                                 std::format("weight file is missing {}", name));
            }
        }
        model.mean_ = Tensor::from_vector({0.485f, 0.456f, 0.406f}, shape_of({1, 3, 1, 1}), device);
        model.std_ = Tensor::from_vector({0.229f, 0.224f, 0.225f}, shape_of({1, 3, 1, 1}), device);
        return model;
    }

    std::size_t Sam2::weights_bytes() const {
        std::size_t bytes = 0;
        for (const auto& [name, tensor] : weights_) {
            (void)name;
            if (tensor.is_valid()) {
                bytes += tensor.bytes();
            }
        }
        return bytes;
    }

    const Tensor& Sam2::w(std::string_view name) const {
        const auto it = weights_.find(std::string(name));
        LFS_ASSERT_MSG(it != weights_.end(), std::format("SAM2 weight {} is not loaded", name));
        return it->second;
    }

    Tensor Sam2::ensure_workspace(std::size_t bytes, const Tensor& like) {
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

    void Sam2::capture_tap(Tensor& slot, const Tensor& src) {
        ActivationArena::bind(nullptr);
        slot = src.to(DataType::Float32);
        ActivationArena::bind(&arena_);
    }

    Tensor Sam2::conv1x1(const Tensor& x, std::string_view weight, std::string_view bias) {
        Conv2dParams p;
        return conv_k(x, weight, bias, p);
    }

    Tensor Sam2::conv_k(const Tensor& x, std::string_view weight, std::string_view bias,
                        const Conv2dParams& params) {
        const auto& ww = w(weight);
        const auto bytes = conv2d_workspace_bytes(x.shape(), ww.shape(), params, x.dtype());
        Tensor ws = ensure_workspace(bytes, x);
        return conv2d(x, ww, &w(bias), params, &ws);
    }

    Tensor Sam2::conv_transpose2x(const Tensor& x, std::string_view weight, std::string_view bias) {
        Conv2dParams p;
        p.stride_h = 2;
        p.stride_w = 2;
        const auto& ww = w(weight);
        const auto bytes = conv_transpose2d_workspace_bytes(x.shape(), ww.shape(), p, x.dtype());
        Tensor ws = ensure_workspace(bytes, x);
        return conv_transpose2d(x, ww, &w(bias), p, &ws);
    }

    Tensor Sam2::lin(const Tensor& x, std::string_view weight, std::string_view bias,
                     Activation activation, const Tensor* residual) {
        return linear(x, w(weight), &w(bias), activation, residual);
    }

    Tensor Sam2::pool_bhwc(const Tensor& x) {
        return max_pool2d_bhwc(x);
    }

    Tensor Sam2::hiera_pos_embed(int token_h, int token_w, cudaStream_t stream) {
        NvtxRange nvtx("sam2/pos_embed");
        if (pos_embed_.is_valid() && pos_embed_h_ == token_h && pos_embed_w_ == token_w) {
            pos_embed_.set_stream(stream);
            return pos_embed_;
        }
        ActivationArena* prev = ActivationArena::current();
        ActivationArena::bind(nullptr);
        auto pe = w("image_encoder.trunk.pos_embed");
        pe.set_stream(stream);
        auto resized = resize2d(pe, token_h, token_w, ResizeMode::Cubic, CoordTransform::HalfPixel);
        auto win = w("image_encoder.trunk.pos_embed_window");
        const int wh = static_cast<int>(win.shape()[2]);
        const int ww = static_cast<int>(win.shape()[3]);
        LFS_ASSERT_MSG(token_h % wh == 0 && token_w % ww == 0, "pos_embed_window does not tile");
        const int nh = token_h / wh;
        const int nw = token_w / ww;
        auto tiled =
            win.reshape(shape_of({1, win.shape()[1], 1, static_cast<std::size_t>(wh), 1,
                                  static_cast<std::size_t>(ww)}))
                .expand(shape_of({1, win.shape()[1], static_cast<std::size_t>(nh),
                                  static_cast<std::size_t>(wh), static_cast<std::size_t>(nw),
                                  static_cast<std::size_t>(ww)}))
                .contiguous()
                .reshape(shape_of({1, win.shape()[1], static_cast<std::size_t>(token_h),
                                   static_cast<std::size_t>(token_w)}));
        auto pos = resized.add(tiled);
        pos_embed_ = pos.permute({0, 2, 3, 1}).contiguous();
        pos_embed_h_ = token_h;
        pos_embed_w_ = token_w;
        ActivationArena::bind(prev);
        return pos_embed_;
    }

    Tensor Sam2::hiera_block(const Tensor& x, int index) {
        char tag[32];
        std::snprintf(tag, sizeof(tag), "sam2/block%d", index);
        NvtxRange block_nvtx(tag);
        auto shortcut = x;
        auto xn = layer_norm(x, w(blk(index, "norm1.weight")), w(blk(index, "norm1.bias")),
                             kHieraEps);
        const bool q_pool = is_q_pool(index);
        if (q_pool) {
            shortcut = pool_bhwc(lin(xn, blk(index, "proj.weight"), blk(index, "proj.bias")));
        }
        const int orig_win = kWindow[static_cast<std::size_t>(index)];
        int H = static_cast<int>(xn.shape()[1]);
        int W = static_cast<int>(xn.shape()[2]);
        const bool fuse_win = orig_win > 0;

        Tensor xa;
        if (fuse_win) {
            auto qkv = lin(xn, blk(index, "attn.qkv.weight"), blk(index, "attn.qkv.bias"));
            const int dim_out = static_cast<int>(qkv.shape()[3]) / 3;
            const int heads = hiera_heads(dim_out);
            const auto& qkv_bias = w(blk(index, "attn.qkv.bias"));
            auto split = split_qkv_window_2d(qkv, heads, orig_win, &qkv_bias);
            Tensor q = std::move(split[0]);
            Tensor k = std::move(split[1]);
            Tensor v = std::move(split[2]);
            int used_win = orig_win;
            if (q_pool) {
                q = max_pool_heads_2d(q, orig_win, orig_win);
                used_win = orig_win / 2;
                H = static_cast<int>(shortcut.shape()[1]);
                W = static_cast<int>(shortcut.shape()[2]);
            }
            auto ctx = attention(q, k, v);
            xa = merge_heads_unwindow_2d(ctx, used_win, H, W);
            xa = lin(xa, blk(index, "attn.proj.weight"), blk(index, "attn.proj.bias"),
                     Activation::None, &shortcut);
        } else {
            const int b = static_cast<int>(xn.shape()[0]);
            auto qkv = lin(xn, blk(index, "attn.qkv.weight"), blk(index, "attn.qkv.bias"));
            const int dim_out = static_cast<int>(qkv.shape()[3]) / 3;
            const int heads = hiera_heads(dim_out);
            auto packed = qkv.reshape(shape_of({static_cast<std::size_t>(b),
                                                static_cast<std::size_t>(H) *
                                                    static_cast<std::size_t>(W),
                                                static_cast<std::size_t>(dim_out) * 3}));
            auto split = split_qkv(packed, heads);
            auto ctx = merge_heads(attention(split[0], split[1], split[2]));
            auto spatial =
                ctx.reshape(shape_of({static_cast<std::size_t>(b), static_cast<std::size_t>(H),
                                      static_cast<std::size_t>(W),
                                      static_cast<std::size_t>(dim_out)}));
            xa = lin(spatial, blk(index, "attn.proj.weight"), blk(index, "attn.proj.bias"),
                     Activation::None, &shortcut);
        }
        auto y = xa;
        auto n2 = layer_norm(y, w(blk(index, "norm2.weight")), w(blk(index, "norm2.bias")),
                             kHieraEps);
        auto h = lin(n2, blk(index, "mlp.layers.0.weight"), blk(index, "mlp.layers.0.bias"),
                     Activation::GeluErf);
        return lin(h, blk(index, "mlp.layers.1.weight"), blk(index, "mlp.layers.1.bias"),
                   Activation::None, &y);
    }

    Tensor Sam2::two_way_attn(const Tensor& q_in, const Tensor& k_in, const Tensor& v_in,
                              std::string_view prefix) {
        auto q = lin(q_in, std::format("{}.q_proj.weight", prefix),
                     std::format("{}.q_proj.bias", prefix));
        auto k = lin(k_in, std::format("{}.k_proj.weight", prefix),
                     std::format("{}.k_proj.bias", prefix));
        auto v = lin(v_in, std::format("{}.v_proj.weight", prefix),
                     std::format("{}.v_proj.bias", prefix));
        const int b = static_cast<int>(q.shape()[0]);
        const int nq = static_cast<int>(q.shape()[1]);
        const int nk = static_cast<int>(k.shape()[1]);
        const int internal = static_cast<int>(q.shape()[2]);
        const int d = internal / kDecHeads;
        auto qh = q.reshape(shape_of({static_cast<std::size_t>(b), static_cast<std::size_t>(nq),
                                      static_cast<std::size_t>(kDecHeads), static_cast<std::size_t>(d)}))
                      .permute({0, 2, 1, 3})
                      .contiguous();
        auto kh = k.reshape(shape_of({static_cast<std::size_t>(b), static_cast<std::size_t>(nk),
                                      static_cast<std::size_t>(kDecHeads), static_cast<std::size_t>(d)}))
                      .permute({0, 2, 1, 3})
                      .contiguous();
        auto vh = v.reshape(shape_of({static_cast<std::size_t>(b), static_cast<std::size_t>(nk),
                                      static_cast<std::size_t>(kDecHeads), static_cast<std::size_t>(d)}))
                      .permute({0, 2, 1, 3})
                      .contiguous();
        auto ctx = merge_heads(attention(qh, kh, vh));
        return lin(ctx, std::format("{}.out_proj.weight", prefix),
                   std::format("{}.out_proj.bias", prefix));
    }

    std::pair<Tensor, Tensor> Sam2::two_way_block(const Tensor& queries, const Tensor& keys,
                                                  const Tensor& query_pe, const Tensor& key_pe,
                                                  int layer) {
        Tensor qy = queries;
        if (layer == 0) {
            auto attn = two_way_attn(qy, qy, qy, dec_layer(layer, "self_attn"));
            qy = attn;
        } else {
            auto q = qy.add(query_pe);
            auto attn = two_way_attn(q, q, qy, dec_layer(layer, "self_attn"));
            qy = qy.add(attn);
        }
        qy = layer_norm(qy, w(dec_layer(layer, "norm1.weight")), w(dec_layer(layer, "norm1.bias")),
                        kDecLnEps);

        {
            auto q = qy.add(query_pe);
            auto k = keys.add(key_pe);
            auto attn = two_way_attn(q, k, keys, dec_layer(layer, "cross_attn_token_to_image"));
            qy = qy.add(attn);
            qy = layer_norm(qy, w(dec_layer(layer, "norm2.weight")), w(dec_layer(layer, "norm2.bias")),
                            kDecLnEps);
        }
        {
            auto mlp_out = mlp(qy, dec_layer(layer, "mlp"), 2, Activation::Relu, false);
            qy = qy.add(mlp_out);
            qy = layer_norm(qy, w(dec_layer(layer, "norm3.weight")), w(dec_layer(layer, "norm3.bias")),
                            kDecLnEps);
        }
        Tensor ky = keys;
        {
            auto q = qy.add(query_pe);
            auto k = ky.add(key_pe);
            auto attn = two_way_attn(k, q, qy, dec_layer(layer, "cross_attn_image_to_token"));
            ky = ky.add(attn);
            ky = layer_norm(ky, w(dec_layer(layer, "norm4.weight")), w(dec_layer(layer, "norm4.bias")),
                            kDecLnEps);
        }
        return {std::move(qy), std::move(ky)};
    }

    std::pair<Tensor, Tensor> Sam2::two_way_transformer(const Tensor& image, const Tensor& image_pe,
                                                        const Tensor& tokens) {
        NvtxRange nvtx("sam2/transformer");
        const int b = static_cast<int>(image.shape()[0]);
        const int c = static_cast<int>(image.shape()[1]);
        const int h = static_cast<int>(image.shape()[2]);
        const int ww = static_cast<int>(image.shape()[3]);
        auto keys = image.reshape(shape_of({static_cast<std::size_t>(b), static_cast<std::size_t>(c),
                                            static_cast<std::size_t>(h) * static_cast<std::size_t>(ww)}))
                        .permute({0, 2, 1})
                        .contiguous();
        auto key_pe = image_pe
                          .reshape(shape_of({static_cast<std::size_t>(b), static_cast<std::size_t>(c),
                                             static_cast<std::size_t>(h) * static_cast<std::size_t>(ww)}))
                          .permute({0, 2, 1})
                          .contiguous();
        auto queries = tokens;
        for (int layer = 0; layer < kDecDepth; ++layer) {
            auto out = two_way_block(queries, keys, tokens, key_pe, layer);
            queries = std::move(out.first);
            keys = std::move(out.second);
        }
        auto q = queries.add(tokens);
        auto k = keys.add(key_pe);
        auto attn = two_way_attn(q, k, keys, "sam_mask_decoder.transformer.final_attn_token_to_image");
        queries = queries.add(attn);
        queries = layer_norm(queries, w("sam_mask_decoder.transformer.norm_final_attn.weight"),
                             w("sam_mask_decoder.transformer.norm_final_attn.bias"), kDecLnEps);
        return {std::move(queries), std::move(keys)};
    }

    Tensor Sam2::mlp(const Tensor& x, std::string_view prefix, int n_layers, Activation hidden_act,
                     bool sigmoid_out) {
        Tensor y = x;
        for (int i = 0; i < n_layers; ++i) {
            const bool last = i + 1 == n_layers;
            y = lin(y, std::format("{}.layers.{}.weight", prefix, i),
                    std::format("{}.layers.{}.bias", prefix, i), last ? Activation::None : hidden_act);
        }
        if (sigmoid_out) {
            y = sigmoid(y);
        }
        return y;
    }

    Tensor Sam2::encode_points(const std::vector<Sam2PointPrompt>& points,
                               const std::optional<std::array<float, 4>>& box) {
        NvtxRange nvtx("sam2/prompt");
        std::vector<Sam2PointPrompt> all;
        all.reserve(points.size() + 3);
        if (box) {
            all.push_back({(*box)[0], (*box)[1], 2});
            all.push_back({(*box)[2], (*box)[3], 3});
        }
        all.insert(all.end(), points.begin(), points.end());
        all.push_back({0.0f, 0.0f, -1});
        const int n = static_cast<int>(all.size());
        std::vector<float> coords(static_cast<std::size_t>(n) * 2);
        const float ow = static_cast<float>(std::max(orig_w_, 1));
        const float oh = static_cast<float>(std::max(orig_h_, 1));
        for (int i = 0; i < n; ++i) {
            const float x = all[static_cast<std::size_t>(i)].x / ow * static_cast<float>(kImage);
            const float y = all[static_cast<std::size_t>(i)].y / oh * static_cast<float>(kImage);
            coords[static_cast<std::size_t>(i) * 2 + 0] = (x + 0.5f) / static_cast<float>(kImage);
            coords[static_cast<std::size_t>(i) * 2 + 1] = (y + 0.5f) / static_cast<float>(kImage);
        }
        auto coord_t = Tensor::from_vector(coords, shape_of({1, static_cast<std::size_t>(n), 2}),
                                           device_);
        coord_t = as_compute(std::move(coord_t), compute_);
        coord_t.set_stream(w("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix").stream());
        auto pe = fourier_pe(coord_t, w("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"));
        Tensor packed;
        for (int i = 0; i < n; ++i) {
            auto row = pe.slice(1, static_cast<std::size_t>(i), static_cast<std::size_t>(i + 1))
                           .contiguous();
            const int label = all[static_cast<std::size_t>(i)].label;
            if (label == -1) {
                row = w("sam_prompt_encoder.not_a_point_embed.weight")
                          .reshape(shape_of({1, 1, static_cast<std::size_t>(kDim)}));
            } else {
                LFS_ASSERT_MSG(label >= 0 && label <= 3, "point label must be -1 or 0..3");
                auto emb = w(std::format("sam_prompt_encoder.point_embeddings.{}.weight", label))
                               .reshape(shape_of({1, 1, static_cast<std::size_t>(kDim)}));
                row = row.add(emb);
            }
            packed = packed.is_valid() ? concat_contiguous(packed, row, 1) : std::move(row);
        }
        return packed;
    }

    lfs::Result<void> Sam2::set_image(const Tensor& image) {
        return run_encoder(image, nullptr);
    }

    lfs::Result<Sam2Taps> Sam2::set_image_with_taps(const Tensor& image) {
        Sam2Taps taps;
        auto st = run_encoder(image, &taps);
        if (!st) {
            return std::move(st.error());
        }
        encoder_taps_ = taps;
        encoder_taps_valid_ = true;
        return taps;
    }

    lfs::Result<Sam2Prediction> Sam2::predict(const std::vector<Sam2PointPrompt>& points,
                                              const std::optional<std::array<float, 4>>& box,
                                              bool multimask) {
        return run_decoder(points, box, multimask, nullptr);
    }

    lfs::Result<std::pair<Sam2Prediction, Sam2Taps>>
    Sam2::predict_with_taps(const std::vector<Sam2PointPrompt>& points,
                            const std::optional<std::array<float, 4>>& box, bool multimask) {
        Sam2Taps taps = encoder_taps_valid_ ? encoder_taps_ : Sam2Taps{};
        auto pred = run_decoder(points, box, multimask, &taps);
        if (!pred) {
            return std::move(pred.error());
        }
        return std::make_pair(std::move(*pred), std::move(taps));
    }

    lfs::Result<void> Sam2::run_encoder(const Tensor& image, Sam2Taps* taps) {
        if (!image.is_valid() || image.ndim() != 4 || image.shape()[1] != 3) {
            return lfs::Result<void>::failure(
                sam_error(lfs::ErrorCode::InvalidArgument,
                          "SAM2 image must be NCHW with 3 channels"));
        }
        if (image.device() != Device::CUDA) {
            return lfs::Result<void>::failure(
                sam_error(lfs::ErrorCode::InvalidArgument, "SAM2 image must be on CUDA"));
        }
        if (image.shape()[0] != 1) {
            return lfs::Result<void>::failure(
                sam_error(lfs::ErrorCode::InvalidArgument, "SAM2 set_image expects batch size 1"));
        }

        orig_h_ = static_cast<int>(image.shape()[2]);
        orig_w_ = static_cast<int>(image.shape()[3]);
        image_set_ = false;
        encoder_taps_valid_ = false;

        NvtxRange forward_nvtx("sam2/set_image");
        const cudaStream_t fwd_stream = image.stream();
        lfs::core::CUDAStreamGuard stream_guard(fwd_stream);
        if (!weights_on_stream_) {
            for (auto& [name, tensor] : weights_) {
                (void)name;
                tensor.set_stream(fwd_stream);
            }
            mean_.set_stream(fwd_stream);
            std_.set_stream(fwd_stream);
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
            NvtxRange nvtx("sam2/preprocess");
            profile.mark("preprocess");
            img = image.contiguous();
            if (img.dtype() != DataType::Float32) {
                img = img.to(DataType::Float32);
            }
            img = resize2d(img, kImage, kImage, ResizeMode::Bilinear, CoordTransform::HalfPixel);
            img = img.sub(mean_).div(std_);
            img = as_compute(std::move(img), compute_);
        }

        Tensor x;
        {
            NvtxRange nvtx("sam2/patch_embed");
            profile.mark("patch_embed");
            Conv2dParams patch;
            patch.stride_h = 4;
            patch.stride_w = 4;
            patch.pad_h = 3;
            patch.pad_w = 3;
            auto conv = conv_k(img, "image_encoder.trunk.patch_embed.proj.weight",
                               "image_encoder.trunk.patch_embed.proj.bias", patch);
            x = conv.permute({0, 2, 3, 1}).contiguous();
        }
        if (taps) {
            capture_tap(taps->patch_embed, x);
        }

        {
            NvtxRange nvtx("sam2/pos_embed_add");
            profile.mark("pos_embed");
            const int th = static_cast<int>(x.shape()[1]);
            const int tw = static_cast<int>(x.shape()[2]);
            auto pe = hiera_pos_embed(th, tw, img.stream());
            x = x.add(pe);
        }

        std::array<Tensor, 4> stages{};
        int stage_i = 0;
        for (int i = 0; i < kDepth; ++i) {
            char mark_name[16];
            std::snprintf(mark_name, sizeof(mark_name), "block%d", i);
            profile.mark(mark_name);
            const std::size_t block_mark = arena_.mark();
            x = hiera_block(x, i);
            if (taps) {
                capture_tap(taps->blocks[static_cast<std::size_t>(i)], x);
            }
            if (i == kStageEnds[static_cast<std::size_t>(stage_i)]) {
                auto nchw = x.permute({0, 3, 1, 2}).contiguous();
                if (taps) {
                    capture_tap(taps->trunk_stage[static_cast<std::size_t>(stage_i)], nchw);
                }
                ActivationArena::bind(nullptr);
                recapture(stage_hold_[static_cast<std::size_t>(stage_i)], nchw);
                ActivationArena::bind(&arena_);
                stages[static_cast<std::size_t>(stage_i)] = stage_hold_[static_cast<std::size_t>(stage_i)];
                ++stage_i;
            }
            ActivationArena::bind(nullptr);
            recapture(token_hold_, x);
            ActivationArena::bind(&arena_);
            arena_.rewind(block_mark);
            x = token_hold_;
        }

        std::array<Tensor, 4> fpn{};
        {
            NvtxRange nvtx("sam2/fpn");
            profile.mark("fpn");
            Tensor prev;
            const int n = 3;
            for (int i = n; i >= 0; --i) {
                auto lateral = conv1x1(stages[static_cast<std::size_t>(i)],
                                       std::format("image_encoder.neck.convs.{}.conv.weight", n - i),
                                       std::format("image_encoder.neck.convs.{}.conv.bias", n - i));
                if ((i == 2 || i == 3) && prev.is_valid()) {
                    const int oh = static_cast<int>(prev.shape()[2]) * 2;
                    const int ow = static_cast<int>(prev.shape()[3]) * 2;
                    auto up = resize2d(prev, oh, ow, ResizeMode::Nearest, CoordTransform::Asymmetric);
                    prev = lateral.add(up);
                } else {
                    prev = std::move(lateral);
                }
                fpn[static_cast<std::size_t>(i)] = prev;
            }
        }
        if (taps) {
            capture_tap(taps->fpn[0], fpn[0]);
            capture_tap(taps->fpn[1], fpn[1]);
            capture_tap(taps->fpn[2], fpn[2]);
        }

        Tensor hr0;
        Tensor hr1;
        Tensor embed;
        {
            NvtxRange nvtx("sam2/neck_proj");
            profile.mark("neck_proj");
            hr0 = conv1x1(fpn[0], "sam_mask_decoder.conv_s0.weight", "sam_mask_decoder.conv_s0.bias");
            hr1 = conv1x1(fpn[1], "sam_mask_decoder.conv_s1.weight", "sam_mask_decoder.conv_s1.bias");
            auto nme = w("no_mem_embed").reshape(shape_of({1, static_cast<std::size_t>(kDim), 1, 1}));
            nme = nme.expand(fpn[2].shape()).contiguous();
            embed = fpn[2].add(nme);
        }
        if (taps) {
            capture_tap(taps->high_res_feats0, hr0);
            capture_tap(taps->high_res_feats1, hr1);
            capture_tap(taps->image_embed, embed);
        }

        if (!dense_pe_.is_valid()) {
            ActivationArena::bind(nullptr);
            dense_pe_ = fourier_pe_grid(
                kEmbed, kEmbed, w("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"),
                compute_, device_, img.stream());
            ActivationArena::bind(&arena_);
        }
        if (taps) {
            capture_tap(taps->dense_pe, dense_pe_);
        }

        {
            ActivationArena::bind(nullptr);
            recapture(image_embed_hold_, embed);
            recapture(high_res0_hold_, hr0);
            recapture(high_res1_hold_, hr1);
            ActivationArena::bind(&arena_);
        }

        image_set_ = true;
        profile.mark("end");
        profile.dump();
        return {};
    }

    lfs::Result<Sam2Prediction> Sam2::run_decoder(const std::vector<Sam2PointPrompt>& points,
                                                  const std::optional<std::array<float, 4>>& box,
                                                  bool multimask, Sam2Taps* taps) {
        if (!image_set_) {
            return sam_error(lfs::ErrorCode::FailedPrecondition,
                             "set_image must be called before predict");
        }
        if (points.empty() && !box) {
            return sam_error(lfs::ErrorCode::InvalidArgument,
                             "predict requires at least one point or a box");
        }

        NvtxRange forward_nvtx("sam2/predict");
        const cudaStream_t fwd_stream = image_embed_hold_.stream();
        lfs::core::CUDAStreamGuard stream_guard(fwd_stream);
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

        Tensor sparse;
        Tensor dense;
        {
            profile.mark("prompt");
            sparse = encode_points(points, box);
            auto no_mask = w("sam_prompt_encoder.no_mask_embed.weight")
                               .reshape(shape_of({1, static_cast<std::size_t>(kDim), 1, 1}));
            dense = no_mask
                        .expand(shape_of({1, static_cast<std::size_t>(kDim),
                                          static_cast<std::size_t>(kEmbed),
                                          static_cast<std::size_t>(kEmbed)}))
                        .contiguous();
        }
        if (taps) {
            capture_tap(taps->sparse_embeddings, sparse);
            capture_tap(taps->dense_embeddings, dense);
        }

        Tensor tokens;
        {
            NvtxRange nvtx("sam2/tokens");
            profile.mark("tokens");
            auto obj = w("sam_mask_decoder.obj_score_token.weight")
                           .reshape(shape_of({1, 1, static_cast<std::size_t>(kDim)}));
            auto iou = w("sam_mask_decoder.iou_token.weight")
                           .reshape(shape_of({1, 1, static_cast<std::size_t>(kDim)}));
            auto mask = w("sam_mask_decoder.mask_tokens.weight")
                            .reshape(shape_of({1, 4, static_cast<std::size_t>(kDim)}));
            tokens = concat_contiguous(concat_contiguous(obj, iou, 1), mask, 1);
            tokens = concat_contiguous(tokens, sparse, 1);
        }

        Tensor hs;
        Tensor src_tokens;
        {
            profile.mark("transformer");
            auto src = image_embed_hold_.add(dense);
            auto tr = two_way_transformer(src, dense_pe_, tokens);
            hs = std::move(tr.first);
            src_tokens = std::move(tr.second);
        }
        if (taps) {
            capture_tap(taps->transformer_tokens, hs);
        }

        Tensor upscaled;
        {
            NvtxRange nvtx("sam2/upscale");
            profile.mark("upscale");
            const int b = static_cast<int>(src_tokens.shape()[0]);
            auto src = src_tokens.permute({0, 2, 1})
                           .contiguous()
                           .reshape(shape_of({static_cast<std::size_t>(b), static_cast<std::size_t>(kDim),
                                              static_cast<std::size_t>(kEmbed),
                                              static_cast<std::size_t>(kEmbed)}));
            auto dc1 = conv_transpose2x(src, "sam_mask_decoder.output_upscaling.0.weight",
                                        "sam_mask_decoder.output_upscaling.0.bias");
            auto ln1 = layer_norm_2d(dc1.add(high_res1_hold_),
                                     w("sam_mask_decoder.output_upscaling.1.weight"),
                                     w("sam_mask_decoder.output_upscaling.1.bias"), kLn2dEps);
            auto act1 = gelu(ln1, GELUApprox::Erf);
            auto dc2 = conv_transpose2x(act1, "sam_mask_decoder.output_upscaling.3.weight",
                                        "sam_mask_decoder.output_upscaling.3.bias");
            upscaled = gelu(dc2.add(high_res0_hold_), GELUApprox::Erf);
        }
        if (taps) {
            capture_tap(taps->upscaled_embedding, upscaled);
        }

        Tensor low_res;
        Tensor iou;
        Tensor obj_logits;
        {
            NvtxRange nvtx("sam2/heads");
            profile.mark("heads");
            const int s = 1;
            auto iou_tok = hs.slice(1, static_cast<std::size_t>(s), static_cast<std::size_t>(s + 1))
                               .squeeze(1)
                               .contiguous();
            auto mask_toks = hs.slice(1, static_cast<std::size_t>(s + 1),
                                      static_cast<std::size_t>(s + 1 + 4));
            Tensor hyper_in;
            for (int i = 0; i < 4; ++i) {
                auto tok = mask_toks.slice(1, static_cast<std::size_t>(i),
                                           static_cast<std::size_t>(i + 1))
                               .squeeze(1)
                               .contiguous();
                auto h = mlp(tok, std::format("sam_mask_decoder.output_hypernetworks_mlps.{}", i), 3,
                             Activation::Relu, false);
                h = h.reshape(shape_of({1, 1, h.shape()[h.ndim() - 1]}));
                hyper_in = hyper_in.is_valid() ? concat_contiguous(hyper_in, h, 1) : std::move(h);
            }
            const int uh = static_cast<int>(upscaled.shape()[2]);
            const int uw = static_cast<int>(upscaled.shape()[3]);
            const int uc = static_cast<int>(upscaled.shape()[1]);
            auto up_flat =
                upscaled
                    .reshape(shape_of({1, static_cast<std::size_t>(uc),
                                       static_cast<std::size_t>(uh) * static_cast<std::size_t>(uw)}));
            auto masks = bmm(hyper_in, up_flat, false, false);
            low_res = masks.reshape(shape_of(
                {1, 4, static_cast<std::size_t>(uh), static_cast<std::size_t>(uw)}));
            iou = mlp(iou_tok, "sam_mask_decoder.iou_prediction_head", 3, Activation::Relu, true);
            auto obj_tok = hs.slice(1, 0, 1).squeeze(1).contiguous();
            obj_logits = mlp(obj_tok, "sam_mask_decoder.pred_obj_score_head", 3, Activation::Relu,
                             false);
        }

        const int start = multimask ? 1 : 0;
        const int count = multimask ? 3 : 1;
        auto sel = low_res.slice(1, static_cast<std::size_t>(start),
                                 static_cast<std::size_t>(start + count))
                       .contiguous();
        auto iou_sel = iou.slice(iou.ndim() == 1 ? 0 : 1, static_cast<std::size_t>(start),
                                 static_cast<std::size_t>(start + count))
                           .contiguous();

        Tensor masks_up;
        Tensor low_out;
        Tensor iou_out;
        {
            NvtxRange nvtx("sam2/postprocess");
            profile.mark("postprocess");
            ActivationArena::bind(nullptr);
            auto nchw = sel.reshape(shape_of({static_cast<std::size_t>(count), 1,
                                              static_cast<std::size_t>(kEmbed * 4),
                                              static_cast<std::size_t>(kEmbed * 4)}));
            auto up = resize2d(nchw.to(DataType::Float32), orig_h_, orig_w_, ResizeMode::Bilinear,
                               CoordTransform::HalfPixel);
            masks_up = up.squeeze(1).contiguous();
            low_out = sel.reshape(shape_of({static_cast<std::size_t>(count),
                                            static_cast<std::size_t>(kEmbed * 4),
                                            static_cast<std::size_t>(kEmbed * 4)}))
                          .to(DataType::Float32)
                          .clamp(-32.0f, 32.0f);
            iou_out = iou_sel.to(DataType::Float32).contiguous();
            if (iou_out.ndim() > 1) {
                iou_out = iou_out.reshape(shape_of({static_cast<std::size_t>(count)}));
            }
            ActivationArena::bind(&arena_);
        }
        if (taps) {
            capture_tap(taps->low_res_logits, low_out);
            capture_tap(taps->iou_predictions, iou_out);
            capture_tap(taps->object_score_logits, obj_logits);
        }

        Sam2Prediction pred;
        pred.masks = masks_up;
        pred.iou = iou_out;
        pred.low_res_logits = low_out;

        profile.mark("end");
        profile.dump();
        return pred;
    }

} // namespace lfs::core::nn::models
