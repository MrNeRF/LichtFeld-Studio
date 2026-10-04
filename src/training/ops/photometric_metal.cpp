/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal PhotometricOps; see ops/photometric_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include "core/assert.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <format>
#include <initializer_list>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::Device;
        using lfs::gpu_ops::PhotoPath;
        using lfs::gpu_ops::PhotoSaved;
        using lfs::gpu_ops::Tensor;
        namespace mk = metal;

        constexpr uint32_t kMaxGroups = 1024;
        constexpr uint32_t kTile = 16;
        // Forward partial-sum modes (kPhotoLossMode in photometric.metal).
        enum : uint32_t { kLossNone = 0,
                          kLossSsimMean = 1,
                          kLossFused = 2,
                          kLossMasked = 3 };

        // The workspace variants of ssim.cu's LossWorkspaceArena, with its byte
        // layout: fields 256-byte aligned in one UInt8 block, exact on variant
        // switches, a same-variant shape high-water otherwise.
        enum class Kind : uint8_t { None,
                                    Fused,
                                    PureSSIM,
                                    Decoupled,
                                    MaskedFused,
                                    MaskedDecoupled };
        enum class Slot : uint8_t { Map,
                                    Full16,
                                    Full32,
                                    Temp1024,
                                    Temp2048,
                                    Scalar };

        struct Views {
            Tensor ssim_map, dm_mu, dm_sigma1, dm_sigma12, raw_dm_mu, dl_dmap, grad, temp, result, mask_sum;
        };
        struct Field {
            Tensor Views::*member;
            Slot slot;
        };

        std::vector<Field> layout(const Kind kind) {
            using V = Views;
            switch (kind) {
            case Kind::Fused:
                return {{&V::ssim_map, Slot::Map}, {&V::dm_mu, Slot::Full16}, {&V::dm_sigma1, Slot::Full16}, {&V::dm_sigma12, Slot::Full16}, {&V::grad, Slot::Full32}, {&V::temp, Slot::Temp1024}, {&V::result, Slot::Scalar}};
            case Kind::PureSSIM:
                return {{&V::ssim_map, Slot::Full32}, {&V::dm_mu, Slot::Full16}, {&V::dm_sigma1, Slot::Full16}, {&V::dm_sigma12, Slot::Full16}, {&V::dl_dmap, Slot::Full32}, {&V::grad, Slot::Full32}, {&V::temp, Slot::Temp1024}, {&V::result, Slot::Scalar}};
            case Kind::Decoupled:
                return {{&V::ssim_map, Slot::Map}, {&V::dm_mu, Slot::Full16}, {&V::raw_dm_mu, Slot::Full16}, {&V::dm_sigma1, Slot::Full16}, {&V::dm_sigma12, Slot::Full16}, {&V::grad, Slot::Full32}, {&V::temp, Slot::Temp1024}, {&V::result, Slot::Scalar}};
            case Kind::MaskedFused:
                return {{&V::ssim_map, Slot::Map}, {&V::dm_mu, Slot::Full16}, {&V::dm_sigma1, Slot::Full16}, {&V::dm_sigma12, Slot::Full16}, {&V::grad, Slot::Full32}, {&V::temp, Slot::Temp2048}, {&V::result, Slot::Scalar}, {&V::mask_sum, Slot::Scalar}};
            case Kind::MaskedDecoupled:
                return {{&V::ssim_map, Slot::Map}, {&V::dm_mu, Slot::Full16}, {&V::raw_dm_mu, Slot::Full16}, {&V::dm_sigma1, Slot::Full16}, {&V::dm_sigma12, Slot::Full16}, {&V::grad, Slot::Full32}, {&V::temp, Slot::Temp2048}, {&V::result, Slot::Scalar}, {&V::mask_sum, Slot::Scalar}};
            case Kind::None:
                return {};
            }
            return {};
        }

        size_t align_up(const size_t bytes) { return (bytes + 255) & ~size_t{255}; }

        std::pair<core::TensorShape, DataType> slot_shape(const Slot slot, const std::vector<size_t>& shape) {
            std::vector<size_t> map = shape;
            map[1] = 1;
            switch (slot) {
            case Slot::Map: return {core::TensorShape(map), DataType::Float32};
            case Slot::Full16: return {core::TensorShape(shape), DataType::Float16};
            case Slot::Full32: return {core::TensorShape(shape), DataType::Float32};
            case Slot::Temp1024: return {core::TensorShape({1024}), DataType::Float32};
            case Slot::Temp2048: return {core::TensorShape({2048}), DataType::Float32};
            case Slot::Scalar: return {core::TensorShape({1}), DataType::Float32};
            }
            return {};
        }

        size_t layout_bytes(const Kind kind, const std::vector<size_t>& shape) {
            size_t total = 0;
            for (const Field& field : layout(kind)) {
                const auto [field_shape, dtype] = slot_shape(field.slot, shape);
                total = align_up(total) + field_shape.elements() * core::dtype_size(dtype);
            }
            return align_up(total);
        }

        struct Arena {
            Tensor storage;
            size_t capacity = 0;
            Kind kind = Kind::None;
            std::vector<size_t> shape;
            Views views;

            [[nodiscard]] size_t required() const { return shape.empty() ? 0 : layout_bytes(kind, shape); }

            void replace(const size_t bytes) {
                views = {};
                storage = Tensor::empty_exact({bytes}, DataType::UInt8);
                capacity = bytes;
                kind = Kind::None;
                shape.clear();
            }

            void bind(const Kind next, const std::vector<size_t>& next_shape) {
                views = {};
                size_t offset = 0;
                for (const Field& field : layout(next)) {
                    auto [field_shape, dtype] = slot_shape(field.slot, next_shape);
                    offset = align_up(offset);
                    const size_t bytes = field_shape.elements() * core::dtype_size(dtype);
                    views.*field.member = Tensor::view_sharing_storage(storage, offset, std::move(field_shape), 0,
                                                                       dtype, "training.photometric.workspace");
                    offset += bytes;
                }
                kind = next;
                shape = next_shape;
            }

            Views& ensure(const Kind next, const std::vector<size_t>& next_shape) {
                const size_t bytes = layout_bytes(next, next_shape);
                if (!storage.is_valid() || (kind != Kind::None && kind != next) || bytes > capacity)
                    replace(bytes);
                if (kind != next || shape != next_shape)
                    bind(next, next_shape);
                return views;
            }

            void shrink_to_required() {
                if (kind == Kind::None || shape.empty()) {
                    reset();
                    return;
                }
                const size_t bytes = required();
                if (storage.is_valid() && capacity == bytes)
                    return;
                const Kind active = kind;
                const std::vector<size_t> active_shape = shape;
                replace(bytes);
                bind(active, active_shape);
            }

            void reset() {
                views = {};
                storage = {};
                capacity = 0;
                kind = Kind::None;
                shape.clear();
            }
        };

        struct MetalPhotoState : lfs::gpu_ops::BackendState {
            Arena arena;
            Tensor cs_map; // outside the arena, as the CUDA workspaces keep it
            // Outside the arena too: the CUDA arena leaves the raw-render gradient to add_raw_gradient.
            Tensor grad_raw;
            Tensor l1_grad, l1_loss, l1_partials;
            Tensor error_ssim_map, error_cs_map;
            // Keeps the last allocating metric's tensors alive until the next call.
            std::vector<Tensor> metric_keep;
        };

        MetalPhotoState& state_of(PhotoSaved& saved) {
            LFS_ASSERT_MSG(saved.backend != nullptr, "photometric op called without a created state (backend=null)");
            return static_cast<MetalPhotoState&>(*saved.backend);
        }

        void publish_maps(PhotoSaved& saved, const Tensor& ssim_map, const Tensor& cs_map) {
            saved.ssim_map = ssim_map;
            saved.cs_map = cs_map;
        }

        void ensure_like(Tensor& buffer, const Tensor& like) {
            if (!buffer.is_valid() || buffer.shape() != like.shape() || !buffer.is_contiguous())
                buffer = Tensor::empty(like.shape(), Device::GPU);
        }

        void validate_weight(const float weight) {
            LFS_ASSERT_MSG(std::isfinite(weight) && weight >= 0.0f && weight <= 1.0f,
                           std::format("SSIM loss weight must be finite and in [0,1] (weight={})", weight));
        }

        // loss_tensor_contract.hpp: [C,H,W] or [N,C,H,W], contiguous, as [N,C,H,W].
        Tensor prepare_image(const Tensor& input, const bool target, const std::string_view name) {
            LFS_ASSERT_MSG(input.is_valid() && input.device() == Device::GPU,
                           std::format("{} must be a valid GPU tensor (valid={})", name, input.is_valid()));
            LFS_ASSERT_MSG(input.ndim() == 3 || input.ndim() == 4,
                           std::format("{} must have shape [C,H,W] or [N,C,H,W] (shape={})", name, input.shape().str()));
            const bool dtype_ok = input.dtype() == DataType::Float32 || (target && input.dtype() == DataType::UInt8);
            LFS_ASSERT_MSG(dtype_ok, std::format("{} has an unsupported dtype ({})", name, static_cast<int>(input.dtype())));
            size_t elements = 1;
            for (size_t dim = 0; dim < input.ndim(); ++dim) {
                const size_t extent = input.shape()[dim];
                LFS_ASSERT_MSG(extent > 0 && extent <= static_cast<size_t>(INT_MAX) &&
                                   elements <= static_cast<size_t>(INT_MAX) / extent,
                               std::format("{} dimensions must be positive and fit the signed kernel-index budget (shape={})",
                                           name, input.shape().str()));
                elements *= extent;
            }
            auto prepared = input.contiguous();
            return prepared.ndim() == 3 ? prepared.unsqueeze(0) : prepared;
        }

        Tensor prepare_mask(const Tensor& input, const Tensor& image) {
            LFS_ASSERT_MSG(input.is_valid() && input.device() == Device::GPU,
                           std::format("Loss mask must be a valid GPU tensor (valid={})", input.is_valid()));
            LFS_ASSERT_MSG(input.dtype() == DataType::Float32 || input.dtype() == DataType::UInt8 ||
                               input.dtype() == DataType::Bool,
                           std::format("Loss mask has an unsupported dtype ({})", static_cast<int>(input.dtype())));
            LFS_ASSERT_MSG(input.ndim() == 2 || (input.ndim() == 3 && input.shape()[0] == 1),
                           std::format("Loss mask must have shape [H,W] or [1,H,W] (shape={})", input.shape().str()));
            auto mask = input.contiguous();
            if (mask.ndim() == 3)
                mask = mask.squeeze(0);
            LFS_ASSERT_MSG(mask.shape()[0] == image.shape()[2] && mask.shape()[1] == image.shape()[3],
                           std::format("Loss mask dimensions must match the image (mask={}, image={})",
                                       mask.shape().str(), image.shape().str()));
            return mask;
        }

        struct Images {
            Tensor prediction, raw, target, mask;
            int n = 0, c = 0, h = 0, w = 0;
        };

        Images prepare(const Tensor& prediction, const Tensor& raw, const Tensor& target, const Tensor& mask) {
            Images images{.prediction = prepare_image(prediction, false, "Loss prediction"),
                          .target = prepare_image(target, true, "Loss target")};
            LFS_ASSERT_MSG(images.prediction.shape() == images.target.shape(),
                           std::format("Loss prediction and target shapes must match (prediction={}, target={})",
                                       images.prediction.shape().str(), images.target.shape().str()));
            if (raw.is_valid()) {
                images.raw = prepare_image(raw, false, "Raw loss image");
                LFS_ASSERT_MSG(images.raw.shape() == images.prediction.shape(),
                               std::format("Decoupled loss image shapes must match (corrected={}, raw={})",
                                           images.prediction.shape().str(), images.raw.shape().str()));
            }
            if (mask.is_valid())
                images.mask = prepare_mask(mask, images.prediction);
            const auto& shape = images.prediction.shape();
            images.n = static_cast<int>(shape[0]);
            images.c = static_cast<int>(shape[1]);
            images.h = static_cast<int>(shape[2]);
            images.w = static_cast<int>(shape[3]);
            return images;
        }

        struct Constants {
            std::array<std::pair<uint32_t, uint32_t>, 4> values;
            Constants(const Images& images, const uint32_t loss_mode, const bool channel_mean)
                : values{{{40, images.target.dtype() == DataType::UInt8 ? 1u : 0u},
                          {41, images.mask.is_valid() && images.mask.dtype() != DataType::Float32 ? 1u : 0u},
                          {42, loss_mode},
                          {43, channel_mean ? 1u : 0u}}} {}
        };

        void launch_with(const std::string_view function, const auto& params, const std::initializer_list<const Tensor*> uses,
                         const std::array<uint32_t, 3> groups, const std::array<uint32_t, 3> group,
                         const Constants& constants) {
            mk::kernels().launch({.function = function,
                                  .params = std::as_bytes(std::span(&params, 1)),
                                  .uses = std::span(uses.begin(), uses.size()),
                                  .groups = groups,
                                  .group = group,
                                  .constants = std::span(constants.values)});
        }

        struct ForwardParams {
            uint64_t prediction, raw, target, mask;
            uint64_t ssim_map, cs_map, dm_mu, dm_sigma1, dm_sigma12, raw_dm_mu, error, partials;
            int32_t height, width, channels, batch;
            uint32_t tiles_x, tiles_y;
            float ssim_weight;
            uint32_t valid_padding;
            uint32_t write_map, write_cs, write_partials, write_error, error_from_cs;
        };

        struct ForwardOutputs {
            Tensor ssim_map, cs_map, dm_mu, dm_sigma1, dm_sigma12, raw_dm_mu, error, partials;
        };

        uint32_t tiles_x(const Images& images) { return (static_cast<uint32_t>(images.w) + kTile - 1) / kTile; }
        uint32_t tiles_y(const Images& images) { return (static_cast<uint32_t>(images.h) + kTile - 1) / kTile; }

        // Tiles spread evenly over at most kMaxGroups groups.
        uint32_t forward_groups(const Images& images) {
            const uint32_t tiles = tiles_x(images) * tiles_y(images) * static_cast<uint32_t>(images.n);
            const uint32_t per_group = (tiles + kMaxGroups - 1) / kMaxGroups;
            return (tiles + per_group - 1) / per_group;
        }

        uint32_t ssim_forward(const Images& images, const ForwardOutputs& out, const uint32_t loss_mode,
                              const bool channel_mean, const float ssim_weight, const bool valid_padding,
                              const bool error_from_cs = false) {
            const bool decoupled = images.raw.is_valid();
            const ForwardParams params{
                .prediction = mk::address(images.prediction),
                .raw = mk::address(images.raw),
                .target = mk::address(images.target),
                .mask = mk::address(images.mask),
                .ssim_map = mk::address(out.ssim_map),
                .cs_map = mk::address(out.cs_map),
                .dm_mu = mk::address(out.dm_mu),
                .dm_sigma1 = mk::address(out.dm_sigma1),
                .dm_sigma12 = mk::address(out.dm_sigma12),
                .raw_dm_mu = mk::address(out.raw_dm_mu),
                .error = mk::address(out.error),
                .partials = mk::address(out.partials),
                .height = images.h,
                .width = images.w,
                .channels = images.c,
                .batch = images.n,
                .tiles_x = tiles_x(images),
                .tiles_y = tiles_y(images),
                .ssim_weight = ssim_weight,
                .valid_padding = valid_padding ? 1u : 0u,
                .write_map = out.ssim_map.is_valid() ? 1u : 0u,
                .write_cs = out.cs_map.is_valid() ? 1u : 0u,
                .write_partials = out.dm_mu.is_valid() ? 1u : 0u,
                .write_error = out.error.is_valid() ? 1u : 0u,
                .error_from_cs = error_from_cs ? 1u : 0u,
            };
            const uint32_t groups = forward_groups(images);
            launch_with(decoupled ? "photo_ssim_forward_decoupled" : "photo_ssim_forward", params,
                        {&images.prediction, &images.raw, &images.target, &images.mask, &out.ssim_map, &out.cs_map,
                         &out.dm_mu, &out.dm_sigma1, &out.dm_sigma12, &out.raw_dm_mu, &out.error, &out.partials},
                        {groups, 1, 1}, {kTile, kTile, 1}, Constants(images, loss_mode, channel_mean));
            return groups;
        }

        struct ReduceParams {
            uint64_t partials, result, mask_sum;
            uint32_t count, masked;
            float channels, scale, divisor, offset;
        };

        // result = offset + scale * (sum(partials[0..count)) / divisor).
        void reduce_final(const Tensor& partials, const uint32_t count, Tensor& result, const float scale,
                          const float divisor, const float offset) {
            const ReduceParams params{mk::address(partials), mk::address(result), 0, count, 0, 0.f, scale, divisor, offset};
            mk::launch("loss_reduce_final", params, {&partials, &result}, 1, kMaxGroups);
        }

        // ssim_reduction.cu normalizes by mask_sum * N * C.
        void reduce_masked(const Tensor& partials, const uint32_t count, Tensor& loss, Tensor& mask_sum, const int channels) {
            const ReduceParams params{mk::address(partials), mk::address(loss), mk::address(mask_sum), count, 1,
                                      static_cast<float>(channels), 1.f, 1.f, 0.f};
            mk::launch("loss_reduce_final", params, {&partials, &loss, &mask_sum}, 1, kMaxGroups);
        }

        // ssim_reduction.cu crops each axis separately.
        double valid_count(const Images& images, const bool valid_padding) {
            const int h = valid_padding && images.h > 10 ? images.h - 10 : images.h;
            const int w = valid_padding && images.w > 10 ? images.w - 10 : images.w;
            return static_cast<double>(images.n) * images.c * h * w;
        }

        struct BackwardParams {
            uint64_t prediction, target, mask, mask_sum, dm_mu, dm_sigma1, dm_sigma12, grad;
            int32_t height, width, channels;
            uint32_t tiles_x, tiles_y;
            float ssim_weight, grad_per_pixel;
            uint32_t valid_padding, masked, has_sigma;
        };

        // ssim.cu crops the backward per axis like the forward.
        float grad_per_pixel(const Images& images, const bool valid_padding) {
            return 1.0f / static_cast<float>(valid_count(images, valid_padding));
        }

        void ssim_backward(const Images& images, const Tensor& prediction, const Tensor& dm_mu, const Tensor& dm_sigma1,
                           const Tensor& dm_sigma12, const Tensor& mask_sum, const Tensor& grad, const float ssim_weight,
                           const bool valid_padding) {
            const bool masked = mask_sum.is_valid();
            const BackwardParams params{
                .prediction = mk::address(prediction),
                .target = mk::address(images.target),
                .mask = mk::address(images.mask),
                .mask_sum = mk::address(mask_sum),
                .dm_mu = mk::address(dm_mu),
                .dm_sigma1 = mk::address(dm_sigma1),
                .dm_sigma12 = mk::address(dm_sigma12),
                .grad = mk::address(grad),
                .height = images.h,
                .width = images.w,
                .channels = images.c,
                .tiles_x = tiles_x(images),
                .tiles_y = tiles_y(images),
                .ssim_weight = ssim_weight,
                .grad_per_pixel = masked ? 0.f : grad_per_pixel(images, valid_padding),
                .valid_padding = valid_padding ? 1u : 0u,
                .masked = masked ? 1u : 0u,
                .has_sigma = dm_sigma1.is_valid() ? 1u : 0u,
            };
            launch_with("photo_ssim_backward", params,
                        {&prediction, &images.target, &images.mask, &mask_sum, &dm_mu, &dm_sigma1, &dm_sigma12, &grad},
                        {tiles_x(images), tiles_y(images) * static_cast<uint32_t>(images.n), 1}, {kTile, kTile, 1},
                        Constants(images, kLossNone, false));
        }

        std::vector<size_t> dims(const Tensor& tensor) {
            return {tensor.shape().dims().begin(), tensor.shape().dims().end()};
        }

        void squeeze_like(Tensor& gradient, const Tensor& input) {
            if (input.ndim() == 3 && gradient.is_valid() && gradient.ndim() == 4)
                gradient = gradient.squeeze(0);
        }

        void evaluate_l1(MetalPhotoState& state, const Tensor& corrected, const Tensor& target, Tensor& loss, Tensor& grad) {
            const Images images = prepare(corrected, {}, target, {});
            const size_t count = images.prediction.numel();
            LFS_ASSERT_MSG(count <= UINT32_MAX, std::format("L1 loss supports at most 2^32-1 elements (count={})", count));
            const uint32_t groups = static_cast<uint32_t>(std::min<size_t>((count + 255) / 256, kMaxGroups));
            if (!state.l1_grad.is_valid() || state.l1_grad.shape() != images.prediction.shape() ||
                state.l1_partials.numel() != groups) {
                state.l1_grad = Tensor::empty(images.prediction.shape(), Device::GPU);
                state.l1_loss = Tensor::zeros({1}, Device::GPU);
                state.l1_partials = Tensor::empty({groups}, Device::GPU);
            }
            struct {
                uint64_t prediction, target, grad, partials;
                uint32_t count;
                float grad_scale;
            } const params{mk::address(images.prediction), mk::address(images.target), mk::address(state.l1_grad),
                           mk::address(state.l1_partials), static_cast<uint32_t>(count), 1.0f / static_cast<float>(count)};
            launch_with("photo_l1", params, {&images.prediction, &images.target, &state.l1_grad, &state.l1_partials},
                        {groups, 1, 1}, {256, 1, 1}, Constants(images, kLossNone, false));
            reduce_final(state.l1_partials, groups, state.l1_loss, 1.0f / static_cast<float>(count), 1.f, 0.f);
            grad = state.l1_grad;
            loss = state.l1_loss;
            squeeze_like(grad, corrected);
        }

        lfs::gpu_ops::State photo_create() { return std::make_unique<MetalPhotoState>(); }

        void photo_evaluate(PhotoSaved& saved, const Tensor& corrected, const Tensor& raw, const Tensor& target,
                            const Tensor& mask, const lfs::gpu_ops::PhotoParams& params, Tensor& loss,
                            Tensor& grad_corrected, Tensor& grad_raw) {
            auto& state = state_of(saved);
            validate_weight(params.ssim_weight);
            const bool decoupled = params.path == PhotoPath::Decoupled || params.path == PhotoPath::MaskedDecoupled;
            const bool masked = params.path == PhotoPath::MaskedFused || params.path == PhotoPath::MaskedDecoupled;
            if (!decoupled && grad_raw.is_valid())
                grad_raw = {};
            if (params.path == PhotoPath::L1) {
                evaluate_l1(state, corrected, target, loss, grad_corrected);
                return;
            }
            LFS_ASSERT_MSG(!decoupled || raw.is_valid(), "Decoupled photometric paths need a raw render (raw=invalid)");
            LFS_ASSERT_MSG(!masked || mask.is_valid(), "Masked photometric paths need a mask (mask=invalid)");
            const Images images = prepare(corrected, decoupled ? raw : Tensor{}, target, masked ? mask : Tensor{});
            const bool padding = params.valid_padding && !masked;
            const Kind kind = params.path == PhotoPath::SSIM        ? Kind::PureSSIM
                              : params.path == PhotoPath::Fused     ? Kind::Fused
                              : params.path == PhotoPath::Decoupled ? Kind::Decoupled
                              : masked && decoupled                 ? Kind::MaskedDecoupled
                                                                    : Kind::MaskedFused;
            Views& ws = state.arena.ensure(kind, dims(images.prediction));
            ensure_like(state.cs_map, ws.ssim_map);
            if (decoupled)
                ensure_like(state.grad_raw, ws.grad);
            else
                state.grad_raw = {};

            const bool pure = kind == Kind::PureSSIM;
            const float weight = pure ? 1.0f : params.ssim_weight;
            const uint32_t mode = pure ? kLossSsimMean : masked ? kLossMasked
                                                                : kLossFused;
            const uint32_t groups = ssim_forward(images,
                                                 {.ssim_map = ws.ssim_map,
                                                  .cs_map = state.cs_map,
                                                  .dm_mu = ws.dm_mu,
                                                  .dm_sigma1 = ws.dm_sigma1,
                                                  .dm_sigma12 = ws.dm_sigma12,
                                                  .raw_dm_mu = ws.raw_dm_mu,
                                                  .partials = ws.temp},
                                                 mode, !pure, weight, padding);
            if (masked)
                reduce_masked(ws.temp, groups, ws.result, ws.mask_sum, images.n * images.c);
            else if (pure) // loss = 1 - mean SSIM
                reduce_final(ws.temp, groups, ws.result, -1.f, static_cast<float>(valid_count(images, padding)), 1.f);
            else
                reduce_final(ws.temp, groups, ws.result, 1.f, static_cast<float>(valid_count(images, padding)), 0.f);

            const Tensor mask_sum = masked ? ws.mask_sum : Tensor{};
            if (decoupled) {
                ssim_backward(images, images.prediction, ws.dm_mu, {}, {}, mask_sum, ws.grad, weight, padding);
                ssim_backward(images, images.raw, ws.raw_dm_mu, ws.dm_sigma1, ws.dm_sigma12, mask_sum, state.grad_raw, 1.f,
                              padding);
            } else {
                ssim_backward(images, images.prediction, ws.dm_mu, ws.dm_sigma1, ws.dm_sigma12, mask_sum, ws.grad, weight,
                              padding);
            }
            loss = ws.result;
            grad_corrected = ws.grad;
            squeeze_like(grad_corrected, corrected);
            if (decoupled) {
                grad_raw = state.grad_raw;
                squeeze_like(grad_raw, corrected);
            }
            publish_maps(saved, ws.ssim_map, state.cs_map);
        }

        Tensor photo_metric(PhotoSaved& saved, const Tensor& predicted, const Tensor& target, const bool maps,
                            const bool valid_padding) {
            auto& state = state_of(saved);
            const Images images = prepare(predicted, {}, target, {});
            ForwardOutputs out{.partials = Tensor::empty({kMaxGroups}, Device::GPU)};
            if (maps) {
                out.ssim_map = Tensor::zeros(images.prediction.shape(), Device::GPU);
                out.cs_map = Tensor::zeros(images.prediction.shape(), Device::GPU);
            }
            const uint32_t groups = ssim_forward(images, out, kLossSsimMean, false, 1.f, valid_padding);
            Tensor value = Tensor::empty({1}, Device::GPU);
            reduce_final(out.partials, groups, value, 1.f, static_cast<float>(valid_count(images, valid_padding)), 0.f);
            if (maps)
                publish_maps(saved, out.ssim_map, out.cs_map);
            state.metric_keep = {images.prediction, images.target, out.ssim_map, out.cs_map, out.partials, value};
            return value;
        }

        void photo_error_map(PhotoSaved& saved, const Tensor& predicted, const Tensor& target, Tensor& error,
                             const bool contrast_structure_only) {
            auto& state = state_of(saved);
            const Images images = prepare(predicted, {}, target, {});
            LFS_ASSERT_MSG(images.n == 1, std::format("SSIM error maps require a single-image batch (shape={})",
                                                      images.prediction.shape().str()));
            if (!state.error_ssim_map.is_valid() || state.error_ssim_map.shape() != images.prediction.shape()) {
                state.error_ssim_map = Tensor::empty(images.prediction.shape(), Device::GPU);
                state.error_cs_map = Tensor::empty(images.prediction.shape(), Device::GPU);
            }
            const size_t h = static_cast<size_t>(images.h), w = static_cast<size_t>(images.w);
            if (!error.is_valid() || error.device() != Device::GPU || error.dtype() != DataType::Float32 ||
                !error.is_contiguous() || error.ndim() != 2 || error.shape()[0] != h || error.shape()[1] != w)
                error = Tensor::empty({h, w}, Device::GPU);
            ssim_forward(images, {.ssim_map = state.error_ssim_map, .cs_map = state.error_cs_map, .error = error},
                         kLossNone, false, 1.f, false, contrast_structure_only);
        }

        void photo_map_to_error(const Tensor& map, Tensor& error) {
            LFS_ASSERT_MSG(map.is_valid() && map.device() == Device::GPU && map.dtype() == DataType::Float32 &&
                               map.is_contiguous() && map.ndim() == 4 && map.shape()[0] == 1 && map.shape()[1] > 0 &&
                               map.shape()[2] > 0 && map.shape()[3] > 0,
                           std::format("SSIM map must be contiguous Float32 GPU [1,C,H,W] (shape={})", map.shape().str()));
            const size_t h = map.shape()[2], w = map.shape()[3];
            LFS_ASSERT_MSG(error.is_valid() && error.device() == Device::GPU && error.dtype() == DataType::Float32 &&
                               error.is_contiguous() && error.ndim() == 2 && error.shape()[0] == h && error.shape()[1] == w,
                           std::format("SSIM error map must be contiguous Float32 GPU [H,W] (map={}, expected=[{}, {}])",
                                       error.shape().str(), h, w));
            struct {
                uint64_t map, error;
                uint32_t plane, channels;
            } const params{mk::address(map), mk::address(error), static_cast<uint32_t>(h * w),
                           static_cast<uint32_t>(map.shape()[1])};
            mk::launch_items("photo_map_to_error", params, {&map, &error}, h * w);
        }

        size_t reserved_bytes(const Tensor& tensor) {
            if (!tensor.is_valid())
                return 0;
            if (tensor.capacity() == 0 || tensor.ndim() == 0)
                return tensor.bytes();
            size_t row = 1;
            for (size_t dim = 1; dim < tensor.ndim(); ++dim)
                row *= tensor.shape()[dim];
            return tensor.capacity() * row * core::dtype_size(tensor.dtype());
        }

        lfs::gpu_ops::PhotoWorkspaceBytes photo_workspace_bytes(const PhotoSaved& saved) {
            const auto* state = static_cast<const MetalPhotoState*>(saved.backend.get());
            if (!state)
                return {};
            return {.required = state->arena.required(),
                    .allocated = state->arena.capacity,
                    .error_map = reserved_bytes(state->error_ssim_map)};
        }

        void photo_shrink_to_required(PhotoSaved& saved) {
            if (saved.backend)
                state_of(saved).arena.shrink_to_required();
        }

        void photo_reset(PhotoSaved& saved) {
            if (saved.backend) {
                state_of(saved).arena.reset();
                state_of(saved).grad_raw = {};
            }
        }
    } // namespace

    const lfs::gpu_ops::PhotometricOps& metal_photometric_ops() {
        static const lfs::gpu_ops::PhotometricOps ops{
            .create = photo_create,
            .evaluate = photo_evaluate,
            .metric = photo_metric,
            .error_map = photo_error_map,
            .map_to_error = photo_map_to_error,
            .workspace_bytes = photo_workspace_bytes,
            .shrink_to_required = photo_shrink_to_required,
            .reset = photo_reset,
        };
        return ops;
    }
} // namespace lfs::training
