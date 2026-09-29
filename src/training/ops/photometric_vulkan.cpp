/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/photometric_vulkan.hpp"
#include "vulkan/dispatch.hpp"
#include <climits>
#include <cmath>

namespace lfs::training {
    namespace {
        using namespace gpu_ops;
        using core::DataType;
        using core::Device;
        using vulkan::address;
        using vulkan::ref;
        struct Params {
            uint64_t corrected, raw, target, mask, horizontal, partials;
            uint64_t ssim, cs, grad, grad_raw, losses, normalizer;
            uint32_t count, channels, height, width, batch, stage, path, target_byte;
            uint32_t mask_byte, valid_padding, partial_stride, reserved;
            float weight;
        };
        static_assert(sizeof(Params) == 152);
        static_assert(offsetof(Params, count) == 96);
        static_assert(offsetof(Params, weight) == 144);
        size_t aligned(size_t bytes) { return (bytes + 255) & ~size_t{255}; }
        bool decoupled(PhotoPath p) { return p == PhotoPath::Decoupled || p == PhotoPath::MaskedDecoupled; }
        bool masked(PhotoPath p) { return p == PhotoPath::MaskedFused || p == PhotoPath::MaskedDecoupled; }
        struct PhotoState : BackendState {
            Tensor arena, map, cs, gradient, raw_gradient;
            Tensor horizontal, full_map, full_cs, losses, normalizer, l1_gradient;
            Tensor error_storage;
            PhotoPath kind = PhotoPath::L1;
            size_t required = 0, capacity = 0, partial_offset = 0, partial_stride = 0;
            core::TensorShape shape;
            void ensure(PhotoPath path, core::TensorShape dims) {
                const size_t e = dims.elements(), pixels = dims[0] * dims[2] * dims[3];
                std::vector<size_t> sizes{4 * (path == PhotoPath::SSIM ? e : pixels)};
                for (int j = 0; j < (decoupled(path) ? 4 : 3); ++j)
                    sizes.push_back(2 * e);
                if (path == PhotoPath::SSIM)
                    sizes.push_back(4 * e);
                sizes.push_back(4 * e);
                if (decoupled(path))
                    sizes.push_back(4 * e);
                sizes.push_back(masked(path) ? 8192 : 4096);
                sizes.push_back(4);
                if (masked(path))
                    sizes.push_back(4);
                required = 0;
                for (size_t bytes : sizes)
                    required = aligned(required) + bytes;
                required = aligned(required);
                if (!arena.is_valid() || kind != path || required > capacity) {
                    arena = Tensor::empty_exact({required / 4});
                    capacity = required;
                }
                kind = path;
                shape = dims;
                // Rebind arena views without invoking Tensor's view-copy assignment.
                map = Tensor{};
                gradient = Tensor{};
                raw_gradient = Tensor{};
                size_t offset = 0;
                auto field = [&](size_t bytes, core::TensorShape field_shape) {
                    offset = aligned(offset);
                    auto result = arena.slice(0, offset / 4, (offset + bytes) / 4).reshape(field_shape);
                    offset += bytes;
                    return result;
                };
                core::TensorShape map_shape{dims[0], path == PhotoPath::SSIM ? dims[1] : 1, dims[2], dims[3]};
                map = field(sizes[0], map_shape);
                partial_offset = aligned(offset);
                partial_stride = aligned(2 * e) / 2;
                offset = partial_offset;
                for (int j = 0; j < (decoupled(path) ? 4 : 3); ++j)
                    offset = aligned(offset) + 2 * e;
                if (path == PhotoPath::SSIM)
                    offset = aligned(offset) + 4 * e;
                gradient = field(4 * e, dims);
                raw_gradient = decoupled(path) ? field(4 * e, dims) : Tensor{};
                cs = Tensor::empty(map_shape, Device::GPU);
            }
        };
        PhotoState& state(PhotoSaved& saved) {
            LFS_ASSERT_MSG(saved.backend != nullptr, "Vulkan photometric requires a created state");
            return static_cast<PhotoState&>(*saved.backend);
        }
        Tensor image(In input, bool target) {
            LFS_ASSERT_MSG(input.is_valid() && input.device() == Device::GPU && (input.ndim() == 3 || input.ndim() == 4),
                           "Photometric input must be GPU CHW or NCHW");
            LFS_ASSERT_MSG(input.dtype() == DataType::Float32 || (target && input.dtype() == DataType::UInt8),
                           "Photometric input has an unsupported dtype");
            size_t count = 1;
            for (size_t dim : input.shape().dims()) {
                LFS_ASSERT_MSG(dim > 0 && dim <= INT_MAX && count <= INT_MAX / dim, "Photometric input exceeds signed indexing");
                count *= dim;
            }
            (void)ref(input);
            auto result = input.contiguous();
            return result.ndim() == 3 ? result.unsqueeze(0) : result;
        }
        void bind(Out dst, In src) { dst = Tensor(src); }
        void run(PhotoState& s, In corrected, In raw, In target, In mask, const PhotoParams& options,
                 Out loss, Out grad, Out raw_grad, bool derivatives = true) {
            LFS_ASSERT_MSG(std::isfinite(options.ssim_weight) && options.ssim_weight >= 0 && options.ssim_weight <= 1,
                           "SSIM weight must be finite and in [0,1]");
            auto a = image(corrected, false), t = image(target, true);
            LFS_ASSERT_MSG(a.shape() == t.shape(), "Photometric image shapes must match");
            Tensor r, m;
            if (decoupled(options.path)) {
                r = image(raw, false);
                LFS_ASSERT_MSG(r.shape() == a.shape(), "Photometric raw image shape must match");
            }
            const auto dims = a.shape();
            if (masked(options.path)) {
                LFS_ASSERT_MSG(mask.is_valid() && mask.device() == Device::GPU &&
                                   (mask.ndim() == 2 || (mask.ndim() == 3 && mask.shape()[0] == 1)) &&
                                   (mask.dtype() == DataType::Float32 || mask.dtype() == DataType::UInt8 || mask.dtype() == DataType::Bool),
                               "Photometric mask must be GPU float, byte or bool HW or 1HW");
                m = mask.contiguous();
                if (m.ndim() == 3)
                    m = m.squeeze(0);
                LFS_ASSERT_MSG(m.shape()[0] == dims[2] && m.shape()[1] == dims[3], "Photometric mask shape must match");
                s.normalizer = (m.dtype() == DataType::Float32 ? m : (m != 0).to(DataType::Float32)).sum();
            }
            Params p{};
            p.corrected = address(a);
            p.raw = address(r);
            p.target = address(t);
            p.mask = address(m);
            p.count = static_cast<uint32_t>(a.numel());
            p.channels = dims[1];
            p.height = dims[2];
            p.width = dims[3];
            p.batch = dims[0];
            p.path = static_cast<uint32_t>(options.path);
            p.target_byte = t.dtype() == DataType::UInt8;
            p.mask_byte = m.is_valid() && m.dtype() != DataType::Float32;
            p.valid_padding = options.valid_padding;
            p.weight = options.path == PhotoPath::SSIM ? 1.f : options.ssim_weight;
            s.losses = Tensor::empty(dims, Device::GPU);
            p.losses = address(s.losses);
            std::vector<core::internal::StorageRef> inputs{ref(a), ref(t)};
            if (r.is_valid())
                inputs.push_back(ref(r));
            if (m.is_valid()) {
                inputs.push_back(ref(m));
                inputs.push_back(ref(s.normalizer));
                p.normalizer = address(s.normalizer);
            }
            if (options.path == PhotoPath::L1) {
                s.l1_gradient = Tensor::empty(dims, Device::GPU);
                p.grad = address(s.l1_gradient);
                p.stage = 4;
                const std::array writes{ref(s.losses), ref(s.l1_gradient)};
                vulkan::dispatch("photometric", p, inputs, writes, vulkan::groups(a.numel()), p.stage | (p.path << 3));
                bind(grad, s.l1_gradient);
            } else {
                if (derivatives)
                    s.ensure(options.path, dims);
                else
                    s.shape = dims;
                s.horizontal = Tensor::empty({6 * a.numel()}, Device::GPU);
                s.full_map = Tensor::empty(dims, Device::GPU);
                s.full_cs = Tensor::empty(dims, Device::GPU);
                p.horizontal = address(s.horizontal);
                p.partials = derivatives ? address(s.arena) + s.partial_offset : 0;
                p.partial_stride = s.partial_stride;
                p.ssim = address(s.full_map);
                p.cs = address(s.full_cs);
                p.grad = address(s.gradient);
                p.grad_raw = address(s.raw_gradient);
                std::vector<core::internal::StorageRef> writes{ref(s.horizontal), ref(s.full_map), ref(s.full_cs), ref(s.losses)};
                if (derivatives)
                    writes.push_back(ref(s.arena));
                auto reads = inputs;
                reads.insert(reads.end(), writes.begin(), writes.end());
                for (uint32_t stage = 0; stage < 5; ++stage) {
                    if (!derivatives && (stage == 2 || stage == 3))
                        continue;
                    p.stage = stage;
                    vulkan::dispatch("photometric", p, reads, writes, vulkan::groups(a.numel()), p.stage | (p.path << 3));
                }
                if (options.path == PhotoPath::SSIM) {
                    if (derivatives) {
                        s.map.copy_(s.full_map);
                        s.cs.copy_(s.full_cs);
                    } else {
                        s.map = Tensor(s.full_map);
                        s.cs = Tensor(s.full_cs);
                    }
                } else {
                    p.stage = 5;
                    p.grad = address(s.map);
                    p.grad_raw = address(s.cs);
                    const std::array map_writes{ref(s.arena), ref(s.cs)};
                    vulkan::dispatch("photometric", p, reads, map_writes, vulkan::groups(a.numel()), p.stage | (p.path << 3));
                }
                bind(grad, s.gradient);
            }
            if (masked(options.path))
                loss = (s.losses.sum() / (s.normalizer * float(p.channels) + 1e-8f)).reshape({1});
            else {
                auto selected = s.losses;
                if (options.path != PhotoPath::L1 && options.valid_padding) {
                    if (p.height > 10)
                        selected = selected.slice(2, 5, p.height - 5);
                    if (p.width > 10)
                        selected = selected.slice(3, 5, p.width - 5);
                }
                loss = selected.mean().reshape({1});
            }
            raw_grad = decoupled(options.path) ? Tensor(s.raw_gradient) : Tensor{};
            if (derivatives && corrected.ndim() == 3) {
                grad = grad.squeeze(0);
                if (raw_grad.is_valid())
                    raw_grad = raw_grad.squeeze(0);
            }
        }
        State create() { return std::make_unique<PhotoState>(); }
        void evaluate(PhotoSaved& saved, In corrected, In raw, In target, In mask, const PhotoParams& options,
                      Out loss, Out grad, Out raw_grad) {
            auto& s = state(saved);
            run(s, corrected, raw, target, mask, options, loss, grad, raw_grad);
            if (options.path != PhotoPath::L1) {
                bind(saved.ssim_map, s.map);
                bind(saved.cs_map, s.cs);
            }
        }
        Tensor metric(PhotoSaved& saved, In predicted, In target, bool maps, bool valid_padding) {
            PhotoState temporary;
            Tensor loss, grad, raw_grad;
            run(temporary, predicted, {}, target, {}, {PhotoPath::SSIM, 1.f, valid_padding}, loss, grad, raw_grad, false);
            if (maps) {
                bind(saved.ssim_map, temporary.map);
                bind(saved.cs_map, temporary.cs);
            }
            return loss * -1.f + 1.f;
        }
        void map_to_error(In map, Out error) {
            LFS_ASSERT_MSG(map.is_valid() && map.dtype() == DataType::Float32 && map.is_contiguous() &&
                               map.ndim() == 4 && map.shape()[0] == 1 && map.shape()[1] > 0 && map.shape()[2] > 0 && map.shape()[3] > 0,
                           "SSIM map must be contiguous float [1,C,H,W]");
            LFS_ASSERT_MSG(error.is_valid() && error.dtype() == DataType::Float32 && error.is_contiguous() &&
                               error.ndim() == 2 && error.shape()[0] == map.shape()[2] && error.shape()[1] == map.shape()[3],
                           "SSIM error must be contiguous float [H,W]");
            Params p{};
            p.ssim = address(map);
            p.grad = address(error);
            p.stage = 6;
            p.count = map.numel();
            p.channels = map.shape()[1];
            p.height = map.shape()[2];
            p.width = map.shape()[3];
            const std::array reads{ref(map)}, writes{ref(error)};
            vulkan::dispatch("photometric", p, reads, writes, vulkan::groups(error.numel()), p.stage | (p.path << 3));
        }
        void error_map(PhotoSaved& saved, In predicted, In target, Out error, bool cs_only) {
            auto& s = state(saved);
            PhotoState temporary;
            Tensor loss, grad, raw_grad;
            run(temporary, predicted, {}, target, {}, {PhotoPath::SSIM, 1.f, false}, loss, grad, raw_grad, false);
            LFS_ASSERT_MSG(temporary.shape[0] == 1, "SSIM error map requires batch one");
            s.error_storage = Tensor::empty(temporary.shape, Device::GPU);
            s.error_storage.copy_(temporary.map);
            const core::TensorShape dims{temporary.shape[2], temporary.shape[3]};
            if (!error.is_valid() || error.shape() != dims || error.dtype() != DataType::Float32 ||
                error.device() != Device::GPU || !error.is_contiguous())
                error = Tensor::empty(dims, Device::GPU);
            map_to_error(cs_only ? temporary.cs : temporary.map, error);
        }
        PhotoWorkspaceBytes workspace_bytes(const PhotoSaved& saved) {
            if (!saved.backend)
                return {};
            const auto& s = static_cast<const PhotoState&>(*saved.backend);
            size_t error = 0;
            if (s.error_storage.is_valid()) {
                error = s.error_storage.capacity();
                for (size_t i = 1; i < s.error_storage.ndim(); ++i)
                    error *= s.error_storage.shape()[i];
                error *= 4;
                if (error == 0)
                    error = s.error_storage.bytes();
            }
            return {s.required, s.capacity, error};
        }
        void shrink(PhotoSaved& saved) {
            if (!saved.backend)
                return;
            auto& s = state(saved);
            if (s.required && s.capacity != s.required) {
                s.arena = {};
                s.ensure(s.kind, s.shape);
            }
        }
        void reset(PhotoSaved& saved) {
            if (!saved.backend)
                return;
            auto& s = state(saved);
            s.arena = {};
            s.map = {};
            s.cs = {};
            s.gradient = {};
            s.raw_gradient = {};
            s.required = s.capacity = 0;
            s.kind = PhotoPath::L1;
        }
        const PhotometricOps ops{create, evaluate, metric, error_map, map_to_error, workspace_bytes, shrink, reset};
    } // namespace
    const gpu_ops::PhotometricOps& vulkan_photometric_ops() { return ops; }
} // namespace lfs::training
