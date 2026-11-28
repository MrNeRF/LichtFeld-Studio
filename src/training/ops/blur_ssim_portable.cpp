/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "blur_ssim_portable.hpp"
#include "blur_ssim_r1_program.hpp"
#include "blur_ssim_r2_program.hpp"
#include "blur_ssim_r3_program.hpp"
#include "blur_ssim_r4_program.hpp"
#include "blur_ssim_r5_program.hpp"
#include "core/assert.hpp"
#include "core/error.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/blur_tolerant_ssim.hpp"

#include <cmath>
#include <cstddef>
#include <format>
#include <span>

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::GpuKernelModule;
        using core::Tensor;

        constexpr uint32_t kTile = 16;
        constexpr uint32_t kMaxGroups = 1024; // blur_ssim.slang's kMaxGroups
        constexpr int kMaxRadius = 5;

        // Parameters in blur_ssim.slang: 16 tensor pointers, then the scalars.
        struct Parameters {
            uint64_t pointers[16]{};
            int32_t width = 0, height = 0, channels = 0, batches = 0;
            uint32_t tiles_x = 0, tiles_y = 0, groups = 0, count = 0;
            float ssim_weight = 0.0f, chain = 0.0f, divisor = 0.0f, denominator = 0.0f;
            std::array<float, kMaxRadius + 1> taps{};
        };
        static_assert(sizeof(Parameters) == 200 && offsetof(Parameters, width) == 128 &&
                      offsetof(Parameters, taps) == 176);
        enum Slot : uint32_t { Corrected,
                               Raw,
                               Photo,
                               PhotoBytes,
                               Mask,
                               MaskBytes,
                               SsimMap,
                               CsMap,
                               Sharp,
                               Blurred,
                               Vu,
                               Partials,
                               Loss,
                               MaskSum,
                               Grad,
                               GradRaw,
                               SlotCount };

        // The tensors one pass reads and writes.
        struct Pass {
            std::array<const Tensor*, SlotCount> slots{};
            uint32_t writes = 0; // bit per written slot

            void read(const Slot s, const Tensor& t) { slots[s] = t.is_valid() ? &t : nullptr; }
            void write(const Slot s, const Tensor& t) {
                read(s, t);
                writes |= 1u << s;
            }
        };

        // One program per blur radius, so every tap loop has a fixed length.
        std::span<const GpuKernelModule::Entry> program_entries(const int radius) {
            switch (radius) {
            case 1: return blur_ssim_r1_program_entries();
            case 2: return blur_ssim_r2_program_entries();
            case 3: return blur_ssim_r3_program_entries();
            case 4: return blur_ssim_r4_program_entries();
            default: return blur_ssim_r5_program_entries();
            }
        }

    } // namespace

    BlurSsimTaps blur_ssim_taps(const float sigma) {
        LFS_ASSERT_MSG(std::isfinite(sigma) && sigma > 0.0f && sigma <= kernels::BLUR_TOLERANT_SSIM_MAX_SIGMA,
                       std::format("Blur-tolerant SSIM sigma must be within (0, 1.5] (sigma={})", sigma));
        BlurSsimTaps taps{.radius = static_cast<int>(std::ceil(3.0 * sigma))};
        double raw[kMaxRadius + 1] = {}, sum = 0.0;
        for (int d = 0; d <= taps.radius; ++d) {
            raw[d] = std::exp(-0.5 * d * d / (static_cast<double>(sigma) * sigma));
            sum += d == 0 ? raw[d] : 2.0 * raw[d];
        }
        for (int d = 0; d <= taps.radius; ++d)
            taps.weight[d] = static_cast<float>(raw[d] / sum);
        return taps;
    }

    void blur_tolerant_ssim(BlurSsimPrograms& programs, const BlurSsimImages& images, const BlurSsimWorkspace& workspace,
                            const float sigma, const float ssim_weight, const float denominator) {
        const auto& shape = images.corrected.shape();
        LFS_ASSERT_MSG(shape.rank() == 4, std::format("Blur-tolerant SSIM images must be [N,C,H,W] (shape={})", shape.str()));
        const BlurSsimTaps taps = blur_ssim_taps(sigma);
        const auto backend = core::gpu_backend_of(images.corrected).value_or(core::default_gpu_backend());
        const core::GpuBackendScope scope(backend);
        auto& program = programs.loaded[taps.radius - 1];
        if (!program) {
            auto loaded = GpuKernelModule::load(program_entries(taps.radius), backend);
            if (!loaded)
                throw lfs::Exception(std::move(loaded).error());
            program = std::move(*loaded);
        }

        const bool masked = images.mask.is_valid();
        const uint32_t n = static_cast<uint32_t>(shape[0]), c = static_cast<uint32_t>(shape[1]);
        const uint32_t h = static_cast<uint32_t>(shape[2]), w = static_cast<uint32_t>(shape[3]);
        const uint32_t tiles_x = (w + kTile - 1) / kTile, tiles_y = (h + kTile - 1) / kTile;
        // Tiles spread evenly over at most kMaxGroups forward groups, one partial sum each.
        const uint32_t tiles = tiles_x * tiles_y * n;
        const uint32_t per_group = (tiles + kMaxGroups - 1) / kMaxGroups;
        const uint32_t groups = (tiles + per_group - 1) / per_group;
        // The valid padding crops each axis on its own, as ssim_reduction.cu.
        const double cropped = static_cast<double>(n) * c * (h > 10 ? h - 10 : h) * (w > 10 ? w - 10 : w);
        Parameters p;
        p.width = static_cast<int32_t>(w);
        p.height = static_cast<int32_t>(h);
        p.channels = static_cast<int32_t>(c);
        p.batches = static_cast<int32_t>(n);
        p.tiles_x = tiles_x;
        p.tiles_y = tiles_y;
        p.groups = groups;
        p.count = groups;
        p.ssim_weight = ssim_weight;
        p.chain = masked ? 0.0f : 1.0f / static_cast<float>(cropped);
        p.divisor = static_cast<float>(cropped);
        p.denominator = masked ? denominator : 0.0f;
        p.taps = taps.weight;

        const bool photo_bytes = images.photo.dtype() != DataType::Float32;
        const bool mask_bytes = masked && images.mask.dtype() != DataType::Float32;
        const auto read_images = [&](Pass& pass) {
            pass.read(Corrected, images.corrected);
            pass.read(Raw, images.raw);
            pass.read(photo_bytes ? PhotoBytes : Photo, images.photo);
            if (masked)
                pass.read(mask_bytes ? MaskBytes : Mask, images.mask);
        };
        const auto run = [&](const char* function, const Pass& pass, const std::array<uint32_t, 3> grid,
                             const std::array<uint32_t, 3> group) {
            std::array<GpuKernelModule::Binding, SlotCount> bindings;
            for (uint32_t s = 0; s < SlotCount; ++s)
                bindings[s] = {s * 8, pass.slots[s],
                               pass.writes & (1u << s) ? GpuKernelModule::Access::ReadWrite : GpuKernelModule::Access::Read};
            auto result = program->dispatch({.function = function,
                                             .arguments = {std::as_bytes(std::span(&p, 1)), bindings},
                                             .groups = grid,
                                             .group = group});
            if (!result)
                throw lfs::Exception(std::move(result).error());
        };
        const std::array<uint32_t, 3> tile{kTile, kTile, 1};

        Pass forward;
        read_images(forward);
        forward.write(SsimMap, workspace.ssim_map);
        forward.write(CsMap, workspace.cs_map);
        forward.write(Sharp, workspace.sharp);
        forward.write(Blurred, workspace.blurred);
        forward.write(Partials, workspace.partials);
        run(masked ? "blur_forward_masked" : "blur_forward", forward, {groups, 1, 1}, tile);

        Pass reduce;
        reduce.read(Partials, workspace.partials);
        reduce.write(Loss, workspace.loss);
        if (masked)
            reduce.write(MaskSum, workspace.mask_sum);
        run("blur_reduce", reduce, {1, 1, 1}, {256, 1, 1});

        const std::array<uint32_t, 3> grid{tiles_x, tiles_y * n, 1};
        Pass branch;
        read_images(branch);
        if (masked)
            branch.read(MaskSum, workspace.mask_sum);
        branch.read(Blurred, workspace.blurred);
        branch.write(Vu, workspace.vu);
        run(masked ? "blur_branch_masked" : "blur_branch", branch, grid, tile);

        // The gradients may overwrite the blurred partials the branch pass has just read.
        Pass gradient;
        read_images(gradient);
        if (masked)
            gradient.read(MaskSum, workspace.mask_sum);
        gradient.read(Sharp, workspace.sharp);
        gradient.read(Vu, workspace.vu);
        gradient.write(Grad, workspace.grad);
        if (workspace.grad_raw.is_valid())
            gradient.write(GradRaw, workspace.grad_raw);
        run(masked ? "blur_gradient_masked" : "blur_gradient", gradient, grid, tile);
    }

} // namespace lfs::training
