/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/popspa_portable.hpp"

#include "core/assert.hpp"
#include "core/error.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "popspa_program.hpp"

#include <array>
#include <format>
#include <limits>

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::Device;
        using core::GpuKernelModule;
        using gpu_ops::Tensor;

        // Parameters in popspa.slang.
        struct Parameters {
            uint64_t scores = 0, a = 0, b = 0, c = 0, active = 0, frozen = 0, keep = 0, out0 = 0, out1 = 0, out2 = 0;
            uint32_t count = 0, flag = 0;
            float rho = 0.f, inv_count = 0.f, rank_weight = 0.f, thin_weight = 0.f, epsilon = 0.f, unused = 0.f;
        };
        static_assert(sizeof(Parameters) == 112 && offsetof(Parameters, count) == 80);

        struct Slots {
            const Tensor *scores = nullptr, *a = nullptr, *b = nullptr, *c = nullptr;
            const Tensor *active = nullptr, *frozen = nullptr, *keep = nullptr;
            const Tensor *out0 = nullptr, *out1 = nullptr, *out2 = nullptr;
        };

        const Tensor* optional(const Tensor& t) {
            return t.is_valid() && t.numel() > 0 ? &t : nullptr;
        }

        void run(const char* function, const Tensor& anchor, Parameters p, const Slots& s) {
            if (p.count == 0)
                return;
            const auto backend = core::gpu_backend_of(anchor).value_or(core::default_gpu_backend());
            const core::GpuBackendScope scope(backend);
            auto loaded = GpuKernelModule::load(popspa_program_entries(), backend);
            if (!loaded)
                throw lfs::Exception(std::move(loaded).error());
            using Binding = GpuKernelModule::Binding;
            using Access = GpuKernelModule::Access;
            const std::array bindings{Binding{0, s.scores, Access::ReadWrite}, Binding{8, s.a}, Binding{16, s.b},
                                      Binding{24, s.c}, Binding{32, s.active}, Binding{40, s.frozen},
                                      Binding{48, s.keep}, Binding{56, s.out0, Access::ReadWrite},
                                      Binding{64, s.out1, Access::ReadWrite}, Binding{72, s.out2, Access::ReadWrite}};
            auto result = (*loaded)->dispatch({.function = function,
                                               .arguments = {std::as_bytes(std::span(&p, 1)), bindings},
                                               .groups = {(p.count + 255) / 256, 1, 1},
                                               .group = {256, 1, 1}});
            if (!result)
                throw lfs::Exception(std::move(result).error());
        }

        uint32_t rows32(size_t n) {
            LFS_ASSERT_MSG(n <= std::numeric_limits<uint32_t>::max(),
                           std::format("POPSpa row count exceeds 32-bit indexing (rows={})", n));
            return static_cast<uint32_t>(n);
        }

        void require_rows(const Tensor& t, size_t n, DataType dtype, const char* name) {
            LFS_ASSERT_MSG(t.is_valid() && t.numel() == n && t.dtype() == dtype && t.is_contiguous() && t.device() == Device::GPU,
                           std::format("POPSpa {} must be contiguous GPU {} with {} values (shape={}, dtype={})", name,
                                       core::dtype_name(dtype), n, t.is_valid() ? t.shape().str() : "none",
                                       t.is_valid() ? core::dtype_name(t.dtype()) : "invalid"));
        }

        // Splits candidates at the need-th largest key: rows above it, rows equal to it,
        // and how many of the equal rows still fit. need is a GPU float scalar, so
        // chained levels never read back to the host.
        struct Cut {
            Tensor above, tied, room;
        };
        Cut cut(const Tensor& key, const Tensor& candidate, const Tensor& need) {
            const auto sorted = key.masked_fill(candidate.logical_not(), -std::numeric_limits<float>::infinity())
                                    .sort(0, /*descending=*/true)
                                    .first;
            const auto threshold = sorted.index_select(0, (need - 1.0f).clamp_min(0.0f).to(DataType::Int64).reshape({1}));
            auto above = candidate.logical_and(key.gt(threshold));
            auto room = need - above.to(DataType::Float32).sum();
            return {std::move(above), candidate.logical_and(key.eq(threshold)), std::move(room)};
        }

        // Frozen rows, then the best other active rows by key, or by (key, tie_key)
        // lexicographically; remaining ties prefer lower rows.
        Tensor keep_mask(const Tensor& key, const Tensor& tie_key, const Tensor& active, const Tensor& frozen, size_t k) {
            // Float32 tie ranks stay exact below 2^24 rows.
            LFS_ASSERT_MSG(key.numel() <= (size_t{1} << 24), std::format("POPSpa selection supports up to 2^24 rows (rows={})", key.numel()));
            const auto need = Tensor::full({1}, static_cast<float>(k), Device::GPU) - frozen.to(DataType::Float32).sum();
            auto level = cut(key, active.logical_and(frozen.logical_not()), need);
            auto keep = frozen.logical_or(level.above);
            if (tie_key.is_valid()) {
                level = cut(tie_key, level.tied, level.room);
                keep = keep.logical_or(level.above);
            }
            return keep.logical_or(level.tied.logical_and(level.tied.to(DataType::Float32).cumsum(0).le(level.room)));
        }

        void keep_indices(const Tensor& scores, const Tensor& active, const Tensor& frozen, size_t k, Tensor& keep) {
            const size_t n = scores.is_valid() ? scores.shape()[0] : 0;
            LFS_ASSERT_MSG(scores.is_valid() && scores.ndim() == 2 && scores.shape()[1] == 8 && scores.dtype() == DataType::UInt8 &&
                               scores.is_contiguous(),
                           std::format("POPSpa scores must be contiguous UInt8 [N,8] (shape={})", scores.is_valid() ? scores.shape().str() : "none"));
            require_rows(active, n, DataType::Bool, "active mask");
            require_rows(frozen, n, DataType::Bool, "frozen mask");
            auto hi = Tensor::empty({n}, Device::GPU, DataType::Float32);
            auto lo = Tensor::empty({n}, Device::GPU, DataType::Float32);
            run("split", scores, {.count = rows32(n)}, {.scores = &scores, .out0 = &hi, .out1 = &lo});
            const auto rows = keep_mask(hi, lo, active, frozen, k).nonzero().flatten();
            LFS_ASSERT_MSG(rows.numel() == k, std::format("POPSpa selection kept {} rows, expected {}", rows.numel(), k));
            keep = rows.to(DataType::Int64);
        }

        void project(const Tensor& raw, Tensor& z, Tensor& u, const Tensor& active, const Tensor& frozen, size_t k,
                     bool update_dual) {
            const size_t n = raw.numel();
            require_rows(z, n, DataType::Float32, "proxy");
            require_rows(u, n, DataType::Float32, "dual");
            require_rows(active, n, DataType::Bool, "active mask");
            require_rows(frozen, n, DataType::Bool, "frozen mask");
            auto values = Tensor::empty({n}, Device::GPU, DataType::Float32);
            run("shifted", raw, {.count = rows32(n)}, {.a = &raw, .b = &u, .out0 = &values});
            const auto keep = keep_mask(values, {}, active, frozen, k);
            run("proxy", raw, {.count = rows32(n), .flag = update_dual ? 1u : 0u},
                {.a = &raw, .b = &values, .active = &active, .frozen = &frozen, .keep = &keep, .out0 = &z, .out1 = &u});
        }

        void penalty(const Tensor& raw, const Tensor& z, const Tensor& u, const Tensor& active, const Tensor& frozen,
                     Tensor& grad, Tensor& loss, float rho) {
            const size_t n = raw.numel();
            require_rows(grad, n, DataType::Float32, "opacity gradient");
            auto rows = Tensor::empty({n}, Device::GPU, DataType::Float32);
            run("penalty", raw, {.count = rows32(n), .rho = rho},
                {.a = &raw, .b = &z, .c = &u, .active = optional(active), .frozen = optional(frozen), .out0 = &grad, .out1 = &rows});
            loss.add_(rows.sum().reshape(loss.shape()));
        }

        void effective_rank(const Tensor& log_scales, Tensor& grad, const Tensor& active, const Tensor& frozen, Tensor& loss,
                            size_t active_count, float rank_weight, float thin_weight, float epsilon) {
            const size_t n = log_scales.is_valid() ? log_scales.shape()[0] : 0;
            if (n == 0 || active_count == 0)
                return;
            require_rows(grad, 3 * n, DataType::Float32, "scale gradient");
            auto rows = Tensor::empty({n}, Device::GPU, DataType::Float32);
            run("effective_rank", log_scales,
                {.count = rows32(n), .inv_count = 1.0f / static_cast<float>(active_count), .rank_weight = rank_weight, .thin_weight = thin_weight, .epsilon = epsilon},
                {.a = &log_scales, .active = optional(active), .frozen = optional(frozen), .out0 = &grad, .out1 = &rows});
            loss.add_(rows.sum().reshape(loss.shape()));
        }

        void export_scores(const Tensor& scores, Tensor& float64) {
            const size_t n = scores.is_valid() ? scores.shape()[0] : 0;
            LFS_ASSERT_MSG(float64.is_valid() && float64.shape() == scores.shape() && float64.dtype() == DataType::UInt8 &&
                               float64.is_contiguous() && float64.device() == Device::GPU,
                           std::format("POPSpa FP64 export needs contiguous GPU UInt8 {} storage, got {}",
                                       scores.shape().str(), float64.is_valid() ? float64.shape().str() : "none"));
            run("export_float64", scores, {.count = rows32(n)}, {.scores = &scores, .out2 = &float64});
        }

        const gpu_ops::PopspaOps kPortablePopspaOps{
            .score_encoding = gpu_ops::PopScoreEncoding::Float32Pair,
            .keep_indices = keep_indices,
            .project = project,
            .penalty = penalty,
            .effective_rank = effective_rank,
            .export_scores = export_scores,
        };
    } // namespace

    const gpu_ops::PopspaOps& portable_popspa_ops() {
        return kPortablePopspaOps;
    }

    void accumulate_pop_view_scores(Tensor& scores, const Tensor& view) {
        const size_t n = scores.is_valid() ? scores.shape()[0] : 0;
        LFS_ASSERT_MSG(scores.is_valid() && scores.ndim() == 2 && scores.shape()[1] == 8 && scores.dtype() == DataType::UInt8 &&
                           scores.is_contiguous() && scores.device() == Device::GPU,
                       std::format("POPSpa scores must be contiguous GPU UInt8 [N,8] (shape={}, dtype={})",
                                   scores.is_valid() ? scores.shape().str() : "none",
                                   scores.is_valid() ? core::dtype_name(scores.dtype()) : "invalid"));
        require_rows(view, n, DataType::Float32, "view scores");
        run("fold", scores, {.count = rows32(n)}, {.scores = &scores, .a = &view});
    }
} // namespace lfs::training
