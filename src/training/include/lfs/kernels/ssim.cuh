/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/tensor.hpp"
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace lfs::training::kernels {

    inline constexpr float SSIM_EPSILON = 1e-8f;

    // The five loss-workspace variants are mutually exclusive per step. One arena
    // holds the active variant exactly on mode switches and keeps a measured
    // same-variant shape high-water until a named shrink boundary.
    class LossWorkspaceArena;

    // Pre-allocated workspace for SSIM computation
    struct SSIMWorkspace {
        // Forward pass buffers
        lfs::core::Tensor ssim_map; // [N, C, H, W] fp32
        // dm_* partials stored fp16 (same proven path as fused L1+SSIM).
        lfs::core::Tensor dm_dmu1;       // [N, C, H, W] fp16
        lfs::core::Tensor dm_dsigma1_sq; // [N, C, H, W] fp16
        lfs::core::Tensor dm_dsigma12;   // [N, C, H, W] fp16

        // Backward pass buffers
        lfs::core::Tensor dL_dmap;  // [N, C, H, W]
        lfs::core::Tensor dL_dimg1; // [N, C, H, W]

        // Tiny reduction buffers for valid-padding mean computation
        lfs::core::Tensor reduction_temp;   // [<=1024]
        lfs::core::Tensor reduction_result; // [1]

        // Track allocated size
        std::vector<size_t> allocated_shape;

        // Non-owning back-pointer when views are carved from a LossWorkspaceArena.
        LossWorkspaceArena* arena = nullptr;

        // Resize workspace if needed (only reallocates if shape changed)
        void ensure_size(const std::vector<size_t>& shape);
    };

    // Context for manual SSIM forward/backward (like RasterizeContext)
    struct SSIMContext {
        lfs::core::Tensor img1;
        lfs::core::Tensor img2;
        lfs::core::Tensor dm_dmu1;
        lfs::core::Tensor dm_dsigma1_sq;
        lfs::core::Tensor dm_dsigma12;
        int original_h;
        int original_w;
        bool apply_valid_padding;
    };

    std::pair<lfs::core::Tensor, SSIMContext> ssim_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        bool apply_valid_padding = true);

    // Version with pre-allocated workspace
    std::pair<lfs::core::Tensor, SSIMContext> ssim_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        SSIMWorkspace& workspace,
        bool apply_valid_padding = true);

    // Per-pixel SSIM map result for masked loss computation
    struct SSIMMapResult {
        lfs::core::Tensor ssim_map;   // [N, C, H, W]
        lfs::core::Tensor ssim_value; // Mean SSIM scalar
        SSIMContext ctx;
    };

    // Lightweight workspace when only the per-pixel SSIM map is needed.
    struct SSIMMapWorkspace {
        lfs::core::Tensor ssim_map; // [N, C, H, W]
        std::vector<size_t> allocated_shape;

        void ensure_size(const std::vector<size_t>& shape) {
            if (allocated_shape != shape) {
                ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
                allocated_shape = shape;
            }
        }
    };

    // Returns per-pixel SSIM map (same padding when apply_valid_padding=false)
    SSIMMapResult ssim_forward_map(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        bool apply_valid_padding = false);

    // Computes a full-resolution error map [H, W] from SSIM without allocating backward buffers.
    void ssim_error_map_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        SSIMMapWorkspace& workspace,
        lfs::core::Tensor& error_map);

    // Manual SSIM backward (no autograd) - computes gradient w.r.t. img1
    lfs::core::Tensor ssim_backward(
        const SSIMContext& ctx,
        float grad_loss); // Gradient of loss w.r.t. SSIM value (scalar)

    // Optimized version with pre-allocated workspace
    lfs::core::Tensor ssim_backward(
        const SSIMContext& ctx,
        SSIMWorkspace& workspace,
        float grad_loss);

    // Per-pixel gradient version for masked SSIM (d(loss)/d(ssim_map) per pixel)
    lfs::core::Tensor ssim_backward_with_grad_map(
        const SSIMContext& ctx,
        const lfs::core::Tensor& dL_dmap); // [N, C, H, W] per-pixel gradient

    // ============================================================================
    // Fused L1+SSIM Loss
    // ============================================================================

    // Workspace for fused L1+SSIM (extends SSIMWorkspace)
    struct FusedL1SSIMWorkspace {
        lfs::core::Tensor ssim_map; // [N, 1, H, W] per-pixel channel-mean SSIM values
        // SSIM partial derivatives, written by forward, read by backward only. Stored
        // fp16: C1/C2 floor A and B so these stay bounded (no overflow), and the ~0.1%
        // quantization is negligible at the typical 0.2 D-SSIM weight. Halves the bulk
        // of the loss workspace at high resolution.
        lfs::core::Tensor dm_dmu1;       // [N, C, H, W] fp16
        lfs::core::Tensor dm_dsigma1_sq; // [N, C, H, W] fp16
        lfs::core::Tensor dm_dsigma12;   // [N, C, H, W] fp16

        // Backward pass buffer (final gradient stays fp32)
        lfs::core::Tensor grad_img; // [N, C, H, W] combined gradient

        // Tiny reduction buffers for valid-padding mean computation
        lfs::core::Tensor reduction_temp;   // [<=1024]
        lfs::core::Tensor reduction_result; // [1]

        // Track allocated size
        std::vector<size_t> allocated_shape;
        LossWorkspaceArena* arena = nullptr;

        void ensure_size(const std::vector<size_t>& shape);
    };

    // Context for fused L1+SSIM backward pass
    struct FusedL1SSIMContext {
        lfs::core::Tensor img1;
        lfs::core::Tensor img2;
        lfs::core::Tensor dm_dmu1;
        lfs::core::Tensor dm_dsigma1_sq;
        lfs::core::Tensor dm_dsigma12;
        float ssim_weight;
        int H, W;
        bool apply_valid_padding;
    };

    // Fused L1+SSIM forward: loss = (1-w)*L1 + w*(1-SSIM)
    std::pair<lfs::core::Tensor, FusedL1SSIMContext> fused_l1_ssim_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        float ssim_weight,
        FusedL1SSIMWorkspace& workspace,
        bool apply_valid_padding = true);

    // Fused L1+SSIM backward
    lfs::core::Tensor fused_l1_ssim_backward(
        const FusedL1SSIMContext& ctx,
        FusedL1SSIMWorkspace& workspace);

    // ============================================================================
    // Decoupled L1+SSIM Loss for appearance modeling
    // ============================================================================

    struct DecoupledGradients {
        lfs::core::Tensor grad_corrected; // Gradient through the appearance-corrected image
        lfs::core::Tensor grad_raw;       // Direct gradient to the raw render (contrast/structure only)
    };

    struct DecoupledFusedL1SSIMWorkspace {
        lfs::core::Tensor ssim_map; // [N, 1, H, W] channel-mean decoupled SSIM map
        // all dm_* partials fp16 (mirror fused path).
        lfs::core::Tensor app_dm_dmu1;       // [N, C, H, W] fp16 d(ssim_map)/d mu(corrected)
        lfs::core::Tensor raw_dm_dmu1;       // [N, C, H, W] fp16 indirect mu(raw) via sigma terms
        lfs::core::Tensor raw_dm_dsigma1_sq; // [N, C, H, W] fp16 lambda-scaled d(ssim)/d sigma^2(raw)
        lfs::core::Tensor raw_dm_dsigma12;   // [N, C, H, W] fp16 lambda-scaled d(ssim)/d sigma12(raw)
        lfs::core::Tensor grad_corrected;    // [N, C, H, W] fp32
        lfs::core::Tensor grad_raw;          // [N, C, H, W] fp32
        lfs::core::Tensor reduction_temp;    // [<=1024]
        lfs::core::Tensor reduction_result;  // [1]

        std::vector<size_t> allocated_shape;
        LossWorkspaceArena* arena = nullptr;

        void ensure_size(const std::vector<size_t>& shape);
    };

    struct DecoupledFusedL1SSIMContext {
        lfs::core::Tensor corrected_img;
        lfs::core::Tensor raw_img;
        lfs::core::Tensor gt_img;
        lfs::core::Tensor app_dm_dmu1;
        lfs::core::Tensor raw_dm_dmu1;
        lfs::core::Tensor raw_dm_dsigma1_sq;
        lfs::core::Tensor raw_dm_dsigma12;
        float ssim_weight;
        int H, W;
        bool apply_valid_padding;
    };

    std::pair<lfs::core::Tensor, DecoupledFusedL1SSIMContext> decoupled_fused_l1_ssim_forward(
        const lfs::core::Tensor& corrected_img,
        const lfs::core::Tensor& raw_img,
        const lfs::core::Tensor& gt_img,
        float ssim_weight,
        DecoupledFusedL1SSIMWorkspace& workspace,
        bool apply_valid_padding = true);

    DecoupledGradients decoupled_fused_l1_ssim_backward(
        const DecoupledFusedL1SSIMContext& ctx,
        DecoupledFusedL1SSIMWorkspace& workspace);

    // ============================================================================
    // Fused Masked L1+SSIM Loss (for segmentation/ignore mask modes)
    // ============================================================================

    struct MaskedFusedL1SSIMWorkspace {
        lfs::core::Tensor ssim_map; // [N, 1, H, W] per-pixel channel-mean SSIM values
        // dm_* partials fp16.
        lfs::core::Tensor dm_dmu1;        // [N, C, H, W] fp16
        lfs::core::Tensor dm_dsigma1_sq;  // [N, C, H, W] fp16
        lfs::core::Tensor dm_dsigma12;    // [N, C, H, W] fp16
        lfs::core::Tensor grad_img;       // [N, C, H, W] fp32
        lfs::core::Tensor reduction_temp; // [<=2048], split into loss and mask partial sums
        lfs::core::Tensor masked_loss;    // [1] scalar
        lfs::core::Tensor mask_sum;       // [1] scalar

        std::vector<size_t> allocated_shape;
        LossWorkspaceArena* arena = nullptr;

        void ensure_size(const std::vector<size_t>& shape);
    };

    struct MaskedFusedL1SSIMContext {
        lfs::core::Tensor img1;
        lfs::core::Tensor img2;
        lfs::core::Tensor mask;
        lfs::core::Tensor dm_dmu1;
        lfs::core::Tensor dm_dsigma1_sq;
        lfs::core::Tensor dm_dsigma12;
        float ssim_weight;
        float mask_sum_value;
        int H, W;
    };

    // Fused masked L1+SSIM forward
    std::pair<lfs::core::Tensor, MaskedFusedL1SSIMContext> masked_fused_l1_ssim_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        const lfs::core::Tensor& mask,
        float ssim_weight,
        MaskedFusedL1SSIMWorkspace& workspace);

    // Fused masked L1+SSIM backward
    lfs::core::Tensor masked_fused_l1_ssim_backward(
        const MaskedFusedL1SSIMContext& ctx,
        MaskedFusedL1SSIMWorkspace& workspace);

    struct MaskedDecoupledFusedL1SSIMWorkspace {
        lfs::core::Tensor ssim_map; // [N, 1, H, W]
        // dm_* partials fp16.
        lfs::core::Tensor app_dm_dmu1;       // [N, C, H, W] fp16
        lfs::core::Tensor raw_dm_dmu1;       // [N, C, H, W] fp16
        lfs::core::Tensor raw_dm_dsigma1_sq; // [N, C, H, W] fp16
        lfs::core::Tensor raw_dm_dsigma12;   // [N, C, H, W] fp16
        lfs::core::Tensor grad_corrected;    // [N, C, H, W] fp32
        lfs::core::Tensor grad_raw;          // [N, C, H, W] fp32
        lfs::core::Tensor reduction_temp;    // [<=2048]
        lfs::core::Tensor masked_loss;       // [1]
        lfs::core::Tensor mask_sum;          // [1]

        std::vector<size_t> allocated_shape;
        LossWorkspaceArena* arena = nullptr;

        void ensure_size(const std::vector<size_t>& shape);
    };

    // =========================================================================
    // LossWorkspaceArena — union storage for exclusive photometric workspaces
    // =========================================================================
    class LossWorkspaceArena {
    public:
        enum class Kind : uint8_t {
            None = 0,
            Fused,
            PureSSIM,
            Decoupled,
            MaskedFused,
            MaskedDecoupled
        };

        LossWorkspaceArena() {
            // Point member workspaces at this arena so ensure_size() rebinds views
            // instead of independently allocating (production path).
            fused_.arena = this;
            pure_ssim_.arena = this;
            decoupled_.arena = this;
            masked_fused_.arena = this;
            masked_decoupled_.arena = this;
        }

        // Non-copyable: workspace arena pointers would dangle after a copy.
        LossWorkspaceArena(const LossWorkspaceArena&) = delete;
        LossWorkspaceArena& operator=(const LossWorkspaceArena&) = delete;
        LossWorkspaceArena(LossWorkspaceArena&& other) noexcept { *this = std::move(other); }
        LossWorkspaceArena& operator=(LossWorkspaceArena&& other) noexcept {
            if (this != &other) {
                storage_ = std::move(other.storage_);
                capacity_bytes_ = other.capacity_bytes_;
                active_kind_ = other.active_kind_;
                active_shape_ = std::move(other.active_shape_);
                fused_ = std::move(other.fused_);
                pure_ssim_ = std::move(other.pure_ssim_);
                decoupled_ = std::move(other.decoupled_);
                masked_fused_ = std::move(other.masked_fused_);
                masked_decoupled_ = std::move(other.masked_decoupled_);
                fused_.arena = this;
                pure_ssim_.arena = this;
                decoupled_.arena = this;
                masked_fused_.arena = this;
                masked_decoupled_.arena = this;
                other.capacity_bytes_ = 0;
                other.active_kind_ = Kind::None;
                other.active_shape_.clear();
                other.fused_.arena = nullptr;
                other.pure_ssim_.arena = nullptr;
                other.decoupled_.arena = nullptr;
                other.masked_fused_.arena = nullptr;
                other.masked_decoupled_.arena = nullptr;
            }
            return *this;
        }

        [[nodiscard]] size_t allocated_bytes() const { return capacity_bytes_; }
        [[nodiscard]] size_t required_bytes() const;
        [[nodiscard]] Kind active_kind() const { return active_kind_; }
        [[nodiscard]] const std::vector<size_t>& active_shape() const { return active_shape_; }

        // Layout oracles (match independent ensure_size field lists).
        [[nodiscard]] static size_t fused_layout_bytes(const std::vector<size_t>& shape);
        [[nodiscard]] static size_t pure_ssim_layout_bytes(const std::vector<size_t>& shape);
        [[nodiscard]] static size_t decoupled_layout_bytes(const std::vector<size_t>& shape);
        [[nodiscard]] static size_t masked_fused_layout_bytes(const std::vector<size_t>& shape);
        [[nodiscard]] static size_t masked_decoupled_layout_bytes(const std::vector<size_t>& shape);
        [[nodiscard]] static size_t max_variant_bytes(const std::vector<size_t>& shape);

        FusedL1SSIMWorkspace& ensure_fused(const std::vector<size_t>& shape);
        SSIMWorkspace& ensure_pure_ssim(const std::vector<size_t>& shape);
        DecoupledFusedL1SSIMWorkspace& ensure_decoupled(const std::vector<size_t>& shape);
        MaskedFusedL1SSIMWorkspace& ensure_masked_fused(const std::vector<size_t>& shape);
        MaskedDecoupledFusedL1SSIMWorkspace& ensure_masked_decoupled(const std::vector<size_t>& shape);

        [[nodiscard]] FusedL1SSIMWorkspace& fused() { return fused_; }
        [[nodiscard]] const FusedL1SSIMWorkspace& fused() const { return fused_; }
        [[nodiscard]] SSIMWorkspace& pure_ssim() { return pure_ssim_; }
        [[nodiscard]] const SSIMWorkspace& pure_ssim() const { return pure_ssim_; }
        [[nodiscard]] DecoupledFusedL1SSIMWorkspace& decoupled() { return decoupled_; }
        [[nodiscard]] const DecoupledFusedL1SSIMWorkspace& decoupled() const { return decoupled_; }
        [[nodiscard]] MaskedFusedL1SSIMWorkspace& masked_fused() { return masked_fused_; }
        [[nodiscard]] const MaskedFusedL1SSIMWorkspace& masked_fused() const { return masked_fused_; }
        [[nodiscard]] MaskedDecoupledFusedL1SSIMWorkspace& masked_decoupled() { return masked_decoupled_; }
        [[nodiscard]] const MaskedDecoupledFusedL1SSIMWorkspace& masked_decoupled() const {
            return masked_decoupled_;
        }

        // EXACT-3 boundary: resize the active variant to its current requirement,
        // or release the arena when no variant is active.
        void shrink_to_required();
        void reset();

    private:
        friend struct SSIMWorkspace;
        friend struct FusedL1SSIMWorkspace;
        friend struct DecoupledFusedL1SSIMWorkspace;
        friend struct MaskedFusedL1SSIMWorkspace;
        friend struct MaskedDecoupledFusedL1SSIMWorkspace;

        void ensure_capacity_for(Kind kind, size_t bytes);
        void replace_storage_exact(size_t bytes);
        void clear_all_views();
        [[nodiscard]] lfs::core::Tensor make_view(size_t& offset,
                                                  const std::vector<size_t>& shape,
                                                  lfs::core::DataType dtype);
        void bind_fused(const std::vector<size_t>& shape);
        void bind_pure_ssim(const std::vector<size_t>& shape);
        void bind_decoupled(const std::vector<size_t>& shape);
        void bind_masked_fused(const std::vector<size_t>& shape);
        void bind_masked_decoupled(const std::vector<size_t>& shape);

        // UInt8 blob: exact on variant switches, same-variant shape high-water.
        lfs::core::Tensor storage_;
        size_t capacity_bytes_ = 0;
        Kind active_kind_ = Kind::None;
        std::vector<size_t> active_shape_;

        FusedL1SSIMWorkspace fused_;
        SSIMWorkspace pure_ssim_;
        DecoupledFusedL1SSIMWorkspace decoupled_;
        MaskedFusedL1SSIMWorkspace masked_fused_;
        MaskedDecoupledFusedL1SSIMWorkspace masked_decoupled_;
    };

    struct MaskedDecoupledFusedL1SSIMContext {
        lfs::core::Tensor corrected_img;
        lfs::core::Tensor raw_img;
        lfs::core::Tensor gt_img;
        lfs::core::Tensor mask;
        lfs::core::Tensor app_dm_dmu1;
        lfs::core::Tensor raw_dm_dmu1;
        lfs::core::Tensor raw_dm_dsigma1_sq;
        lfs::core::Tensor raw_dm_dsigma12;
        float ssim_weight;
        float mask_sum_value;
        int H, W;
    };

    std::pair<lfs::core::Tensor, MaskedDecoupledFusedL1SSIMContext> masked_decoupled_fused_l1_ssim_forward(
        const lfs::core::Tensor& corrected_img,
        const lfs::core::Tensor& raw_img,
        const lfs::core::Tensor& gt_img,
        const lfs::core::Tensor& mask,
        float ssim_weight,
        MaskedDecoupledFusedL1SSIMWorkspace& workspace);

    DecoupledGradients masked_decoupled_fused_l1_ssim_backward(
        const MaskedDecoupledFusedL1SSIMContext& ctx,
        MaskedDecoupledFusedL1SSIMWorkspace& workspace);

    // Fused SSIM map → error map: error_map[i] = max(0, 1 - mean_c(ssim_map[c, i]))
    // Replaces .neg().add(1).mean({1}).squeeze(0).clamp_min(0).contiguous() chain
    void launch_ssim_to_error_map(
        const lfs::core::Tensor& ssim_map,
        lfs::core::Tensor& error_map);

} // namespace lfs::training::kernels
