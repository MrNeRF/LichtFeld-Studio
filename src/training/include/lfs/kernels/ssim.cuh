/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/tensor.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
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
        lfs::core::Tensor cs_map;   // [N, C, H, W] contrast×structure (independent of arena)
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
        lfs::core::Tensor cs_map;     // [N, C, H, W] contrast×structure
        lfs::core::Tensor ssim_value; // Mean SSIM scalar
        SSIMContext ctx;
    };

    // Lightweight workspace when only the per-pixel SSIM map is needed.
    struct SSIMMapWorkspace {
        lfs::core::Tensor ssim_map; // [N, C, H, W]
        lfs::core::Tensor cs_map;   // [N, C, H, W] contrast×structure
        std::vector<size_t> allocated_shape;

        void ensure_size(const std::vector<size_t>& shape) {
            if (allocated_shape != shape) {
                ssim_map = lfs::core::Tensor::empty(lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
                cs_map = lfs::core::Tensor::empty(lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
                allocated_shape = shape;
            }
        }
    };

    // Returns per-pixel SSIM map (same padding when apply_valid_padding=false)
    SSIMMapResult ssim_forward_map(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        bool apply_valid_padding = false);

    // Evaluation-only normalized masked windows, [N,C,H,W], without backward buffers.
    lfs::core::Tensor masked_ssim_map_forward(
        const lfs::core::Tensor& pred, const lfs::core::Tensor& target,
        const lfs::core::Tensor& mask);

    // Computes a full-resolution error map [H, W] from SSIM without allocating backward buffers.
    void ssim_error_map_forward(
        const lfs::core::Tensor& img1,
        const lfs::core::Tensor& img2,
        SSIMMapWorkspace& workspace,
        lfs::core::Tensor& error_map,
        bool contrast_structure_only = false);

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
        lfs::core::Tensor cs_map;   // [N, 1, H, W] channel-mean contrast×structure
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
    };

    // The decoupled loss's contrast/structure gradient with respect to the raw render. It is added straight into
    // the render gradient after the appearance backward, so it needs no buffer of its own. The views stay valid
    // until the loss workspace is used again.
    struct DecoupledRawGradient {
        lfs::core::Tensor raw_img;
        lfs::core::Tensor gt_img;
        lfs::core::Tensor mask; // Empty for the unmasked loss
        lfs::core::Tensor raw_dm_dmu1;
        lfs::core::Tensor raw_dm_dsigma1_sq;
        lfs::core::Tensor raw_dm_dsigma12;
        float mask_sum_value = 0.0f;
        int H = 0;
        int W = 0;
        bool apply_valid_padding = true;
        // When valid, the gradient itself, added as is (the blur-tolerant SSIM forms it in its own pass).
        lfs::core::Tensor precomputed;
    };

    struct DecoupledFusedL1SSIMWorkspace {
        lfs::core::Tensor ssim_map; // [N, 1, H, W] channel-mean decoupled SSIM map
        lfs::core::Tensor cs_map;   // [N, 1, H, W] channel-mean CS on the raw render
        // all dm_* partials fp16 (mirror fused path).
        lfs::core::Tensor app_dm_dmu1;       // [N, C, H, W] fp16 d(ssim_map)/d mu(corrected)
        lfs::core::Tensor raw_dm_dmu1;       // [N, C, H, W] fp16 indirect mu(raw) via sigma terms
        lfs::core::Tensor raw_dm_dsigma1_sq; // [N, C, H, W] fp16 lambda-scaled d(ssim)/d sigma^2(raw)
        lfs::core::Tensor raw_dm_dsigma12;   // [N, C, H, W] fp16 lambda-scaled d(ssim)/d sigma12(raw)
        lfs::core::Tensor grad_corrected;    // [N, C, H, W] fp32
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
        lfs::core::Tensor cs_map;   // [N, 1, H, W] channel-mean contrast×structure
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
        MaskedFusedL1SSIMWorkspace& workspace, float denominator = 0.0f);

    // Fused masked L1+SSIM backward
    lfs::core::Tensor masked_fused_l1_ssim_backward(
        const MaskedFusedL1SSIMContext& ctx,
        MaskedFusedL1SSIMWorkspace& workspace);

    // Forward-only masked L1+SSIM for reporting a loss over part of an image, such as the
    // crop box region: the value masked_fused_l1_ssim_forward, or with `raw`
    // masked_decoupled_fused_l1_ssim_forward, returns for the same inputs. It keeps no
    // gradient state and owns its buffers, so it never aliases the training loss arena;
    // the result stays on the device.
    struct MaskedL1SSIMValueWorkspace {
        lfs::core::Tensor ssim_map;       // [N, 1, H, W]
        lfs::core::Tensor reduction_temp; // [<=2048], split into loss and mask partial sums
        lfs::core::Tensor loss;           // [1] weighted mean
        lfs::core::Tensor mask_sum;       // [1] normalization, below 1 when no pixel counts
    };

    void masked_fused_l1_ssim_value(
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& raw,
        const lfs::core::Tensor& target,
        const lfs::core::Tensor& mask,
        float ssim_weight,
        MaskedL1SSIMValueWorkspace& workspace,
        float denominator = 0.0f);

    struct MaskedDecoupledFusedL1SSIMWorkspace {
        lfs::core::Tensor ssim_map; // [N, 1, H, W]
        lfs::core::Tensor cs_map;   // [N, 1, H, W] channel-mean CS on the raw render
        // dm_* partials fp16.
        lfs::core::Tensor app_dm_dmu1;       // [N, C, H, W] fp16
        lfs::core::Tensor raw_dm_dmu1;       // [N, C, H, W] fp16
        lfs::core::Tensor raw_dm_dsigma1_sq; // [N, C, H, W] fp16
        lfs::core::Tensor raw_dm_dsigma12;   // [N, C, H, W] fp16
        lfs::core::Tensor grad_corrected;    // [N, C, H, W] fp32
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
                active_step_partials_ = other.active_step_partials_;
                step_scratch_ = std::move(other.step_scratch_);
                step_fallback_ = std::move(other.step_fallback_);
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
                other.active_step_partials_ = false;
                other.step_scratch_ = {};
                other.fused_.arena = nullptr;
                other.pure_ssim_.arena = nullptr;
                other.decoupled_.arena = nullptr;
                other.masked_fused_.arena = nullptr;
                other.masked_decoupled_.arena = nullptr;
            }
            return *this;
        }

        [[nodiscard]] size_t allocated_bytes() const { return capacity_bytes_ + step_fallback_.numel(); }
        [[nodiscard]] size_t required_bytes() const;
        [[nodiscard]] Kind active_kind() const { return active_kind_; }
        [[nodiscard]] const std::vector<size_t>& active_shape() const { return active_shape_; }

        // Layout oracles (match independent ensure_size field lists). With
        // step_partials the fp16 dm_* partials are left to the step allocator.
        [[nodiscard]] static size_t fused_layout_bytes(const std::vector<size_t>& shape, bool step_partials = false);
        [[nodiscard]] static size_t pure_ssim_layout_bytes(const std::vector<size_t>& shape, bool step_partials = false);
        [[nodiscard]] static size_t decoupled_layout_bytes(const std::vector<size_t>& shape, bool step_partials = false);
        [[nodiscard]] static size_t masked_fused_layout_bytes(const std::vector<size_t>& shape,
                                                              bool step_partials = false);
        [[nodiscard]] static size_t masked_decoupled_layout_bytes(const std::vector<size_t>& shape,
                                                                  bool step_partials = false);
        [[nodiscard]] static size_t max_variant_bytes(const std::vector<size_t>& shape);

        // The fp16 dm_* partials are read only until the loss gradients are formed, including the
        // raw-render part. While a step allocator is bound, each step carves them from it and the
        // arena keeps only the outputs (SSIM map, image gradient, scalars); an allocator returning
        // null throws MemoryAllocationError. Releasing drops every view; the next ensure_* rebinds.
        void bind_step_scratch(const std::function<char*(size_t)>& allocator);
        void release_step_scratch();
        // Device bytes a loss reads only until its gradients are formed: from the bound step allocator, else from
        // a grow-only block beside the variants, which shrink_to_required() and reset() release. Each call while
        // an allocator is bound takes new bytes; without one, every call returns the same block.
        [[nodiscard]] char* take_step_scratch(size_t bytes, const char* operation);

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
        [[nodiscard]] lfs::core::Tensor partial_view(size_t& offset, const std::vector<size_t>& shape);
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
        bool active_step_partials_ = false;
        std::function<char*(size_t)> step_scratch_;
        lfs::core::Tensor step_fallback_; // UInt8, take_step_scratch without a step allocator

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
        MaskedDecoupledFusedL1SSIMWorkspace& workspace, float denominator = 0.0f);

    DecoupledGradients masked_decoupled_fused_l1_ssim_backward(
        const MaskedDecoupledFusedL1SSIMContext& ctx,
        MaskedDecoupledFusedL1SSIMWorkspace& workspace);

    [[nodiscard]] DecoupledRawGradient decoupled_raw_gradient(const DecoupledFusedL1SSIMContext& ctx);
    [[nodiscard]] DecoupledRawGradient decoupled_raw_gradient(const MaskedDecoupledFusedL1SSIMContext& ctx);
    // grad_image += the raw-render gradient; grad_image has the raw render's element count.
    void accumulate_decoupled_raw_gradient(const DecoupledRawGradient& raw, lfs::core::Tensor& grad_image);

    // ============================================================================
    // Blur-tolerant SSIM (MRNF)
    // ============================================================================
    //
    // Per channel and window two SSIM branches share the photo's statistics: S0 = ssim(x, y) and
    // S1 = ssim(G x, y), with G a normalised Gaussian blur of the render only, over the kept pixels
    // (inside the image and, with a weight map, weighted). The loss takes their soft maximum,
    // S = max(S0, S1) + (log(1 + exp(-T |S0 - S1|)) - log 2) / T, so a render sharper than its photo
    // is not pulled towards the photo's blur: thin structure (a branch, a wire, an edge) that lands a
    // pixel off in a photo through small pose or calibration errors keeps its sharpness, while a
    // render blurrier than its photo still meets plain SSIM. Photo windows flatter than
    // BLUR_TOLERANT_SSIM_FLAT_VARIANCE keep plain SSIM, where the blurred branch could only smooth
    // noise. The blur grows with the photo: sigma = 1.5 x short side / 3000, clamped to [0.5, 1.5];
    // photos with a short side below 1200 pixels carry too little blur for it and train with plain
    // SSIM (sigma 0).

    inline constexpr float BLUR_TOLERANT_SSIM_TEMPERATURE = 50.0f;
    inline constexpr float BLUR_TOLERANT_SSIM_FLAT_VARIANCE = 0.0005f;
    inline constexpr int BLUR_TOLERANT_SSIM_MIN_SHORT_SIDE = 1200;
    inline constexpr float BLUR_TOLERANT_SSIM_REFERENCE_SHORT_SIDE = 3000.0f;
    inline constexpr float BLUR_TOLERANT_SSIM_MIN_SIGMA = 0.5f;
    inline constexpr float BLUR_TOLERANT_SSIM_MAX_SIGMA = 1.5f;

    // The blur sigma of a photo of this size; 0 means plain SSIM.
    [[nodiscard]] inline float blur_tolerant_ssim_sigma(const int height, const int width) {
        const int short_side = std::min(height, width);
        if (short_side < BLUR_TOLERANT_SSIM_MIN_SHORT_SIDE)
            return 0.0f;
        return std::clamp(BLUR_TOLERANT_SSIM_MAX_SIGMA * static_cast<float>(short_side) /
                              BLUR_TOLERANT_SSIM_REFERENCE_SHORT_SIDE,
                          BLUR_TOLERANT_SSIM_MIN_SIGMA, BLUR_TOLERANT_SSIM_MAX_SIGMA);
    }

    struct BlurTolerantResult {
        lfs::core::Tensor loss;           // [1] device scalar
        lfs::core::Tensor grad_corrected; // [N, C, H, W], the arena's decoupled workspace
        // The decoupled loss's raw-render gradient, precomputed in the step scratch: read it before the
        // scratch is released. Empty for the plain loss, whose grad_corrected already holds both parts.
        std::optional<DecoupledRawGradient> raw;
        float mask_sum_value = 0.0f; // normalization of a weighted loss
    };

    // L1 + blur-tolerant SSIM of one image and its gradients, blur sigma in (0, 1.5]. The decoupled form
    // takes luminance from `corrected` and contrast-structure from `raw`; an empty `raw` is the plain
    // loss. Without a weight map the loss keeps the fused loss's valid padding; with one it is the masked
    // loss, normalized by `denominator` when positive. The arena's decoupled (or masked decoupled)
    // workspace holds the gradient and the maps the densification error reads: ssim_map the sharp
    // branch's channel-mean SSIM, cs_map the channel-mean contrast-structure term of the blend,
    // w0 cs(x, y) + w1 cs(G x, y) with w0, w1 the soft-maximum weights.
    BlurTolerantResult blur_tolerant_loss(const lfs::core::Tensor& corrected, const lfs::core::Tensor& raw,
                                          const lfs::core::Tensor& target, const lfs::core::Tensor& weight,
                                          float ssim_weight, float blur_sigma, LossWorkspaceArena& arena,
                                          float denominator = 0.0f);

    // Fused SSIM map → error map: error_map[i] = max(0, 1 - mean_c(ssim_map[c, i])), squared on request.
    // Replaces .neg().add(1).mean({1}).squeeze(0).clamp_min(0).contiguous() chain
    void launch_ssim_to_error_map(
        const lfs::core::Tensor& ssim_map,
        lfs::core::Tensor& error_map,
        bool square = false);

} // namespace lfs::training::kernels
