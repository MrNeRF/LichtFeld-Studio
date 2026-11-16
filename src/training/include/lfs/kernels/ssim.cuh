/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <unordered_map>
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
        lfs::core::Tensor precomputed; // when valid: the gradient itself, added as is (pyramid losses)
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
        MaskedDecoupledFusedL1SSIMWorkspace& workspace, float denominator = 0.0f);

    DecoupledGradients masked_decoupled_fused_l1_ssim_backward(
        const MaskedDecoupledFusedL1SSIMContext& ctx,
        MaskedDecoupledFusedL1SSIMWorkspace& workspace);

    [[nodiscard]] DecoupledRawGradient decoupled_raw_gradient(const DecoupledFusedL1SSIMContext& ctx);
    [[nodiscard]] DecoupledRawGradient decoupled_raw_gradient(const MaskedDecoupledFusedL1SSIMContext& ctx);
    // grad_image += the raw-render gradient; grad_image has the raw render's element count.
    void accumulate_decoupled_raw_gradient(const DecoupledRawGradient& raw, lfs::core::Tensor& grad_image);

    // ============================================================================
    // Blur-tolerant SSIM and multi-scale photometric loss
    // ============================================================================
    //
    // Blur-tolerant SSIM, per channel and window: S0 = ssim(x, y), S1 = ssim(G x, y) with G a
    // normalised Gaussian blur of the render only, and S the soft maximum of the two with
    // temperature 50 (S = S0 when the branches are equal). A render sharper than the photo is not
    // pulled towards the photo's blur; where the render is the blurrier one S is plain SSIM.
    //
    // The pyramid averages the photometric loss over K scales; scale s halves render, photo and
    // weight map s times by 2x2 averaging (an odd last row or column averages the pixels it has, so
    // a pixel is kept when any of its pixels is kept):
    //   loss = (1 / K) sum_s [(1 - w) mean|x_s - y_s| + w (1 - mean S_s)]
    // with S_s plain SSIM, or blur-tolerant SSIM with blur sigma 1.5 / 2^s.
    //
    // With appearance correction (decoupled loss: luminance from the corrected image, contrast and
    // structure from the raw render) the blurred branch blurs both images, and the pyramid pools both.

    inline constexpr float BLUR_TOLERANT_SSIM_TEMPERATURE = 50.0f;
    inline constexpr float BLUR_TOLERANT_SSIM_SIGMA = 1.5f;

    // K = 1 when min(H, W) < 1920, else min(4, floor(log2(min(H, W) / 1920)) + 2).
    [[nodiscard]] int loss_pyramid_levels(int height, int width);

    // Grow-only buffers of the pyramid losses; empty while they are never called. Scale 0 keeps its
    // SSIM partials and maps in the loss arena (the workspace today's loss uses), so the
    // densification error map still reads the plain full-resolution SSIM of the unblurred render.
    struct PhotometricPyramidWorkspace {
        struct Level {
            lfs::core::Tensor prediction; // [N, C, h, w] pooled render (scales >= 1)
            lfs::core::Tensor target;     // [N, C, h, w] pooled photo, Float32 (scales >= 1)
            lfs::core::Tensor mask;       // [h, w] pooled weight (scales >= 1, weighted loss)
            // Scales >= 1: the SSIM workspace of the loss variant, grown to the largest image like the
            // scale-0 arena.
            LossWorkspaceArena arena;
            // Decoupled loss
            lfs::core::Tensor raw;      // [N, C, h, w] pooled raw render (scales >= 1)
            lfs::core::Tensor grad_raw; // [N, C, h, w] raw-render gradient
            // Blur-tolerant branch
            lfs::core::Tensor blurred;         // [N, C, h, w] blurred render
            lfs::core::Tensor ssim_sharp;      // [N, C, h, w] SSIM of the render, then the blurred-branch gradient
            lfs::core::Tensor ssim_blurred;    // [N, C, h, w] SSIM of the blurred render, also blur scratch;
                                               // fused: the blurred raw-render gradient (decoupled loss)
            lfs::core::Tensor inv_support;     // [h, w] 1 / blurred keep mask, 0 where nothing is kept
            lfs::core::Tensor support_scratch; // [h, w]
            lfs::core::Tensor blend_map;       // [N, 1, h, w] channel-mean blended SSIM
            lfs::core::Tensor dm_dmu1;         // [N, C, h, w] fp16 blurred-branch partials
            lfs::core::Tensor dm_dsigma1_sq;
            lfs::core::Tensor dm_dsigma12;
            lfs::core::Tensor blurred_raw; // [N, C, h, w] blurred raw render (decoupled loss)
            lfs::core::Tensor app_dm_dmu1; // [N, C, h, w] fp16 blurred-branch corrected-image partial (decoupled)
            lfs::core::Tensor cs_error;    // [h, w] contrast-structure error of this scale (scales >= 1)
            // Device storage behind the tensors above, by field offset: the photos differ in size, and
            // the tensors are views into the largest allocation so far instead of new ones per image.
            std::unordered_map<std::ptrdiff_t, lfs::core::Tensor> storage;
        };
        std::vector<Level> levels;
        lfs::core::Tensor loss; // [1]
        // Blur-tolerant scales run as one blur (which also pools the next scale's render), one
        // two-branch forward, one two-branch backward and one blur adjoint; set, they run the per-branch
        // kernels instead (the reference the fused ones are tested against).
        bool reference_kernels = false;

        // Contrast-structure error map of the loss, filled when want_cs_error_map is set (single image):
        // per pixel 1 - mean_c(w0 cs(x, y) + w1 cs(G x, y)) with w0, w1 the soft-maximum weights of the
        // two branches (w0 = 1 without the blur-tolerant term), averaged over the scales, each coarse
        // map repeated over the pixels it pools. cs_error_map_ready tells whether the last pyramid loss
        // call wrote it.
        bool want_cs_error_map = false;
        bool cs_error_map_ready = false;
        lfs::core::Tensor cs_error_map; // [H, W]
        // With want_photo_sigma_map, the scale-0 pass of the contrast-structure map above also writes the photo's
        // local standard deviation (its SSIM window statistics, mean over the channels) to photo_sigma_map;
        // photo_sigma_map_ready tells whether the last pyramid loss call wrote it.
        bool want_photo_sigma_map = false;
        bool photo_sigma_map_ready = false;
        lfs::core::Tensor photo_sigma_map; // [H, W]
    };

    struct PhotometricPyramidResult {
        lfs::core::Tensor loss;      // [1] device scalar; aliases the workspace or the arena
        lfs::core::Tensor grad;      // [N, C, H, W] gradient w.r.t. the render (corrected image); aliases the arena
        lfs::core::Tensor grad_raw;  // [N, C, H, W] decoupled loss only: gradient w.r.t. the raw render
        float mask_sum_value = 0.0f; // scale-0 normalization of a weighted loss, 0 without weights
    };

    // L1+SSIM loss and its gradient over `levels` scales, plain or blur-tolerant SSIM. Without a
    // weight map it is the fused loss (valid padding at every scale); with one it is the masked fused
    // loss, the weight map pooled with the images and `denominator` (when positive) scaled by the
    // pixel count of each scale. levels == 1 without the blur-tolerant term is exactly the fused or
    // masked fused loss.
    PhotometricPyramidResult photometric_pyramid_loss(
        const lfs::core::Tensor& prediction,
        const lfs::core::Tensor& target,
        const lfs::core::Tensor& weight, // empty, or [H, W] / [1, H, W]
        float ssim_weight,
        int levels,
        bool blur_tolerant,
        LossWorkspaceArena& arena,
        PhotometricPyramidWorkspace& workspace,
        float denominator = 0.0f);

    // The same for the decoupled appearance loss (as decoupled_fused_l1_ssim_forward without a
    // weight map, masked_decoupled_fused_l1_ssim_forward with one). The raw-render gradient is
    // computed eagerly into grad_raw; carry it with decoupled_raw_gradient(result, raw).
    PhotometricPyramidResult decoupled_photometric_pyramid_loss(
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& raw,
        const lfs::core::Tensor& target,
        const lfs::core::Tensor& weight,
        float ssim_weight,
        int levels,
        bool blur_tolerant,
        LossWorkspaceArena& arena,
        PhotometricPyramidWorkspace& workspace,
        float denominator = 0.0f);
    [[nodiscard]] DecoupledRawGradient decoupled_raw_gradient(const PhotometricPyramidResult& result,
                                                              const lfs::core::Tensor& raw);

    // Normalised Gaussian blur over kept pixels (weight > 0, or every pixel of the image without a
    // weight map) and its adjoint, for [N, C, H, W] images. Taps: sigma, radius min(6, ceil(3 sigma)).
    [[nodiscard]] lfs::core::Tensor normalized_gaussian_blur(
        const lfs::core::Tensor& image, const lfs::core::Tensor& weight, float sigma);
    [[nodiscard]] lfs::core::Tensor normalized_gaussian_blur_adjoint(
        const lfs::core::Tensor& grad, const lfs::core::Tensor& weight, float sigma);

    // The photo's local standard deviation as the SSIM window sees it (11-tap Gaussian, sigma 1.5, zero padding),
    // per pixel the mean over the channels of sqrt(max(E[y^2] - E[y]^2, 0)). target is [C, H, W] or [1, C, H, W]
    // (Float32 or UInt8); out becomes [H, W]. The pyramid loss writes the same map with want_photo_sigma_map.
    void photo_local_sigma(const lfs::core::Tensor& target, lfs::core::Tensor& out);

    // Fused SSIM map → error map: error_map[i] = max(0, 1 - mean_c(ssim_map[c, i]))
    // Replaces .neg().add(1).mean({1}).squeeze(0).clamp_min(0).contiguous() chain
    void launch_ssim_to_error_map(
        const lfs::core::Tensor& ssim_map,
        lfs::core::Tensor& error_map);

} // namespace lfs::training::kernels
