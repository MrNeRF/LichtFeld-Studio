/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/nn/activation_arena.hpp"
#include "core/nn/ops.hpp"
#include "core/nn/weight_file.hpp"
#include "core/tensor.hpp"

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lfs::core::nn::models {

    enum class InputScaling {
        Identity,
        Normalize,
    };

    enum class LpipsComputeMode {
        Exact,
        Fast,
    };

    struct LpipsTaps {
        std::array<Tensor, 5> normalized_features{};
        std::array<float, 5> scalars{};
    };

    // LPIPS v0.1 with the VGG16 ImageNet backbone, matching the lpips package.
    class LFS_CORE_API Lpips {
    public:
        static lfs::Result<Lpips> load(const std::filesystem::path& weights, Device device,
                                       std::optional<DataType> compute = std::nullopt,
                                       InputScaling scaling = InputScaling::Identity,
                                       std::size_t activation_budget_bytes = 1536ULL * 1024ULL * 1024ULL);

        // RGB fp32 [0,1] on CUDA, supplied as [3,H,W] or [1,3,H,W].
        lfs::Result<float> forward(const Tensor& pred, const Tensor& target,
                                   std::optional<InputScaling> scaling = std::nullopt);

        lfs::Result<LpipsTaps> forward_with_taps(const Tensor& pred, const Tensor& target,
                                                 std::optional<InputScaling> scaling = std::nullopt);

        [[nodiscard]] DataType compute_dtype() const { return compute_; }
        [[nodiscard]] LpipsComputeMode compute_mode() const {
            return compute_ == DataType::Float16 ? LpipsComputeMode::Fast : LpipsComputeMode::Exact;
        }
        [[nodiscard]] Device device() const { return device_; }
        [[nodiscard]] std::size_t weights_bytes() const;
        [[nodiscard]] std::size_t arena_bytes() const { return arena_.capacity(); }
        [[nodiscard]] std::size_t workspace_bytes() const { return workspace_.bytes(); }
        [[nodiscard]] std::size_t activation_budget_bytes() const { return activation_budget_bytes_; }
        [[nodiscard]] std::size_t tile_size_for(int height, int width) const;
        [[nodiscard]] std::size_t estimated_peak_bytes(int height, int width) const;
        void release_activations();

    private:
        lfs::Result<float> run(const Tensor& pred, const Tensor& target, InputScaling scaling,
                               LpipsTaps* taps);
        lfs::Result<float> run_untiled(const Tensor& pred, const Tensor& target,
                                       InputScaling scaling, LpipsTaps* taps);
        lfs::Result<float> run_tiled(const Tensor& pred, const Tensor& target,
                                     InputScaling scaling);
        lfs::Result<float> run_tiled_fast(const Tensor& pred, const Tensor& target,
                                          InputScaling scaling);
        // Fast mode without taps: fused kernels writing ping-pong feature buffers,
        // no activation arena, no intermediate copies.
        lfs::Result<float> run_fast(const Tensor& pred, const Tensor& target,
                                    InputScaling scaling);
        std::optional<lfs::Error> validate_pair(const Tensor& pred, const Tensor& target) const;
        void bind_weights_to_stream(cudaStream_t stream);
        const Tensor& w(std::string_view name) const;
        Tensor conv(const Tensor& x, std::string_view weight, std::string_view bias = {});
        Tensor conv(const Tensor& x, const Tensor& weight);
        Tensor normalize_feature(const Tensor& x);
        Tensor ensure_workspace(std::size_t bytes, const Tensor& like);

        std::unordered_map<std::string, Tensor> weights_;
        Tensor workspace_;
        Tensor feature_x_hold_;
        Tensor feature_y_hold_;
        Tensor normalized_x_hold_;
        Tensor normalized_y_hold_;
        std::array<Tensor, 4> fast_features_;
        Tensor fast_scores_;
        Tensor fast_weight_taps_;
        std::array<std::size_t, 13> fast_weight_tap_offsets_{};
        std::array<float, 3> scaling_shift_{};
        std::array<float, 3> scaling_scale_{};
        ActivationArena arena_;
        bool weights_on_stream_ = false;
        Device device_ = Device::CUDA;
        DataType compute_ = DataType::Float32;
        InputScaling scaling_ = InputScaling::Identity;
        std::size_t activation_budget_bytes_ = 1536ULL * 1024ULL * 1024ULL;
    };

} // namespace lfs::core::nn::models
