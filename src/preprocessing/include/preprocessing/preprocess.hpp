/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/nn/weight_file.hpp"
#include "core/parameters.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace lfs::preprocessing {

    enum class CachedModel : unsigned char {
        Sam2,
        Moge2,
        Lpips,
        LlmDet
    };

    struct ModelDescriptor {
        CachedModel id;
        std::string_view name;
        std::string_view category_key;
        std::string_view filename;
        std::string_view url;
        std::string_view sha256;
    };

    struct ModelCacheState {
        std::filesystem::path path;
        std::uintmax_t bytes = 0;
        bool present = false;
        bool busy = false;
    };

    [[nodiscard]] std::span<const ModelDescriptor> model_catalog();
    [[nodiscard]] lfs::Result<ModelCacheState> inspect_cached_model(CachedModel model);
    enum class ModelPreparationPhase { Waiting,
                                       Downloading,
                                       Verifying };
    // Invoked synchronously on the acquisition worker. A zero total is unknown;
    // bytes describe the current phase, not overall operation completion.
    using ModelPreparationCallback = std::function<void(ModelPreparationPhase, std::uintmax_t, std::uintmax_t)>;
    // Verification reads only the managed archive; acquisition may migrate it.
    enum class ModelPreparationMode { Acquire,
                                      Verify };
    [[nodiscard]] lfs::Result<core::nn::WeightFile> prepare_cached_model(
        CachedModel model, bool allow_download, const std::function<bool()>& cancel_requested = {},
        const ModelPreparationCallback& progress = {}, ModelPreparationMode mode = ModelPreparationMode::Acquire);
    [[nodiscard]] lfs::Result<bool> remove_cached_model(CachedModel model);

    [[nodiscard]] lfs::Result<core::nn::WeightFile> ensure_lpips_weights(bool allow_download);

    [[nodiscard]] lfs::Result<core::nn::WeightFile> ensure_sam2_weights(
        bool allow_download, const std::function<bool()>& cancel_requested = {});

    [[nodiscard]] lfs::Result<core::nn::WeightFile> ensure_llmdet_weights(
        bool allow_download, const std::function<bool()>& cancel_requested = {});

    using PreprocessProgressCallback = std::function<void(
        std::size_t done, std::size_t total, std::string_view filename)>;

    struct PreprocessRunResult {
        bool ok = true;
        std::string error;
        std::size_t processed = 0;
        std::size_t skipped = 0;
    };

    int run_preprocess(const lfs::core::param::PreprocessParameters& params);

    PreprocessRunResult run_preprocess_ex(
        const lfs::core::param::PreprocessParameters& params,
        const PreprocessProgressCallback& progress = {});

} // namespace lfs::preprocessing
