/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

namespace lfs::core {

    constexpr uint32_t CHECKPOINT_MAGIC = 0x4C464B50; // "LFKP"
    constexpr uint32_t CHECKPOINT_VERSION = 1;
    constexpr uint64_t MAX_CHECKPOINT_FILE_BYTES = 256ULL * 1024ULL * 1024ULL * 1024ULL;
    constexpr uint64_t MAX_CHECKPOINT_JSON_BYTES = 16ULL * 1024ULL * 1024ULL;
    constexpr uint32_t MAX_CHECKPOINT_STRATEGY_NAME_BYTES = 64;
    constexpr uint32_t MAX_CHECKPOINT_GAUSSIANS = 1'000'000'000;

    enum class CheckpointFlags : uint32_t {
        NONE = 0,
        HAS_BILATERAL_GRID = 1 << 0,
        HAS_PPISP = 1 << 1,
        HAS_PPISP_CONTROLLER = 1 << 2,
    };

    constexpr CheckpointFlags operator|(const CheckpointFlags a, const CheckpointFlags b) {
        return static_cast<CheckpointFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }

    constexpr bool has_flag(const CheckpointFlags flags, const CheckpointFlags flag) {
        return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
    }

    struct CheckpointHeader {
        uint32_t magic = CHECKPOINT_MAGIC;
        uint32_t version = CHECKPOINT_VERSION;
        int32_t iteration = 0;
        uint32_t num_gaussians = 0;
        int32_t sh_degree = 0;
        CheckpointFlags flags = CheckpointFlags::NONE;
        uint64_t params_json_offset = 0;
        uint64_t params_json_size = 0;
    };

    LFS_CORE_API std::expected<void, std::string> validate_checkpoint_header(
        const CheckpointHeader& header,
        uint64_t file_size);

    LFS_CORE_API std::expected<CheckpointHeader, std::string> load_checkpoint_header(
        const std::filesystem::path& path);

    LFS_CORE_API std::expected<SplatData, std::string> load_checkpoint_splat_data(
        const std::filesystem::path& path,
        SplatTensorAllocator tensor_allocator = {});

    LFS_CORE_API std::expected<param::TrainingParameters, std::string> load_checkpoint_params(
        const std::filesystem::path& path);

} // namespace lfs::core
