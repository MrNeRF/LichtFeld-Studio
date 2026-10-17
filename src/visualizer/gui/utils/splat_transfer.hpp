/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <atomic>
#include <chrono>
#include <core/export.hpp>
#include <core/splat_data.hpp>
#include <filesystem>
#include <functional>
#include <future>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
namespace lfs::vis::gui {
    class LFS_VIS_API SplatTransferError : public std::runtime_error {
    public:
        const size_t item_index;
        SplatTransferError(size_t index, const std::string& detail) : std::runtime_error(detail), item_index(index) {}
    };
    struct SplatTransferOptions {
        std::string format = "ply";
        std::string name = "splat";
        int sh_degree = -1;
        int sog_iterations = 30;
        bool z_up = true;
        std::function<bool(float)> progress;
        // Optional owned staging root for embedding and isolated IO contracts.
        std::filesystem::path staging_root;
    };
    struct SplatTransferSource {
        const core::SplatData* model;
        glm::mat4 world;
        SplatTransferOptions options;
    };
    LFS_VIS_API std::future<std::vector<std::filesystem::path>> prepareSplatTransferBatch(
        std::span<const SplatTransferSource> sources, std::shared_ptr<std::atomic<bool>> cancelled);
    LFS_VIS_API void cleanupSplatTransfers(const std::filesystem::path& root, std::chrono::hours age = std::chrono::hours(24 * 7));
    // UI-thread CPU snapshot; worker transforms and atomically commits the selected file format.
    LFS_VIS_API std::future<std::filesystem::path> prepareSplatTransfer(
        const core::SplatData& model, glm::mat4 world,
        std::shared_ptr<std::atomic<bool>> cancelled,
        SplatTransferOptions options = {});
} // namespace lfs::vis::gui
