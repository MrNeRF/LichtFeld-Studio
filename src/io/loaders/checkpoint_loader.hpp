/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/checkpoint_format.hpp"
#include "io/loader_interface.hpp"

namespace lfs::io {

    /**
     * @brief Loader for LichtFeld checkpoint files (.resume)
     *
     * Checkpoint files contain complete training state:
     * - SplatData (Gaussian parameters)
     * - Optimizer state (Adam moments)
     * - Scheduler state
     * - Training parameters
     *
     * This loader extracts only the SplatData for viewing.
     * For full training resumption, use Trainer::load_checkpoint() directly.
     */
    class CheckpointLoader : public IDataLoader {
    public:
        CheckpointLoader() = default;
        ~CheckpointLoader() override = default;

        [[nodiscard]] Result<LoadResult> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) override;

        bool canLoad(const std::filesystem::path& path) const override;
        std::string name() const override;
        std::vector<std::string> supportedExtensions() const override;
        int priority() const override;

        /**
         * @brief Load checkpoint header only (for inspection without full load)
         */
        static std::expected<lfs::core::CheckpointHeader, std::string> loadHeader(
            const std::filesystem::path& path);
    };

} // namespace lfs::io
