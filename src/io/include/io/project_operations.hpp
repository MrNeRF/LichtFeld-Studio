/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/project_inspector.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>

namespace lfs::io::project {

    using ProjectOperationProgress =
        std::function<void(float progress, const std::string& stage)>;
    using ProjectOperationCancel = std::function<bool()>;

    enum class ProjectVerificationStatus {
        Verified,
        Canceled,
        Failed,
    };

    struct LFS_IO_API ProjectVerificationResult {
        ProjectVerificationStatus status = ProjectVerificationStatus::Failed;
        std::uint64_t verified_chunks = 0;
        std::optional<std::string> first_mismatch;
    };

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    restore_save(const std::filesystem::path& path,
                 std::uint64_t generation,
                 const std::filesystem::path& destination);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    rebind_checkpoint(const std::filesystem::path& path,
                      const lfs::core::Uuid& checkpoint_instance_uuid);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    compact_project_file(const std::filesystem::path& path,
                         ProjectOperationProgress progress = {},
                         ProjectOperationCancel cancel = {});

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectVerificationResult>
    verify_project_file(const std::filesystem::path& path,
                        ProjectOperationProgress progress = {},
                        ProjectOperationCancel cancel = {});

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    set_project_preview(const std::filesystem::path& path,
                        std::span<const std::byte> png_bytes);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    preview_from_first_dataset_image(const std::filesystem::path& path);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    preview_from_first_embedded_image(const std::filesystem::path& path);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    set_project_license(const std::filesystem::path& path,
                        const std::string& identifier,
                        const std::string& notice = {});

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    clear_project_license(const std::filesystem::path& path);

    [[nodiscard]] LFS_IO_API lfs::Result<ProjectInspectorCard>
    set_project_title(const std::filesystem::path& path,
                      const std::string& title);

} // namespace lfs::io::project
