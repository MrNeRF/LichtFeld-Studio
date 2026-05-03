/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include <filesystem>
#include <string>

namespace lfs::vis::gui {

    LFS_VIS_API std::filesystem::path OpenImageFileDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path OpenEnvironmentMapFileDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path PickFolderDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path OpenPointCloudFileDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path OpenMeshFileDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path OpenCheckpointFileDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path OpenPPISPFileDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path OpenDatasetFolderDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path OpenJsonFileDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path OpenCsvFileDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path OpenVideoFileDialog(const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path OpenPythonFileDialog(const std::filesystem::path& defaultPath = {});

    LFS_VIS_API std::filesystem::path SavePlyFileDialog(const std::string& defaultName,
                                                        const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path SaveJsonFileDialog(const std::string& defaultName,
                                                         const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path SaveTextFileDialog(const std::string& defaultName,
                                                         const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path SaveSogFileDialog(const std::string& defaultName,
                                                        const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path SaveSpzFileDialog(const std::string& defaultName,
                                                        const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path SaveUsdFileDialog(const std::string& defaultName,
                                                        const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path SaveUsdzFileDialog(const std::string& defaultName,
                                                         const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path SaveHtmlFileDialog(const std::string& defaultName,
                                                         const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path SaveRadFileDialog(const std::string& defaultName,
                                                        const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path SaveMp4FileDialog(const std::string& defaultName,
                                                        const std::filesystem::path& defaultPath = {});
    LFS_VIS_API std::filesystem::path SavePythonFileDialog(const std::string& defaultName,
                                                           const std::filesystem::path& defaultPath = {});

} // namespace lfs::vis::gui
