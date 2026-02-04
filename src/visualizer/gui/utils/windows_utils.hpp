/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include <filesystem>
#include <string>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <Shobjidl.h>
#include <Windows.h>
#endif

namespace lfs::vis::gui {

#ifdef WIN32
    namespace utils {
        HRESULT selectFileNative(PWSTR& strDirectory,
                                 COMDLG_FILTERSPEC rgSpec[] = nullptr,
                                 UINT cFileTypes = 0,
                                 bool blnDirectory = false);
        HRESULT saveFileNative(PWSTR& outPath,
                               COMDLG_FILTERSPEC rgSpec[] = nullptr,
                               UINT cFileTypes = 0,
                               const wchar_t* defaultName = nullptr);
    } // namespace utils
#endif

    // Cross-platform file open dialogs (return path, empty if cancelled)
    LFS_VIS_API std::filesystem::path OpenPlyFileDialogNative(const std::filesystem::path& startDir = {});
    LFS_VIS_API std::filesystem::path OpenCheckpointFileDialog();
    LFS_VIS_API std::filesystem::path OpenDatasetFolderDialogNative();

    // Cross-platform file save/open dialogs
    LFS_VIS_API std::filesystem::path OpenImageFileDialog(const std::filesystem::path& startDir = {});
    LFS_VIS_API std::filesystem::path SavePlyFileDialog(const std::string& defaultName);
    LFS_VIS_API std::filesystem::path SaveJsonFileDialog(const std::string& defaultName);
    LFS_VIS_API std::filesystem::path OpenJsonFileDialog();
    LFS_VIS_API std::filesystem::path SaveSogFileDialog(const std::string& defaultName);
    LFS_VIS_API std::filesystem::path SaveSpzFileDialog(const std::string& defaultName);
    LFS_VIS_API std::filesystem::path SaveHtmlFileDialog(const std::string& defaultName);
    LFS_VIS_API std::filesystem::path SaveMp4FileDialog(const std::string& defaultName);
    LFS_VIS_API std::filesystem::path OpenVideoFileDialog();
    LFS_VIS_API std::filesystem::path SelectFolderDialog(const std::string& title = "Select Folder",
                                                         const std::filesystem::path& startDir = {});

    // Python script file dialogs
    LFS_VIS_API std::filesystem::path OpenPythonFileDialog(const std::filesystem::path& startDir = {});
    LFS_VIS_API std::filesystem::path SavePythonFileDialog(const std::string& defaultName);

} // namespace lfs::vis::gui
