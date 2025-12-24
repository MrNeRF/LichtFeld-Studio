/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

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
    std::filesystem::path OpenPlyFileDialogNative();
    std::filesystem::path OpenCheckpointFileDialog();
    std::filesystem::path OpenDatasetFolderDialogNative();

    // Cross-platform file save/open dialogs
    std::filesystem::path SavePlyFileDialog(const std::string& defaultName);
    std::filesystem::path SaveJsonFileDialog(const std::string& defaultName);
    std::filesystem::path OpenJsonFileDialog();
    std::filesystem::path SaveSogFileDialog(const std::string& defaultName);
    std::filesystem::path SaveSpzFileDialog(const std::string& defaultName);
    std::filesystem::path SaveHtmlFileDialog(const std::string& defaultName);
    std::filesystem::path SelectFolderDialog(const std::string& title = "Select Folder",
                                             const std::filesystem::path& startDir = {});

} // namespace lfs::vis::gui
