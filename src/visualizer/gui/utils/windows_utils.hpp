/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#ifdef WIN32
#include <Shobjidl.h>
#include <Windows.h>
#endif

namespace gs::gui {

#ifdef WIN32

    namespace utils {
        /**
         * Opens a native Windows file/folder selection dialog
         * @param strDirectory Output path selected by the user
         * @param rgSpec File type filters (can be nullptr)
         * @param cFileTypes Number of file type filters
         * @param blnDirectory True to select folders, false for files
         * @return HRESULT indicating success or failure
         */
        HRESULT selectFileNative(PWSTR& strDirectory,
                                 COMDLG_FILTERSPEC rgSpec[] = nullptr,
                                 UINT cFileTypes = 0,
                                 bool blnDirectory = false);
    } // namespace utils

    // in windows- open file browser that search for lfs project
    void OpenProjectFileDialog();
    // in windows- open file browser that search for ply files
    void OpenPlyFileDialog();
    // in windows- open file browser that search directories
    void OpenDatasetFolderDialog();
#endif
} // namespace gs::gui