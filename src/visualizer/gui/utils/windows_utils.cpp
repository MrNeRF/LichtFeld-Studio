/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/events.hpp"
#include "core/logger.hpp"

#include "gui/utils/windows_utils.hpp"

#ifdef WIN32
#include <ShlObj.h>
#include <windows.h>
#endif // WIN32

namespace gs::gui {
#ifdef WIN32

    namespace utils {
        HRESULT selectFileNative(PWSTR& strDirectory,
                                 COMDLG_FILTERSPEC rgSpec[],
                                 UINT cFileTypes,
                                 bool blnDirectory) {

            HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
            if (FAILED(hr)) {
                LOG_ERROR("Failed to initialize COM: {:#x}", static_cast<unsigned int>(hr));
            } else {
                // Create the FileOpenDialog instance
                IFileOpenDialog* pFileOpen = nullptr;
                hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
                                      IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));

                if (SUCCEEDED(hr)) {
                    DWORD dwOptions;

                    if (SUCCEEDED(pFileOpen->GetOptions(&dwOptions))) {
                        if (blnDirectory) {
                            pFileOpen->SetOptions(dwOptions | FOS_PICKFOLDERS);
                        } else {
                            if (rgSpec != nullptr && cFileTypes > 0) {
                                hr = pFileOpen->SetFileTypes(cFileTypes, rgSpec);
                                if (SUCCEEDED(hr)) {
                                    pFileOpen->SetOptions(dwOptions | FOS_NOCHANGEDIR | FOS_FILEMUSTEXIST);
                                    pFileOpen->SetFileTypeIndex(1);
                                } else {
                                    LOG_ERROR("Failed to set file types: {:#x}", static_cast<unsigned int>(hr));
                                }
                            } else {
                                pFileOpen->SetOptions(dwOptions | FOS_NOCHANGEDIR | FOS_FILEMUSTEXIST);
                            }
                        }
                    }

                    // Show the Open File dialog
                    hr = pFileOpen->Show(NULL);

                    if (SUCCEEDED(hr)) {
                        IShellItem* pItem;
                        hr = pFileOpen->GetResult(&pItem);
                        if (SUCCEEDED(hr)) {
                            PWSTR filePath = nullptr;
                            hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &filePath);

                            if (SUCCEEDED(hr)) {
                                strDirectory = filePath;
                                CoTaskMemFree(filePath);
                            }
                            pItem->Release();
                        }
                    }
                    pFileOpen->Release();
                } else {
                    LOG_ERROR("Failed to create FileOpenDialog: {:#x}", static_cast<unsigned int>(hr));
                    CoUninitialize();
                }
                CoUninitialize();
            }
            return hr;
        }
    } // namespace utils

    void OpenProjectFileDialog() {
        // show native windows file dialog for project file selection
        PWSTR filePath = nullptr;

        COMDLG_FILTERSPEC rgSpec[] =
            {
                {L"LichtFeldStudio Project File", L"*.lfs;*.ls"},
            };

        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path project_path(filePath);
            events::cmd::LoadProject{.path = project_path}.emit();
            LOG_INFO("Loading project file : {}", std::filesystem::path(project_path).string());
        }
    }

    void OpenPlyFileDialog() {
        // show native windows file dialog for PLY file selection
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] =
            {
                {L"Point Cloud", L"*.ply;"},
            };
        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path ply_path(filePath);
            events::cmd::LoadFile{.path = ply_path}.emit();
            LOG_INFO("Loading PLY file : {}", std::filesystem::path(ply_path).string()); // FIXED: Changed from "Loading project file"
        }
    }

    void OpenDatasetFolderDialog() {
        // show native windows file dialog for folder selection
        PWSTR filePath = nullptr;
        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, nullptr, 0, true))) {
            std::filesystem::path dataset_path(filePath);
            if (std::filesystem::is_directory(dataset_path)) {
                events::cmd::LoadFile{.path = dataset_path, .is_dataset = true}.emit();
                LOG_INFO("Loading dataset : {}", std::filesystem::path(dataset_path).string());
            }
        }
    }

    void SaveProjectFileDialog(bool* p_open) {
        // show native windows file dialog for project directory selection
        PWSTR filePath = nullptr;
        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, nullptr, 0, true))) {
            std::filesystem::path project_path(filePath);
            events::cmd::SaveProject{project_path}.emit();
            LOG_INFO("Saving project file into : {}", std::filesystem::path(project_path).string());
            *p_open = false;
        }
    }

#endif // WIN32

} // namespace gs::gui