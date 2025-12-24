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

namespace lfs::vis::gui {
#ifdef WIN32

    namespace utils {
        HRESULT selectFileNative(PWSTR& strDirectory,
                                 COMDLG_FILTERSPEC rgSpec[],
                                 UINT cFileTypes,
                                 bool blnDirectory) {

            HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
            if (FAILED(hr)) {
                LOG_ERROR("Failed to initialize COM: {:#x}", static_cast<unsigned int>(hr));
            } else {
                IFileOpenDialog* pFileOpen = nullptr;
                hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL,
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

                    hr = pFileOpen->Show(nullptr);

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

    namespace utils {
        HRESULT saveFileNative(PWSTR& outPath,
                               COMDLG_FILTERSPEC rgSpec[],
                               UINT cFileTypes,
                               const wchar_t* defaultName) {
            HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
            if (FAILED(hr)) {
                LOG_ERROR("Failed to initialize COM: {:#x}", static_cast<unsigned int>(hr));
                return hr;
            }

            IFileSaveDialog* pFileSave = nullptr;
            hr = CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_ALL,
                                  IID_IFileSaveDialog, reinterpret_cast<void**>(&pFileSave));

            if (SUCCEEDED(hr)) {
                if (rgSpec != nullptr && cFileTypes > 0) {
                    pFileSave->SetFileTypes(cFileTypes, rgSpec);
                    pFileSave->SetFileTypeIndex(1);
                    pFileSave->SetDefaultExtension(L"ply");
                }

                if (defaultName != nullptr) {
                    pFileSave->SetFileName(defaultName);
                }

                hr = pFileSave->Show(nullptr);

                if (SUCCEEDED(hr)) {
                    IShellItem* pItem;
                    hr = pFileSave->GetResult(&pItem);
                    if (SUCCEEDED(hr)) {
                        PWSTR filePath = nullptr;
                        hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &filePath);
                        if (SUCCEEDED(hr)) {
                            outPath = filePath;
                        }
                        pItem->Release();
                    }
                }
                pFileSave->Release();
            }
            CoUninitialize();
            return hr;
        }
    } // namespace utils

#endif // WIN32

    namespace {
#ifndef _WIN32
        constexpr size_t DIALOG_BUFFER_SIZE = 4096;

        // Execute dialog command, trying fallback if primary fails
        std::string runDialogCommand(const std::string& primary_cmd, const std::string& fallback_cmd) {
            FILE* pipe = popen(primary_cmd.c_str(), "r");
            if (!pipe && !fallback_cmd.empty()) {
                pipe = popen(fallback_cmd.c_str(), "r");
            }
            if (!pipe)
                return {};

            char buffer[DIALOG_BUFFER_SIZE];
            std::string result;
            while (fgets(buffer, sizeof(buffer), pipe)) {
                result += buffer;
            }
            pclose(pipe);

            // Trim trailing newlines
            while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                result.pop_back();
            }
            return result;
        }
#endif
    } // namespace

    std::filesystem::path SaveJsonFileDialog(const std::string& defaultName) {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"JSON File", L"*.json"}};
        const std::wstring wDefaultName(defaultName.begin(), defaultName.end());

        if (SUCCEEDED(utils::saveFileNative(filePath, rgSpec, 1, wDefaultName.c_str()))) {
            std::filesystem::path result(filePath);
            CoTaskMemFree(filePath);
            if (result.extension() != ".json") {
                result += ".json";
            }
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --save --confirm-overwrite "
                                    "--file-filter='JSON files|*.json' "
                                    "--filename='" +
                                    defaultName + ".json' 2>/dev/null";
        const std::string fallback = "kdialog --getsavefilename . 'JSON files (*.json)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        if (result.empty())
            return {};

        std::filesystem::path path(result);
        if (path.extension() != ".json") {
            path += ".json";
        }
        return path;
#endif
    }

    std::filesystem::path OpenJsonFileDialog() {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"JSON File", L"*.json"}};

        if (SUCCEEDED(utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path result(filePath);
            CoTaskMemFree(filePath);
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --file-filter='JSON files|*.json' 2>/dev/null";
        const std::string fallback = "kdialog --getopenfilename . 'JSON files (*.json)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        if (result.empty())
            return {};
        return std::filesystem::path(result);
#endif
    }

    std::filesystem::path SavePlyFileDialog(const std::string& defaultName) {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"PLY Point Cloud", L"*.ply"}};
        const std::wstring wDefaultName(defaultName.begin(), defaultName.end());

        if (SUCCEEDED(utils::saveFileNative(filePath, rgSpec, 1, wDefaultName.c_str()))) {
            std::filesystem::path result(filePath);
            CoTaskMemFree(filePath);
            if (result.extension() != ".ply") {
                result += ".ply";
            }
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --save --confirm-overwrite "
                                    "--file-filter='PLY files|*.ply' "
                                    "--filename='" +
                                    defaultName + ".ply' 2>/dev/null";
        const std::string fallback = "kdialog --getsavefilename . 'PLY files (*.ply)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        if (result.empty())
            return {};

        std::filesystem::path path(result);
        if (path.extension() != ".ply") {
            path += ".ply";
        }
        return path;
#endif
    }

    std::filesystem::path SaveSogFileDialog(const std::string& defaultName) {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"SOG File (SuperSplat)", L"*.sog"}};
        const std::wstring wDefaultName(defaultName.begin(), defaultName.end());

        if (SUCCEEDED(utils::saveFileNative(filePath, rgSpec, 1, wDefaultName.c_str()))) {
            std::filesystem::path result(filePath);
            CoTaskMemFree(filePath);
            if (result.extension() != ".sog") {
                result += ".sog";
            }
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --save --confirm-overwrite "
                                    "--file-filter='SOG files (SuperSplat)|*.sog' "
                                    "--filename='" +
                                    defaultName + ".sog' 2>/dev/null";
        const std::string fallback = "kdialog --getsavefilename . 'SOG files (*.sog)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        if (result.empty())
            return {};

        std::filesystem::path path(result);
        if (path.extension() != ".sog") {
            path += ".sog";
        }
        return path;
#endif
    }

    std::filesystem::path SaveSpzFileDialog(const std::string& defaultName) {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"SPZ File (Niantic)", L"*.spz"}};
        const std::wstring wDefaultName(defaultName.begin(), defaultName.end());

        if (SUCCEEDED(utils::saveFileNative(filePath, rgSpec, 1, wDefaultName.c_str()))) {
            std::filesystem::path result(filePath);
            CoTaskMemFree(filePath);
            if (result.extension() != ".spz") {
                result += ".spz";
            }
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --save --confirm-overwrite "
                                    "--file-filter='SPZ files (Niantic)|*.spz' "
                                    "--filename='" +
                                    defaultName + ".spz' 2>/dev/null";
        const std::string fallback = "kdialog --getsavefilename . 'SPZ files (*.spz)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        if (result.empty())
            return {};

        std::filesystem::path path(result);
        if (path.extension() != ".spz") {
            path += ".spz";
        }
        return path;
#endif
    }

    std::filesystem::path SaveHtmlFileDialog(const std::string& defaultName) {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"HTML Viewer", L"*.html"}};
        const std::wstring wDefaultName(defaultName.begin(), defaultName.end());

        if (SUCCEEDED(utils::saveFileNative(filePath, rgSpec, 1, wDefaultName.c_str()))) {
            std::filesystem::path result(filePath);
            CoTaskMemFree(filePath);
            if (result.extension() != ".html") {
                result += ".html";
            }
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --save --confirm-overwrite "
                                    "--file-filter='HTML files|*.html' "
                                    "--filename='" +
                                    defaultName + ".html' 2>/dev/null";
        const std::string fallback = "kdialog --getsavefilename . 'HTML files (*.html)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        if (result.empty())
            return {};

        std::filesystem::path path(result);
        if (path.extension() != ".html") {
            path += ".html";
        }
        return path;
#endif
    }

    std::filesystem::path SelectFolderDialog(const std::string& title, const std::filesystem::path& startDir) {
#ifdef _WIN32
        HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (FAILED(hr)) {
            LOG_ERROR("COM init failed: {:#x}", static_cast<unsigned int>(hr));
            return {};
        }

        IFileOpenDialog* pFileOpen = nullptr;
        hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL,
                              IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));

        std::filesystem::path result;
        if (SUCCEEDED(hr)) {
            DWORD dwOptions = 0;
            if (SUCCEEDED(pFileOpen->GetOptions(&dwOptions))) {
                pFileOpen->SetOptions(dwOptions | FOS_PICKFOLDERS);
            }

            if (!startDir.empty() && std::filesystem::exists(startDir)) {
                IShellItem* pStartFolder = nullptr;
                const std::wstring wStartDir = startDir.wstring();
                hr = SHCreateItemFromParsingName(wStartDir.c_str(), nullptr, IID_PPV_ARGS(&pStartFolder));
                if (SUCCEEDED(hr)) {
                    pFileOpen->SetFolder(pStartFolder);
                    pStartFolder->Release();
                }
            }

            hr = pFileOpen->Show(nullptr);
            if (SUCCEEDED(hr)) {
                IShellItem* pItem = nullptr;
                hr = pFileOpen->GetResult(&pItem);
                if (SUCCEEDED(hr)) {
                    PWSTR filePath = nullptr;
                    hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &filePath);
                    if (SUCCEEDED(hr)) {
                        result = std::filesystem::path(filePath);
                        CoTaskMemFree(filePath);
                    }
                    pItem->Release();
                }
            }
            pFileOpen->Release();
        }
        CoUninitialize();
        return result;
#else
        std::filesystem::path abs_start_dir = startDir;
        if (!abs_start_dir.empty()) {
            if (abs_start_dir.is_relative()) {
                abs_start_dir = std::filesystem::absolute(abs_start_dir);
            }
        }

        const bool has_valid_start = !abs_start_dir.empty() &&
                                     std::filesystem::exists(abs_start_dir) &&
                                     std::filesystem::is_directory(abs_start_dir);

        const std::string start_arg = has_valid_start
                                          ? " --filename='" + abs_start_dir.string() + "/'"
                                          : "";

        const std::string primary = "zenity --file-selection --directory "
                                    "--title='" +
                                    title + "'" + start_arg + " 2>/dev/null";
        const std::string fallback = "kdialog --getexistingdirectory '" +
                                     (has_valid_start ? abs_start_dir.string() : ".") + "' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        return result.empty() ? std::filesystem::path{} : std::filesystem::path(result);
#endif
    }

    std::filesystem::path OpenPlyFileDialogNative() {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"Point Cloud", L"*.ply;*.sog;*.spz"}};

        if (SUCCEEDED(utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path result(filePath);
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection "
                                    "--file-filter='Point Cloud|*.ply *.sog *.spz' "
                                    "--title='Open Point Cloud' 2>/dev/null";
        const std::string fallback = "kdialog --getopenfilename . 'Point Cloud (*.ply *.sog *.spz)' 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        return result.empty() ? std::filesystem::path{} : std::filesystem::path(result);
#endif
    }

    std::filesystem::path OpenCheckpointFileDialog() {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {{L"Checkpoint", L"*.resume"}};

        if (SUCCEEDED(utils::selectFileNative(filePath, rgSpec, 1, false))) {
            return std::filesystem::path(filePath);
        }
        return {};
#else
        const std::string result = runDialogCommand(
            "zenity --file-selection --file-filter='Checkpoint|*.resume' --title='Open Checkpoint' 2>/dev/null",
            "kdialog --getopenfilename . 'Checkpoint (*.resume)' 2>/dev/null");
        return result.empty() ? std::filesystem::path{} : std::filesystem::path(result);
#endif
    }

    std::filesystem::path OpenDatasetFolderDialogNative() {
#ifdef _WIN32
        PWSTR filePath = nullptr;
        if (SUCCEEDED(utils::selectFileNative(filePath, nullptr, 0, true))) {
            std::filesystem::path result(filePath);
            return result;
        }
        return {};
#else
        const std::string primary = "zenity --file-selection --directory "
                                    "--title='Select Dataset Folder' 2>/dev/null";
        const std::string fallback = "kdialog --getexistingdirectory . 2>/dev/null";

        const std::string result = runDialogCommand(primary, fallback);
        return result.empty() ? std::filesystem::path{} : std::filesystem::path(result);
#endif
    }

} // namespace lfs::vis::gui
