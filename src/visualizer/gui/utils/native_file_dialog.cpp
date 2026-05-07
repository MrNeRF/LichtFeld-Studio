/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/utils/native_file_dialog.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"

#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_video.h>
#include <nfd.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace lfs::vis::gui {

    namespace {

        enum class DialogKind : uint8_t {
            OpenFile,
            SaveFile,
            PickFolder,
        };

        struct DialogFilter {
            std::string name;
            std::vector<std::string> extensions;
        };

        struct DialogRequest {
            DialogKind kind;
            std::vector<DialogFilter> filters;
            std::filesystem::path default_path;
            std::string default_name;
            std::string required_extension;
        };

        [[nodiscard]] DialogFilter makeFilter(std::string name,
                                              std::vector<std::string> extensions) {
            return DialogFilter{
                .name = std::move(name),
                .extensions = std::move(extensions),
            };
        }

        [[nodiscard]] DialogRequest makeOpenFileRequest(std::vector<DialogFilter> filters,
                                                        std::filesystem::path defaultPath) {
            return DialogRequest{
                .kind = DialogKind::OpenFile,
                .filters = std::move(filters),
                .default_path = std::move(defaultPath),
                .default_name = {},
                .required_extension = {},
            };
        }

        [[nodiscard]] DialogRequest makeSaveFileRequest(std::vector<DialogFilter> filters,
                                                        std::filesystem::path defaultPath,
                                                        std::string defaultName,
                                                        std::string requiredExtension) {
            return DialogRequest{
                .kind = DialogKind::SaveFile,
                .filters = std::move(filters),
                .default_path = std::move(defaultPath),
                .default_name = std::move(defaultName),
                .required_extension = std::move(requiredExtension),
            };
        }

        [[nodiscard]] DialogRequest makePickFolderRequest(std::filesystem::path defaultPath) {
            return DialogRequest{
                .kind = DialogKind::PickFolder,
                .filters = {},
                .default_path = std::move(defaultPath),
                .default_name = {},
                .required_extension = {},
            };
        }

        class ThreadLocalNfdContext {
        public:
            ThreadLocalNfdContext() {
                const nfdresult_t result = NFD_Init();
                initialized_ = (result == NFD_OKAY);
                if (!initialized_) {
                    const char* error = NFD_GetError();
                    LOG_ERROR("Failed to initialize native file dialogs: {}",
                              error ? error : "unknown error");
                }
            }

            ~ThreadLocalNfdContext() {
                if (initialized_) {
                    NFD_Quit();
                }
            }

            [[nodiscard]] bool initialized() const {
                return initialized_;
            }

        private:
            bool initialized_ = false;
        };

        [[nodiscard]] bool ensureDialogBackendInitialized() {
            static thread_local ThreadLocalNfdContext context;
            return context.initialized();
        }

        [[nodiscard]] std::string normalizeExtension(std::string extension) {
            while (!extension.empty() &&
                   (extension.front() == '*' || extension.front() == '.')) {
                extension.erase(extension.begin());
            }
            return extension;
        }

        [[nodiscard]] std::string ensureDefaultExtension(std::string defaultName,
                                                         const std::string_view extension) {
            const std::filesystem::path normalizedPath =
                defaultName.empty() ? std::filesystem::path{}
                                    : lfs::core::utf8_to_path(defaultName).filename();
            const std::string normalizedName =
                normalizedPath.empty() ? std::string{} : lfs::core::path_to_utf8(normalizedPath);
            if (normalizedName.empty() || extension.empty() ||
                normalizedPath.extension() == extension) {
                return normalizedName;
            }
            return normalizedName + std::string(extension);
        }

        [[nodiscard]] std::filesystem::path appendRequiredExtension(
            std::filesystem::path path,
            const std::string_view extension) {
            if (path.empty() || extension.empty() || path.extension() == extension) {
                return path;
            }
            path += std::string(extension);
            return path;
        }

        [[nodiscard]] std::filesystem::path normalizeDefaultDirectory(
            const std::filesystem::path& inputPath) {
            if (inputPath.empty()) {
                return {};
            }

            std::error_code ec;
            if (std::filesystem::exists(inputPath, ec)) {
                if (std::filesystem::is_directory(inputPath, ec)) {
                    return inputPath;
                }
                return inputPath.parent_path();
            }

            const std::filesystem::path parent = inputPath.parent_path();
            if (!parent.empty() && std::filesystem::exists(parent, ec) &&
                std::filesystem::is_directory(parent, ec)) {
                return parent;
            }
            return {};
        }

        [[nodiscard]] bool isPortalAbruptDismissal(const char* error) {
            if (error == nullptr) {
                return false;
            }

            const std::string_view message(error);
            return message.find("D-Bus file dialog interaction was ended abruptly") !=
                       std::string_view::npos &&
                   message.find("response code 2") != std::string_view::npos;
        }

        class FilterListStorage {
        public:
            explicit FilterListStorage(const std::vector<DialogFilter>& filters) {
                friendly_names_.reserve(filters.size());
                filter_specs_.reserve(filters.size());

                for (const DialogFilter& filter : filters) {
                    std::vector<std::string> normalizedExtensions;
                    normalizedExtensions.reserve(filter.extensions.size());
                    for (const std::string& extension : filter.extensions) {
                        const std::string normalized = normalizeExtension(extension);
                        if (!normalized.empty()) {
                            normalizedExtensions.push_back(normalized);
                        }
                    }

                    if (normalizedExtensions.empty()) {
                        continue;
                    }

                    friendly_names_.push_back(filter.name);

                    std::string joined;
                    for (size_t i = 0; i < normalizedExtensions.size(); ++i) {
                        if (i > 0) {
                            joined += ',';
                        }
                        joined += normalizedExtensions[i];
                    }
                    filter_specs_.push_back(std::move(joined));
                }

                items_.reserve(friendly_names_.size());
                for (size_t i = 0; i < friendly_names_.size(); ++i) {
                    items_.push_back(
                        nfdu8filteritem_t{friendly_names_[i].c_str(), filter_specs_[i].c_str()});
                }
            }

            [[nodiscard]] const nfdu8filteritem_t* data() const {
                return items_.empty() ? nullptr : items_.data();
            }

            [[nodiscard]] nfdfiltersize_t size() const {
                return static_cast<nfdfiltersize_t>(items_.size());
            }

        private:
            std::vector<std::string> friendly_names_;
            std::vector<std::string> filter_specs_;
            std::vector<nfdu8filteritem_t> items_;
        };

        [[nodiscard]] nfdwindowhandle_t currentParentWindowHandle() {
            nfdwindowhandle_t handle{};
            SDL_Window* const window = SDL_GetKeyboardFocus();
            if (!window) {
                return handle;
            }

            const SDL_PropertiesID props = SDL_GetWindowProperties(window);
            if (!props) {
                return handle;
            }

            if (void* hwnd =
                    SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr)) {
                handle.type = NFD_WINDOW_HANDLE_TYPE_WINDOWS;
                handle.handle = hwnd;
                return handle;
            }

            if (void* cocoaWindow =
                    SDL_GetPointerProperty(props, SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr)) {
                handle.type = NFD_WINDOW_HANDLE_TYPE_COCOA;
                handle.handle = cocoaWindow;
                return handle;
            }

            const Sint64 x11Window = SDL_GetNumberProperty(
                props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
            if (x11Window != 0) {
                handle.type = NFD_WINDOW_HANDLE_TYPE_X11;
                handle.handle = reinterpret_cast<void*>(static_cast<uintptr_t>(x11Window));
                return handle;
            }

            // SDL exposes Wayland properties, but nativefiledialog-extended does not currently
            // provide a Wayland parent handle type, so dialogs will be unparented on Wayland.
            return handle;
        }

        bool runDialog(const DialogRequest& request, std::filesystem::path& resultPath) {
            resultPath.clear();
            if (!ensureDialogBackendInitialized()) {
                return false;
            }

            const FilterListStorage filters(request.filters);
            const std::filesystem::path defaultDirectory =
                normalizeDefaultDirectory(request.default_path);
            const std::string defaultDirectoryUtf8 =
                defaultDirectory.empty() ? std::string{} : lfs::core::path_to_utf8(defaultDirectory);
            const char* const defaultDirectoryArg =
                defaultDirectoryUtf8.empty() ? nullptr : defaultDirectoryUtf8.c_str();

            const nfdwindowhandle_t parentWindow = currentParentWindowHandle();
            nfdresult_t dialogResult = NFD_ERROR;
            nfdu8char_t* selectedPath = nullptr;
            NFD_ClearError();

            if (request.kind == DialogKind::OpenFile) {
                const nfdopendialogu8args_t args{
                    filters.data(),
                    filters.size(),
                    defaultDirectoryArg,
                    parentWindow,
                };
                dialogResult = NFD_OpenDialogU8_With(&selectedPath, &args);
            } else if (request.kind == DialogKind::SaveFile) {
                const std::string defaultName =
                    ensureDefaultExtension(request.default_name, request.required_extension);
                const char* const defaultNameArg =
                    defaultName.empty() ? nullptr : defaultName.c_str();
                const nfdsavedialogu8args_t args{
                    filters.data(),
                    filters.size(),
                    defaultDirectoryArg,
                    defaultNameArg,
                    parentWindow,
                };
                dialogResult = NFD_SaveDialogU8_With(&selectedPath, &args);
            } else {
                const nfdpickfolderu8args_t args{
                    defaultDirectoryArg,
                    parentWindow,
                };
                dialogResult = NFD_PickFolderU8_With(&selectedPath, &args);
            }

            if (dialogResult == NFD_CANCEL) {
                NFD_ClearError();
                return false;
            }

            if (dialogResult != NFD_OKAY) {
                const char* error = NFD_GetError();
                if (isPortalAbruptDismissal(error)) {
                    LOG_DEBUG("Native file dialog dismissed by portal: {}", error);
                    NFD_ClearError();
                    return false;
                }

                LOG_ERROR("Native file dialog failed: {}",
                          error ? error : "unknown error");
                NFD_ClearError();
                return false;
            }

            resultPath = lfs::core::utf8_to_path(selectedPath);
            NFD_FreePathU8(selectedPath);
            resultPath = appendRequiredExtension(resultPath, request.required_extension);
            return !resultPath.empty();
        }

        [[nodiscard]] std::vector<DialogFilter> imageFilters() {
            return {makeFilter("Image Files",
                               {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".hdr", ".exr"})};
        }

        [[nodiscard]] std::vector<DialogFilter> environmentMapFilters() {
            return {makeFilter("Environment Maps", {".hdr", ".exr"})};
        }

        [[nodiscard]] std::vector<DialogFilter> pointCloudFilters() {
            return {makeFilter("Point Cloud Files", {".ply", ".sog", ".spz", ".usd", ".usda", ".usdc", ".usdz"})};
        }

        [[nodiscard]] std::vector<DialogFilter> meshFilters() {
            return {makeFilter("3D Mesh Files",
                               {".obj", ".fbx", ".gltf", ".glb", ".stl", ".dae", ".3ds", ".ply"})};
        }

        [[nodiscard]] std::vector<DialogFilter> checkpointFilters() {
            return {makeFilter("Checkpoint Files", {".resume"})};
        }

        [[nodiscard]] std::vector<DialogFilter> ppispFilters() {
            return {makeFilter("PPISP Sidecar Files", {".ppisp"})};
        }

        [[nodiscard]] std::vector<DialogFilter> jsonFilters() {
            return {makeFilter("JSON Files", {".json"})};
        }

        [[nodiscard]] std::vector<DialogFilter> csvFilters() {
            return {makeFilter("CSV Files", {".csv"})};
        }

        [[nodiscard]] std::vector<DialogFilter> lasFilters() {
            return {makeFilter("LAS Point Cloud Files", {".las"})};
        }

        [[nodiscard]] std::vector<DialogFilter> lazFilters() {
            return {makeFilter("LAZ Compressed Point Cloud Files", {".laz"})};
        }

        [[nodiscard]] std::vector<DialogFilter> lasLazFilters() {
            return {makeFilter("LAS/LAZ Point Cloud Files", {".las", ".laz"})};
        }

        [[nodiscard]] std::vector<DialogFilter> videoFilters() {
            return {makeFilter("Video Files", {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"})};
        }

        [[nodiscard]] std::vector<DialogFilter> pythonFilters() {
            return {makeFilter("Python Files", {".py"})};
        }

        [[nodiscard]] std::vector<DialogFilter> singleExtensionFilter(
            const std::string& name,
            const std::string& extension) {
            return {makeFilter(name, {extension})};
        }

    } // namespace

    std::filesystem::path OpenImageFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(imageFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path OpenEnvironmentMapFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(environmentMapFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path PickFolderDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makePickFolderRequest(defaultPath), result);
        return result;
    }

    std::filesystem::path OpenPointCloudFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(pointCloudFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path OpenMeshFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(meshFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path OpenCheckpointFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(checkpointFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path OpenPPISPFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(ppispFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path OpenDatasetFolderDialog(const std::filesystem::path& defaultPath) {
        return PickFolderDialog(defaultPath);
    }

    std::filesystem::path OpenJsonFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(jsonFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path OpenCsvFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(csvFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path OpenLasFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(lasLazFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path SaveLasFileDialog(const std::string& defaultName,
                                            const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(lasFilters(), defaultPath, defaultName, ".las"), result);
        return result;
    }

    std::filesystem::path SaveLazFileDialog(const std::string& defaultName,
                                            const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(lazFilters(), defaultPath, defaultName, ".laz"), result);
        return result;
    }

    std::filesystem::path OpenVideoFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(videoFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path OpenPythonFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(pythonFilters(), defaultPath), result);
        return result;
    }

    std::filesystem::path SavePlyFileDialog(const std::string& defaultName,
                                            const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("PLY Files", ".ply"),
                                      defaultPath,
                                      defaultName,
                                      ".ply"),
                  result);
        return result;
    }

    std::filesystem::path SaveJsonFileDialog(const std::string& defaultName,
                                             const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(jsonFilters(), defaultPath, defaultName, ".json"), result);
        return result;
    }

    std::filesystem::path SaveTextFileDialog(const std::string& defaultName,
                                             const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("Text Files", ".txt"),
                                      defaultPath,
                                      defaultName,
                                      ".txt"),
                  result);
        return result;
    }

    std::filesystem::path SaveSogFileDialog(const std::string& defaultName,
                                            const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("SOG Files", ".sog"),
                                      defaultPath,
                                      defaultName,
                                      ".sog"),
                  result);
        return result;
    }

    std::filesystem::path SaveSpzFileDialog(const std::string& defaultName,
                                            const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("SPZ Files", ".spz"),
                                      defaultPath,
                                      defaultName,
                                      ".spz"),
                  result);
        return result;
    }

    std::filesystem::path SaveUsdFileDialog(const std::string& defaultName,
                                            const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("USD Files", ".usd"),
                                      defaultPath,
                                      defaultName,
                                      ".usd"),
                  result);
        return result;
    }

    std::filesystem::path SaveUsdzFileDialog(const std::string& defaultName,
                                             const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("USDZ Files", ".usdz"),
                                      defaultPath,
                                      defaultName,
                                      ".usdz"),
                  result);
        return result;
    }

    std::filesystem::path SaveHtmlFileDialog(const std::string& defaultName,
                                             const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("HTML Files", ".html"),
                                      defaultPath,
                                      defaultName,
                                      ".html"),
                  result);
        return result;
    }

    std::filesystem::path SaveRadFileDialog(const std::string& defaultName,
                                            const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("RAD Files", ".rad"),
                                      defaultPath,
                                      defaultName,
                                      ".rad"),
                  result);
        return result;
    }

    std::filesystem::path SaveMp4FileDialog(const std::string& defaultName,
                                            const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("MP4 Video", ".mp4"),
                                      defaultPath,
                                      defaultName,
                                      ".mp4"),
                  result);
        return result;
    }

    std::filesystem::path SavePythonFileDialog(const std::string& defaultName,
                                               const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(pythonFilters(), defaultPath, defaultName, ".py"), result);
        return result;
    }

} // namespace lfs::vis::gui
