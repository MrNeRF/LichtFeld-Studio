/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/utils/native_file_dialog.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/formats/colmap.hpp"
#include "io/video/video_extensions.hpp"

#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_video.h>
#include <nfd.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef __linux__
#include <cstdlib>
#include <gtk/gtk.h>
#endif

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

#ifdef __linux__
        // GTK is sensitive to in-process state. GDK picks a windowing backend
        // on first touch, and GTK auto-loads accessibility / IM modules that
        // drag in their own dynamic state. Both routinely misbehave in a
        // process that already owns SDL3 + Vulkan surfaces and an embedded
        // Python interpreter. Pin the environment to a known-good config
        // before GTK initializes:
        //   - GDK_BACKEND=x11 routes GTK through X11 / XWayland, isolated
        //     from our Vulkan-on-Wayland surfaces,
        //   - NO_AT_BRIDGE=1 disables the AT-SPI accessibility bridge,
        //   - GTK_MODULES is cleared so distro-injected modules
        //     (canberra-gtk-module, pk-gtk-module, ...) can't run inside
        //     our address space.
        // All defaults respect a caller-set value.
        void preflightGtkEnvironmentOnce() {
            static const bool applied = []() {
                ::setenv("GDK_BACKEND", "x11", /*overwrite=*/0);
                ::setenv("NO_AT_BRIDGE", "1", /*overwrite=*/0);
                ::unsetenv("GTK_MODULES");
                ::unsetenv("GTK3_MODULES");
                return true;
            }();
            (void)applied;
        }
#endif

        class ThreadLocalNfdContext {
        public:
            ThreadLocalNfdContext() {
#ifdef __linux__
                preflightGtkEnvironmentOnce();
#endif
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

        [[nodiscard]] std::filesystem::path absoluteDialogDirectory(
            const std::filesystem::path& directory) {
            std::error_code ec;
            auto canonical = std::filesystem::weakly_canonical(directory, ec);
            if (!ec && !canonical.empty()) {
                return canonical;
            }

            ec.clear();
            auto absolute = std::filesystem::absolute(directory, ec);
            if (!ec && !absolute.empty()) {
                return absolute.lexically_normal();
            }

            return directory;
        }

        [[nodiscard]] std::filesystem::path normalizeDefaultDirectory(
            const std::filesystem::path& inputPath) {
            if (inputPath.empty()) {
                return {};
            }

            std::error_code ec;
            if (std::filesystem::exists(inputPath, ec)) {
                if (std::filesystem::is_directory(inputPath, ec)) {
                    return absoluteDialogDirectory(inputPath);
                }
                return absoluteDialogDirectory(inputPath.parent_path());
            }

            const std::filesystem::path parent = inputPath.parent_path();
            if (!parent.empty() && std::filesystem::exists(parent, ec) &&
                std::filesystem::is_directory(parent, ec)) {
                return absoluteDialogDirectory(parent);
            }
            return {};
        }

        [[nodiscard]] std::filesystem::path resolveColmapSparseDialogDirectory(
            const std::filesystem::path& inputPath) {
            if (inputPath.empty()) {
                return {};
            }

            if (auto sparsePath = lfs::io::find_colmap_sparse_model_path(inputPath); sparsePath) {
                return *sparsePath;
            }

            return inputPath;
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

#ifndef __linux__
        void warnUnsupportedDialogTitleOnce() {
            static std::once_flag flag;
            std::call_once(flag, []() {
                LOG_WARN("Custom native file-dialog titles are unavailable on this platform");
            });
        }
#endif

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

#ifndef __linux__
            // On Linux, NFD's GTK backend takes the X11 parent handle and feeds it to
            // gdk_x11_window_foreign_new_for_display, which crashes when the supplied
            // window belongs to SDL3's own X11/XWayland connection. The dialog works
            // fine unparented (top-level window grabs focus via the WM), so we leave
            // the handle UNSET on Linux. Windows and macOS pickers still get a proper
            // parent.
            const Sint64 x11Window = SDL_GetNumberProperty(
                props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
            if (x11Window != 0) {
                handle.type = NFD_WINDOW_HANDLE_TYPE_X11;
                handle.handle = reinterpret_cast<void*>(static_cast<uintptr_t>(x11Window));
                return handle;
            }
#endif

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
                               {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tga", ".hdr", ".exr"})};
        }

        [[nodiscard]] std::vector<DialogFilter> environmentMapFilters() {
            return {makeFilter("Environment Maps", {".hdr", ".exr"})};
        }

        [[nodiscard]] std::vector<DialogFilter> pointCloudFilters() {
            return {makeFilter("Point Cloud Files", {".ply", ".sog", ".spz", ".rad", ".usd", ".usda", ".usdc", ".usdz"})};
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

        [[nodiscard]] std::vector<DialogFilter> xmlFilters() {
            return {makeFilter("Metashape XML Files", {".xml"})};
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
            std::vector<std::string> extensions;
            extensions.reserve(lfs::io::video::kSupportedVideoExtensions.size());
            for (const std::string_view ext : lfs::io::video::kSupportedVideoExtensions)
                extensions.emplace_back(ext);
            return {makeFilter("Video Files", std::move(extensions))};
        }

        [[nodiscard]] std::vector<DialogFilter> pythonFilters() {
            return {makeFilter("Python Files", {".py"})};
        }

        [[nodiscard]] std::vector<DialogFilter> singleExtensionFilter(
            const std::string& name,
            const std::string& extension) {
            return {makeFilter(name, {extension})};
        }

#ifdef __linux__
        [[nodiscard]] std::filesystem::path runLinuxGtkPathPicker(
            const std::filesystem::path& defaultDirectory,
            const char* title,
            const GtkFileChooserAction action,
            const bool returnCurrentFolder) {
            if (!ensureDialogBackendInitialized()) {
                return {};
            }

            const char* const acceptLabel =
                action == GTK_FILE_CHOOSER_ACTION_OPEN ? "_Open" : "_Select";
            GtkWidget* const dialog = gtk_file_chooser_dialog_new(
                title,
                nullptr,
                action,
                "_Cancel", GTK_RESPONSE_CANCEL,
                acceptLabel, GTK_RESPONSE_ACCEPT,
                nullptr);
            if (dialog == nullptr) {
                LOG_ERROR("Failed to construct GTK file chooser dialog");
                return {};
            }

            if (!defaultDirectory.empty()) {
                const std::string utf8 = lfs::core::path_to_utf8(defaultDirectory);
                gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), utf8.c_str());
            }

            std::filesystem::path result;
            if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
                gchar* selected =
                    returnCurrentFolder
                        ? gtk_file_chooser_get_current_folder(GTK_FILE_CHOOSER(dialog))
                        : gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
                if (selected != nullptr) {
                    result = lfs::core::utf8_to_path(selected);
                    g_free(selected);
                }
            }

            gtk_widget_destroy(dialog);
            // Drain queued GTK events so destroyed widgets are torn down
            // before control returns to SDL3's loop.
            while (gtk_events_pending()) {
                gtk_main_iteration();
            }
            return result;
        }

        // The COLMAP export picker selects the directory currently being
        // browsed, rather than a highlighted child inside that directory.
        [[nodiscard]] std::filesystem::path runLinuxGtkFolderPicker(
            const std::filesystem::path& defaultDirectory,
            const char* title) {
            return runLinuxGtkPathPicker(
                defaultDirectory, title, GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, true);
        }
#endif

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

    std::filesystem::path OpenFileDialog(const std::filesystem::path& defaultPath,
                                         const std::string& dialogTitle) {
#ifdef __linux__
        if (!dialogTitle.empty()) {
            return runLinuxGtkPathPicker(
                normalizeDefaultDirectory(defaultPath), dialogTitle.c_str(),
                GTK_FILE_CHOOSER_ACTION_OPEN, false);
        }
#else
        if (!dialogTitle.empty())
            warnUnsupportedDialogTitleOnce();
#endif
        std::filesystem::path result;
        runDialog(makeOpenFileRequest({}, defaultPath), result);
        return result;
    }

    std::filesystem::path PickFolderDialog(const std::filesystem::path& defaultPath,
                                           const std::string& dialogTitle) {
#ifdef __linux__
        if (!dialogTitle.empty()) {
            return runLinuxGtkPathPicker(
                normalizeDefaultDirectory(defaultPath), dialogTitle.c_str(),
                GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, false);
        }
#else
        if (!dialogTitle.empty())
            warnUnsupportedDialogTitleOnce();
#endif
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

    std::filesystem::path PickColmapSparseFolderDialog(const std::filesystem::path& defaultSparsePath) {
        const std::filesystem::path defaultDirectory =
            normalizeDefaultDirectory(resolveColmapSparseDialogDirectory(defaultSparsePath));
        if (!defaultDirectory.empty()) {
            LOG_INFO("Opening COLMAP sparse folder picker at: {}",
                     lfs::core::path_to_utf8(defaultDirectory));
        }

#ifdef __linux__
        return runLinuxGtkFolderPicker(defaultDirectory, "Select COLMAP sparse export folder");
#else
        std::filesystem::path result;
        runDialog(makePickFolderRequest(defaultDirectory), result);
        return result;
#endif
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

    std::filesystem::path OpenXmlFileDialog(const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeOpenFileRequest(xmlFilters(), defaultPath), result);
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

    std::filesystem::path SavePngFileDialog(const std::string& defaultName,
                                            const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("PNG Image", ".png"),
                                      defaultPath,
                                      defaultName,
                                      ".png"),
                  result);
        return result;
    }

    std::filesystem::path SaveJpgFileDialog(const std::string& defaultName,
                                            const std::filesystem::path& defaultPath) {
        std::filesystem::path result;
        runDialog(makeSaveFileRequest(singleExtensionFilter("JPEG Image", ".jpg"),
                                      defaultPath,
                                      defaultName,
                                      ".jpg"),
                  result);
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
