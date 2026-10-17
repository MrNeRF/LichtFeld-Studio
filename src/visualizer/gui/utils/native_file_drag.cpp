/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "native_file_drag.hpp"
#include <SDL3/SDL.h>
#include <limits>
#include <vector>
#ifdef __linux__
#include <gdk/gdkx.h>
#include <gtk/gtk.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstring>
#include <ole2.h>
#include <shellapi.h>
#include <shlobj.h>
#include <windows.h>
#endif

namespace lfs::vis::gui {
#ifdef _WIN32
    namespace {
        class CopySource final : public IDropSource {
        public:
            HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id, void** out) override {
                if (!out)
                    return E_POINTER;
                *out = nullptr;
                if (id != IID_IUnknown && id != IID_IDropSource)
                    return E_NOINTERFACE;
                *out = static_cast<IDropSource*>(this);
                AddRef();
                return S_OK;
            }
            ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
            ULONG STDMETHODCALLTYPE Release() override {
                const auto remaining = --refs_;
                if (!remaining)
                    delete this;
                return remaining;
            }
            HRESULT STDMETHODCALLTYPE QueryContinueDrag(BOOL escape, DWORD keys) override {
                if (escape || (keys & MK_RBUTTON))
                    return DRAGDROP_S_CANCEL;
                return (keys & MK_LBUTTON) ? S_OK : DRAGDROP_S_DROP;
            }
            HRESULT STDMETHODCALLTYPE GiveFeedback(DWORD) override {
                return DRAGDROP_S_USEDEFAULTCURSORS;
            }

        private:
            ULONG refs_ = 1;
        };

        struct OleScope {
            HRESULT result = OleInitialize(nullptr);
            ~OleScope() {
                if (SUCCEEDED(result))
                    OleUninitialize();
            }
        };

        // The Shell data object supplies the standard COM format enumeration.
        // Each GetData consumer receives its own storage; ownership of SetData's
        // HGLOBAL passes to the data object only on success.
        HRESULT createData(const std::vector<std::filesystem::path>& files, IDataObject** out) {
            *out = nullptr;
            std::vector<wchar_t> list;
            for (const auto& path : files) {
                const auto& name = path.native();
                list.insert(list.end(), name.begin(), name.end());
                list.push_back(L'\0');
            }
            list.push_back(L'\0');
            if (list.size() > (std::numeric_limits<SIZE_T>::max() - sizeof(DROPFILES)) / sizeof(wchar_t))
                return E_OUTOFMEMORY;
            auto memory = GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT,
                                      sizeof(DROPFILES) + list.size() * sizeof(wchar_t));
            if (!memory)
                return E_OUTOFMEMORY;
            auto* header = static_cast<DROPFILES*>(GlobalLock(memory));
            if (!header) {
                GlobalFree(memory);
                return E_OUTOFMEMORY;
            }
            header->pFiles = sizeof(DROPFILES);
            header->fWide = TRUE;
            std::memcpy(reinterpret_cast<char*>(header) + sizeof(DROPFILES),
                        list.data(), list.size() * sizeof(wchar_t));
            GlobalUnlock(memory);
            IDataObject* data = nullptr;
            auto hr = SHCreateDataObject(nullptr, 0, nullptr, nullptr, IID_IDataObject,
                                         reinterpret_cast<void**>(&data));
            if (FAILED(hr)) {
                GlobalFree(memory);
                return hr;
            }
            FORMATETC format{CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL};
            STGMEDIUM storage{};
            storage.tymed = TYMED_HGLOBAL;
            storage.hGlobal = memory;
            hr = data->SetData(&format, &storage, TRUE);
            if (FAILED(hr)) {
                GlobalFree(memory);
                data->Release();
                return hr;
            }
            *out = data;
            return S_OK;
        }
    } // namespace
#endif

    std::optional<bool> pointerOutsideApplication(SDL_Window* source) {
#ifdef _WIN32
        (void)source;
        POINT point;
        if (!GetCursorPos(&point))
            return std::nullopt;
        const auto target = WindowFromPoint(point);
        if (!target)
            return true;
        DWORD process = 0;
        GetWindowThreadProcessId(target, &process);
        return process != GetCurrentProcessId();
#elif defined(__linux__)
        const auto* driver = SDL_GetCurrentVideoDriver();
        if (!source || !driver || std::string_view(driver) != "x11")
            return std::nullopt;
        const auto properties = SDL_GetWindowProperties(source);
        auto* display = static_cast<Display*>(SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr));
        if (!display)
            return std::nullopt;
        std::vector<Window> own;
        int count = 0;
        if (auto** windows = SDL_GetWindows(&count)) {
            for (int i = 0; i < count; ++i)
                own.push_back(static_cast<Window>(SDL_GetNumberProperty(SDL_GetWindowProperties(windows[i]), SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0)));
            SDL_free(windows);
        }
        Window window = DefaultRootWindow(display);
        for (int depth = 0; depth < 64; ++depth) {
            for (const auto id : own)
                if (id && id == window)
                    return false;
            Window root = 0, child = 0;
            int rx, ry, wx, wy;
            unsigned mask;
            if (!XQueryPointer(display, window, &root, &child, &rx, &ry, &wx, &wy, &mask))
                return std::nullopt;
            if (!child || child == window)
                return true;
            window = child;
        }
        return std::nullopt;
#else
        (void)source;
        return std::nullopt;
#endif
    }

    FileDragResult dragReadyFiles(std::span<const std::filesystem::path> files, SDL_Window* window) {
        if (files.empty())
            return {FileDragStatus::Failed, "No files prepared"};
        std::vector<std::filesystem::path> absolute;
        absolute.reserve(files.size());
        for (const auto& file : files) {
            std::error_code ec;
            auto path = std::filesystem::absolute(file, ec);
            if (ec || !std::filesystem::is_regular_file(path, ec) || ec)
                return {FileDragStatus::Failed, "Drag requires existing, committed regular files"};
            absolute.push_back(std::move(path));
        }
#ifdef _WIN32
        (void)window;
        // Preparation may have outlasted the original press. Never manufacture
        // a drop when the user has already released the gesture.
        if (!(GetAsyncKeyState(VK_LBUTTON) & 0x8000))
            return {FileDragStatus::Cancelled, "The left mouse button is no longer held"};
        // A cancelled OLE loop can dispatch callbacks that attempt another drag.
        static thread_local bool active = false;
        if (active)
            return {FileDragStatus::Failed, "A native file drag is already active"};
        struct ActiveScope {
            bool& value;
            explicit ActiveScope(bool& flag) : value(flag) { value = true; }
            ~ActiveScope() { value = false; }
        } active_scope(active);
        OleScope ole;
        if (FAILED(ole.result))
            return {FileDragStatus::Failed, "OLE initialization failed on the calling UI thread"};
        IDataObject* data = nullptr;
        if (FAILED(createData(absolute, &data)))
            return {FileDragStatus::Failed, "Cannot create the native CF_HDROP payload"};
        auto* source = new CopySource();
        DWORD effect = DROPEFFECT_NONE;
        // SDL and GHOST can hold automatic capture while the button is down.
        // Hand it to OLE explicitly so the pointer can enter another app.
        ReleaseCapture();
        const auto hr = DoDragDrop(data, source, DROPEFFECT_COPY, &effect);
        source->Release();
        data->Release();
        if (hr == DRAGDROP_S_DROP && (effect & DROPEFFECT_COPY))
            return {FileDragStatus::Copied, {}};
        if (hr == DRAGDROP_S_CANCEL || (hr == DRAGDROP_S_DROP && effect == DROPEFFECT_NONE))
            return {FileDragStatus::Cancelled, {}};
        return {FileDragStatus::Failed, "Native file drag failed"};
#elif defined(__linux__)
        // GTK owns an independent X11 connection and serves URI selection requests
        // while its native drag loop runs. Do not steal SDL's X11 event queue.
        // A Wayland source requires the compositor's original input serial; an
        // unrelated GTK connection cannot safely manufacture that serial.
        const auto* driver = SDL_GetCurrentVideoDriver();
        if (!window || !driver || std::string_view(driver) != "x11")
            return {FileDragStatus::Unsupported, "Splat drag currently requires Linux X11 / XWayland (SDL_VIDEODRIVER=x11)"};
        float mouse_x, mouse_y;
        if (!(SDL_GetGlobalMouseState(&mouse_x, &mouse_y) & SDL_BUTTON_LMASK))
            return {FileDragStatus::Cancelled, {}};
        if (!gtk_init_check(nullptr, nullptr))
            return {FileDragStatus::Failed, "Cannot initialize GTK for file drag"};
        if (!GDK_IS_X11_DISPLAY(gdk_display_get_default()))
            return {FileDragStatus::Unsupported, "GTK must use X11 for splat drag (GDK_BACKEND=x11)"};
        static thread_local bool active = false;
        if (active)
            return {FileDragStatus::Failed, "A file drag is already active"};
        active = true;
        struct State {
            bool done = false;
            bool failed = false;
            std::vector<std::string> uris;
        } state;
        for (const auto& path : absolute) {
            auto* uri = g_filename_to_uri(path.c_str(), nullptr, nullptr);
            if (!uri) {
                active = false;
                return {FileDragStatus::Failed, "Cannot encode file URI"};
            }
            state.uris.emplace_back(uri);
            g_free(uri);
        }
        auto* widget = gtk_invisible_new();
        gtk_widget_realize(widget);
        gtk_widget_show(widget);
        g_signal_connect(widget, "drag-data-get", G_CALLBACK(+[](GtkWidget*, GdkDragContext*, GtkSelectionData* selection, guint, guint, gpointer data) {
                             auto& s = *static_cast<State*>(data);
                             std::vector<gchar*> uris;
                             for (auto& uri : s.uris)
                                 uris.push_back(uri.data());
                             uris.push_back(nullptr);
                             gtk_selection_data_set_uris(selection, uris.data());
                         }),
                         &state);
        g_signal_connect(widget, "drag-failed", G_CALLBACK(+[](GtkWidget*, GdkDragContext*, GtkDragResult, gpointer data) -> gboolean {
                             static_cast<State*>(data)->failed = true;
                             return FALSE;
                         }),
                         &state);
        g_signal_connect(widget, "drag-end", G_CALLBACK(+[](GtkWidget*, GdkDragContext*, gpointer data) {
                             static_cast<State*>(data)->done = true;
                         }),
                         &state);
        auto* targets = gtk_target_list_new(nullptr, 0);
        gtk_target_list_add_uri_targets(targets, 0);
        auto* drag = gtk_drag_begin_with_coordinates(widget, targets, GDK_ACTION_COPY, 1, nullptr, -1, -1);
        gtk_target_list_unref(targets);
        if (drag) {
            gtk_drag_set_icon_name(drag, "text-x-generic", 0, 0);
            while (!state.done)
                g_main_context_iteration(nullptr, TRUE);
        }
        gtk_widget_destroy(widget);
        active = false;
        if (!drag)
            return {FileDragStatus::Failed, "Cannot start GTK file drag"};
        return {state.failed ? FileDragStatus::Cancelled : FileDragStatus::Copied, {}};
#else
        (void)window;
        return {FileDragStatus::Unsupported, "Native drag source is not implemented for this platform"};
#endif
    }
} // namespace lfs::vis::gui
