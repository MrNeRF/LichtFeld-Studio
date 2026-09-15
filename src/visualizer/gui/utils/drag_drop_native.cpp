/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "drag_drop_native.hpp"
#include "core/logger.hpp"

// Platform detection and native includes
#ifdef _WIN32
#include <SDL3/SDL.h>
#include <oleidl.h>
#include <shellapi.h>
#include <shlobj.h>
#elif defined(__linux__)
#include <SDL3/SDL.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <utility>
#else
#include <SDL3/SDL.h>
#endif

namespace lfs::vis::gui {

// ============================================================================
// Windows Implementation
// ============================================================================
#ifdef _WIN32

    class DropTarget : public IDropTarget {
    public:
        explicit DropTarget(NativeDragDrop* owner) : owner_(owner) {}

        // IUnknown
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
            if (riid == IID_IUnknown || riid == IID_IDropTarget) {
                *ppv = static_cast<IDropTarget*>(this);
                AddRef();
                return S_OK;
            }
            *ppv = nullptr;
            return E_NOINTERFACE;
        }

        ULONG STDMETHODCALLTYPE AddRef() override { return ++ref_count_; }

        ULONG STDMETHODCALLTYPE Release() override {
            ULONG count = --ref_count_;
            if (count == 0)
                delete this;
            return count;
        }

        // IDropTarget
        HRESULT STDMETHODCALLTYPE DragEnter(IDataObject* pDataObj, DWORD grfKeyState,
                                            POINTL pt, DWORD* pdwEffect) override {
            (void)pDataObj;
            (void)grfKeyState;
            (void)pt;
            *pdwEffect = DROPEFFECT_COPY;
            if (owner_)
                owner_->setDragHovering(true);
            return S_OK;
        }

        HRESULT STDMETHODCALLTYPE DragOver(DWORD grfKeyState, POINTL pt,
                                           DWORD* pdwEffect) override {
            (void)grfKeyState;
            (void)pt;
            *pdwEffect = DROPEFFECT_COPY;
            return S_OK;
        }

        HRESULT STDMETHODCALLTYPE DragLeave() override {
            if (owner_)
                owner_->setDragHovering(false);
            return S_OK;
        }

        HRESULT STDMETHODCALLTYPE Drop(IDataObject* const pDataObj, DWORD /*grfKeyState*/,
                                       POINTL /*pt*/, DWORD* const pdwEffect) override {
            *pdwEffect = DROPEFFECT_COPY;
            LOG_DEBUG("IDropTarget::Drop called");

            std::vector<std::string> paths;
            constexpr FORMATETC FMT = {CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL};
            STGMEDIUM stg = {TYMED_HGLOBAL};

            HRESULT hr = pDataObj->GetData(const_cast<FORMATETC*>(&FMT), &stg);
            if (SUCCEEDED(hr)) {
                if (const HDROP hdrop = static_cast<HDROP>(GlobalLock(stg.hGlobal))) {
                    const UINT count = DragQueryFileW(hdrop, 0xFFFFFFFF, nullptr, 0);
                    LOG_DEBUG("Drop contains {} files", count);
                    paths.reserve(count);
                    for (UINT i = 0; i < count; ++i) {
                        const UINT len = DragQueryFileW(hdrop, i, nullptr, 0);
                        if (len == 0)
                            continue;
                        std::wstring wpath(len + 1, L'\0');
                        DragQueryFileW(hdrop, i, wpath.data(), len + 1);
                        const int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, nullptr, 0, nullptr, nullptr);
                        if (utf8_len <= 0) {
                            LOG_WARN("Failed to convert dropped path to UTF-8");
                            continue;
                        }
                        std::string utf8_path(utf8_len - 1, '\0');
                        WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, utf8_path.data(), utf8_len, nullptr, nullptr);
                        LOG_DEBUG("Dropped file: {}", utf8_path);
                        paths.push_back(std::move(utf8_path));
                    }
                    GlobalUnlock(stg.hGlobal);
                } else {
                    LOG_WARN("GlobalLock failed for dropped data");
                }
                ReleaseStgMedium(&stg);
            } else {
                LOG_WARN("GetData failed for CF_HDROP format: 0x{:08X}", static_cast<unsigned>(hr));
            }

            if (owner_) {
                owner_->setDragHovering(false);
                if (!paths.empty()) {
                    owner_->handleFileDrop(paths);
                } else {
                    LOG_WARN("No valid files extracted from drop");
                }
            }
            return S_OK;
        }

    private:
        NativeDragDrop* owner_;
        ULONG ref_count_ = 1;
    };

    struct NativeDragDrop::PlatformData {
        HWND hwnd = nullptr;
        DropTarget* drop_target = nullptr;
        bool ole_initialized = false;
    };

    bool NativeDragDrop::init(SDL_Window* window) {
        if (initialized_)
            return true;
        window_ = window;

        platform_data_ = new PlatformData();
        platform_data_->hwnd = (HWND)SDL_GetPointerProperty(
            SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);

        if (!platform_data_->hwnd) {
            LOG_ERROR("Failed to get Win32 window handle");
            return false;
        }

        // Initialize OLE (required for drag-drop)
        HRESULT hr = OleInitialize(nullptr);
        if (FAILED(hr) && hr != S_FALSE) { // S_FALSE means already initialized
            LOG_ERROR("OleInitialize failed: 0x{:08X}", static_cast<unsigned>(hr));
            return false;
        }
        platform_data_->ole_initialized = true;

        // Revoke any existing drop target (a previous init call may have registered one)
        RevokeDragDrop(platform_data_->hwnd);

        // Create and register our drop target
        platform_data_->drop_target = new DropTarget(this);
        hr = RegisterDragDrop(platform_data_->hwnd, platform_data_->drop_target);
        if (FAILED(hr)) {
            LOG_ERROR("RegisterDragDrop failed: 0x{:08X}", static_cast<unsigned>(hr));
            platform_data_->drop_target->Release();
            platform_data_->drop_target = nullptr;
            return false;
        }

        initialized_ = true;
        LOG_DEBUG("Native drag-drop initialized (Windows IDropTarget)");
        return true;
    }

    void NativeDragDrop::shutdown() {
        if (!initialized_)
            return;

        if (platform_data_) {
            if (platform_data_->hwnd) {
                RevokeDragDrop(platform_data_->hwnd);
            }
            if (platform_data_->drop_target) {
                platform_data_->drop_target->Release();
            }
            if (platform_data_->ole_initialized) {
                OleUninitialize();
            }
            delete platform_data_;
            platform_data_ = nullptr;
        }

        initialized_ = false;
        drag_hovering_ = false;
    }

    void NativeDragDrop::pollEvents() {
        // Windows handles events via COM callbacks, no polling needed
    }

// ============================================================================
// Linux/X11 Implementation
// ============================================================================
#elif defined(__linux__)

    struct NativeDragDrop::PlatformData {
        Display* display = nullptr;
        Window xwindow = 0;

        // XDnD atoms
        Atom xdnd_aware = 0;
        Atom xdnd_enter = 0;
        Atom xdnd_position = 0;
        Atom xdnd_status = 0;
        Atom xdnd_leave = 0;
        Atom xdnd_drop = 0;
        Atom xdnd_finished = 0;
        Atom xdnd_selection = 0;
        Atom xdnd_type_list = 0;
        Atom text_uri_list = 0;

        Window source_window = 0;
    };

    namespace {

        constexpr long kXdndEnterUseTypeList = 1;

        // SDL 3.4.2 takes the first XDND target it recognises. mutter forwards a
        // GTK4 (Nautilus) drag with text/plain;charset=utf-8 listed ahead of
        // text/uri-list, so the drop arrives as SDL_EVENT_DROP_TEXT holding a
        // bare path instead of SDL_EVENT_DROP_FILE. Rewrite XdndEnter so
        // text/uri-list is the first offered type whenever the source has it
        // (upstream fix: SDL commit 1df279a04, after 3.4.2).
        void preferUriListTarget(Display* const dpy, XClientMessageEvent& msg, const Atom type_list_atom,
                                 const Atom uri_list) {
            long* const l = msg.data.l;
            if ((l[1] & kXdndEnterUseTypeList) == 0) {
                for (int i = 3; i <= 4; ++i) {
                    if (static_cast<Atom>(l[i]) == uri_list) {
                        std::swap(l[2], l[i]);
                        break;
                    }
                }
                return;
            }

            Atom actual_type = None;
            int actual_format = 0;
            unsigned long count = 0;
            unsigned long bytes_after = 0;
            unsigned char* data = nullptr;
            if (XGetWindowProperty(dpy, static_cast<Window>(l[0]), type_list_atom, 0, 4096, False, XA_ATOM,
                                   &actual_type, &actual_format, &count, &bytes_after, &data) != Success ||
                !data) {
                return;
            }
            bool has_uri_list = false;
            if (actual_type == XA_ATOM && actual_format == 32) {
                const Atom* const atoms = reinterpret_cast<const Atom*>(data);
                for (unsigned long i = 0; i < count && !has_uri_list; ++i)
                    has_uri_list = atoms[i] == uri_list;
            }
            XFree(data);
            if (!has_uri_list)
                return;

            l[1] &= ~kXdndEnterUseTypeList;
            l[2] = static_cast<long>(uri_list);
            l[3] = 0;
            l[4] = 0;
        }

    } // namespace

    bool NativeDragDrop::init(SDL_Window* window) {
        if (initialized_)
            return true;
        window_ = window;

        platform_data_ = new PlatformData();
        SDL_PropertiesID props = SDL_GetWindowProperties(window);
        platform_data_->display =
            (Display*)SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
        platform_data_->xwindow =
            (Window)SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);

        if (!platform_data_->display || !platform_data_->xwindow) {
            LOG_ERROR("Failed to get X11 display/window");
            return false;
        }

        Display* dpy = platform_data_->display;

        // Get XDnD atoms
        platform_data_->xdnd_aware = XInternAtom(dpy, "XdndAware", False);
        platform_data_->xdnd_enter = XInternAtom(dpy, "XdndEnter", False);
        platform_data_->xdnd_position = XInternAtom(dpy, "XdndPosition", False);
        platform_data_->xdnd_status = XInternAtom(dpy, "XdndStatus", False);
        platform_data_->xdnd_leave = XInternAtom(dpy, "XdndLeave", False);
        platform_data_->xdnd_drop = XInternAtom(dpy, "XdndDrop", False);
        platform_data_->xdnd_finished = XInternAtom(dpy, "XdndFinished", False);
        platform_data_->xdnd_selection = XInternAtom(dpy, "XdndSelection", False);
        platform_data_->xdnd_type_list = XInternAtom(dpy, "XdndTypeList", False);
        platform_data_->text_uri_list = XInternAtom(dpy, "text/uri-list", False);

        // Observe XDnD traffic inside SDL's own pump. Dequeuing ClientMessages
        // ourselves (XCheckTypedEvent + XPutBackEvent) reordered them ahead of
        // the SelectionNotify SDL waits for after XdndDrop.
        SDL_SetX11EventHook(&NativeDragDrop::x11EventHook, this);

        initialized_ = true;
        LOG_DEBUG("Native drag-drop initialized (X11 XDnD)");
        return true;
    }

    void NativeDragDrop::shutdown() {
        if (!initialized_)
            return;

        SDL_SetX11EventHook(nullptr, nullptr);
        if (platform_data_) {
            delete platform_data_;
            platform_data_ = nullptr;
        }

        initialized_ = false;
        drag_hovering_ = false;
    }

    void NativeDragDrop::pollEvents() {
        // XDnD state is tracked from SDL's event hook, nothing to poll.
    }

    bool NativeDragDrop::x11EventHook(void* userdata, XEvent* xevent) {
        auto* const self = static_cast<NativeDragDrop*>(userdata);
        if (!self || !self->platform_data_ || xevent->type != ClientMessage ||
            xevent->xclient.window != self->platform_data_->xwindow) {
            return true;
        }

        PlatformData& pd = *self->platform_data_;
        XClientMessageEvent& msg = xevent->xclient;
        if (msg.message_type == pd.xdnd_enter) {
            LOG_DEBUG("XDnD enter: source=0x{:x} version={} type_list={}", static_cast<unsigned long>(msg.data.l[0]),
                      msg.data.l[1] >> 24, (msg.data.l[1] & kXdndEnterUseTypeList) != 0);
            preferUriListTarget(pd.display, msg, pd.xdnd_type_list, pd.text_uri_list);
            pd.source_window = static_cast<Window>(msg.data.l[0]);
            self->setDragHovering(true);
        } else if (msg.message_type == pd.xdnd_drop || msg.message_type == pd.xdnd_leave) {
            LOG_DEBUG("XDnD {}: source=0x{:x}", msg.message_type == pd.xdnd_drop ? "drop" : "leave",
                      static_cast<unsigned long>(msg.data.l[0]));
            self->setDragHovering(false);
            pd.source_window = 0;
        }
        return true;
    }

// ============================================================================
// Unsupported Platform
// ============================================================================
#else

    struct NativeDragDrop::PlatformData {};

    bool NativeDragDrop::init(SDL_Window* window) {
        window_ = window;
        LOG_WARN("Native drag-drop not implemented for this platform");
        return false;
    }

    void NativeDragDrop::shutdown() {
        initialized_ = false;
    }

    void NativeDragDrop::pollEvents() {}

#endif

    // ============================================================================
    // Common Implementation
    // ============================================================================

    NativeDragDrop::~NativeDragDrop() {
        shutdown();
    }

    void NativeDragDrop::setDragHovering(bool hovering) {
        if (drag_hovering_ == hovering)
            return;

        drag_hovering_ = hovering;
    }

    void NativeDragDrop::handleFileDrop(const std::vector<std::string>& paths) {
        if (on_file_drop_) {
            on_file_drop_(paths);
        }
    }

} // namespace lfs::vis::gui
