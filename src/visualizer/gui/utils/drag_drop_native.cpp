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

        Window source_window = 0;
    };

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

        initialized_ = true;
        LOG_DEBUG("Native drag-drop initialized (X11 XDnD)");
        return true;
    }

    void NativeDragDrop::shutdown() {
        if (!initialized_)
            return;

        if (platform_data_) {
            delete platform_data_;
            platform_data_ = nullptr;
        }

        initialized_ = false;
        drag_hovering_ = false;
    }

    void NativeDragDrop::pollEvents() {
        if (!initialized_ || !platform_data_)
            return;

        Display* const dpy = platform_data_->display;
        if (XPending(dpy) <= 0)
            return;

        // Extract all ClientMessage events to find XDnD state changes.
        // XPeekEvent only checks the first event; XDnD messages may be
        // buried behind mouse/keyboard/expose events in the queue.
        XEvent event;
        std::vector<XEvent> extracted;
        while (XCheckTypedEvent(dpy, ClientMessage, &event)) {
            const Atom msg_type = event.xclient.message_type;
            if (msg_type == platform_data_->xdnd_enter && !drag_hovering_) {
                platform_data_->source_window = static_cast<Window>(event.xclient.data.l[0]);
                setDragHovering(true);
            } else if ((msg_type == platform_data_->xdnd_leave || msg_type == platform_data_->xdnd_drop) && drag_hovering_) {
                setDragHovering(false);
                platform_data_->source_window = 0;
            }
            extracted.push_back(event);
        }

        // Restore events for SDL to process (reverse order preserves queue ordering)
        for (auto it = extracted.rbegin(); it != extracted.rend(); ++it) {
            XPutBackEvent(dpy, &*it);
        }
    }

// ============================================================================
// Unsupported Platform (macOS, etc.)
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

        if (hovering && on_drag_enter_) {
            on_drag_enter_({});
        } else if (!hovering && on_drag_leave_) {
            on_drag_leave_();
        }
    }

    void NativeDragDrop::handleFileDrop(const std::vector<std::string>& paths) {
        if (on_file_drop_) {
            on_file_drop_(paths);
        }
    }

} // namespace lfs::vis::gui
