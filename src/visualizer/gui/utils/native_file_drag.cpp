/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "native_file_drag.hpp"
#include <SDL3/SDL.h>
#include <array>
#include <atomic>
#include <core/native_fs.hpp>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

#include <vector>
#ifdef __linux__
#include "native_drag_icon.hpp"
#include "native_drop_pause.hpp"
#include <X11/Xutil.h>
#include <gdk/gdkx.h>
#include <gtk/gtk.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstring>
#include <cwchar>
#include <ole2.h>
#include <shellapi.h>
#include <shlobj.h>
#include <windows.h>
#endif

namespace lfs::vis::gui {
    namespace {
        thread_local bool preserve_drag_input = false;
    }
    bool preserveNativeDragInput() { return preserve_drag_input; }
    bool mayShowTransferProgress(const NativeDragContext* context) {
#ifdef _WIN32
        (void)context;
        return !(GetAsyncKeyState(VK_LBUTTON) & 0x8000);
#elif defined(__linux__)
        const auto* driver = SDL_GetCurrentVideoDriver();
        if (driver && std::string_view(driver) == "wayland")
            return context && context->dropPerformed();
        float x = 0, y = 0;
        return !(SDL_GetGlobalMouseState(&x, &y) & SDL_BUTTON_LMASK);
#else
        (void)context;
        return false;
#endif
    }
#ifdef _WIN32
    namespace {
        template <class T>
        struct ComRelease {
            void operator()(T* value) const noexcept {
                if (value)
                    value->Release();
            }
        };
        template <class T>
        using ComPtr = std::unique_ptr<T, ComRelease<T>>;
        FileDragTarget targetFromExecutable(const wchar_t* path) noexcept {
            if (!path)
                return FileDragTarget::Unknown;
            const auto* slash = wcsrchr(path, L'\\');
            const auto* name = slash ? slash + 1 : path;
            return _wcsicmp(name, L"blender.exe") == 0 ? FileDragTarget::Blender : FileDragTarget::Unknown;
        }
        FileDragTarget targetFromWindow(HWND window) noexcept {
            DWORD pid = 0;
            if (!window || !GetWindowThreadProcessId(window, &pid) || !pid)
                return FileDragTarget::Unknown;
            constexpr DWORD capacity = 32768;
            std::unique_ptr<wchar_t[]> path(new (std::nothrow) wchar_t[capacity]);
            if (!path)
                return FileDragTarget::Unknown;
            const auto process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
            if (!process)
                return FileDragTarget::Unknown;
            DWORD length = capacity;
            const bool found = QueryFullProcessImageNameW(process, 0, path.get(), &length);
            CloseHandle(process);
            if (!found || length == 0 || length >= capacity)
                return FileDragTarget::Unknown;
            path[length] = L'\0';
            return targetFromExecutable(path.get());
        }
        struct DragTargetState {
            FileDragTarget target = FileDragTarget::Unknown;
            bool frozen = false;
        };
        class CopySource final : public IDropSource, public IDropSourceNotify {
            static HCURSOR makeIconCursor(HWND window, bool allowed) {
                const auto dpi = window ? GetDpiForWindow(window) : 96;
                const int size = MulDiv(64, dpi ? dpi : 96, 96);
                auto icon = static_cast<HICON>(LoadImageW(GetModuleHandle(nullptr), L"IDI_ICON1", IMAGE_ICON, size, size, LR_DEFAULTCOLOR));
                if (!icon) {
                    SHFILEINFOW info{};
                    if (SHGetFileInfoW(L"file", FILE_ATTRIBUTE_NORMAL, &info, sizeof(info), SHGFI_USEFILEATTRIBUTES | SHGFI_ICON | SHGFI_LARGEICON))
                        icon = info.hIcon;
                }
                if (!icon)
                    return nullptr;
                BITMAPINFO info{};
                info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                info.bmiHeader.biWidth = size;
                info.bmiHeader.biHeight = -size;
                info.bmiHeader.biPlanes = 1;
                info.bmiHeader.biBitCount = 32;
                info.bmiHeader.biCompression = BI_RGB;
                void* pixels = nullptr;
                const auto bitmap = CreateDIBSection(nullptr, &info, DIB_RGB_COLORS, &pixels, nullptr, 0);
                const auto dc = CreateCompatibleDC(nullptr);
                HCURSOR cursor = nullptr;
                if (bitmap && dc) {
                    std::memset(pixels, 0, static_cast<size_t>(size) * size * 4);
                    const auto previous = SelectObject(dc, bitmap);
                    const bool drawn = DrawIconEx(dc, 0, 0, icon, size, size, 0, nullptr, DI_NORMAL);
                    SelectObject(dc, previous);
                    auto* output = static_cast<uint32_t*>(pixels);
                    // A readable badge on the existing app icon, never an arrow
                    // rasterized by hand. Both states retain the full thumbnail.
                    for (int y = size * 5 / 8; y < size; ++y)
                        for (int x = size * 5 / 8; x < size; ++x) {
                            const int dx = (x - size * 53 / 64) * 64 / size;
                            const int dy = (y - size * 53 / 64) * 64 / size;
                            if (dx * dx + dy * dy > 100)
                                continue;
                            const bool mark = allowed ? (std::abs(dx) <= 1 && std::abs(dy) <= 6) || (std::abs(dy) <= 1 && std::abs(dx) <= 6)
                                                      : std::abs(dx + dy) <= 1 && std::abs(dx) <= 7;
                            output[y * size + x] = mark ? 0xffffffff : allowed ? 0xff28843b
                                                                               : 0xffb82b32;
                        }
                    const auto mask = CreateBitmap(size, size, 1, 1, nullptr);
                    if (mask) {
                        const auto old_mask = SelectObject(dc, mask);
                        const bool cleared = PatBlt(dc, 0, 0, size, size, BLACKNESS);
                        SelectObject(dc, old_mask);
                        if (drawn && cleared) {
                            ICONINFO cursor_info{FALSE, 0, 0, mask, bitmap};
                            cursor = static_cast<HCURSOR>(CreateIconIndirect(&cursor_info));
                        }
                    }
                    if (mask)
                        DeleteObject(mask);
                }
                if (dc)
                    DeleteDC(dc);
                if (bitmap)
                    DeleteObject(bitmap);
                DestroyIcon(icon);
                return cursor;
            }
            HCURSOR previous_cursor_ = GetCursor();
            HCURSOR copy_cursor_;
            HCURSOR forbidden_cursor_;
            std::shared_ptr<DragTargetState> target_;

        public:
            explicit CopySource(HWND window = nullptr, std::shared_ptr<DragTargetState> target = {})
                : copy_cursor_(makeIconCursor(window, true)), forbidden_cursor_(makeIconCursor(window, false)), target_(std::move(target)) {}
            ~CopySource() {
                if (GetCursor() == copy_cursor_ || GetCursor() == forbidden_cursor_)
                    SetCursor(previous_cursor_);
                if (copy_cursor_)
                    DestroyCursor(copy_cursor_);
                if (forbidden_cursor_)
                    DestroyCursor(forbidden_cursor_);
            }
#ifdef LFS_NATIVE_DRAG_TESTS
            bool readablePreview() const {
                ICONINFO info{};
                if (!copy_cursor_ || !forbidden_cursor_ || !GetIconInfo(copy_cursor_, &info))
                    return false;
                BITMAP bitmap{};
                const bool readable = GetObject(info.hbmColor, sizeof(bitmap), &bitmap) && bitmap.bmWidth >= 64 && bitmap.bmHeight >= 64;
                DeleteObject(info.hbmColor);
                DeleteObject(info.hbmMask);
                return readable;
            }
#endif
            HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id, void** out) override {
                if (!out)
                    return E_POINTER;
                *out = nullptr;
                if (id == IID_IUnknown || id == IID_IDropSource)
                    *out = static_cast<IDropSource*>(this);
                else if (id == IID_IDropSourceNotify)
                    *out = static_cast<IDropSourceNotify*>(this);
                else
                    return E_NOINTERFACE;
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
                if (!(keys & MK_LBUTTON) && target_)
                    target_->frozen = true;
                return (keys & MK_LBUTTON) ? S_OK : DRAGDROP_S_DROP;
            }
            HRESULT STDMETHODCALLTYPE DragEnterTarget(HWND window) override {
                if (target_ && !target_->frozen)
                    target_->target = targetFromWindow(window);
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE DragLeaveTarget() override {
                if (target_ && !target_->frozen)
                    target_->target = FileDragTarget::Unknown;
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE GiveFeedback(DWORD effect) override {
                const auto cursor = effect & DROPEFFECT_COPY ? copy_cursor_ : forbidden_cursor_;
                if (!cursor)
                    return DRAGDROP_S_USEDEFAULTCURSORS;
                SetCursor(cursor);
                return S_OK;
            }

        private:
            std::atomic<ULONG> refs_{1};
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
        class DeferredData final : public IDataObject {
            std::atomic<ULONG> refs_{1};
            FileProvider provider_;
            std::array<ComPtr<IDataObject>, 2> data_;
            std::shared_ptr<DragTargetState> target_;
            bool failed_ = false;
            bool cancelled_ = false;
            bool preparing_ = false;

            size_t variant() const { return target_ && target_->target == FileDragTarget::Blender ? 1u : 0u; }

        public:
            bool failed() const { return failed_; }
            bool prepared() const { return bool(data_[variant()]); }
            void seal() noexcept { provider_ = {}; }
            void cancel() noexcept {
                cancelled_ = true;
                provider_ = {};
                for (auto& data : data_)
                    data.reset();
            }
            explicit DeferredData(FileProvider provider, std::shared_ptr<DragTargetState> target = {})
                : provider_(std::move(provider)), target_(std::move(target)) {}
            HRESULT STDMETHODCALLTYPE QueryInterface(REFIID id, void** out) override {
                if (!out)
                    return E_POINTER;
                *out = nullptr;
                if (id != IID_IUnknown && id != IID_IDataObject)
                    return E_NOINTERFACE;
                *out = static_cast<IDataObject*>(this);
                AddRef();
                return S_OK;
            }
            ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
            ULONG STDMETHODCALLTYPE Release() override {
                auto n = --refs_;
                if (!n)
                    delete this;
                return n;
            }
            HRESULT STDMETHODCALLTYPE QueryGetData(FORMATETC* f) override {
                if (!f)
                    return E_POINTER;
                return f->cfFormat == CF_HDROP && f->dwAspect == DVASPECT_CONTENT && (f->tymed & TYMED_HGLOBAL) && f->lindex == -1 ? S_OK : DV_E_FORMATETC;
            }
            HRESULT STDMETHODCALLTYPE GetData(FORMATETC* f, STGMEDIUM* out) override {
                if (!out)
                    return E_POINTER;
                *out = {};
                if (cancelled_)
                    return E_ABORT;
                if (FAILED(QueryGetData(f)))
                    return DV_E_FORMATETC;
                if (preparing_)
                    return E_PENDING;
                if (failed_)
                    return E_FAIL;
                const auto key = variant();
                try {
                    if (!data_[key]) {
                        if (!provider_)
                            return E_FAIL;
                        struct PreparationScope {
                            bool& active;
                            explicit PreparationScope(bool& value) : active(value) { active = true; }
                            ~PreparationScope() { active = false; }
                        } preparation(preparing_);
                        struct InputScope {
                            bool previous = preserve_drag_input;
                            explicit InputScope(bool preserve) { preserve_drag_input = previous || preserve; }
                            ~InputScope() { preserve_drag_input = previous; }
                        } input_scope(target_ && !target_->frozen);
                        const auto files = provider_(key ? FileDragTarget::Blender : FileDragTarget::Unknown);
                        IDataObject* result = nullptr;
                        const auto hr = createData(files, &result);
                        if (FAILED(hr)) {
                            failed_ = true;
                            return hr;
                        }
                        data_[key].reset(result);
                    }
                    const auto hr = data_[key]->GetData(f, out);
                    if (FAILED(hr))
                        failed_ = true;
                    return hr;
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): HRESULT is the COM failure contract; never unwind into OLE.
                    failed_ = true;
                    return E_FAIL;
                }
            }
            HRESULT STDMETHODCALLTYPE EnumFormatEtc(DWORD direction, IEnumFORMATETC** out) override {
                if (!out)
                    return E_POINTER;
                *out = nullptr;
                if (direction != DATADIR_GET)
                    return E_NOTIMPL;
                FORMATETC f{CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL};
                return SHCreateStdEnumFmtEtc(1, &f, out);
            }
            HRESULT STDMETHODCALLTYPE GetDataHere(FORMATETC*, STGMEDIUM*) override { return DATA_E_FORMATETC; }
            HRESULT STDMETHODCALLTYPE GetCanonicalFormatEtc(FORMATETC*, FORMATETC* out) override {
                if (!out)
                    return E_POINTER;
                out->ptd = nullptr;
                return DATA_S_SAMEFORMATETC;
            }
            HRESULT STDMETHODCALLTYPE SetData(FORMATETC*, STGMEDIUM*, BOOL) override { return E_NOTIMPL; }
            HRESULT STDMETHODCALLTYPE DAdvise(FORMATETC*, DWORD, IAdviseSink*, DWORD*) override { return OLE_E_ADVISENOTSUPPORTED; }
            HRESULT STDMETHODCALLTYPE DUnadvise(DWORD) override { return OLE_E_ADVISENOTSUPPORTED; }
            HRESULT STDMETHODCALLTYPE EnumDAdvise(IEnumSTATDATA** out) override {
                if (out)
                    *out = nullptr;
                return OLE_E_ADVISENOTSUPPORTED;
            }
        };
    } // namespace
#endif

#ifdef __linux__
    namespace {
        FileDragTarget targetFromX11Window(GdkWindow* window) noexcept {
            if (!window || gdk_window_is_destroyed(window))
                return FileDragTarget::Unknown;
            auto* display = gdk_window_get_display(window);
            if (!GDK_IS_X11_DISPLAY(display))
                return FileDragTarget::Unknown;
            auto* x11 = gdk_x11_display_get_xdisplay(display);
            Window current = gdk_x11_window_get_xid(window);
            FileDragTarget target = FileDragTarget::Unknown;
            gdk_x11_display_error_trap_push(display);
            for (int level = 0; current && level < 16; ++level) {
                XClassHint hint{};
                const bool found = XGetClassHint(x11, current, &hint) && hint.res_class &&
                                   g_ascii_strcasecmp(hint.res_class, "Blender") == 0;
                if (hint.res_name)
                    XFree(hint.res_name);
                if (hint.res_class)
                    XFree(hint.res_class);
                if (found) {
                    target = FileDragTarget::Blender;
                    break;
                }
                Window root = 0, parent = 0, *children = nullptr;
                unsigned int count = 0;
                const bool queried = XQueryTree(x11, current, &root, &parent, &children, &count);
                if (children)
                    XFree(children);
                if (!queried || current == root || parent == current)
                    break;
                current = parent;
            }
            return gdk_x11_display_error_trap_pop(display) ? FileDragTarget::Unknown : target;
        }

        class X11FileData {
            FileProvider provider_;
            std::array<std::vector<std::string>, 2> uris_;
            std::array<std::vector<gchar*>, 2> pointers_;

        public:
            explicit X11FileData(FileProvider provider) : provider_(std::move(provider)) {}
            X11FileData(const X11FileData&) = delete;
            X11FileData& operator=(const X11FileData&) = delete;
            std::vector<gchar*>& get(FileDragTarget target) {
                const size_t key = target == FileDragTarget::Blender ? 1u : 0u;
                auto& pointers = pointers_[key];
                if (pointers.empty()) {
                    std::vector<std::string> uris;
                    for (const auto& path : provider_(target)) {
                        std::unique_ptr<gchar, decltype(&g_free)> uri(g_filename_to_uri(path.c_str(), nullptr, nullptr), g_free);
                        if (!uri)
                            throw std::runtime_error("Cannot encode file URI");
                        uris.emplace_back(uri.get());
                    }
                    if (uris.empty())
                        throw std::runtime_error("No files prepared");
                    uris_[key] = std::move(uris);
                    pointers.reserve(uris_[key].size() + 1);
                    for (auto& uri : uris_[key])
                        pointers.push_back(uri.data());
                    pointers.push_back(nullptr);
                }
                return pointers;
            }
        };
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

    FileDragResult dragReadyFiles(std::span<const std::filesystem::path> files, SDL_Window* window, NativeDragContext* context) {
        if (files.empty())
            return {FileDragStatus::Failed, "No files prepared"};
        std::vector<std::filesystem::path> owned(files.begin(), files.end());
        for (const auto& file : owned) {
            std::error_code ec;
            if (!core::native_fs::is_regular_file(file, ec) || ec)
                return {FileDragStatus::Failed, "Missing committed file"};
        }
        return dragFiles([owned = std::move(owned)](FileDragTarget) { return owned; }, window, context);
    }
    FileDragResult dragFiles(FileProvider provider, SDL_Window* window, NativeDragContext* context) {
        if (!provider)
            return {FileDragStatus::Failed, "Missing file provider"};
        auto resolve = [provider = std::move(provider), cached = std::array<std::optional<std::vector<std::filesystem::path>>, 2>{}](FileDragTarget target) mutable -> const std::vector<std::filesystem::path>& {
            auto& entry = cached[target == FileDragTarget::Blender ? 1u : 0u];
            if (!entry) {
                auto files = provider(target);
                if (files.empty())
                    throw std::runtime_error("No files prepared");
                for (auto& file : files) {
                    std::error_code ec;
                    file = std::filesystem::absolute(file, ec);
                    if (ec || !core::native_fs::is_regular_file(file, ec) || ec)
                        throw std::runtime_error("Missing committed file");
                }
                entry = std::move(files);
            }
            return *entry;
        };
#ifdef _WIN32
        (void)window;
        (void)context;
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
        auto target = std::make_shared<DragTargetState>();
        auto* deferred = new DeferredData([resolve = std::move(resolve)](FileDragTarget destination) mutable { return resolve(destination); }, target);
        ComPtr<IDataObject> owned_data(deferred);
        const auto hwnd = static_cast<HWND>(SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
        ComPtr<IDropSource> source(new CopySource(hwnd, target));
        DWORD effect = DROPEFFECT_NONE;
        // SDL and GHOST can hold automatic capture while the button is down.
        // Hand it to OLE explicitly so the pointer can enter another app.
        ReleaseCapture();
        const auto hr = DoDragDrop(owned_data.get(), source.get(), DROPEFFECT_COPY, &effect);
        if (hr == DRAGDROP_S_DROP && (effect & DROPEFFECT_COPY) && !deferred->failed() && !deferred->prepared()) {
            // An OLE receiver may retain IDataObject and defer all reads. Resolve
            // before leaving this completed-frame action so a later GetData can
            // only read the cache, never try to advance a live render callback.
            FORMATETC format{CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL};
            STGMEDIUM medium{};
            if (SUCCEEDED(deferred->GetData(&format, &medium)))
                ReleaseStgMedium(&medium);
        }
        if (hr != DRAGDROP_S_DROP || !(effect & DROPEFFECT_COPY) || deferred->failed())
            deferred->cancel();
        else
            deferred->seal();
        if (deferred->failed())
            return {FileDragStatus::Failed, "Deferred file preparation failed"};
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
        NativeDropPause pause;
        if (driver && std::string_view(driver) == "wayland" && context)
            return context->drag([resolve = std::move(resolve)](FileDragTarget destination) mutable { return resolve(destination); }, window);
        if (!window || !driver || std::string_view(driver) != "x11")
            return {FileDragStatus::Unsupported, "Splat drag currently requires Linux X11 / XWayland (SDL_VIDEODRIVER=x11)"};
        float mouse_x, mouse_y;
        if (!(SDL_GetGlobalMouseState(&mouse_x, &mouse_y) & SDL_BUTTON_LMASK))
            return {FileDragStatus::Cancelled, {}};
        if (!gtk_init_check(nullptr, nullptr))
            return {FileDragStatus::Failed, "Cannot initialize GTK for file drag"};
        auto* x11 = static_cast<Display*>(SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr));
        if (!x11)
            return {FileDragStatus::Failed, "SDL X11 display unavailable"};
        // The press that started this gesture left an implicit pointer grab on SDL's connection. GTK
        // grabs the pointer on its own connection to run the drag, which fails while SDL holds it.
        XUngrabPointer(x11, CurrentTime);
        XFlush(x11);
        auto* display = gdk_display_get_default();
        bool close_display = false;
        if (!GDK_IS_X11_DISPLAY(display) || std::string_view(DisplayString(gdk_x11_display_get_xdisplay(display))) != DisplayString(x11)) {
            // A Wayland desktop may give GTK a different default backend than
            // SDL. Open only our matching X11 connection; never replace GTK's
            // global default display or rewrite GDK_BACKEND for other dialogs.
            display = gdk_display_open(DisplayString(x11));
            close_display = display && display != gdk_display_get_default();
        }
        struct DisplayScope {
            GdkDisplay* display;
            bool owned;
            ~DisplayScope() {
                if (display && owned)
                    gdk_display_close(display);
            }
        } display_scope{display, close_display};
        if (!display || !GDK_IS_X11_DISPLAY(display))
            return {FileDragStatus::Unsupported, "GTK cannot open SDL's X11 display; allow X11 in GDK_BACKEND"};
        static thread_local bool active = false;
        if (active)
            return {FileDragStatus::Failed, "A file drag is already active"};
        struct ActiveScope {
            bool& flag;
            explicit ActiveScope(bool& f) : flag(f) { flag = true; }
            ~ActiveScope() { flag = false; }
        } active_scope(active);
        struct State {
            bool done = false;
            bool failed = false;
            bool provider_failed = false;
            bool preparing = false;
            X11FileData files;
        } state{false, false, false, false, X11FileData([resolve = std::move(resolve)](FileDragTarget destination) mutable { return resolve(destination); })};
        auto* widget = gtk_invisible_new_for_screen(gdk_display_get_default_screen(display));
        if (!widget)
            return {FileDragStatus::Failed, "Cannot create GTK drag source"};
        struct WidgetScope {
            GtkWidget* widget;
            ~WidgetScope() { gtk_widget_destroy(widget); }
        } widget_scope{widget};
        gtk_widget_realize(widget);
        gtk_widget_show(widget);
        g_signal_connect(widget, "drag-data-get", G_CALLBACK((+[](GtkWidget*, GdkDragContext* drag_context, GtkSelectionData* selection, guint, guint, gpointer data) {
                             auto& s = *static_cast<State*>(data);
                             if (s.failed || s.provider_failed || s.preparing)
                                 return;
                             struct PreparationScope {
                                 bool& active;
                                 explicit PreparationScope(bool& value) : active(value) { active = true; }
                                 ~PreparationScope() { active = false; }
                             } preparation(s.preparing);
                             try {
                                 const auto target = targetFromX11Window(gdk_drag_context_get_dest_window(drag_context));
                                 gtk_selection_data_set_uris(selection, s.files.get(target).data());
                             } catch (...) {
                                 // LFS-CENSUS-OK(empty-catch): GTK receives an empty failed selection; never unwind across C.
                                 s.failed = true;
                                 s.provider_failed = true;
                             }
                         })),
                         &state);
        g_signal_connect(widget, "drag-failed", G_CALLBACK((+[](GtkWidget*, GdkDragContext*, GtkDragResult, gpointer data) -> gboolean {
                             static_cast<State*>(data)->failed = true;
                             return FALSE;
                         })),
                         &state);
        g_signal_connect(widget, "drag-end", G_CALLBACK((+[](GtkWidget*, GdkDragContext*, gpointer data) {
                             static_cast<State*>(data)->done = true;
                         })),
                         &state);
        auto* targets = gtk_target_list_new(nullptr, 0);
        gtk_target_list_add_uri_targets(targets, 0);
        auto* drag = gtk_drag_begin_with_coordinates(widget, targets, GDK_ACTION_COPY, 1, nullptr, -1, -1);
        gtk_target_list_unref(targets);
        if (drag) {
            if (auto* icon = loadNativeDragIcon(64)) {
                gtk_drag_set_icon_pixbuf(drag, icon, -12, -12);
                g_object_unref(icon);
            } else
                gtk_drag_set_icon_name(drag, "text-x-generic", 0, 0);
            struct Timeout {
                State* state;
                GdkDragContext* drag;
            } timeout{&state, drag};
            const auto timer = g_timeout_add_seconds(120, +[](gpointer value) -> gboolean {
                auto& timeout = *static_cast<Timeout*>(value);
                timeout.state->failed = true;
                gtk_drag_cancel(timeout.drag);
                timeout.state->done = true;
                return G_SOURCE_CONTINUE; // Removed by scope, including this path.
            },
                                                     &timeout);
            struct TimerScope {
                guint id;
                ~TimerScope() {
                    if (id)
                        g_source_remove(id);
                }
            } timer_scope{timer};
            while (!state.done)
                g_main_context_iteration(nullptr, TRUE);
        }
        if (!drag)
            return {FileDragStatus::Failed, "Cannot start GTK file drag"};
        if (state.provider_failed)
            return {FileDragStatus::Failed, "Deferred file preparation failed"};
        return {state.failed ? FileDragStatus::Cancelled : FileDragStatus::Copied, {}};
#else
        (void)window;
        (void)context;
        (void)resolve;
        return {FileDragStatus::Unsupported, "Native drag source is not implemented for this platform"};
#endif
    }

#if defined(__linux__) && defined(LFS_NATIVE_DRAG_TESTS)
    int nativeX11DragContract() {
        if (!gtk_init_check(nullptr, nullptr))
            return 77;
        auto* display = gdk_display_get_default();
        if (!GDK_IS_X11_DISPLAY(display))
            return 77;
        auto* x11 = gdk_x11_display_get_xdisplay(display);
        const auto window = XCreateSimpleWindow(x11, DefaultRootWindow(x11), 0, 0, 8, 8, 0, 0, 0);
        const auto child = XCreateSimpleWindow(x11, window, 0, 0, 4, 4, 0, 0, 0);
        XSync(x11, False);
        auto* parent_wrapper = gdk_x11_window_foreign_new_for_display(display, window);
        auto* child_wrapper = gdk_x11_window_foreign_new_for_display(display, child);
        const auto classify = [&](const char* name) {
            XClassHint hint{const_cast<char*>("blender"), const_cast<char*>(name)};
            XSetClassHint(x11, window, &hint);
            XSync(x11, False);
            return targetFromX11Window(parent_wrapper);
        };
        bool valid = targetFromX11Window(nullptr) == FileDragTarget::Unknown;
        for (int i = 0; i < 100; ++i) {
            valid = valid && classify("Blender") == FileDragTarget::Blender;
            valid = valid && targetFromX11Window(child_wrapper) == FileDragTarget::Blender;
            valid = valid && classify("bLeNdEr") == FileDragTarget::Blender;
            valid = valid && classify("MyBlender") == FileDragTarget::Unknown;
            valid = valid && classify("Blender-helper") == FileDragTarget::Unknown;
            valid = valid && classify("Nautilus") == FileDragTarget::Unknown;
        }
        XDeleteProperty(x11, window, XInternAtom(x11, "WM_CLASS", False));
        XSync(x11, False);
        valid = valid && targetFromX11Window(parent_wrapper) == FileDragTarget::Unknown;
        XDestroyWindow(x11, child);
        XSync(x11, False);
        valid = valid && targetFromX11Window(child_wrapper) == FileDragTarget::Unknown;
        g_object_unref(child_wrapper);
        g_object_unref(parent_wrapper);
        XDestroyWindow(x11, window);
        XSync(x11, False);
        if (!valid)
            return 1;
        std::array<int, 2> calls{};
        const auto original = std::filesystem::path("/tmp/drag original #.ply");
        const auto blender = std::filesystem::path("/tmp/drag blender +.ply");
        X11FileData files([&](FileDragTarget target) {
            ++calls[target == FileDragTarget::Blender ? 1 : 0];
            return std::vector<std::filesystem::path>{target == FileDragTarget::Blender ? blender : original};
        });
        auto* original_uri = files.get(FileDragTarget::Unknown).front();
        const std::string stable_uri = original_uri;
        auto* blender_uri = files.get(FileDragTarget::Blender).front();
        std::unique_ptr<gchar, decltype(&g_free)> decoded(g_filename_from_uri(blender_uri, nullptr, nullptr), g_free);
        if (!decoded || std::filesystem::path(decoded.get()) != blender ||
            files.get(FileDragTarget::Unknown).front() != original_uri ||
            stable_uri != original_uri || calls != std::array<int, 2>{1, 1})
            return 2;
        bool rejected = false;
        try {
            X11FileData empty([](FileDragTarget) { return std::vector<std::filesystem::path>{}; });
            empty.get(FileDragTarget::Unknown);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        return rejected ? 0 : 3;
    }
#endif

#if defined(_WIN32) && defined(LFS_NATIVE_DRAG_TESTS)
    int nativeDragDataContract(const std::filesystem::path& path) {
        OleScope ole;
        if (FAILED(ole.result))
            return 1;
        // Explorer can read before Drop. The provider renders nested frames,
        // which must not pump away the native release OLE is waiting for.
        for (const bool frozen : {false, true}) {
            for (const bool fail : {false, true}) {
                auto target = std::make_shared<DragTargetState>();
                target->frozen = frozen;
                bool saw_expected_input_policy = false;
                ComPtr<IDataObject> data(new DeferredData([&](FileDragTarget) {
                    saw_expected_input_policy = preserveNativeDragInput() == !frozen;
                    if (fail)
                        throw std::runtime_error("Synthetic hover-time export failure");
                    return std::vector<std::filesystem::path>{path};
                },
                                                          target));
                FORMATETC format{CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL};
                STGMEDIUM storage{};
                const auto result = data->GetData(&format, &storage);
                if (SUCCEEDED(result))
                    ReleaseStgMedium(&storage);
                if (!saw_expected_input_policy || preserveNativeDragInput() || (FAILED(result) != fail))
                    return 27;
            }
        }
        if (targetFromExecutable(L"C:\\Apps\\BLENDER.EXE") != FileDragTarget::Blender ||
            targetFromExecutable(L"C:\\blender.exe\\other.exe") != FileDragTarget::Unknown ||
            targetFromExecutable(L"C:\\Apps\\my-blender.exe") != FileDragTarget::Unknown ||
            targetFromExecutable(L"C:\\Apps\\blender.exe.old") != FileDragTarget::Unknown ||
            targetFromWindow(nullptr) != FileDragTarget::Unknown ||
            targetFromWindow(GetDesktopWindow()) != FileDragTarget::Unknown)
            return 20;
        {
            auto target = std::make_shared<DragTargetState>();
            auto* source_impl = new CopySource(nullptr, target);
            ComPtr<IDropSource> source(source_impl);
            IDropSourceNotify* raw_notify = nullptr;
            if (FAILED(source->QueryInterface(IID_IDropSourceNotify, reinterpret_cast<void**>(&raw_notify))))
                return 21;
            ComPtr<IDropSourceNotify> notify(raw_notify);
            std::array<int, 2> calls{};
            auto variant_path = std::filesystem::path(path.native() + L".blender");
            auto* delayed = new DeferredData([&](FileDragTarget destination) {
                ++calls[destination == FileDragTarget::Blender ? 1 : 0];
                return std::vector<std::filesystem::path>{destination == FileDragTarget::Blender ? variant_path : path};
            },
                                             target);
            ComPtr<IDataObject> data(delayed);
            FORMATETC format{CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL};
            const auto read = [&](const std::filesystem::path& expected) {
                STGMEDIUM storage{};
                if (FAILED(data->GetData(&format, &storage)))
                    return false;
                const auto length = DragQueryFileW(static_cast<HDROP>(storage.hGlobal), 0, nullptr, 0);
                std::vector<wchar_t> name(length + 1);
                const bool decoded = DragQueryFileW(static_cast<HDROP>(storage.hGlobal), 0, name.data(), static_cast<UINT>(name.size())) == length;
                const bool equal = decoded && std::filesystem::path(name.data()) == expected;
                ReleaseStgMedium(&storage);
                return equal;
            };
            if (!read(path))
                return 22;
            target->target = FileDragTarget::Blender;
            if (!read(variant_path))
                return 23;
            notify->DragLeaveTarget();
            if (!read(path) || calls != std::array<int, 2>{1, 1})
                return 24;
            target->target = FileDragTarget::Blender;
            if (source->QueryContinueDrag(FALSE, 0) != DRAGDROP_S_DROP)
                return 25;
            notify->DragLeaveTarget();
            notify->DragEnterTarget(GetDesktopWindow());
            delayed->seal();
            if (target->target != FileDragTarget::Blender || !read(variant_path) || calls != std::array<int, 2>{1, 1})
                return 26;
        }
        for (int i = 0; i < 1000; ++i) {
            auto* preview = new CopySource();
            ComPtr<IDropSource> source(preview);
            if (!preview->readablePreview())
                return 15;
            int calls = 0;
            auto* delayed = new DeferredData([&](FileDragTarget) {
                ++calls;
                if (i % 4 == 3)
                    throw std::runtime_error("Synthetic provider failure");
                return std::vector<std::filesystem::path>{path};
            });
            ComPtr<IDataObject> data(delayed);
            FORMATETC format{CF_HDROP, nullptr, DVASPECT_CONTENT, -1, TYMED_HGLOBAL};
            if (FAILED(data->QueryGetData(&format)) || calls)
                return 2;
            IEnumFORMATETC* enumeration = nullptr;
            if (FAILED(data->EnumFormatEtc(DATADIR_GET, &enumeration)))
                return 3;
            ComPtr<IEnumFORMATETC> formats(enumeration);
            FORMATETC item{};
            ULONG count = 0;
            if (formats->Next(1, &item, &count) != S_OK || count != 1 || item.cfFormat != CF_HDROP || calls)
                return 4;
            if (item.ptd)
                CoTaskMemFree(item.ptd);
            if (i % 4 == 1) {
                delayed->cancel();
                STGMEDIUM cancelled{};
                if (SUCCEEDED(data->GetData(&format, &cancelled)) || calls)
                    return 14;
                continue; // A retained object must not generate after cancellation.
            }
            STGMEDIUM storage{};
            const auto hr = data->GetData(&format, &storage);
            if (i % 4 == 3) {
                if (SUCCEEDED(hr) || calls != 1)
                    return 5;
                if (SUCCEEDED(data->GetData(&format, &storage)) || calls != 1)
                    return 6;
                continue;
            }
            if (FAILED(hr) || calls != 1)
                return 7;
            if (DragQueryFileW(static_cast<HDROP>(storage.hGlobal), 0xFFFFFFFF, nullptr, 0) != 1)
                return 8;
            STGMEDIUM second{};
            if (FAILED(data->GetData(&format, &second)) || calls != 1)
                return 9;
            ReleaseStgMedium(&storage);
            const auto length = DragQueryFileW(static_cast<HDROP>(second.hGlobal), 0, nullptr, 0);
            std::vector<wchar_t> decoded(length + 1);
            if (DragQueryFileW(static_cast<HDROP>(second.hGlobal), 0, decoded.data(), static_cast<UINT>(decoded.size())) != length)
                return 12;
            if (std::wstring(decoded.data(), length) != path.native())
                return 13;
            ReleaseStgMedium(&second);
        }
        return 0;
    }
#endif
} // namespace lfs::vis::gui
