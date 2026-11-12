/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "win32_trackpad.hpp"

#ifdef _WIN32
#include "core/logger.hpp"
#include <SDL3/SDL.h>
#include <chrono>

// clang-format off: windows.h must precede the other Windows headers.
#include <windows.h>
#include <commctrl.h>
#include <directmanipulation.h>
#include <wrl/client.h>
// clang-format on

#ifndef DM_POINTERHITTEST
#define DM_POINTERHITTEST 0x0250
#endif
#endif

namespace lfs::vis {

#ifdef _WIN32

    namespace {
        using Microsoft::WRL::ComPtr;

        // The viewport is a large virtual surface reset to the identity after
        // every gesture, so content transforms read as offsets from the gesture
        // start and never reach its edges.
        constexpr float kViewportExtent = 10000.0f;
        constexpr UINT_PTR kSubclassId = 0x4C465354; // "LFST"
        // A contact that never starts a gesture (a tap) stops keeping the loop awake.
        constexpr auto kContactTimeout = std::chrono::milliseconds(250);
    } // namespace

    struct Win32Trackpad::Impl {
        class Handler;

        HWND hwnd = nullptr;
        SDL_Window* window = nullptr;
        std::function<bool()> wants_gesture;
        std::function<void()> wake;
        bool com_initialized = false;
        bool subclassed = false;
        DWORD handler_cookie = 0;
        ComPtr<IDirectManipulationManager> manager;
        ComPtr<IDirectManipulationUpdateManager> update_manager;
        ComPtr<IDirectManipulationViewport> viewport;
        ComPtr<Handler> handler;

        DIRECTMANIPULATION_STATUS status = DIRECTMANIPULATION_BUILDING;
        bool contact_pending = false;
        std::chrono::steady_clock::time_point contact_time{};

        input::TrackpadGestureTracker gesture;

        void resetGesture(IDirectManipulationViewport* target) {
            if (target)
                target->ZoomToRect(0.0f, 0.0f, kViewportExtent, kViewportExtent, FALSE);
            gesture.endGesture();
        }

        // Direct Manipulation reports physical pixels in a DPI-aware window.
        void onContentTransform(const float scale, const float x, const float y) {
            const float display_scale = window ? SDL_GetWindowDisplayScale(window) : 1.0f;
            const float to_logical = display_scale > 0.0f ? 1.0f / display_scale : 1.0f;
            gesture.update(scale, glm::vec2(x, y) * to_logical);
        }

        void onPointerHitTest(const WPARAM wparam) {
            const UINT32 pointer_id = GET_POINTERID_WPARAM(wparam);
            POINTER_INPUT_TYPE type = PT_POINTER;
            if (!GetPointerType(pointer_id, &type) || type != PT_TOUCHPAD)
                return;
            if (!viewport || !wants_gesture || !wants_gesture())
                return;
            if (FAILED(viewport->SetContact(pointer_id)))
                return;
            contact_pending = true;
            contact_time = std::chrono::steady_clock::now();
            if (wake)
                wake();
        }

        static LRESULT CALLBACK subclassProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam,
                                             UINT_PTR /*id*/, DWORD_PTR ref) {
            if (msg == DM_POINTERHITTEST)
                reinterpret_cast<Impl*>(ref)->onPointerHitTest(wparam);
            return DefSubclassProc(hwnd, msg, wparam, lparam);
        }
    };

    class Win32Trackpad::Impl::Handler final : public IDirectManipulationViewportEventHandler {
    public:
        explicit Handler(Impl* owner) : owner_(owner) {}
        void detach() { owner_ = nullptr; }

        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
            if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirectManipulationViewportEventHandler)) {
                *out = static_cast<IDirectManipulationViewportEventHandler*>(this);
                AddRef();
                return S_OK;
            }
            *out = nullptr;
            return E_NOINTERFACE;
        }
        ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&ref_count_); }
        ULONG STDMETHODCALLTYPE Release() override {
            const LONG count = InterlockedDecrement(&ref_count_);
            if (count == 0)
                delete this;
            return static_cast<ULONG>(count);
        }

        HRESULT STDMETHODCALLTYPE OnViewportStatusChanged(IDirectManipulationViewport* viewport,
                                                          DIRECTMANIPULATION_STATUS current,
                                                          DIRECTMANIPULATION_STATUS /*previous*/) override {
            if (!owner_)
                return S_OK;
            owner_->status = current;
            if (current == DIRECTMANIPULATION_RUNNING)
                owner_->contact_pending = false;
            else if (current == DIRECTMANIPULATION_READY)
                owner_->resetGesture(viewport);
            return S_OK;
        }

        HRESULT STDMETHODCALLTYPE OnViewportUpdated(IDirectManipulationViewport* /*viewport*/) override {
            return S_OK;
        }

        HRESULT STDMETHODCALLTYPE OnContentUpdated(IDirectManipulationViewport* /*viewport*/,
                                                   IDirectManipulationContent* content) override {
            if (!owner_ || !content)
                return S_OK;
            float transform[6] = {};
            if (SUCCEEDED(content->GetContentTransform(transform, ARRAYSIZE(transform))))
                owner_->onContentTransform(transform[0], transform[4], transform[5]);
            return S_OK;
        }

    private:
        Impl* owner_;
        LONG ref_count_ = 1;
    };

    Win32Trackpad::Win32Trackpad() = default;

    Win32Trackpad::~Win32Trackpad() { shutdown(); }

    bool Win32Trackpad::init(SDL_Window* window, std::function<bool()> wants_gesture,
                             std::function<void()> wake) {
        shutdown();
        const auto hwnd = static_cast<HWND>(SDL_GetPointerProperty(
            SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
        if (!hwnd)
            return false;

        auto impl = std::make_unique<Impl>();
        impl->hwnd = hwnd;
        impl->window = window;
        impl->wants_gesture = std::move(wants_gesture);
        impl->wake = std::move(wake);

        // Direct Manipulation runs on this UI thread's single-threaded apartment.
        const HRESULT com = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        impl->com_initialized = SUCCEEDED(com);

        const auto fail = [&](const char* step, const HRESULT hr) {
            LOG_WARN("Trackpad: Direct Manipulation unavailable ({} failed: 0x{:08X}); "
                     "two-finger swipes use scroll-wheel input",
                     step, static_cast<unsigned>(hr));
            impl_ = std::move(impl);
            shutdown();
            return false;
        };

        HRESULT hr = CoCreateInstance(__uuidof(DirectManipulationManager), nullptr, CLSCTX_INPROC_SERVER,
                                      IID_PPV_ARGS(&impl->manager));
        if (FAILED(hr))
            return fail("CoCreateInstance", hr);
        if (FAILED(hr = impl->manager->GetUpdateManager(IID_PPV_ARGS(&impl->update_manager))))
            return fail("GetUpdateManager", hr);
        if (FAILED(hr = impl->manager->CreateViewport(nullptr, hwnd, IID_PPV_ARGS(&impl->viewport))))
            return fail("CreateViewport", hr);
        constexpr DIRECTMANIPULATION_CONFIGURATION configuration =
            DIRECTMANIPULATION_CONFIGURATION_INTERACTION |
            DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X |
            DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y |
            DIRECTMANIPULATION_CONFIGURATION_SCALING;
        if (FAILED(hr = impl->viewport->ActivateConfiguration(configuration)))
            return fail("ActivateConfiguration", hr);
        // No DirectComposition visual is attached, so updates are pumped manually.
        if (FAILED(hr = impl->viewport->SetViewportOptions(DIRECTMANIPULATION_VIEWPORT_OPTIONS_MANUALUPDATE)))
            return fail("SetViewportOptions", hr);
        impl->handler.Attach(new Impl::Handler(impl.get()));
        if (FAILED(hr = impl->viewport->AddEventHandler(hwnd, impl->handler.Get(), &impl->handler_cookie)))
            return fail("AddEventHandler", hr);
        const RECT rect{0, 0, static_cast<LONG>(kViewportExtent), static_cast<LONG>(kViewportExtent)};
        if (FAILED(hr = impl->viewport->SetViewportRect(&rect)))
            return fail("SetViewportRect", hr);
        if (FAILED(hr = impl->manager->Activate(hwnd)))
            return fail("Activate", hr);
        if (FAILED(hr = impl->viewport->Enable()))
            return fail("Enable", hr);
        impl->resetGesture(impl->viewport.Get());
        if (!SetWindowSubclass(hwnd, &Impl::subclassProc, kSubclassId, reinterpret_cast<DWORD_PTR>(impl.get())))
            return fail("SetWindowSubclass", HRESULT_FROM_WIN32(GetLastError()));
        impl->subclassed = true;

        impl_ = std::move(impl);
        LOG_INFO("Trackpad: Direct Manipulation enabled for precision-touchpad navigation");
        return true;
    }

    void Win32Trackpad::shutdown() {
        if (!impl_)
            return;
        auto& impl = *impl_;
        if (impl.subclassed)
            RemoveWindowSubclass(impl.hwnd, &Impl::subclassProc, kSubclassId);
        if (impl.viewport) {
            impl.viewport->Stop();
            if (impl.handler_cookie)
                impl.viewport->RemoveEventHandler(impl.handler_cookie);
            impl.viewport->Abandon();
        }
        if (impl.manager)
            impl.manager->Deactivate(impl.hwnd);
        if (impl.handler)
            impl.handler->detach();
        impl.handler.Reset();
        impl.viewport.Reset();
        impl.update_manager.Reset();
        impl.manager.Reset();
        const bool uninitialize = impl.com_initialized;
        impl_.reset();
        if (uninitialize)
            CoUninitialize();
    }

    void Win32Trackpad::update() {
        if (!impl_ || !impl_->update_manager)
            return;
        auto& impl = *impl_;
        if (impl.contact_pending && std::chrono::steady_clock::now() - impl.contact_time > kContactTimeout)
            impl.contact_pending = false;
        if (active())
            impl.update_manager->Update(nullptr);
    }

    bool Win32Trackpad::active() const {
        return impl_ && (impl_->contact_pending || impl_->status == DIRECTMANIPULATION_RUNNING ||
                         impl_->status == DIRECTMANIPULATION_INERTIA);
    }

    Win32Trackpad::Motion Win32Trackpad::takeMotion() {
        if (!impl_)
            return {};
        return impl_->gesture.take();
    }

#else

    struct Win32Trackpad::Impl {};

    Win32Trackpad::Win32Trackpad() = default;
    Win32Trackpad::~Win32Trackpad() = default;
    bool Win32Trackpad::init(SDL_Window*, std::function<bool()>, std::function<void()>) { return false; }
    void Win32Trackpad::shutdown() {}
    void Win32Trackpad::update() {}
    bool Win32Trackpad::active() const { return false; }
    Win32Trackpad::Motion Win32Trackpad::takeMotion() { return {}; }

#endif

} // namespace lfs::vis
