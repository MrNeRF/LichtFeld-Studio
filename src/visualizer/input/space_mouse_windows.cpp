/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "space_mouse_device.hpp"
#include <SDL3/SDL.h>
#include <SDL3/SDL_system.h>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>

namespace lfs::vis::input {
    namespace {
        class RawInputSource final : public SpaceMouseSource {
        public:
            explicit RawInputSource(SDL_Window* window) {
                if (!window)
                    return;
                window_ = static_cast<HWND>(SDL_GetPointerProperty(
                    SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
                if (!window_)
                    return;
                const RAWINPUTDEVICE device{1, 8, RIDEV_DEVNOTIFY, window_};
                registered_ = RegisterRawInputDevices(&device, 1, sizeof(device)) != FALSE;
                if (registered_)
                    SDL_SetWindowsMessageHook(messageHook, this);
            }

            ~RawInputSource() override {
                if (registered_) {
                    SDL_SetWindowsMessageHook(nullptr, nullptr);
                    const RAWINPUTDEVICE device{1, 8, RIDEV_REMOVE, nullptr};
                    RegisterRawInputDevices(&device, 1, sizeof(device));
                }
            }

            std::optional<SpaceMouseConnection> open() override {
                if (!registered_)
                    return std::nullopt;
                UINT count = 0;
                if (GetRawInputDeviceList(nullptr, &count, sizeof(RAWINPUTDEVICELIST)) != 0)
                    return std::nullopt;
                std::vector<RAWINPUTDEVICELIST> devices(count);
                const UINT found = GetRawInputDeviceList(devices.data(), &count, sizeof(RAWINPUTDEVICELIST));
                if (found == UINT(-1))
                    return std::nullopt;
                for (UINT i = 0; i < found; ++i) {
                    if (devices[i].dwType != RIM_TYPEHID)
                        continue;
                    RID_DEVICE_INFO info{};
                    info.cbSize = sizeof(info);
                    UINT size = sizeof(info);
                    if (GetRawInputDeviceInfoW(devices[i].hDevice, RIDI_DEVICEINFO, &info, &size) == UINT(-1) ||
                        info.hid.usUsagePage != 1 || info.hid.usUsage != 8)
                        continue;
                    const auto* profile = spaceMouseProfile(static_cast<std::uint16_t>(info.hid.dwVendorId),
                                                            static_cast<std::uint16_t>(info.hid.dwProductId));
                    const std::scoped_lock lock(mutex_);
                    selected_ = devices[i].hDevice;
                    full_scale_ = profile ? profile->full_scale : 350.0f;
                    disconnected_ = false;
                    reports_.clear();
                    return SpaceMouseConnection{profile ? profile->name : "HID " + std::to_string(info.hid.dwVendorId) + ":" + std::to_string(info.hid.dwProductId), "Raw Input"};
                }
                return std::nullopt;
            }

            SpaceMouseRead read(SpaceMouseReport& report) override {
                std::unique_lock lock(mutex_);
                ready_.wait_for(lock, std::chrono::milliseconds(20), [this] { return disconnected_ || !reports_.empty(); });
                if (disconnected_)
                    return SpaceMouseRead::Disconnected;
                if (reports_.empty())
                    return SpaceMouseRead::Timeout;
                report = reports_.front();
                reports_.pop_front();
                return SpaceMouseRead::Report;
            }

            void close() override {
                const std::scoped_lock lock(mutex_);
                selected_ = nullptr;
                reports_.clear();
            }

        private:
            static bool SDLCALL messageHook(void* context, MSG* message) {
                static_cast<RawInputSource*>(context)->handleMessage(*message);
                // SDL/DefWindowProc must still perform WM_INPUT cleanup.
                return true;
            }

            void handleMessage(const MSG& message) {
                if (message.hwnd != window_)
                    return;
                const std::scoped_lock lock(mutex_);
                if (!selected_)
                    return;
                if (message.message == WM_INPUT_DEVICE_CHANGE && message.wParam == GIDC_REMOVAL &&
                    reinterpret_cast<HANDLE>(message.lParam) == selected_) {
                    disconnected_ = true;
                    ready_.notify_one();
                    return;
                }
                if (message.message != WM_INPUT)
                    return;
                const auto handle = reinterpret_cast<HRAWINPUT>(message.lParam);
                UINT size = 0;
                constexpr std::size_t payload_offset = offsetof(RAWINPUT, data) + offsetof(RAWHID, bRawData);
                if (GetRawInputData(handle, RID_INPUT, nullptr, &size, sizeof(RAWINPUTHEADER)) != 0 ||
                    size <= payload_offset || size > 65536)
                    return;
                std::vector<std::max_align_t> storage((size + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
                if (GetRawInputData(handle, RID_INPUT, storage.data(), &size, sizeof(RAWINPUTHEADER)) != size)
                    return;
                const auto& raw = *reinterpret_cast<const RAWINPUT*>(storage.data());
                if (raw.header.dwType != RIM_TYPEHID || raw.header.hDevice != selected_)
                    return;
                const auto& hid = raw.data.hid;
                const auto* bytes = reinterpret_cast<const std::uint8_t*>(storage.data());
                const std::size_t offset = hid.bRawData - bytes;
                if (offset > size || hid.dwSizeHid == 0 || hid.dwCount > (size - offset) / hid.dwSizeHid)
                    return;
                for (DWORD i = 0; i < hid.dwCount; ++i) {
                    const auto packet = std::span(hid.bRawData + std::size_t(i) * hid.dwSizeHid, hid.dwSizeHid);
                    if (const auto report = decodeSpaceMouseHid(packet, full_scale_)) {
                        if (reports_.size() >= 256) {
                            reports_.clear();
                            disconnected_ = true;
                            break;
                        }
                        reports_.push_back(*report);
                    }
                }
                ready_.notify_one();
            }

            HWND window_ = nullptr;
            bool registered_ = false;
            std::mutex mutex_;
            std::condition_variable ready_;
            HANDLE selected_ = nullptr;
            float full_scale_ = 350.0f;
            bool disconnected_ = false;
            std::deque<SpaceMouseReport> reports_;
        };
    } // namespace

    std::unique_ptr<SpaceMouseSource> createSpaceMouseRawInput(SDL_Window* window) {
        return std::make_unique<RawInputSource>(window);
    }
} // namespace lfs::vis::input
#endif
