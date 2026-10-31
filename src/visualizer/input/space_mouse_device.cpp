/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "space_mouse_device.hpp"
#include <SDL3/SDL.h>
#include <SDL3/SDL_hidapi.h>
#include <array>
#include <condition_variable>
#include <core/logger.hpp>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

#ifdef __linux__
#include <cerrno>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace lfs::vis::input {
    namespace {
        class HidSource final : public SpaceMouseSource {
        public:
            explicit HidSource(SDL_Window* window) {
#if defined(__linux__) || defined(_WIN32)
                hid_initialized_ = SDL_hid_init() == 0;
#endif
#ifdef _WIN32
                fallback_ = createSpaceMouseRawInput(window);
#else
                (void)window;
#endif
            }

            ~HidSource() override {
                close();
                if (hid_initialized_)
                    SDL_hid_exit();
            }

            std::optional<SpaceMouseConnection> open() override {
                bool found = false;
                std::string name;
                if (hid_initialized_) {
                    auto* devices = SDL_hid_enumerate(0, 0);
                    for (auto* item = devices; item; item = item->next) {
                        if (!isSpaceMouse(item->vendor_id, item->product_id, item->usage_page, item->usage))
                            continue;
                        found = true;
                        const auto* profile = spaceMouseProfile(item->vendor_id, item->product_id);
                        name = profile ? profile->name : "HID " + std::to_string(item->vendor_id) + ":" + std::to_string(item->product_id);
                        if (item->product_string) {
                            if (char* utf8 = SDL_iconv_wchar_utf8(item->product_string)) {
                                if (*utf8)
                                    name = utf8;
                                SDL_free(utf8);
                            }
                        }
                        device_ = SDL_hid_open_path(item->path);
                        if (device_) {
                            full_scale_ = profile ? profile->full_scale : 350.0f;
                            std::array<unsigned char, 4096> descriptor{};
                            const int size = SDL_hid_get_report_descriptor(device_, descriptor.data(), descriptor.size());
                            if (size > 0)
                                full_scale_ = spaceMouseHidScale(std::span(descriptor).first(static_cast<std::size_t>(size)), full_scale_);
                            break;
                        }
                    }
                    SDL_hid_free_enumeration(devices);
                }
                if (device_)
                    return SpaceMouseConnection{name, "HID"};
#ifdef __linux__
                socket_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
                if (socket_ >= 0) {
                    sockaddr_un address{};
                    address.sun_family = AF_UNIX;
                    constexpr char socket_path[] = "/var/run/spnav.sock";
                    static_assert(sizeof(socket_path) <= sizeof(address.sun_path));
                    std::memcpy(address.sun_path, socket_path, sizeof(socket_path));
                    if (::connect(socket_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0)
                        return SpaceMouseConnection{name.empty() ? "spacenavd" : name, "spacenavd"};
                    ::close(socket_);
                    socket_ = -1;
                }
#endif
                if (fallback_) {
                    if (auto connection = fallback_->open()) {
                        using_fallback_ = true;
                        return connection;
                    }
                }
                if (found && !access_warning_logged_) {
#ifdef __linux__
                    LOG_INFO("3D mouse: device found but cannot be opened; install a hidraw udev rule with TAG+=\"uaccess\" for vendors 256f/046d, or start spacenavd (see the 3D mouse setup guide)");
#elif defined(_WIN32)
                    LOG_INFO("3D mouse: device found but cannot be opened; check the 3DxWare application profile or stop 3DxWare (see the 3D mouse setup guide)");
#endif
                    access_warning_logged_ = true;
                }
                return std::nullopt;
            }

            SpaceMouseRead read(SpaceMouseReport& report) override {
                if (using_fallback_)
                    return fallback_->read(report);
                if (device_) {
                    std::array<unsigned char, 128> bytes{};
                    const int size = SDL_hid_read_timeout(device_, bytes.data(), bytes.size(), 20);
                    if (size < 0)
                        return SpaceMouseRead::Disconnected;
                    if (const auto decoded = decodeSpaceMouseHid(
                            std::span(bytes).first(static_cast<std::size_t>(size)), full_scale_)) {
                        report = *decoded;
                        return SpaceMouseRead::Report;
                    }
                }
#ifdef __linux__
                if (socket_ >= 0) {
                    pollfd descriptor{socket_, POLLIN, 0};
                    const int ready = ::poll(&descriptor, 1, 20);
                    if (ready < 0)
                        return errno == EINTR ? SpaceMouseRead::Timeout : SpaceMouseRead::Disconnected;
                    if (ready == 0)
                        return SpaceMouseRead::Timeout;
                    const auto count = ::recv(socket_, packet_.data() + packet_used_, packet_.size() - packet_used_, 0);
                    if (count <= 0) {
                        if (count < 0 && (errno == EAGAIN || errno == EINTR))
                            return SpaceMouseRead::Timeout;
                        return SpaceMouseRead::Disconnected;
                    }
                    packet_used_ += static_cast<std::size_t>(count);
                    if (packet_used_ == packet_.size()) {
                        packet_used_ = 0;
                        if (const auto decoded = decodeSpaceMouseSpnav(packet_)) {
                            report = *decoded;
                            return SpaceMouseRead::Report;
                        }
                    }
                }
#endif
                return SpaceMouseRead::Timeout;
            }

            void close() override {
                if (device_) {
                    SDL_hid_close(device_);
                    device_ = nullptr;
                }
#ifdef __linux__
                if (socket_ >= 0) {
                    ::close(socket_);
                    socket_ = -1;
                }
                packet_used_ = 0;
#endif
                if (using_fallback_) {
                    fallback_->close();
                    using_fallback_ = false;
                }
            }

        private:
            bool hid_initialized_ = false;
            bool access_warning_logged_ = false;
            SDL_hid_device* device_ = nullptr;
            float full_scale_ = 350.0f;
            std::unique_ptr<SpaceMouseSource> fallback_;
            bool using_fallback_ = false;
#ifdef __linux__
            int socket_ = -1;
            std::array<std::uint8_t, 32> packet_{};
            std::size_t packet_used_ = 0;
#endif
        };
    } // namespace

    struct SpaceMouseDevice::Impl {
        HidSource source;
        std::mutex mutex;
        SpaceMouseFrame frame;
        bool input_enabled = false;
        std::jthread reader;

        Impl(SDL_Window* window, std::function<void()> wake) : source(window) {
            reader = std::jthread([this, wake = std::move(wake)](const std::stop_token stop) {
                SpaceMouseSession session(source);
                std::optional<SpaceMouseConnection> last_connection;
                std::condition_variable_any wait;
                std::mutex wait_mutex;
                while (!stop.stop_requested()) {
                    session.poll(SpaceMouseSession::Clock::now());
                    auto next = session.takeFrame();
                    if (next.connection != last_connection) {
                        if (next.connection)
                            LOG_INFO("3D mouse: {} via {}", next.connection->name, next.connection->backend);
                        last_connection = next.connection;
                    }
                    bool changed;
                    {
                        const std::scoped_lock lock(mutex);
                        if (!input_enabled) {
                            next.motion = {};
                            next.buttons = 0;
                            next.pressed = 0;
                        } else if (next.connection && next.connection == frame.connection) {
                            next.pressed |= frame.pressed;
                        }
                        changed = spaceMouseStep(next.motion, 1, 1, 1, false, false) !=
                                      spaceMouseStep(frame.motion, 1, 1, 1, false, false) ||
                                  next.buttons != frame.buttons ||
                                  next.pressed != frame.pressed || next.connection != frame.connection;
                        frame = next;
                    }
                    if (changed)
                        wake();
                    if (!next.connection) {
                        std::unique_lock lock(wait_mutex);
                        wait.wait_for(lock, stop, std::chrono::milliseconds(50), [] { return false; });
                    }
                }
            });
        }
    };

    SpaceMouseDevice::SpaceMouseDevice(SDL_Window* window, std::function<void()> wake)
        : impl_(std::make_unique<Impl>(window, std::move(wake))) {}
    SpaceMouseDevice::~SpaceMouseDevice() = default;

    SpaceMouseFrame SpaceMouseDevice::takeFrame() {
        const std::scoped_lock lock(impl_->mutex);
        const auto frame = impl_->frame;
        impl_->frame.pressed = 0;
        return frame;
    }

    std::optional<SpaceMouseConnection> SpaceMouseDevice::connection() const {
        const std::scoped_lock lock(impl_->mutex);
        return impl_->frame.connection;
    }

    void SpaceMouseDevice::setInputEnabled(const bool enabled) {
        const std::scoped_lock lock(impl_->mutex);
        impl_->input_enabled = enabled;
        if (!enabled) {
            impl_->frame.motion = {};
            impl_->frame.buttons = 0;
            impl_->frame.pressed = 0;
        }
    }
} // namespace lfs::vis::input
