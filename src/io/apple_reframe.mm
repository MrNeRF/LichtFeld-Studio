/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "io/apple_reframe.hpp"
#include "apple_reframe_conversion.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"

#import <Foundation/Foundation.h>
#include <TargetConditionals.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <limits>
#include <thread>

namespace lfs::io {
    namespace {
        std::filesystem::path helperPath() {
            uint32_t length = 0;
            _NSGetExecutablePath(nullptr, &length);
            std::vector<char> buffer(length);
            if (_NSGetExecutablePath(buffer.data(), &length) != 0)
                return {};
            const auto directory = std::filesystem::weakly_canonical(buffer.data()).parent_path();
            for (const auto& path : {directory / "lfs-reframe", directory / "bin/lfs-reframe"}) {
                if (access(path.c_str(), X_OK) == 0)
                    return path;
            }
            return {};
        }

        bool supportedHost() {
#if defined(__aarch64__) && !TARGET_OS_IPHONE
            if (@available(macOS 27.0, *))
                return true;
#endif
            return false;
        }

        struct TemporaryDirectory {
            std::filesystem::path path;
            TemporaryDirectory() {
                auto pattern = (std::filesystem::temp_directory_path() / "lichtfeld-reframe-XXXXXX").string();
                std::vector<char> chars(pattern.begin(), pattern.end());
                chars.push_back('\0');
                if (!mkdtemp(chars.data()))
                    throw std::runtime_error("Cannot create reconstruction workspace");
                path = chars.data();
            }
            ~TemporaryDirectory() {
                std::error_code ignored;
                std::filesystem::remove_all(path, ignored);
            }
        };

        void runHelper(const std::filesystem::path& helper, NSArray<NSString*>* arguments,
                       const std::filesystem::path& log, const LoadOptions& options,
                       std::chrono::seconds timeout) {
            @autoreleasepool {
                NSString* log_path = [NSString stringWithUTF8String:log.c_str()];
                if (![NSFileManager.defaultManager createFileAtPath:log_path contents:nil attributes:@{NSFilePosixPermissions : @0600}])
                    throw std::runtime_error("Cannot create reconstruction diagnostics");
                NSFileHandle* output = [NSFileHandle fileHandleForWritingAtPath:log_path];
                NSTask* task = [NSTask new];
                task.executableURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:helper.c_str()]];
                task.arguments = arguments;
                task.standardOutput = output;
                task.standardError = output;
                task.standardInput = NSFileHandle.fileHandleWithNullDevice;
                NSError* error = nil;
                if (![task launchAndReturnError:&error]) {
                    [output closeFile];
                    const char* detail = error.localizedDescription.UTF8String;
                    throw std::runtime_error(detail ? detail : "Cannot launch Reframe helper");
                }
                const auto deadline = std::chrono::steady_clock::now() + timeout;
                bool cancelled = false, timed_out = false;
                while (task.running) {
                    cancelled = is_load_cancel_requested(options);
                    timed_out = std::chrono::steady_clock::now() >= deadline;
                    if (cancelled || timed_out) {
                        [task terminate];
                        const auto grace = std::chrono::steady_clock::now() + std::chrono::seconds(1);
                        while (task.running && std::chrono::steady_clock::now() < grace)
                            std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        if (task.running)
                            kill(task.processIdentifier, SIGKILL);
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                [task waitUntilExit];
                [output closeFile];
                if (cancelled)
                    throw LoadCancelledError("Photo reconstruction cancelled");
                if (timed_out)
                    throw std::runtime_error("Apple Reframe did not finish within its time limit");
                if (task.terminationStatus != 0 || task.terminationReason != NSTaskTerminationReasonExit) {
                    std::ifstream stream(log);
                    std::string message(4096, '\0');
                    stream.read(message.data(), message.size());
                    message.resize(static_cast<size_t>(stream.gcount()));
                    throw std::runtime_error(message.empty() ? "Apple Reframe helper failed" : message);
                }
            }
        }

        std::vector<float> readHalfArray(std::ifstream& stream, size_t size) {
            std::vector<uint16_t> packed(size);
            if (!stream.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(size * 2)))
                throw std::runtime_error("Incomplete Reframe buffer");
            std::vector<float> values(size);
            for (size_t i = 0; i < size; ++i) {
                __fp16 value;
                std::memcpy(&value, &packed[i], 2);
                values[i] = static_cast<float>(value);
                if (!std::isfinite(values[i]))
                    throw std::runtime_error("Non-finite Reframe Gaussian data");
            }
            return values;
        }
    } // namespace

    bool appleReframeAvailable() {
        if (!supportedHost())
            return false;
        // Both successful and failed probes expire so installing Photos assets
        // or an OS update can change capability without restarting the app.
        static std::mutex mutex;
        static auto checked = std::chrono::steady_clock::time_point::min();
        static bool available = false;
        const std::lock_guard lock(mutex);
        const auto now = std::chrono::steady_clock::now();
        if (checked != std::chrono::steady_clock::time_point::min() && now - checked < std::chrono::seconds(30))
            return available;
        available = false;
        checked = now;
        try {
            const auto helper = helperPath();
            if (helper.empty())
                return false;
            TemporaryDirectory workspace;
            runHelper(helper, @[ @"--check" ], workspace.path / "check.log", {}, std::chrono::seconds(3));
            available = true;
        } catch (...) { /* Unsupported private ABI or unavailable assets: hide the feature. */
        }
        return available;
    }

    Result<LoadResult> createAppleReframeSplat(const std::filesystem::path& photo, const LoadOptions& options) {
        const auto started = std::chrono::steady_clock::now();
        try {
            throw_if_load_cancel_requested(options);
            const auto helper = supportedHost() ? helperPath() : std::filesystem::path{};
            if (helper.empty())
                return make_error(ErrorCode::UNSUPPORTED_FORMAT, "Apple Reframe is unavailable on this Mac", photo);
            if (!std::filesystem::is_regular_file(photo))
                return make_error(ErrorCode::NOT_A_FILE, "Photo does not exist", photo);
            TemporaryDirectory workspace;
            const auto payload = workspace.path / "gaussians.bin";
            if (options.progress)
                options.progress(5, "Creating splats with Apple Reframe…");
            runHelper(helper, @[ [NSString stringWithUTF8String:photo.c_str()], [NSString stringWithUTF8String:payload.c_str()] ],
                      workspace.path / "helper.log", options, std::chrono::seconds(180));
            throw_if_load_cancel_requested(options);
            if (options.progress)
                options.progress(80, "Preparing Gaussian tensors…");
            std::ifstream stream(payload, std::ios::binary);
            char magic[8];
            uint64_t count = 0;
            uint32_t width = 0, height = 0;
            if (!stream.read(magic, 8) || std::memcmp(magic, "LFSRFR02", 8) ||
                !stream.read(reinterpret_cast<char*>(&count), 8) || count == 0 || count > 10'000'000 ||
                !stream.read(reinterpret_cast<char*>(&width), 4) || !stream.read(reinterpret_cast<char*>(&height), 4) ||
                width == 0 || height == 0 || width > 32768 || height > 32768 ||
                std::filesystem::file_size(payload) != 24 + count * 28)
                throw std::runtime_error("Incompatible Reframe Gaussian payload");
            auto means = readHalfArray(stream, count * 3);
            auto rotation = readHalfArray(stream, count * 4);
            auto scaling = readHalfArray(stream, count * 3);
            auto sh0 = readHalfArray(stream, count * 3);
            auto opacity = readHalfArray(stream, count);
            for (size_t i = 0; i < count; ++i) {
                if ((i & 4095) == 0)
                    throw_if_load_cancel_requested(options);
                // Apple uses scalar-first (w,x,y,z), matching SplatData. Verified
                // with an anisotropic splat rotated 90 degrees around X.
                const auto quaternion = reframe::normalizedRotation({rotation[4 * i], rotation[4 * i + 1], rotation[4 * i + 2], rotation[4 * i + 3]});
                std::copy(quaternion.begin(), quaternion.end(), rotation.begin() + 4 * i);
                for (size_t axis = 0; axis < 3; ++axis) {
                    scaling[3 * i + axis] = reframe::logScale(scaling[3 * i + axis]);
                    sh0[3 * i + axis] = reframe::viewerSh0(sh0[3 * i + axis]);
                }
                opacity[i] = reframe::opacityLogit(opacity[i]);
            }
            using core::Device;
            using core::Tensor;
            auto splat = std::make_shared<core::SplatData>(
                0, Tensor::from_vector(means, {count, 3}, Device::GPU),
                Tensor::from_vector(sh0, {count, 1, 3}, Device::GPU),
                Tensor::zeros({count, 0, 3}, Device::GPU),
                Tensor::from_vector(scaling, {count, 3}, Device::GPU),
                Tensor::from_vector(rotation, {count, 4}, Device::GPU),
                Tensor::from_vector(opacity, {count, 1}, Device::GPU), 1.0f);
            if (options.splat_tensor_allocator) {
                if (auto migrated = migrateSplatTensorsToAllocator(*splat, options.splat_tensor_allocator); !migrated)
                    return std::unexpected(migrated.error());
            }
            throw_if_load_cancel_requested(options);
            if (options.progress)
                options.progress(100, "Photo reconstruction complete");
            LoadResult result;
            result.data = std::move(splat);
            result.scene_center = Tensor::zeros({3}, Device::CPU);
            result.loader_used = "Apple Reframe";
            PhotoReconstructionView view;
            view.source_aspect = float(width) / float(height);
            view.bounds_min.fill(std::numeric_limits<float>::infinity());
            view.bounds_max.fill(-std::numeric_limits<float>::infinity());
            bool has_visible_geometry = false;
            for (size_t i = 0; i < count; ++i) {
                if ((i & 4095) == 0)
                    throw_if_load_cancel_requested(options);
                if (means[3 * i + 2] <= 0.01f || opacity[i] <= -2.944439f)
                    continue;
                // Bound the rotated Gaussian ellipsoid at three sigma.
                const auto radii = reframe::supportRadii(
                    {scaling[3 * i], scaling[3 * i + 1], scaling[3 * i + 2]},
                    {rotation[4 * i], rotation[4 * i + 1], rotation[4 * i + 2], rotation[4 * i + 3]});
                for (size_t axis = 0; axis < 3; ++axis) {
                    const float position = means[3 * i + axis];
                    if (!std::isfinite(position))
                        throw std::runtime_error("Reframe returned a non-finite position");
                    view.bounds_min[axis] = std::min(view.bounds_min[axis], position - radii[axis]);
                    view.bounds_max[axis] = std::max(view.bounds_max[axis], position + radii[axis]);
                }
                has_visible_geometry = true;
            }
            if (!has_visible_geometry)
                throw std::runtime_error("Reframe returned no positive scene depth");
            result.photo_view = view;
            result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started);
            return result;
        } catch (const LoadCancelledError& error) {
            return make_error(ErrorCode::CANCELLED, error.what(), photo);
        } catch (const std::exception& error) {
            return make_error(ErrorCode::READ_FAILURE, error.what(), photo);
        }
    }
} // namespace lfs::io
