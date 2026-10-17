/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include <RmlUi/Core/Types.h>
#include <RmlUi/Core/Vector2.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <vector>

namespace lfs::vis::gui {
    // `preview://key=value&...` image sources are decoded on worker threads; the
    // renderer draws a transparent placeholder until the pixels are uploaded.
    struct PreviewTextureRequest {
        std::filesystem::path path;
        int max_size = 0;
        int width = 0;
        int height = 0;
        bool embedded_project_preview = false;

        // RmlUi builds and caches cover/contain geometry from the dimensions
        // LoadTexture reports, so report the decoded aspect up front.
        [[nodiscard]] Rml::Vector2i layoutDimensions() const {
            return width > 0 && height > 0 ? Rml::Vector2i{width, height} : Rml::Vector2i{1, 1};
        }
    };

    struct PreviewTexturePixels {
        std::vector<Rml::byte> pixels; // premultiplied RGBA8
        int width = 0;
        int height = 0;
    };

    [[nodiscard]] LFS_VIS_API bool isPreviewTextureSource(std::string_view source);
    [[nodiscard]] LFS_VIS_API std::optional<PreviewTextureRequest> parsePreviewTextureRequest(std::string_view source);
    [[nodiscard]] LFS_VIS_API PreviewTexturePixels decodePreviewTexture(std::filesystem::path path, int max_size,
                                                                        bool embedded_project_preview);

    // Decode workers and the decoded previews waiting for upload. Everything except
    // the workers runs on the render thread.
    class LFS_VIS_API PreviewTextureLoader {
    public:
        // Replaces the texture's pixels; returns false when the upload failed.
        using Upload = std::function<bool(Rml::TextureHandle texture, const Rml::String& source,
                                          PreviewTexturePixels& pixels)>;

        ~PreviewTextureLoader(); // stop()
        void load(Rml::TextureHandle texture, const Rml::String& source, const PreviewTextureRequest& request);
        void cancel(Rml::TextureHandle texture); // call before the texture is released
        void stop() noexcept;                    // join workers, drop pending; a later load() restarts them
        void uploadReady(const Upload& upload);  // at frame start
        [[nodiscard]] std::uint64_t generation() const;

    private:
        struct State {
            Rml::TextureHandle texture = 0;
            Rml::String source;
            std::mutex mutex;
            PreviewTexturePixels result;
            std::atomic<bool> ready = false;
            std::atomic<bool> cancelled = false;
        };
        struct Work {
            std::shared_ptr<State> state;
            std::filesystem::path path;
            int max_size = 0;
            bool embedded_project_preview = false;
        };
        void ensureWorkers();
        void enqueue(Work work);
        std::vector<std::shared_ptr<State>> pending_;
        std::mutex queue_mutex_;
        std::condition_variable queue_cv_;
        std::deque<Work> queue_;
        std::vector<std::thread> workers_;
        bool stopping_ = false;
        std::atomic<std::uint64_t> generation_{0};
    };
} // namespace lfs::vis::gui
