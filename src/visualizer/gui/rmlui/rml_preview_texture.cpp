/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_preview_texture.hpp"

#include "core/error.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/project_container.hpp"
#include "python/python_runtime.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>

namespace lfs::vis::gui {
    namespace {
        int hexValue(char c) {
            if (c >= '0' && c <= '9')
                return c - '0';
            if (c >= 'a' && c <= 'f')
                return 10 + c - 'a';
            if (c >= 'A' && c <= 'F')
                return 10 + c - 'A';
            return -1;
        }

        std::string percentDecode(std::string_view value) {
            std::string decoded;
            decoded.reserve(value.size());
            for (size_t i = 0; i < value.size(); ++i) {
                if (value[i] == '%' && i + 2 < value.size()) {
                    const int hi = hexValue(value[i + 1]);
                    const int lo = hexValue(value[i + 2]);
                    if (hi >= 0 && lo >= 0) {
                        decoded.push_back(static_cast<char>((hi << 4) | lo));
                        i += 2;
                        continue;
                    }
                }
                decoded.push_back(value[i] == '+' ? ' ' : value[i]);
            }
            return decoded;
        }

        int parseIntParam(std::string_view value) {
            int parsed = 0;
            const auto* first = value.data();
            const auto* last = value.data() + value.size();
            const auto [ptr, ec] = std::from_chars(first, last, parsed);
            if (ec != std::errc() || ptr != last)
                return 0;
            return std::max(parsed, 0);
        }
    } // namespace

    bool isPreviewTextureSource(std::string_view source) {
        return source.rfind("preview://", 0) == 0;
    }

    std::optional<PreviewTextureRequest> parsePreviewTextureRequest(std::string_view source) {
        if (!isPreviewTextureSource(source))
            return std::nullopt;

        source.remove_prefix(std::string_view("preview://").size());
        PreviewTextureRequest request;
        int thumb_size = 0;
        int preview_size = 0;
        int dataset_size = 0;

        while (!source.empty()) {
            const size_t sep = source.find('&');
            const std::string_view part = sep == std::string_view::npos ? source : source.substr(0, sep);
            if (const size_t eq = part.find('='); eq != std::string_view::npos) {
                const std::string_view key = part.substr(0, eq);
                const std::string_view value = part.substr(eq + 1);
                if (key == "path") {
                    request.path = lfs::core::utf8_to_path(percentDecode(value));
                } else if (key == "kind") {
                    request.embedded_project_preview = value == "licht";
                } else if (key == "thumb") {
                    thumb_size = parseIntParam(value);
                } else if (key == "pmw") {
                    preview_size = parseIntParam(value);
                } else if (key == "mw") {
                    dataset_size = parseIntParam(value);
                } else if (key == "w") {
                    request.width = parseIntParam(value);
                } else if (key == "h") {
                    request.height = parseIntParam(value);
                }
            }

            if (sep == std::string_view::npos)
                break;
            source.remove_prefix(sep + 1);
        }

        if (request.path.empty())
            return std::nullopt;

        request.max_size = thumb_size > 0 ? thumb_size : (preview_size > 0 ? preview_size : dataset_size);
        return request;
    }

    PreviewTexturePixels decodePreviewTexture(
        std::filesystem::path path,
        const int max_size,
        const bool embedded_project_preview) {
        PreviewTexturePixels result;
        unsigned char* data = nullptr;
        try {
            int width = 0;
            int height = 0;
            int channels = 0;
            unsigned char* pixels = nullptr;
            if (embedded_project_preview) {
                auto reader = lfs::io::project::ProjectReader::open(path);
                if (!reader) {
                    // A catalog entry can outlive its file until the catalog marks it missing.
                    if (reader.error().code() != lfs::ErrorCode::NotFound)
                        LOG_WARN("Failed to inspect embedded project preview '{}': {}",
                                 lfs::core::path_to_utf8(path),
                                 lfs::format_for_developer(reader.error()));
                    return result;
                }
                auto preview = reader->read_preview();
                if (!preview) {
                    LOG_WARN("Failed to read embedded project preview '{}': {}",
                             lfs::core::path_to_utf8(path),
                             lfs::format_for_developer(preview.error()));
                    return result;
                }
                const auto* bytes = reinterpret_cast<const std::uint8_t*>(preview->data());
                std::tie(pixels, width, height, channels) =
                    lfs::core::load_image_from_memory(bytes, preview->size());
            } else {
                std::tie(pixels, width, height, channels) =
                    lfs::core::load_image_thumbnail(path, max_size);
            }
            data = pixels;
            if (pixels && width > 0 && height > 0 && channels > 0) {
                int output_width = width;
                int output_height = height;
                if (embedded_project_preview && max_size > 0 &&
                    std::max(width, height) > max_size) {
                    const double scale = static_cast<double>(max_size) /
                                         static_cast<double>(std::max(width, height));
                    output_width = std::max(1, static_cast<int>(std::lround(width * scale)));
                    output_height = std::max(1, static_cast<int>(std::lround(height * scale)));
                }
                const std::size_t pixel_count =
                    static_cast<std::size_t>(output_width) *
                    static_cast<std::size_t>(output_height);
                result.pixels.resize(pixel_count * 4u);
                const auto sample = [&](const double x, const double y, const int channel) {
                    const double sx = std::clamp(x, 0.0, static_cast<double>(width - 1));
                    const double sy = std::clamp(y, 0.0, static_cast<double>(height - 1));
                    const int x0 = static_cast<int>(sx);
                    const int y0 = static_cast<int>(sy);
                    const int x1 = std::min(width - 1, x0 + 1);
                    const int y1 = std::min(height - 1, y0 + 1);
                    const double fx = sx - x0;
                    const double fy = sy - y0;
                    const auto at = [&](const int px, const int py) {
                        return static_cast<double>(pixels[(static_cast<std::size_t>(py) * width + px) * channels + channel]);
                    };
                    return (at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx) * (1.0 - fy) +
                           (at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx) * fy;
                };
                for (int y = 0; y < output_height; ++y) {
                    for (int x = 0; x < output_width; ++x) {
                        const double source_x = (x + 0.5) * width / output_width - 0.5;
                        const double source_y = (y + 0.5) * height / output_height - 0.5;
                        const unsigned char r = static_cast<unsigned char>(std::lround(sample(source_x, source_y, 0)));
                        const unsigned char g = static_cast<unsigned char>(std::lround(sample(source_x, source_y, channels > 1 ? 1 : 0)));
                        const unsigned char b = static_cast<unsigned char>(std::lround(sample(source_x, source_y, channels > 2 ? 2 : 0)));
                        const unsigned char a = channels > 3 ? static_cast<unsigned char>(std::lround(sample(source_x, source_y, 3))) : 255;
                        Rml::byte* dst = result.pixels.data() +
                                         (static_cast<std::size_t>(y) *
                                              static_cast<std::size_t>(output_width) +
                                          static_cast<std::size_t>(x)) *
                                             4u;
                        dst[0] = static_cast<Rml::byte>(
                            (static_cast<unsigned int>(r) * a + 127u) / 255u);
                        dst[1] = static_cast<Rml::byte>(
                            (static_cast<unsigned int>(g) * a + 127u) / 255u);
                        dst[2] = static_cast<Rml::byte>(
                            (static_cast<unsigned int>(b) * a + 127u) / 255u);
                        dst[3] = static_cast<Rml::byte>(a);
                    }
                }
                result.width = output_width;
                result.height = output_height;
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to decode preview texture '{}': {}", lfs::core::path_to_utf8(path), e.what());
        }
        if (data)
            lfs::core::free_image(data);
        return result;
    }

    PreviewTextureLoader::~PreviewTextureLoader() { stop(); }

    void PreviewTextureLoader::load(Rml::TextureHandle texture, const Rml::String& source,
                                    const PreviewTextureRequest& request) {
        auto state = std::make_shared<State>();
        state->texture = texture;
        state->source = source;
        pending_.push_back(state);

        try {
            ensureWorkers();
            enqueue(Work{
                state, request.path, request.max_size, request.embedded_project_preview});
        } catch (const std::exception& e) {
            LOG_WARN("Failed to queue async preview texture worker for '{}': {}",
                     lfs::core::path_to_utf8(request.path), e.what());
            cancel(texture);
        }
    }

    void PreviewTextureLoader::ensureWorkers() {
        std::lock_guard lock(queue_mutex_);
        if (!workers_.empty())
            return;
        stopping_ = false;
        constexpr std::size_t kWorkerCount = 3;
        for (std::size_t i = 0; i < kWorkerCount; ++i) {
            workers_.emplace_back([this] {
                for (;;) {
                    Work work;
                    {
                        std::unique_lock queue_lock(queue_mutex_);
                        queue_cv_.wait(queue_lock, [this] {
                            return stopping_ || !queue_.empty();
                        });
                        if (stopping_ && queue_.empty())
                            return;
                        work = std::move(queue_.front());
                        queue_.pop_front();
                    }
                    if (!work.state || work.state->cancelled.load(std::memory_order_acquire))
                        continue;
                    auto result = decodePreviewTexture(
                        std::move(work.path), work.max_size, work.embedded_project_preview);
                    if (work.state->cancelled.load(std::memory_order_acquire))
                        continue;
                    {
                        std::lock_guard result_lock(work.state->mutex);
                        work.state->result = std::move(result);
                    }
                    work.state->ready.store(true, std::memory_order_release);
                    lfs::python::request_redraw();
                }
            });
        }
    }

    void PreviewTextureLoader::enqueue(Work work) {
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_)
                throw std::runtime_error("preview worker pool is stopping");
            queue_.emplace_back(std::move(work));
        }
        queue_cv_.notify_one();
    }

    void PreviewTextureLoader::stop() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
            queue_.clear();
        }
        queue_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable())
                worker.join();
        }
        workers_.clear();
        pending_.clear();
    }

    void PreviewTextureLoader::cancel(Rml::TextureHandle texture) {
        if (!texture)
            return;
        pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                      [texture](const std::shared_ptr<State>& state) {
                                          if (state && state->texture == texture) {
                                              state->cancelled.store(true, std::memory_order_release);
                                              state->texture = 0;
                                              return true;
                                          }
                                          return false;
                                      }),
                       pending_.end());
    }

    void PreviewTextureLoader::uploadReady(const Upload& upload) {
        if (pending_.empty())
            return;

        constexpr int kUploadBudgetPerFrame = 2;
        int uploads_remaining = kUploadBudgetPerFrame;
        bool uploaded_any = false;
        bool needs_followup_redraw = false;

        for (auto it = pending_.begin(); it != pending_.end();) {
            const std::shared_ptr<State>& state = *it;
            if (!state || !state->texture) {
                it = pending_.erase(it);
                continue;
            }
            if (!state->ready.load(std::memory_order_acquire)) {
                ++it;
                continue;
            }
            if (uploads_remaining <= 0) {
                needs_followup_redraw = true;
                break;
            }

            PreviewTexturePixels result;
            {
                std::lock_guard lock(state->mutex);
                result = std::move(state->result);
            }

            if (!result.pixels.empty() && result.width > 0 && result.height > 0) {
                if (upload(state->texture, state->source, result)) {
                    --uploads_remaining;
                    uploaded_any = true;
                }
            }

            it = pending_.erase(it);
        }

        if (uploaded_any)
            generation_.fetch_add(1, std::memory_order_acq_rel);
        if (needs_followup_redraw)
            lfs::python::request_redraw();
    }

    std::uint64_t PreviewTextureLoader::generation() const {
        return generation_.load(std::memory_order_acquire);
    }
} // namespace lfs::vis::gui
