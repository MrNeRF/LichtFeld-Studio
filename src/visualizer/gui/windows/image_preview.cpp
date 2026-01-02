/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/windows/image_preview.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <future>
#include <glad/glad.h>
#include <stdexcept>
#include <thread>
#include <imgui.h>

namespace lfs::vis::gui {

    using namespace lichtfeld::Strings;

    // EXIF tag IDs
    namespace exif_tags {
        constexpr uint16_t MAKE = 0x010F;
        constexpr uint16_t MODEL = 0x0110;
        constexpr uint16_t ORIENTATION = 0x0112;
        constexpr uint16_t SOFTWARE = 0x0131;
        constexpr uint16_t DATE_TIME = 0x0132;
        constexpr uint16_t EXIF_IFD = 0x8769;
        constexpr uint16_t EXPOSURE_TIME = 0x829A;
        constexpr uint16_t F_NUMBER = 0x829D;
        constexpr uint16_t ISO = 0x8827;
        constexpr uint16_t DATE_TIME_ORIGINAL = 0x9003;
        constexpr uint16_t FOCAL_LENGTH = 0xA404;
        constexpr uint16_t FOCAL_LENGTH_35MM = 0xA405;
        constexpr uint16_t LENS_MODEL = 0xA434;
    } // namespace exif_tags

    ExifData parseExifData(const std::filesystem::path& path) {
        ExifData result;

        std::string ext = path.extension().string();
        for (char& c : ext)
            c = static_cast<char>(std::tolower(c));
        if (ext != ".jpg" && ext != ".jpeg")
            return result;

        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, std::ios::binary, file))
            return result;

        uint8_t buf[2];
        file.read(reinterpret_cast<char*>(buf), 2);
        if (buf[0] != 0xFF || buf[1] != 0xD8)
            return result;

        while (file) {
            file.read(reinterpret_cast<char*>(buf), 2);
            if (buf[0] != 0xFF)
                break;

            if (buf[1] == 0xE1) {
                uint8_t len_buf[2];
                file.read(reinterpret_cast<char*>(len_buf), 2);
                const uint16_t segment_len = (len_buf[0] << 8) | len_buf[1];

                std::vector<uint8_t> segment(segment_len - 2);
                file.read(reinterpret_cast<char*>(segment.data()), segment.size());

                if (segment.size() < 14 || std::memcmp(segment.data(), "Exif\0\0", 6) != 0)
                    continue;

                const uint8_t* const tiff = segment.data() + 6;
                const size_t tiff_size = segment.size() - 6;
                const bool big_endian = (tiff[0] == 'M' && tiff[1] == 'M');
                if (!big_endian && !(tiff[0] == 'I' && tiff[1] == 'I'))
                    continue;

                const auto read16 = [big_endian, tiff, tiff_size](const size_t offset) -> uint16_t {
                    if (offset + 2 > tiff_size)
                        return 0;
                    if (big_endian)
                        return (tiff[offset] << 8) | tiff[offset + 1];
                    return tiff[offset] | (tiff[offset + 1] << 8);
                };

                const auto read32 = [big_endian, tiff, tiff_size](const size_t offset) -> uint32_t {
                    if (offset + 4 > tiff_size)
                        return 0;
                    if (big_endian)
                        return (tiff[offset] << 24) | (tiff[offset + 1] << 16) |
                               (tiff[offset + 2] << 8) | tiff[offset + 3];
                    return tiff[offset] | (tiff[offset + 1] << 8) |
                           (tiff[offset + 2] << 16) | (tiff[offset + 3] << 24);
                };

                const auto read_string = [tiff, tiff_size](const size_t offset, const size_t count) {
                    if (offset + count > tiff_size)
                        return std::string{};
                    std::string s(reinterpret_cast<const char*>(tiff + offset), count);
                    while (!s.empty() && (s.back() == '\0' || s.back() == ' '))
                        s.pop_back();
                    return s;
                };

                const auto read_rational = [&read32](const size_t offset) {
                    return std::pair{read32(offset), read32(offset + 4)};
                };

                const uint32_t ifd0_offset = read32(4);
                if (ifd0_offset + 2 > tiff_size)
                    continue;

                uint32_t exif_ifd_offset = 0;

                const auto parse_ifd = [&](const uint32_t ifd_offset, const bool is_exif_ifd) {
                    if (ifd_offset + 2 > tiff_size)
                        return;
                    const uint16_t num_entries = read16(ifd_offset);
                    size_t entry_offset = ifd_offset + 2;

                    for (uint16_t i = 0; i < num_entries && entry_offset + 12 <= tiff_size; ++i, entry_offset += 12) {
                        const uint16_t tag = read16(entry_offset);
                        const uint16_t type = read16(entry_offset + 2);
                        const uint32_t count = read32(entry_offset + 4);
                        uint32_t value_offset = entry_offset + 8;

                        size_t data_size = count;
                        if (type == 3)
                            data_size *= 2;
                        else if (type == 4)
                            data_size *= 4;
                        else if (type == 5)
                            data_size *= 8;

                        if (data_size > 4)
                            value_offset = read32(entry_offset + 8);

                        switch (tag) {
                        case exif_tags::MAKE:
                            result.camera_make = read_string(value_offset, count);
                            break;
                        case exif_tags::MODEL:
                            result.camera_model = read_string(value_offset, count);
                            break;
                        case exif_tags::SOFTWARE:
                            result.software = read_string(value_offset, count);
                            break;
                        case exif_tags::DATE_TIME:
                        case exif_tags::DATE_TIME_ORIGINAL:
                            if (result.date_time.empty())
                                result.date_time = read_string(value_offset, count);
                            break;
                        case exif_tags::ORIENTATION:
                            result.orientation = read16(value_offset);
                            break;
                        case exif_tags::EXIF_IFD:
                            exif_ifd_offset = read32(value_offset);
                            break;
                        case exif_tags::EXPOSURE_TIME:
                            if (is_exif_ifd) {
                                const auto [num, den] = read_rational(value_offset);
                                if (den > 0) {
                                    result.exposure_time = (num == 1)
                                                               ? std::format("1/{}s", den)
                                                               : std::format("{:.1f}s", static_cast<double>(num) / den);
                                }
                            }
                            break;
                        case exif_tags::F_NUMBER:
                            if (is_exif_ifd) {
                                const auto [num, den] = read_rational(value_offset);
                                if (den > 0)
                                    result.f_number = std::format("f/{:.1f}", static_cast<double>(num) / den);
                            }
                            break;
                        case exif_tags::ISO:
                            if (is_exif_ifd)
                                result.iso = std::format("ISO {}", read16(value_offset));
                            break;
                        case exif_tags::FOCAL_LENGTH:
                            if (is_exif_ifd) {
                                const auto [num, den] = read_rational(value_offset);
                                if (den > 0)
                                    result.focal_length = std::format("{:.0f}mm", static_cast<double>(num) / den);
                            }
                            break;
                        case exif_tags::FOCAL_LENGTH_35MM:
                            if (is_exif_ifd)
                                result.focal_length_35mm = std::format("{}mm (35mm eq.)", read16(value_offset));
                            break;
                        case exif_tags::LENS_MODEL:
                            if (is_exif_ifd)
                                result.lens_model = read_string(value_offset, count);
                            break;
                        default: break;
                        }
                    }
                };

                parse_ifd(ifd0_offset, false);
                if (exif_ifd_offset > 0)
                    parse_ifd(exif_ifd_offset, true);

                result.valid = !result.camera_make.empty() || !result.camera_model.empty() ||
                               !result.exposure_time.empty() || !result.f_number.empty();
                break;
            } else if (buf[1] >= 0xE0 && buf[1] <= 0xEF) {
                uint8_t len_buf[2];
                file.read(reinterpret_cast<char*>(len_buf), 2);
                file.seekg((len_buf[0] << 8 | len_buf[1]) - 2, std::ios::cur);
            } else if (buf[1] == 0xDA) {
                break;
            } else {
                uint8_t len_buf[2];
                file.read(reinterpret_cast<char*>(len_buf), 2);
                file.seekg((len_buf[0] << 8 | len_buf[1]) - 2, std::ios::cur);
            }
        }

        return result;
    }

    ImagePreview::ImagePreview() = default;

    ImagePreview::~ImagePreview() {
        close();
    }

    void ImagePreview::ensureMaxTextureSizeInitialized() {
        static std::once_flag initialized;
        std::call_once(initialized, [this]() {
            if (glGetIntegerv) { // Check if OpenGL is initialized
                glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size_);
                LOG_DEBUG("Max texture size: {}x{}", max_texture_size_, max_texture_size_);
            }
        });
    }

    void ImagePreview::open(const std::vector<std::filesystem::path>& image_paths, size_t initial_index) {
        LOG_TIMER_TRACE("ImagePreview::open");

        if (image_paths.empty()) {
            LOG_WARN("No images to preview");
            return;
        }

        // Clear any existing state
        close();

        image_paths_ = image_paths;
        current_index_ = std::min(initial_index, image_paths.size() - 1);
        is_open_ = true;
        focus_on_next_frame_ = true;

        // Reset view
        zoom_ = 1.0f;
        pan_x_ = 0.0f;
        pan_y_ = 0.0f;
        fit_to_window_ = true;

        LOG_DEBUG("Opening image preview with {} images, starting at index {}",
                  image_paths.size(), current_index_);

        // Load current image
        if (!loadImage(image_paths_[current_index_])) {
            LOG_ERROR("Failed to load initial image: {}", load_error_);
            throw std::runtime_error(std::format("Failed to load image: {}", load_error_));
        }

        // Start preloading adjacent images
        preloadAdjacentImages();

        LOG_INFO("Opened image {}/{}: {}",
                 current_index_ + 1,
                 image_paths_.size(),
                 lfs::core::path_to_utf8(image_paths_[current_index_].filename()));
    }

    void ImagePreview::open(const std::filesystem::path& image_path) {
        open(std::vector{image_path}, 0);
    }

    void ImagePreview::openWithOverlay(const std::vector<std::filesystem::path>& image_paths,
                                       const std::vector<std::filesystem::path>& overlay_paths,
                                       const size_t initial_index) {
        open(image_paths, initial_index);
        overlay_paths_ = overlay_paths;
        loadCurrentOverlay();
    }

    void ImagePreview::loadCurrentOverlay() {
        overlay_texture_.reset();
        if (current_index_ < overlay_paths_.size() && !overlay_paths_[current_index_].empty()) {
            try {
                auto data = loadImageData(overlay_paths_[current_index_]);
                overlay_texture_ = createTexture(std::move(*data), overlay_paths_[current_index_]);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load overlay: {}", e.what());
            }
        }
    }

    bool ImagePreview::hasValidOverlay() const {
        return current_index_ < overlay_paths_.size() &&
               !overlay_paths_[current_index_].empty() &&
               overlay_texture_ && overlay_texture_->texture.valid();
    }

    void ImagePreview::close() {
        is_open_ = false;
        image_paths_.clear();
        overlay_paths_.clear();
        current_texture_.reset();
        previous_texture_.reset();
        overlay_texture_.reset();
        prev_texture_.reset();
        next_texture_.reset();

        {
            std::lock_guard lock(preload_mutex_);
            prev_result_.reset();
            next_result_.reset();
        }

        load_error_.clear();
        show_overlay_ = false;
    }

    std::unique_ptr<ImageData> ImagePreview::loadImageData(const std::filesystem::path& path) {
        LOG_TIMER("LoadImageData");

        LOG_TRACE("Loading image data from: {}", lfs::core::path_to_utf8(path));

        // Load image (max_width = -1 disables downscaling for preview)
        auto [data, width, height, channels] = [&]() {
            LOG_TIMER("load_image call");
            return lfs::core::load_image(path, -1, -1);
        }();

        // Wrap in RAII immediately
        auto image_data = [&]() {
            LOG_TIMER("ImageData construction");
            return std::make_unique<ImageData>(data, width, height, channels);
        }();

        // Validate
        {
            LOG_TIMER("Image validation");
            if (!image_data->valid()) {
                LOG_ERROR("Failed to load image data from: {}", lfs::core::path_to_utf8(path));
                throw std::runtime_error("Failed to load image data");
            }

            if (width <= 0 || height <= 0) {
                LOG_ERROR("Invalid image dimensions: {}x{} for: {}", width, height, lfs::core::path_to_utf8(path));
                throw std::runtime_error(std::format("Invalid image dimensions: {}x{}", width, height));
            }

            if (channels < 1 || channels > 4) {
                LOG_ERROR("Invalid number of channels: {} for: {}", channels, lfs::core::path_to_utf8(path));
                throw std::runtime_error(std::format("Invalid number of channels: {}", channels));
            }
        }

        LOG_TRACE("Loaded image: {}x{}, {} channels", width, height, channels);
        return image_data;
    }

    std::unique_ptr<ImagePreview::ImageTexture> ImagePreview::createTexture(
        ImageData&& data, const std::filesystem::path& path) {
        LOG_TIMER("CreateTexture");

        {
            LOG_TIMER("ensureMaxTextureSizeInitialized");
            ensureMaxTextureSizeInitialized();
        }

        int width = data.width();
        int height = data.height();
        int channels = data.channels();

        // Check if we need to downscale
        if (width > max_texture_size_ || height > max_texture_size_) {
            // Calculate scale factor
            int scale_factor = 1;
            while (width / scale_factor > max_texture_size_ ||
                   height / scale_factor > max_texture_size_) {
                scale_factor *= 2;
            }

            LOG_DEBUG("Image too large ({}x{}), downscaling by factor of {}",
                      width, height, scale_factor);

            // Reload at lower resolution
            auto scaled_data = loadImageData(path);
            if (!scaled_data) {
                LOG_ERROR("Failed to reload image at lower resolution");
                throw std::runtime_error("Failed to reload image at lower resolution");
            }

            // Move the scaled data
            data = std::move(*scaled_data);
            width = data.width();
            height = data.height();
            channels = data.channels();
        }

        auto texture = std::make_unique<ImageTexture>();
        texture->width = width;
        texture->height = height;
        texture->path = path;

        {
            LOG_TIMER("GL texture generation");
            // Generate texture
            if (!texture->texture.generate()) {
                LOG_ERROR("Failed to generate texture ID for: {}", lfs::core::path_to_utf8(path));
                throw std::runtime_error("Failed to generate texture ID");
            }
        }

        {
            LOG_TIMER("GL texture bind and setup");
            texture->texture.bind();

            // Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            // Set pixel alignment
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        }

        // Determine format
        GLenum format = GL_RGB;
        GLenum internal_format = GL_RGB8;

        switch (channels) {
        case 1:
            format = GL_RED;
            internal_format = GL_R8;
            break;
        case 3:
            format = GL_RGB;
            internal_format = GL_RGB8;
            break;
        case 4:
            format = GL_RGBA;
            internal_format = GL_RGBA8;
            break;
        default:
            LOG_ERROR("Unsupported channel count: {} for: {}", channels, lfs::core::path_to_utf8(path));
            throw std::runtime_error(std::format("Unsupported channel count: {}", channels));
        }

        LOG_TRACE("Creating {}x{} texture with {} channels", width, height, channels);

        // Upload texture
        {
            LOG_TIMER("glTexImage2D upload");
            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0,
                         format, GL_UNSIGNED_BYTE, data.data());
        }

        {
            // Check for errors
            GLenum error = glGetError();
            if (error != GL_NO_ERROR) {
                std::string error_str;
                switch (error) {
                case GL_INVALID_ENUM: error_str = "GL_INVALID_ENUM"; break;
                case GL_INVALID_VALUE: error_str = "GL_INVALID_VALUE"; break;
                case GL_INVALID_OPERATION: error_str = "GL_INVALID_OPERATION"; break;
                case GL_OUT_OF_MEMORY: error_str = "GL_OUT_OF_MEMORY"; break;
                default: error_str = std::format("0x{:X}", error); break;
                }
                LOG_ERROR("OpenGL error creating texture: {} for: {}", error_str, lfs::core::path_to_utf8(path));
                throw std::runtime_error(std::format("OpenGL error: {}", error_str));
            }

            // Set swizzle for grayscale
            if (channels == 1) {
                GLint swizzleMask[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
                glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
            }
        }

        // State guard will restore texture binding automatically
        LOG_DEBUG("Created texture {}x{} for: {}", width, height, lfs::core::path_to_utf8(path.filename()));
        return texture;
    }

    bool ImagePreview::loadImage(const std::filesystem::path& path) {
        try {
            is_loading_ = true;
            load_error_.clear();

            LOG_DEBUG("Loading image: {}", lfs::core::path_to_utf8(path));

            // Load image data with RAII
            auto image_data = loadImageData(path);

            // Create new texture first, then swap (keeps old texture visible during load)
            auto new_texture = createTexture(std::move(*image_data), path);
            current_texture_ = std::move(new_texture);

            is_loading_ = false;
            return true;

        } catch (const std::exception& e) {
            load_error_ = e.what();
            is_loading_ = false;
            LOG_ERROR("Error loading image '{}': {}", lfs::core::path_to_utf8(path), e.what());
            return false;
        }
    }

    void ImagePreview::preloadAdjacentImages() {
        LOG_TIMER("PreloadAdjacentImages");

        // Don't start new preload if one is already in progress
        bool expected = false;
        if (!preload_in_progress_.compare_exchange_strong(expected, true)) {
            LOG_TRACE("Preload already in progress, skipping");
            return;
        }

        // Clear existing preloaded data
        {
            std::lock_guard<std::mutex> lock(preload_mutex_);
            prev_result_.reset();
            next_result_.reset();
        }
        prev_texture_.reset();
        next_texture_.reset();

        // Ensure we have max texture size for the background threads
        ensureMaxTextureSizeInitialized();

        // Capture needed values
        GLint max_size = max_texture_size_;

        // Preload previous image
        if (current_index_ > 0) {
            LOG_TRACE("Starting preload of previous image (index {})", current_index_ - 1);
            std::thread([this, prev_idx = current_index_ - 1, max_size]() {
                try {
                    LOG_TIMER("Preload prev image data");
                    auto image_data = loadImageData(image_paths_[prev_idx]);

                    // Check if downscaling is needed
                    if (image_data->width() > max_size || image_data->height() > max_size) {
                        int scale = 2;
                        while (image_data->width() / scale > max_size ||
                               image_data->height() / scale > max_size) {
                            scale *= 2;
                        }

                        LOG_TRACE("Preloaded image needs downscaling by factor of {}", scale);
                        // Reload at lower resolution (max_width = -1 disables additional downscaling)
                        auto [data, w, h, c] = lfs::core::load_image(image_paths_[prev_idx], scale, -1);
                        image_data = std::make_unique<ImageData>(data, w, h, c);
                    }

                    auto result = std::make_unique<LoadResult>();
                    auto preloaded = std::make_unique<PreloadedImage>();
                    preloaded->data = std::move(*image_data);
                    preloaded->path = image_paths_[prev_idx];
                    result->image = std::move(preloaded);

                    std::lock_guard<std::mutex> lock(preload_mutex_);
                    prev_result_ = std::move(result);
                    LOG_TRACE("Successfully preloaded previous image data");
                } catch (const std::exception& e) {
                    auto result = std::make_unique<LoadResult>();
                    result->error = e.what();

                    std::lock_guard<std::mutex> lock(preload_mutex_);
                    prev_result_ = std::move(result);
                    LOG_WARN("Failed to preload previous image: {}", e.what());
                }
            }).detach();
        }

        // Preload next image
        if (current_index_ + 1 < image_paths_.size()) {
            LOG_TRACE("Starting preload of next image (index {})", current_index_ + 1);
            std::thread([this, next_idx = current_index_ + 1, max_size]() {
                try {
                    LOG_TIMER("Preload next image data");
                    auto image_data = loadImageData(image_paths_[next_idx]);

                    // Check if downscaling is needed
                    if (image_data->width() > max_size || image_data->height() > max_size) {
                        int scale = 2;
                        while (image_data->width() / scale > max_size ||
                               image_data->height() / scale > max_size) {
                            scale *= 2;
                        }

                        LOG_TRACE("Preloaded image needs downscaling by factor of {}", scale);
                        // Reload at lower resolution (max_width = -1 disables additional downscaling)
                        auto [data, w, h, c] = lfs::core::load_image(image_paths_[next_idx], scale, -1);
                        image_data = std::make_unique<ImageData>(data, w, h, c);
                    }

                    auto result = std::make_unique<LoadResult>();
                    auto preloaded = std::make_unique<PreloadedImage>();
                    preloaded->data = std::move(*image_data);
                    preloaded->path = image_paths_[next_idx];
                    result->image = std::move(preloaded);

                    std::lock_guard<std::mutex> lock(preload_mutex_);
                    next_result_ = std::move(result);
                    LOG_TRACE("Successfully preloaded next image data");
                } catch (const std::exception& e) {
                    auto result = std::make_unique<LoadResult>();
                    result->error = e.what();

                    std::lock_guard<std::mutex> lock(preload_mutex_);
                    next_result_ = std::move(result);
                    LOG_WARN("Failed to preload next image: {}", e.what());
                }

                preload_in_progress_ = false;
            }).detach();
        } else {
            preload_in_progress_ = false;
        }
    }

    void ImagePreview::checkPreloadedImages() {
        LOG_TIMER("CheckPreloadedImages");
        std::lock_guard<std::mutex> lock(preload_mutex_);

        // Check previous image
        if (prev_result_ && !prev_texture_) {
            if (prev_result_->image) {
                try {
                    LOG_TIMER("Create prev texture");
                    prev_texture_ = createTexture(
                        std::move(prev_result_->image->data),
                        prev_result_->image->path);
                    LOG_TRACE("Created texture for preloaded previous image");
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to create prev texture: {}", e.what());
                }
            }
            prev_result_.reset();
        }

        // Check next image
        if (next_result_ && !next_texture_) {
            if (next_result_->image) {
                try {
                    LOG_TIMER("Create next texture");
                    next_texture_ = createTexture(
                        std::move(next_result_->image->data),
                        next_result_->image->path);
                    LOG_TRACE("Created texture for preloaded next image");
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to create next texture: {}", e.what());
                }
            }
            next_result_.reset();
        }
    }

    void ImagePreview::nextImage() {
        if (image_paths_.empty() || current_index_ + 1 >= image_paths_.size()) {
            return;
        }
        if (!next_texture_ && is_loading_) {
            return; // Skip if no preload and currently loading
        }

        current_index_++;
        previous_texture_ = std::move(current_texture_); // Keep old texture in GPU memory

        if (next_texture_) {
            current_texture_ = std::move(next_texture_);
        } else {
            loadImage(image_paths_[current_index_]);
        }

        loadCurrentOverlay();
        preloadAdjacentImages();
    }

    void ImagePreview::previousImage() {
        if (image_paths_.empty() || current_index_ == 0)
            return;
        if (!prev_texture_ && is_loading_)
            return;

        --current_index_;
        previous_texture_ = std::move(current_texture_);

        if (prev_texture_) {
            current_texture_ = std::move(prev_texture_);
        } else {
            loadImage(image_paths_[current_index_]);
        }

        loadCurrentOverlay();
        preloadAdjacentImages();
    }

    void ImagePreview::goToImage(const size_t index) {
        if (index >= image_paths_.size())
            return;

        current_index_ = index;
        zoom_ = 1.0f;
        pan_x_ = 0.0f;
        pan_y_ = 0.0f;

        previous_texture_ = std::move(current_texture_);
        loadImage(image_paths_[current_index_]);
        loadCurrentOverlay();

        preloadAdjacentImages();
    }

    std::pair<float, float> ImagePreview::calculateDisplaySize(int window_width, int window_height) const {
        if (!current_texture_) {
            return {0.0f, 0.0f};
        }

        float img_width = static_cast<float>(current_texture_->width);
        float img_height = static_cast<float>(current_texture_->height);

        if (fit_to_window_) {
            float scale_x = window_width / img_width;
            float scale_y = window_height / img_height;
            float scale = std::min(scale_x, scale_y) * 0.9f;

            return {img_width * scale * zoom_, img_height * scale * zoom_};
        } else {
            return {img_width * zoom_, img_height * zoom_};
        }
    }

    void ImagePreview::render(bool* p_open) {
        if (!is_open_ || !p_open || !*p_open) {
            close();
            return;
        }

        // Check for preloaded images and convert to textures
        checkPreloadedImages();

        // Initial size: half viewport, centered
        const auto* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowSize({vp->Size.x * 0.5f, vp->Size.y * 0.5f}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos({vp->Pos.x + vp->Size.x * 0.5f, vp->Pos.y + vp->Size.y * 0.5f},
                                ImGuiCond_FirstUseEver, {0.5f, 0.5f});

        constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoScrollbar |
                                                  ImGuiWindowFlags_NoScrollWithMouse |
                                                  ImGuiWindowFlags_MenuBar;

        const std::string title = image_paths_.empty()
                                      ? "Image Preview###ImagePreview"
                                      : std::format("Image Preview - {}/{} - {}###ImagePreview",
                                                    current_index_ + 1, image_paths_.size(),
                                                    lfs::core::path_to_utf8(image_paths_[current_index_].filename()));

        if (focus_on_next_frame_) {
            ImGui::SetNextWindowFocus();
            focus_on_next_frame_ = false;
        }

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.17f, 1.0f));
        if (!ImGui::Begin(title.c_str(), p_open, WINDOW_FLAGS)) {
            ImGui::PopStyleColor();
            ImGui::End();
            return;
        }

        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu(LOC(lichtfeld::Strings::ImagePreview::VIEW))) {
                ImGui::MenuItem(LOC(lichtfeld::Strings::ImagePreview::FIT_TO_WINDOW), "F", &fit_to_window_);
                ImGui::MenuItem(LOC(lichtfeld::Strings::ImagePreview::SHOW_INFO_PANEL), "I", &show_info_panel_);
                if (hasValidOverlay()) {
                    ImGui::MenuItem(LOC(lichtfeld::Strings::ImagePreview::SHOW_MASK_OVERLAY), "M", &show_overlay_);
                }
                ImGui::Separator();
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::ImagePreview::RESET_VIEW), "R") || ImGui::MenuItem(LOC(lichtfeld::Strings::ImagePreview::ACTUAL_SIZE), "1")) {
                    zoom_ = 1.0f;
                    pan_x_ = 0.0f;
                    pan_y_ = 0.0f;
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu(LOC(lichtfeld::Strings::ImagePreview::NAVIGATE))) {
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::ImagePreview::PREVIOUS), "Left", nullptr, current_index_ > 0)) {
                    previousImage();
                }
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::ImagePreview::NEXT), "Right", nullptr,
                                    current_index_ + 1 < image_paths_.size())) {
                    nextImage();
                }
                ImGui::Separator();
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::ImagePreview::FIRST), "Home", nullptr, !image_paths_.empty())) {
                    goToImage(0);
                }
                if (ImGui::MenuItem(LOC(lichtfeld::Strings::ImagePreview::LAST), "End", nullptr, !image_paths_.empty())) {
                    goToImage(image_paths_.size() - 1);
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();
            if (!image_paths_.empty()) {
                ImGui::Text("Image %zu/%zu", current_index_ + 1, image_paths_.size());
            }

            ImGui::EndMenuBar();
        }

        const ImVec2 content_size = ImGui::GetContentRegionAvail();

        if (!load_error_.empty() && !current_texture_) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", load_error_.c_str());
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }

        if (!current_texture_ || !current_texture_->texture.valid()) {
            ImGui::Text(is_loading_ ? "Loading..." : "No image loaded");
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }

        const float scale = getDpiScale();
        const float INFO_PANEL_WIDTH = 260.0f * scale;
        const float PANEL_MARGIN = 8.0f * scale;

        // Reserve space for info panel if visible
        const float available_width = show_info_panel_ 
            ? content_size.x - INFO_PANEL_WIDTH - PANEL_MARGIN * 2.0f
            : content_size.x;

        const auto [display_width, display_height] = calculateDisplaySize(
            static_cast<int>(available_width), static_cast<int>(content_size.y));

        // Center image in available space (excluding info panel if visible)
        const float x_offset = (available_width - display_width) * 0.5f + pan_x_;
        const float y_offset = (content_size.y - display_height) * 0.5f + pan_y_;
        const float base_cursor_y = ImGui::GetCursorPosY();

        // Invisible button for mouse interaction
        ImGui::SetCursorPos(ImVec2(0, base_cursor_y));
        ImGui::InvisibleButton("##ImageArea", content_size);
        const bool image_hovered = ImGui::IsItemHovered();
        const bool image_active = ImGui::IsItemActive();

        ImGui::SetCursorPos(ImVec2(x_offset, y_offset + base_cursor_y));
        ImGui::Image(static_cast<ImTextureID>(current_texture_->texture.id()),
                     ImVec2(display_width, display_height));

        // Draw mask overlay
        if (show_overlay_ && hasValidOverlay()) {
            constexpr ImVec4 OVERLAY_TINT{1.0f, 0.2f, 0.2f, 0.5f};
            ImGui::SetCursorPos(ImVec2(x_offset, y_offset + base_cursor_y));
            ImGui::Image(static_cast<ImTextureID>(overlay_texture_->texture.id()),
                         ImVec2(display_width, display_height),
                         ImVec2(0, 0), ImVec2(1, 1), OVERLAY_TINT, ImVec4(0, 0, 0, 0));
        }

        // Floating info panel
        if (show_info_panel_ && current_texture_) {
            const float MAX_PANEL_HEIGHT = 400.0f * scale;
            const float panel_height = std::min(content_size.y - 2.0f * PANEL_MARGIN, MAX_PANEL_HEIGHT);
            ImGui::SetCursorPos(ImVec2(content_size.x - INFO_PANEL_WIDTH - PANEL_MARGIN, base_cursor_y + PANEL_MARGIN));

            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.1f, 0.12f, 0.85f));
            ImGui::BeginChild("##ImageInfoPanel", ImVec2(INFO_PANEL_WIDTH, panel_height), true);

            const auto& path = image_paths_[current_index_];
            const std::string filename = lfs::core::path_to_utf8(path.filename());
            std::string ext = path.extension().string();
            if (!ext.empty() && ext[0] == '.')
                ext = ext.substr(1);
            for (char& c : ext)
                c = static_cast<char>(std::toupper(c));

            // File info section
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "FILE");
            ImGui::Separator();
            ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::NAME), filename.c_str());
            ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::FORMAT), ext.c_str());
            if (std::filesystem::exists(path)) {
                const auto file_size = std::filesystem::file_size(path);
                if (file_size >= 1024 * 1024)
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::SIZE_MB), file_size / (1024.0 * 1024.0));
                else if (file_size >= 1024)
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::SIZE_KB), file_size / 1024.0);
                else
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::SIZE_BYTES), file_size);

                const auto ftime = std::filesystem::last_write_time(path);
                const auto sys_time = std::chrono::clock_cast<std::chrono::system_clock>(ftime);
                const auto time_t = std::chrono::system_clock::to_time_t(sys_time);
                char time_buf[64];
                std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M", std::localtime(&time_t));
                ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::MODIFIED), time_buf);
            }
            ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::PATH), lfs::core::path_to_utf8(path.parent_path()).c_str());

            ImGui::Spacing();
            ImGui::Spacing();

            // Image info section
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "IMAGE");
            ImGui::Separator();
            ImGui::Text("Dimensions: %dx%d", current_texture_->width, current_texture_->height);
            ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::MEGAPIXELS),
                        (current_texture_->width * current_texture_->height) / 1e6);

            // Infer channels from extension
            const char* channels = "RGB (3)";
            const char* color_space = "sRGB";
            if (ext == "PNG" || ext == "WEBP" || ext == "TIFF" || ext == "TIF") {
                channels = "RGBA (4)";
            } else if (ext == "EXR" || ext == "HDR") {
                channels = "RGBA (4)";
                color_space = "Linear";
            }
            ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::CHANNELS), channels);
            ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::COLOR_SPACE), color_space);

            // Aspect ratio
            const float aspect = static_cast<float>(current_texture_->width) /
                                 static_cast<float>(current_texture_->height);
            const char* aspect_name = "Custom";
            if (std::abs(aspect - 16.0f / 9.0f) < 0.01f)
                aspect_name = "16:9";
            else if (std::abs(aspect - 4.0f / 3.0f) < 0.01f)
                aspect_name = "4:3";
            else if (std::abs(aspect - 3.0f / 2.0f) < 0.01f)
                aspect_name = "3:2";
            else if (std::abs(aspect - 1.0f) < 0.01f)
                aspect_name = "1:1";
            else if (std::abs(aspect - 21.0f / 9.0f) < 0.02f)
                aspect_name = "21:9";
            ImGui::Text("Aspect Ratio: %s (%.2f)", aspect_name, aspect);

            // EXIF section (only for JPEG files with valid EXIF)
            if (exif_cache_index_ != current_index_) {
                current_exif_ = parseExifData(path);
                exif_cache_index_ = current_index_;
            }

            if (current_exif_.valid) {
                ImGui::Spacing();
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "EXIF");
                ImGui::Separator();

                if (!current_exif_.camera_make.empty() || !current_exif_.camera_model.empty()) {
                    std::string camera;
                    if (!current_exif_.camera_make.empty())
                        camera = current_exif_.camera_make;
                    if (!current_exif_.camera_model.empty()) {
                        if (!camera.empty())
                            camera += " ";
                        camera += current_exif_.camera_model;
                    }
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::CAMERA), camera.c_str());
                }
                if (!current_exif_.lens_model.empty())
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::LENS), current_exif_.lens_model.c_str());
                if (!current_exif_.focal_length.empty())
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::FOCAL_LENGTH), current_exif_.focal_length.c_str());
                if (!current_exif_.focal_length_35mm.empty())
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::FOCAL_35MM), current_exif_.focal_length_35mm.c_str());
                if (!current_exif_.exposure_time.empty())
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::EXPOSURE), current_exif_.exposure_time.c_str());
                if (!current_exif_.f_number.empty())
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::APERTURE), current_exif_.f_number.c_str());
                if (!current_exif_.iso.empty())
                    ImGui::Text("%s", current_exif_.iso.c_str());
                if (!current_exif_.date_time.empty())
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::DATE), current_exif_.date_time.c_str());
                if (!current_exif_.software.empty())
                    ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::SOFTWARE), current_exif_.software.c_str());
            }

            ImGui::Spacing();
            ImGui::Spacing();

            // View info section
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "VIEW");
            ImGui::Separator();
            ImGui::Text("Zoom: %.0f%%", zoom_ * 100.0f);
            ImGui::Text("Pan: %.0f, %.0f", pan_x_, pan_y_);
            ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::FIT_STATUS), fit_to_window_ ? LOC(Training::Status::YES) : LOC(Training::Status::NO));

            if (hasValidOverlay()) {
                ImGui::Spacing();
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", LOC(lichtfeld::Strings::ImagePreview::MASK_SECTION));
                ImGui::Separator();
                ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::OVERLAY_STATUS), show_overlay_ ? LOC(lichtfeld::Strings::ImagePreview::VISIBLE) : LOC(lichtfeld::Strings::ImagePreview::HIDDEN));
                ImGui::Text(LOC(lichtfeld::Strings::ImagePreview::FILE_LABEL), lfs::core::path_to_utf8(overlay_paths_[current_index_].filename()).c_str());
            }

            ImGui::EndChild();
            ImGui::PopStyleColor();
        }

        // Mouse interaction
        if (image_hovered) {
            if (const float wheel = ImGui::GetIO().MouseWheel; wheel != 0.0f) {
                constexpr float ZOOM_SPEED = 0.15f;
                zoom_ = std::clamp(zoom_ + wheel * ZOOM_SPEED, 0.1f, 10.0f);
            }
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                zoom_ = 1.0f;
                pan_x_ = 0.0f;
                pan_y_ = 0.0f;
            }
        }

        if (image_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            const ImVec2 delta = ImGui::GetIO().MouseDelta;
            pan_x_ += delta.x;
            pan_y_ += delta.y;
        }

        if (ImGui::IsWindowFocused()) {
            // Disable key repeat for navigation (false = no repeat)
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false)) {
                previousImage();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false)) {
                nextImage();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Home)) {
                goToImage(0);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_End) && !image_paths_.empty()) {
                goToImage(image_paths_.size() - 1);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_F)) {
                fit_to_window_ = !fit_to_window_;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_I)) {
                show_info_panel_ = !show_info_panel_;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_M) && hasValidOverlay()) {
                show_overlay_ = !show_overlay_;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_R) || ImGui::IsKeyPressed(ImGuiKey_1)) {
                zoom_ = 1.0f;
                pan_x_ = 0.0f;
                pan_y_ = 0.0f;
            }

            if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl)) {
                if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd)) {
                    zoom_ = std::clamp(zoom_ + 0.1f, 0.1f, 10.0f);
                }
                if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract)) {
                    zoom_ = std::clamp(zoom_ - 0.1f, 0.1f, 10.0f);
                }
                if (ImGui::IsKeyPressed(ImGuiKey_0) || ImGui::IsKeyPressed(ImGuiKey_Keypad0)) {
                    zoom_ = 1.0f;
                }
            }
        }

        ImGui::End();
        ImGui::PopStyleColor();
    }

} // namespace lfs::vis::gui