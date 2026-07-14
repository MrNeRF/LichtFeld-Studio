/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preprocessing/preprocess.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "depth_anchor_cache.hpp"
#include "io/loader.hpp"

#include "indicators.hpp"
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>
#include <curl/curl.h>
#include <onnxruntime_cxx_api.h>
#include <openssl/evp.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

    constexpr std::string_view kDefaultModelFile = "moge-2-vitb-normal.onnx";
    constexpr std::string_view kDefaultModelUrl =
        "https://github.com/MrNeRF/LichtFeld-Studio/releases/download/model-moge2-v1/moge-2-vitb-normal.onnx";
    constexpr std::string_view kFallbackModelUrl =
        "https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx/resolve/main/model.onnx";
    constexpr std::string_view kDefaultModelSha256 =
        "bbf14e07a30f11e69d36ab861590123f5598ababcbc8946a063eb4a966f35a21";

    bool stdout_is_tty() {
#ifdef _WIN32
        return _isatty(_fileno(stdout)) != 0;
#else
        return isatty(fileno(stdout)) != 0;
#endif
    }

    void style_progress_bar(indicators::ProgressBar& bar, std::string prefix) {
        bar.set_option(indicators::option::Start("["));
#ifdef _WIN32
        bar.set_option(indicators::option::BarWidth(38));
        bar.set_option(indicators::option::Fill("="));
        bar.set_option(indicators::option::Lead(">"));
        bar.set_option(indicators::option::Remainder(" "));
#else
        bar.set_option(indicators::option::BarWidth(40));
        bar.set_option(indicators::option::Fill("█"));
        bar.set_option(indicators::option::Lead("▌"));
        bar.set_option(indicators::option::Remainder("░"));
#endif
        bar.set_option(indicators::option::End("]"));
        bar.set_option(indicators::option::PrefixText(std::move(prefix)));
        bar.set_option(indicators::option::ShowPercentage(true));
        bar.set_option(indicators::option::ShowElapsedTime(true));
        bar.set_option(indicators::option::ShowRemainingTime(true));
        bar.set_option(indicators::option::ForegroundColor(indicators::Color::cyan));
        bar.set_option(indicators::option::FontStyles(
            std::vector<indicators::FontStyle>{indicators::FontStyle::bold}));
    }

    struct Image {
        int width = 0;
        int height = 0;
        std::vector<float> rgb_hwc;
    };

    struct SpatialMap {
        int width = 0;
        int height = 0;
        std::vector<float> values;
    };

    struct VectorMap {
        int width = 0;
        int height = 0;
        std::vector<float> xyz_hwc;
    };

    struct OrtOutputs {
        SpatialMap mask;
        VectorMap points;
        VectorMap normals;
    };

    std::string path_to_string(const fs::path& path) {
        return lfs::core::path_to_utf8(path);
    }

    std::string lower_copy(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    }

    bool is_image_file(const fs::path& path) {
        const auto ext = lower_copy(path.extension().string());
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
               ext == ".bmp" || ext == ".tif" || ext == ".tiff" ||
               ext == ".webp";
    }

    fs::path home_directory() {
#ifdef _WIN32
        if (const char* profile = std::getenv("USERPROFILE"); profile && profile[0])
            return fs::path(profile);
        const char* drive = std::getenv("HOMEDRIVE");
        const char* homepath = std::getenv("HOMEPATH");
        if (drive && drive[0] && homepath && homepath[0])
            return fs::path(std::string(drive) + homepath);
#else
        if (const char* home = std::getenv("HOME"); home && home[0])
            return fs::path(home);
#endif
        return fs::temp_directory_path();
    }

    fs::path default_model_path() {
        return home_directory() / ".lichtfeld" / "onnx" / std::string(kDefaultModelFile);
    }

    fs::path legacy_model_path() {
        fs::path root;
#ifdef _WIN32
        if (const char* local = std::getenv("LOCALAPPDATA"); local && local[0])
            root = fs::path(local) / "LichtFeld";
        else if (const char* temp = std::getenv("TEMP"); temp && temp[0])
            root = fs::path(temp) / "LichtFeld";
        else
            root = fs::temp_directory_path() / "LichtFeld";
#else
        if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0])
            root = fs::path(xdg) / "lichtfeld";
        else if (const char* home = std::getenv("HOME"); home && home[0])
            root = fs::path(home) / ".cache" / "lichtfeld";
        else
            root = fs::temp_directory_path() / "lichtfeld";
#endif
        return root / "models" / "Ruicheng" / "moge-2-vitb-normal-onnx" / "model.onnx";
    }

    class DownloadIntegrityError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    std::string sha256_file(const fs::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("Could not open " + path_to_string(path) + " for hashing");

        std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
        if (!ctx)
            throw std::runtime_error("EVP_MD_CTX_new failed");
        if (EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("EVP_DigestInit_ex failed");

        std::vector<char> buffer(1 << 20);
        while (input) {
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const auto count = input.gcount();
            if (count > 0 && EVP_DigestUpdate(ctx.get(), buffer.data(), static_cast<std::size_t>(count)) != 1)
                throw std::runtime_error("EVP_DigestUpdate failed");
        }

        std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
        unsigned int digest_len = 0;
        if (EVP_DigestFinal_ex(ctx.get(), digest.data(), &digest_len) != 1)
            throw std::runtime_error("EVP_DigestFinal_ex failed");

        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (unsigned int i = 0; i < digest_len; ++i) {
            const auto byte = digest[i];
            out << std::setw(2) << static_cast<int>(byte);
        }
        return out.str();
    }

    void remove_file_if_exists(const fs::path& path) {
        std::error_code ec;
        fs::remove(path, ec);
    }

    void require_sha256(const fs::path& path,
                        std::string_view expected_hash,
                        std::string_view label) {
        const auto hash = sha256_file(path);
        if (hash == expected_hash)
            return;

        throw DownloadIntegrityError(std::string(label) + " SHA-256 mismatch for " +
                                     path_to_string(path) + ": expected " +
                                     std::string(expected_hash) + ", got " + hash);
    }

    void replace_file(const fs::path& source, const fs::path& destination) {
        std::error_code ec;
        fs::rename(source, destination, ec);
        if (!ec)
            return;

        std::error_code exists_ec;
        if (fs::exists(destination, exists_ec)) {
            ec.clear();
            fs::remove(destination, ec);
            if (ec) {
                remove_file_if_exists(source);
                throw std::runtime_error("Could not replace " + path_to_string(destination) +
                                         ": " + ec.message());
            }

            ec.clear();
            fs::rename(source, destination, ec);
        }

        if (ec) {
            remove_file_if_exists(source);
            throw std::runtime_error("Could not move verified download to " +
                                     path_to_string(destination) + ": " + ec.message());
        }
    }

    size_t curl_write(void* ptr, size_t size, size_t nmemb, void* userdata) {
        auto* out = static_cast<std::ofstream*>(userdata);
        const auto bytes = size * nmemb;
        out->write(static_cast<const char*>(ptr), static_cast<std::streamsize>(bytes));
        return out->good() ? bytes : 0;
    }

    struct DownloadProgress {
        std::unique_ptr<indicators::ProgressBar> bar;
        int last_percent = -1;
        int next_plain_print = 10;
    };

    int curl_progress(void* userdata, curl_off_t dltotal, curl_off_t dlnow, curl_off_t, curl_off_t) {
        auto* progress = static_cast<DownloadProgress*>(userdata);
        if (dltotal <= 0)
            return 0;
        const int percent = static_cast<int>(dlnow * 100 / dltotal);
        if (percent == progress->last_percent)
            return 0;
        progress->last_percent = percent;
        if (progress->bar) {
            progress->bar->set_option(indicators::option::PostfixText(
                std::format("{}/{} MiB", dlnow >> 20, dltotal >> 20)));
            progress->bar->set_progress(static_cast<size_t>(percent));
        } else if (percent >= progress->next_plain_print) {
            std::cout << "  " << percent << "% (" << (dlnow >> 20) << "/" << (dltotal >> 20) << " MiB)\n";
            progress->next_plain_print = percent - percent % 10 + 10;
        }
        return 0;
    }

    void download_verified_file(std::string_view url,
                                const fs::path& destination,
                                std::string_view expected_hash) {
        fs::create_directories(destination.parent_path());
        const fs::path tmp_path = destination.string() + ".tmp";

        std::ofstream output(tmp_path, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("Could not open " + path_to_string(tmp_path) + " for writing");

        curl_global_init(CURL_GLOBAL_DEFAULT);
        CURL* curl = curl_easy_init();
        if (!curl) {
            curl_global_cleanup();
            output.close();
            remove_file_if_exists(tmp_path);
            throw std::runtime_error("curl_easy_init failed");
        }

        const std::string url_string(url);
        curl_easy_setopt(curl, CURLOPT_URL, url_string.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "LichtFeld-Studio/preprocess");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &output);

        DownloadProgress progress;
        if (stdout_is_tty()) {
            progress.bar = std::make_unique<indicators::ProgressBar>();
            style_progress_bar(*progress.bar, "Downloading ");
        }
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_progress);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress);

        const CURLcode result = curl_easy_perform(curl);
        if (progress.bar && !progress.bar->is_completed()) {
            progress.bar->mark_as_completed();
            std::cout << "\n";
        }
        long response_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        output.close();

        if (result != CURLE_OK) {
            remove_file_if_exists(tmp_path);
            throw std::runtime_error("Download failed: " + std::string(curl_easy_strerror(result)));
        }
        if (response_code >= 400) {
            remove_file_if_exists(tmp_path);
            throw std::runtime_error("Download failed with HTTP " + std::to_string(response_code));
        }

        try {
            require_sha256(tmp_path, expected_hash, "Downloaded model");
        } catch (...) {
            remove_file_if_exists(tmp_path);
            throw;
        }

        replace_file(tmp_path, destination);
    }

    fs::path ensure_default_model(bool no_download) {
        const fs::path path = default_model_path();
        if (fs::is_regular_file(path)) {
            try {
                require_sha256(path, kDefaultModelSha256, "Cached model");
                return path;
            } catch (const DownloadIntegrityError& e) {
                if (no_download)
                    throw;

                std::cerr << e.what() << "\n";
                remove_file_if_exists(path);
                std::cerr << "Removed untrusted cached model; re-downloading "
                          << path_to_string(path) << "\n";
            }
        } else {
            const fs::path legacy = legacy_model_path();
            if (fs::is_regular_file(legacy)) {
                try {
                    require_sha256(legacy, kDefaultModelSha256, "Legacy cached model");
                    fs::create_directories(path.parent_path());
                    std::error_code ec;
                    fs::rename(legacy, path, ec);
                    if (ec)
                        fs::copy_file(legacy, path, fs::copy_options::overwrite_existing);
                    require_sha256(path, kDefaultModelSha256, "Cached model");
                    std::cout << "Migrated cached model to " << path_to_string(path) << "\n";
                    return path;
                } catch (const DownloadIntegrityError& e) {
                    std::cerr << e.what() << "\n";
                    std::cerr << "Ignoring untrusted legacy cached model at "
                              << path_to_string(legacy) << "\n";
                }
            }
            if (no_download) {
                throw std::runtime_error("Default model is not cached: " + path_to_string(path));
            }
        }

        std::cout << "Downloading MoGe-2 ViT-B normal model (MIT license, (c) Microsoft) to "
                  << path_to_string(path) << "\n";
        try {
            download_verified_file(kDefaultModelUrl, path, kDefaultModelSha256);
        } catch (const DownloadIntegrityError&) {
            throw;
        } catch (const std::exception& e) {
            std::cerr << "Primary download failed (" << e.what() << "); retrying from "
                      << kFallbackModelUrl << "\n";
            download_verified_file(kFallbackModelUrl, path, kDefaultModelSha256);
        }
        require_sha256(path, kDefaultModelSha256, "Cached model");
        return path;
    }

    Image load_image_rgb(const fs::path& path) {
        auto input = OIIO::ImageInput::open(path_to_string(path));
        if (!input)
            throw std::runtime_error("Failed to open image: " + path_to_string(path) + ": " + OIIO::geterror());

        const OIIO::ImageSpec spec = input->spec();
        if (spec.width <= 0 || spec.height <= 0 || spec.nchannels <= 0)
            throw std::runtime_error("Invalid image shape for " + path_to_string(path));

        Image out;
        out.width = spec.width;
        out.height = spec.height;
        out.rgb_hwc.resize(static_cast<std::size_t>(out.width) * out.height * 3);

        auto fill_rgb = [&](const auto& raw, float scale) {
            for (int y = 0; y < out.height; ++y) {
                for (int x = 0; x < out.width; ++x) {
                    const std::size_t src = (static_cast<std::size_t>(y) * out.width + x) * spec.nchannels;
                    const std::size_t dst = (static_cast<std::size_t>(y) * out.width + x) * 3;
                    out.rgb_hwc[dst + 0] = static_cast<float>(raw[src + 0]) * scale;
                    out.rgb_hwc[dst + 1] = static_cast<float>(spec.nchannels > 1 ? raw[src + 1] : raw[src + 0]) * scale;
                    out.rgb_hwc[dst + 2] = static_cast<float>(spec.nchannels > 2 ? raw[src + 2] : raw[src + 0]) * scale;
                }
            }
        };

        const std::size_t raw_values = static_cast<std::size_t>(spec.width) * spec.height * spec.nchannels;
        if (spec.format == OIIO::TypeDesc::UINT8) {
            std::vector<std::uint8_t> raw(raw_values);
            if (!input->read_image(0, 0, 0, spec.nchannels, OIIO::TypeDesc::UINT8, raw.data())) {
                const auto error = input->geterror();
                input->close();
                throw std::runtime_error("Failed to read image: " + path_to_string(path) + ": " + error);
            }
            input->close();
            fill_rgb(raw, 1.0f / 255.0f);
            return out;
        }

        if (spec.format == OIIO::TypeDesc::UINT16) {
            std::vector<std::uint16_t> raw(raw_values);
            if (!input->read_image(0, 0, 0, spec.nchannels, OIIO::TypeDesc::UINT16, raw.data())) {
                const auto error = input->geterror();
                input->close();
                throw std::runtime_error("Failed to read image: " + path_to_string(path) + ": " + error);
            }
            input->close();
            fill_rgb(raw, 1.0f / 65535.0f);
            return out;
        }

        std::vector<float> raw(raw_values);
        if (!input->read_image(0, 0, 0, spec.nchannels, OIIO::TypeDesc::FLOAT, raw.data())) {
            const auto error = input->geterror();
            input->close();
            throw std::runtime_error("Failed to read image: " + path_to_string(path) + ": " + error);
        }
        input->close();

        float max_channel = 0.0f;
        for (int y = 0; y < out.height; ++y) {
            for (int x = 0; x < out.width; ++x) {
                const std::size_t src = (static_cast<std::size_t>(y) * out.width + x) * spec.nchannels;
                const std::size_t dst = (static_cast<std::size_t>(y) * out.width + x) * 3;
                out.rgb_hwc[dst + 0] = raw[src + 0];
                out.rgb_hwc[dst + 1] = spec.nchannels > 1 ? raw[src + 1] : raw[src + 0];
                out.rgb_hwc[dst + 2] = spec.nchannels > 2 ? raw[src + 2] : raw[src + 0];
                max_channel = std::max({max_channel, out.rgb_hwc[dst + 0], out.rgb_hwc[dst + 1], out.rgb_hwc[dst + 2]});
            }
        }

        const float scale = max_channel > 255.5f ? (1.0f / 65535.0f)
                            : max_channel > 1.5f ? (1.0f / 255.0f)
                                                 : 1.0f;
        if (scale != 1.0f) {
            for (float& channel : out.rgb_hwc)
                channel = std::clamp(channel * scale, 0.0f, 1.0f);
        }
        return out;
    }

    int resolve_thread_count(int requested_threads) {
        if (requested_threads > 0)
            return requested_threads;
        return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }

    int round_to_patch_multiple(int value) {
        constexpr int patch = 14;
        return std::max(patch, (value + patch / 2) / patch * patch);
    }

    Image resize_for_inference(const Image& image, int max_side) {
        if (max_side == 0)
            return image;

        int new_width = image.width;
        int new_height = image.height;

        if (std::max(image.width, image.height) > max_side) {
            const float scale = static_cast<float>(max_side) /
                                static_cast<float>(std::max(image.width, image.height));
            new_width = static_cast<int>(std::round(image.width * scale));
            new_height = static_cast<int>(std::round(image.height * scale));
        }

        new_width = round_to_patch_multiple(new_width);
        new_height = round_to_patch_multiple(new_height);

        if (new_width == image.width && new_height == image.height)
            return image;

        Image resized;
        resized.width = new_width;
        resized.height = new_height;
        resized.rgb_hwc.resize(static_cast<std::size_t>(new_width) * new_height * 3);

        OIIO::ImageBuf src(OIIO::ImageSpec(image.width, image.height, 3, OIIO::TypeDesc::FLOAT),
                           const_cast<float*>(image.rgb_hwc.data()));
        OIIO::ImageBuf dst(OIIO::ImageSpec(new_width, new_height, 3, OIIO::TypeDesc::FLOAT),
                           resized.rgb_hwc.data());
        OIIO::ROI roi(0, new_width, 0, new_height, 0, 1, 0, 3);
        if (!OIIO::ImageBufAlgo::resample(dst, src, true, roi, 0))
            throw std::runtime_error("Image resize failed: " + dst.geterror());
        return resized;
    }

    std::vector<float> hwc_to_nchw(const Image& image) {
        std::vector<float> chw(static_cast<std::size_t>(3) * image.width * image.height);
        const std::size_t plane = static_cast<std::size_t>(image.width) * image.height;
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                const std::size_t src = (static_cast<std::size_t>(y) * image.width + x) * 3;
                const std::size_t dst = static_cast<std::size_t>(y) * image.width + x;
                chw[dst] = image.rgb_hwc[src + 0];
                chw[plane + dst] = image.rgb_hwc[src + 1];
                chw[2 * plane + dst] = image.rgb_hwc[src + 2];
            }
        }
        return chw;
    }

    std::vector<fs::path> collect_images(const fs::path& images_dir, bool recursive) {
        if (!fs::is_directory(images_dir))
            throw std::runtime_error("Images directory does not exist: " + path_to_string(images_dir));

        std::vector<fs::path> images;
        const auto add_if_image = [&images](const fs::directory_entry& entry) {
            if (entry.is_regular_file() && is_image_file(entry.path()))
                images.push_back(entry.path());
        };
        if (recursive) {
            for (fs::recursive_directory_iterator it(images_dir), end; it != end; ++it)
                add_if_image(*it);
        } else {
            for (fs::directory_iterator it(images_dir), end; it != end; ++it)
                add_if_image(*it);
        }
        std::sort(images.begin(), images.end());
        return images;
    }

    bool is_same_directory(const fs::path& lhs, const fs::path& rhs) {
        std::error_code ec;
        if (fs::equivalent(lhs, rhs, ec) && !ec)
            return true;
        return lhs.lexically_normal() == rhs.lexically_normal();
    }

    fs::path output_path_for(const fs::path& dataset_root,
                             std::string_view folder,
                             const fs::path& image_path,
                             const fs::path& images_dir) {
        fs::path rel = image_path.lexically_relative(images_dir);
        rel.replace_extension(".png");
        return dataset_root / std::string(folder) / rel;
    }

    template <typename PixelT>
    void write_png(const fs::path& path,
                   int width,
                   int height,
                   int channels,
                   const std::vector<PixelT>& data,
                   int png_compression) {
        static_assert(std::is_same_v<PixelT, uint8_t> || std::is_same_v<PixelT, uint16_t>);
        constexpr auto kFormat = std::is_same_v<PixelT, uint16_t> ? OIIO::TypeDesc::UINT16
                                                                  : OIIO::TypeDesc::UINT8;
        if (channels < 1 || channels > 4)
            throw std::runtime_error("Internal error: invalid image channel count");
        if (data.size() != static_cast<std::size_t>(width) * height * channels)
            throw std::runtime_error("Internal error: image buffer has wrong size");

        fs::create_directories(path.parent_path());
        auto output = OIIO::ImageOutput::create(path_to_string(path));
        if (!output)
            throw std::runtime_error("Failed to create image output: " + path_to_string(path));

        OIIO::ImageSpec spec(width, height, channels, kFormat);
        const int compression = std::clamp(png_compression, 0, 9);
        spec.attribute("png:compressionLevel", compression);
        if (compression == 0) {
            spec.attribute("compression", "none");
        } else if (compression == 1) {
            spec.attribute("compression", "pngfast");
        }
        if (!output->open(path_to_string(path), spec)) {
            const auto error = output->geterror();
            output->close();
            throw std::runtime_error("Failed to open image output: " + path_to_string(path) + ": " + error);
        }
        if (!output->write_image(kFormat, data.data())) {
            const auto error = output->geterror();
            output->close();
            throw std::runtime_error("Failed to write image output: " + path_to_string(path) + ": " + error);
        }
        output->close();
    }

    std::vector<std::string> get_input_names(Ort::Session& session) {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> names;
        const auto count = session.GetInputCount();
        names.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            names.emplace_back(name.get());
        }
        return names;
    }

    std::vector<std::string> get_output_names(Ort::Session& session) {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> names;
        const auto count = session.GetOutputCount();
        names.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            names.emplace_back(name.get());
        }
        return names;
    }

    std::vector<int64_t> tensor_shape(const Ort::Value& value) {
        return value.GetTensorTypeAndShapeInfo().GetShape();
    }

    std::size_t tensor_element_count(const Ort::Value& value) {
        return value.GetTensorTypeAndShapeInfo().GetElementCount();
    }

    SpatialMap extract_mask(const Ort::Value& value, int fallback_width, int fallback_height) {
        const auto shape = tensor_shape(value);
        const auto count = tensor_element_count(value);
        const float* data = value.GetTensorData<float>();

        int width = fallback_width;
        int height = fallback_height;

        if (shape.size() == 2) {
            height = static_cast<int>(shape[0]);
            width = static_cast<int>(shape[1]);
        } else if (shape.size() == 3) {
            height = static_cast<int>(shape[shape.size() - 2]);
            width = static_cast<int>(shape[shape.size() - 1]);
        } else if (shape.size() == 4) {
            if (shape[3] == 1) {
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
            } else if (shape[1] == 1) {
                height = static_cast<int>(shape[2]);
                width = static_cast<int>(shape[3]);
            }
        }

        const std::size_t expected = static_cast<std::size_t>(width) * height;
        if (count < expected)
            throw std::runtime_error("Mask output has fewer elements than expected");

        SpatialMap mask;
        mask.width = width;
        mask.height = height;
        mask.values.assign(data, data + expected);
        return mask;
    }

    VectorMap extract_vector3(const Ort::Value& value,
                              int fallback_width,
                              int fallback_height,
                              std::string_view label) {
        const auto shape = tensor_shape(value);
        const auto count = tensor_element_count(value);
        const float* data = value.GetTensorData<float>();

        int width = fallback_width;
        int height = fallback_height;
        bool chw = false;

        if (shape.size() == 3 && shape[2] == 3) {
            height = static_cast<int>(shape[0]);
            width = static_cast<int>(shape[1]);
        } else if (shape.size() == 4 && shape[3] == 3) {
            height = static_cast<int>(shape[1]);
            width = static_cast<int>(shape[2]);
        } else if (shape.size() == 4 && shape[1] == 3) {
            chw = true;
            height = static_cast<int>(shape[2]);
            width = static_cast<int>(shape[3]);
        }

        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        if (count < pixels * 3)
            throw std::runtime_error(std::string(label) + " output has fewer elements than expected");

        VectorMap out;
        out.width = width;
        out.height = height;
        out.xyz_hwc.resize(pixels * 3);

        if (!chw) {
            std::copy(data, data + pixels * 3, out.xyz_hwc.begin());
        } else {
            for (std::size_t i = 0; i < pixels; ++i) {
                out.xyz_hwc[i * 3 + 0] = data[i];
                out.xyz_hwc[i * 3 + 1] = data[pixels + i];
                out.xyz_hwc[i * 3 + 2] = data[2 * pixels + i];
            }
        }
        return out;
    }

    Ort::SessionOptions make_session_options(int thread_count, bool use_cuda) {
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetIntraOpNumThreads(thread_count);
        if (use_cuda) {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
            cuda_options.do_copy_in_default_stream = 1;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
        return session_options;
    }

    class MogeOnnxSession {
    public:
        MogeOnnxSession(const fs::path& model_path, int thread_count, bool force_cpu)
            : env_(ORT_LOGGING_LEVEL_WARNING, "LichtFeld-Studio-preprocess") {
#ifdef _WIN32
            const auto model_string = model_path.wstring();
#else
            const auto model_string = path_to_string(model_path);
#endif
            bool use_cuda = !force_cpu;
            while (!session_) {
                try {
                    const auto session_options = make_session_options(thread_count, use_cuda);
                    session_ = std::make_unique<Ort::Session>(env_, model_string.c_str(), session_options);
                } catch (const Ort::Exception& e) {
                    if (!use_cuda)
                        throw;
                    std::cerr << "CUDA execution provider unavailable, falling back to CPU: " << e.what() << "\n";
                    use_cuda = false;
                }
            }
            provider_ = use_cuda ? "CUDA" : "CPU";
            input_names_ = get_input_names(*session_);
            output_names_ = get_output_names(*session_);

            bool has_image = false;
            for (const auto& name : input_names_)
                has_image = has_image || name == "image";
            if (!has_image)
                throw std::runtime_error("ONNX model has no 'image' input");
        }

        OrtOutputs run(const Image& image, int64_t num_tokens) {
            std::vector<float> chw = hwc_to_nchw(image);
            std::array<int64_t, 4> image_shape = {1, 3, image.height, image.width};
            std::vector<int64_t> scalar_shape;
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            std::vector<Ort::Value> input_values;
            input_values.reserve(input_names_.size());
            for (const auto& name : input_names_) {
                if (name == "image") {
                    input_values.emplace_back(Ort::Value::CreateTensor<float>(
                        memory_info, chw.data(), chw.size(), image_shape.data(), image_shape.size()));
                } else if (name == "num_tokens") {
                    input_values.emplace_back(Ort::Value::CreateTensor<int64_t>(
                        memory_info, &num_tokens, 1, nullptr, scalar_shape.size()));
                } else {
                    throw std::runtime_error("Unsupported ONNX input: " + name);
                }
            }

            std::vector<const char*> input_name_ptrs;
            input_name_ptrs.reserve(input_names_.size());
            for (const auto& name : input_names_)
                input_name_ptrs.push_back(name.c_str());

            std::vector<const char*> output_name_ptrs;
            output_name_ptrs.reserve(output_names_.size());
            for (const auto& name : output_names_)
                output_name_ptrs.push_back(name.c_str());

            Ort::RunOptions run_options;
            auto outputs = session_->Run(run_options,
                                         input_name_ptrs.data(),
                                         input_values.data(),
                                         input_values.size(),
                                         output_name_ptrs.data(),
                                         output_name_ptrs.size());

            std::unordered_map<std::string, std::size_t> output_index;
            for (std::size_t i = 0; i < output_names_.size(); ++i)
                output_index.emplace(output_names_[i], i);

            auto get_output = [&](std::string_view name) -> const Ort::Value& {
                const auto it = output_index.find(std::string(name));
                if (it == output_index.end())
                    throw std::runtime_error("ONNX model did not produce '" + std::string(name) + "'");
                return outputs[it->second];
            };

            return OrtOutputs{
                .mask = extract_mask(get_output("mask"), image.width, image.height),
                .points = extract_vector3(get_output("points"), image.width, image.height, "points"),
                .normals = extract_vector3(get_output("normal"), image.width, image.height, "normal"),
            };
        }

        std::string_view provider() const { return provider_; }

    private:
        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
        std::vector<std::string> input_names_;
        std::vector<std::string> output_names_;
        std::string_view provider_;
    };

    template <typename PixelT>
    std::vector<PixelT> build_depth_png(const SpatialMap& mask,
                                        const VectorMap& points,
                                        int out_width,
                                        int out_height) {
        constexpr long kMaxValue = static_cast<long>(std::numeric_limits<PixelT>::max());
        if (mask.width != points.width || mask.height != points.height)
            throw std::runtime_error("Mask and point outputs have different spatial sizes");

        const std::size_t pixels = static_cast<std::size_t>(points.width) * points.height;
        float min_z = std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();

        for (std::size_t i = 0; i < pixels; ++i) {
            const float valid = mask.values[i];
            const float z = points.xyz_hwc[i * 3 + 2];
            if (std::isfinite(valid) && valid >= 0.5f && std::isfinite(z) && z > 0.0f) {
                min_z = std::min(min_z, z);
                max_z = std::max(max_z, z);
            }
        }

        std::vector<PixelT> low(pixels, 0);
        if (std::isfinite(min_z) && std::isfinite(max_z)) {
            const float denom = std::max(max_z - min_z, 1e-6f);
            for (std::size_t i = 0; i < pixels; ++i) {
                const float valid = mask.values[i];
                const float z = points.xyz_hwc[i * 3 + 2];
                if (std::isfinite(valid) && valid >= 0.5f && std::isfinite(z) && z > 0.0f) {
                    const float normalized = 0.02f + 0.98f * ((z - min_z) / denom);
                    low[i] = static_cast<PixelT>(std::clamp(std::lround(normalized * kMaxValue), 1l, kMaxValue));
                }
            }
        }

        std::vector<PixelT> out(static_cast<std::size_t>(out_width) * out_height);
        for (int y = 0; y < out_height; ++y) {
            const int sy = std::min(points.height - 1, static_cast<int>((static_cast<int64_t>(y) * points.height) / out_height));
            for (int x = 0; x < out_width; ++x) {
                const int sx = std::min(points.width - 1, static_cast<int>((static_cast<int64_t>(x) * points.width) / out_width));
                out[static_cast<std::size_t>(y) * out_width + x] =
                    low[static_cast<std::size_t>(sy) * points.width + sx];
            }
        }
        return out;
    }

    template <typename PixelT>
    std::vector<PixelT> build_normals_png(const SpatialMap& mask,
                                          const VectorMap& normals,
                                          int out_width,
                                          int out_height) {
        constexpr long kMaxValue = static_cast<long>(std::numeric_limits<PixelT>::max());
        constexpr PixelT kNeutral = static_cast<PixelT>((kMaxValue + 1) / 2);
        if (mask.width != normals.width || mask.height != normals.height)
            throw std::runtime_error("Mask and normal outputs have different spatial sizes");

        std::vector<PixelT> low(static_cast<std::size_t>(normals.width) * normals.height * 3, kNeutral);
        for (int y = 0; y < normals.height; ++y) {
            for (int x = 0; x < normals.width; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * normals.width + x;
                if (!std::isfinite(mask.values[i]) || mask.values[i] < 0.5f)
                    continue;

                for (int c = 0; c < 3; ++c) {
                    const float n = normals.xyz_hwc[i * 3 + c];
                    const float encoded = std::isfinite(n) ? (n * 0.5f + 0.5f) : 0.5f;
                    low[i * 3 + c] = static_cast<PixelT>(std::clamp(std::lround(encoded * kMaxValue), 0l, kMaxValue));
                }
            }
        }

        std::vector<PixelT> out(static_cast<std::size_t>(out_width) * out_height * 3);
        for (int y = 0; y < out_height; ++y) {
            const int sy = std::min(normals.height - 1, static_cast<int>((static_cast<int64_t>(y) * normals.height) / out_height));
            for (int x = 0; x < out_width; ++x) {
                const int sx = std::min(normals.width - 1, static_cast<int>((static_cast<int64_t>(x) * normals.width) / out_width));
                const std::size_t dst = (static_cast<std::size_t>(y) * out_width + x) * 3;
                const std::size_t src = (static_cast<std::size_t>(sy) * normals.width + sx) * 3;
                out[dst + 0] = low[src + 0];
                out[dst + 1] = low[src + 1];
                out[dst + 2] = low[src + 2];
            }
        }
        return out;
    }

    bool needs_depth(lfs::core::param::PreprocessOutputMode mode) {
        return mode == lfs::core::param::PreprocessOutputMode::Depth ||
               mode == lfs::core::param::PreprocessOutputMode::Both;
    }

    bool needs_normals(lfs::core::param::PreprocessOutputMode mode) {
        return mode == lfs::core::param::PreprocessOutputMode::Normals ||
               mode == lfs::core::param::PreprocessOutputMode::Both;
    }

    // Fits each camera's depth prior against the COLMAP point cloud once and
    // writes a sidecar next to the depth maps, so depth-loss training skips the
    // per-camera anchor fit at startup. Requires a COLMAP scene; a bare image
    // folder is skipped (the trainer fits and caches it on first run instead).
    void precompute_depth_anchors(const lfs::core::param::PreprocessParameters& params) {
        if (!needs_depth(params.mode)) {
            return;
        }
        try {
            auto loader = lfs::io::Loader::create();
            lfs::io::LoadOptions options;
            options.images_folder = params.images_folder;
            auto result = loader->load(params.dataset_path, options);
            if (!result) {
                LOG_INFO("Depth anchors: scene load failed ({}); trainer will fit at startup",
                         result.error().format());
                return;
            }
            const auto* scene = std::get_if<lfs::io::LoadedScene>(&result->data);
            if (!scene || !scene->point_cloud || scene->cameras.empty()) {
                LOG_INFO("Depth anchors: no COLMAP point cloud; trainer will fit at startup");
                return;
            }

            const auto sidecar = lfs::training::depthAnchorSidecarPath(scene->cameras);
            if (sidecar.empty()) {
                LOG_INFO("Depth anchors: no depth priors resolved for any camera; skipping");
                return;
            }

            auto means = scene->point_cloud->means;
            if (!means.is_valid() || means.numel() == 0) {
                LOG_INFO("Depth anchors: empty point cloud; skipping");
                return;
            }
            means = means.to(lfs::core::Device::CUDA);

            const auto fingerprint = lfs::training::computeAnchorFingerprint(scene->cameras);

            std::optional<indicators::ProgressBar> anchor_bar;
            lfs::training::DepthAnchorProgress progress;
            if (stdout_is_tty()) {
                anchor_bar.emplace();
                style_progress_bar(*anchor_bar, "Depth anchors ");
                progress = [&anchor_bar](std::size_t done, std::size_t total) {
                    if (total == 0)
                        return;
                    anchor_bar->set_option(indicators::option::PostfixText(std::format("{}/{}", done, total)));
                    anchor_bar->set_progress(done * 100 / total);
                };
            }

            const auto anchors = lfs::training::computeRawDepthAnchors(means, scene->cameras, -1, 0, progress);

            if (anchor_bar && !anchor_bar->is_completed()) {
                anchor_bar->set_progress(100);
                anchor_bar->mark_as_completed();
                std::cout << std::endl;
            }

            if (lfs::training::writeDepthAnchorSidecar(sidecar, anchors, fingerprint)) {
                std::cout << "Depth anchors: wrote " << anchors.size() << " anchors to "
                          << path_to_string(sidecar) << "\n";
            } else {
                LOG_WARN("Depth anchors: failed to write sidecar {}", path_to_string(sidecar));
            }
        } catch (const std::exception& e) {
            LOG_WARN("Depth anchors: precompute failed: {}", e.what());
        }
    }

    bool should_write_output(bool output_requested,
                             bool overwrite,
                             const fs::path& output_path) {
        return output_requested && (overwrite || !fs::is_regular_file(output_path));
    }

    struct PreprocessJob {
        fs::path image_path;
        fs::path depth_path;
        fs::path normals_path;
        bool write_depth = false;
        bool write_normals = false;
    };

    struct PreprocessPlan {
        fs::path dataset_root;
        fs::path images_dir;
        std::vector<fs::path> images;
        std::vector<PreprocessJob> jobs;
        std::size_t skipped = 0;
    };

    struct LoadedImage {
        Image original;
        Image inference;
    };

    constexpr std::ptrdiff_t kMaxPendingWrites = 4;
    constexpr std::size_t kPrefetchDepth = 3;

    class PreprocessProgressBar {
    public:
        explicit PreprocessProgressBar(std::size_t total) : total_(total) {
            style_progress_bar(bar_, "Preprocessing ");
        }

        void report(std::size_t done, const std::string& name, double inference_ms) {
            bar_.set_option(indicators::option::PostfixText(
                std::format("{}/{} {} {:.0f} ms/img", done, total_, name, inference_ms)));
            bar_.set_progress(done * 100 / total_);
        }

        void complete() {
            if (!bar_.is_completed()) {
                bar_.set_progress(100);
                bar_.mark_as_completed();
                std::cout << std::endl;
            }
        }

        ~PreprocessProgressBar() {
            if (!bar_.is_completed()) {
                bar_.mark_as_completed();
                std::cout << std::endl;
            }
        }

    private:
        indicators::ProgressBar bar_;
        std::size_t total_;
    };

    PreprocessPlan build_preprocess_plan(const lfs::core::param::PreprocessParameters& params) {
        PreprocessPlan plan;
        plan.dataset_root = fs::absolute(params.dataset_path);
        plan.images_dir = plan.dataset_root / params.images_folder;

        // Flat COLMAP exports keep the images next to the sparse model files. Scan the
        // root non-recursively there so generated depth/normals outputs are never
        // picked up as inputs on a rerun.
        bool flat_layout = is_same_directory(plan.images_dir, plan.dataset_root);
        if (!flat_layout && !fs::is_directory(plan.images_dir) &&
            fs::is_directory(plan.dataset_root) &&
            !collect_images(plan.dataset_root, false).empty()) {
            std::cout << "Images folder '" << params.images_folder
                      << "' not found; using images in the dataset root (flat layout)\n";
            flat_layout = true;
        }
        if (flat_layout)
            plan.images_dir = plan.dataset_root;

        plan.images = collect_images(plan.images_dir, !flat_layout);
        if (plan.images.empty())
            throw std::runtime_error("No images found under " + path_to_string(plan.images_dir));

        plan.jobs.reserve(plan.images.size());
        for (const auto& image_path : plan.images) {
            PreprocessJob job{
                .image_path = image_path,
                .depth_path = output_path_for(plan.dataset_root, "depth", image_path, plan.images_dir),
                .normals_path = output_path_for(plan.dataset_root, "normals", image_path, plan.images_dir),
            };
            job.write_depth = should_write_output(needs_depth(params.mode), params.overwrite, job.depth_path);
            job.write_normals = should_write_output(needs_normals(params.mode), params.overwrite, job.normals_path);
            if (job.write_depth || job.write_normals)
                plan.jobs.push_back(std::move(job));
        }
        plan.skipped = plan.images.size() - plan.jobs.size();
        return plan;
    }

    void print_plan_summary(const lfs::core::param::PreprocessParameters& params,
                            const PreprocessPlan& plan,
                            const fs::path* model_path) {
        if (model_path)
            std::cout << "Model: " << path_to_string(*model_path) << "\n";
        std::cout << "Images: " << plan.images.size() << " under " << path_to_string(plan.images_dir) << "\n";
        std::cout << "Threads: " << resolve_thread_count(params.threads) << "\n";
        std::cout << "PNG compression: " << params.png_compression << "\n";
    }

    void process_dataset(const lfs::core::param::PreprocessParameters& params,
                         const fs::path& model_path,
                         const PreprocessPlan& plan) {
        print_plan_summary(params, plan, &model_path);

        MogeOnnxSession session(model_path, resolve_thread_count(params.threads), params.force_cpu);
        std::cout << "Execution provider: " << session.provider() << "\n";

        const auto load_job = [&params](const PreprocessJob& job) {
            LoadedImage loaded{.original = load_image_rgb(job.image_path)};
            loaded.inference = resize_for_inference(loaded.original, params.max_side);
            return loaded;
        };

        std::counting_semaphore<kMaxPendingWrites> write_slots(kMaxPendingWrites);
        std::deque<std::future<void>> writes;

        std::optional<PreprocessProgressBar> bar;
        if (stdout_is_tty() && !plan.jobs.empty())
            bar.emplace(plan.jobs.size());

        std::deque<std::future<LoadedImage>> pending_loads;
        std::size_t next_load = 0;
        const auto top_up_loads = [&] {
            while (next_load < plan.jobs.size() && pending_loads.size() < kPrefetchDepth) {
                pending_loads.push_back(std::async(std::launch::async, load_job, std::cref(plan.jobs[next_load])));
                ++next_load;
            }
        };
        top_up_loads();

        for (std::size_t i = 0; i < plan.jobs.size(); ++i) {
            const PreprocessJob& job = plan.jobs[i];
            if (!bar)
                std::cout << "[" << (i + 1) << "/" << plan.jobs.size() << "] " << path_to_string(job.image_path) << "\n";

            const LoadedImage loaded = pending_loads.front().get();
            pending_loads.pop_front();
            top_up_loads();

            const auto inference_start = std::chrono::steady_clock::now();
            auto outputs = std::make_shared<const OrtOutputs>(session.run(loaded.inference, params.num_tokens));
            const double inference_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - inference_start).count();
            if (bar)
                bar->report(i + 1, job.image_path.filename().string(), inference_ms);

            while (!writes.empty() && writes.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                writes.front().get();
                writes.pop_front();
            }

            write_slots.acquire();
            writes.push_back(std::async(
                std::launch::async,
                [&job, &params, &write_slots, outputs,
                 width = loaded.original.width, height = loaded.original.height] {
                    struct SlotRelease {
                        std::counting_semaphore<kMaxPendingWrites>& slots;
                        ~SlotRelease() { slots.release(); }
                    } release{write_slots};

                    if (params.bit_depth == 16) {
                        if (job.write_depth) {
                            const auto depth = build_depth_png<uint16_t>(outputs->mask, outputs->points, width, height);
                            write_png(job.depth_path, width, height, 1, depth, params.png_compression);
                        }
                        if (job.write_normals) {
                            const auto normals = build_normals_png<uint16_t>(outputs->mask, outputs->normals, width, height);
                            write_png(job.normals_path, width, height, 3, normals, params.png_compression);
                        }
                    } else {
                        if (job.write_depth) {
                            const auto depth = build_depth_png<uint8_t>(outputs->mask, outputs->points, width, height);
                            write_png(job.depth_path, width, height, 1, depth, params.png_compression);
                        }
                        if (job.write_normals) {
                            const auto normals = build_normals_png<uint8_t>(outputs->mask, outputs->normals, width, height);
                            write_png(job.normals_path, width, height, 3, normals, params.png_compression);
                        }
                    }
                }));
        }

        for (auto& write : writes)
            write.get();

        if (bar)
            bar->complete();
        std::cout << "Done. processed=" << plan.jobs.size() << " skipped=" << plan.skipped << "\n";
    }

} // namespace

namespace lfs::preprocessing {

    int run_preprocess(const lfs::core::param::PreprocessParameters& params) {
        try {
            if (params.download_only) {
                fs::path model_path = params.model_path;
                if (model_path.empty())
                    model_path = ensure_default_model(params.no_download);
                std::cout << "Cached model: " << path_to_string(model_path) << "\n";
                return 0;
            }

            const PreprocessPlan plan = build_preprocess_plan(params);
            if (plan.jobs.empty()) {
                print_plan_summary(params, plan, nullptr);
                std::cout << "No outputs need preprocessing; model inference skipped.\n";
                precompute_depth_anchors(params);
                std::cout << "Done. processed=0 skipped=" << plan.skipped << "\n";
                return 0;
            }

            fs::path model_path = params.model_path;
            if (model_path.empty())
                model_path = ensure_default_model(params.no_download);

            if (!fs::is_regular_file(model_path))
                throw std::runtime_error("Model file does not exist: " + path_to_string(model_path));

            process_dataset(params, model_path, plan);
            precompute_depth_anchors(params);
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "preprocess: " << e.what() << "\n";
            return 1;
        }
    }

} // namespace lfs::preprocessing
