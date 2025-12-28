/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>

#include "core/logger.hpp"
#include <algorithm>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

    // Helper function to convert filesystem path to UTF-8 string for OIIO
    // On Windows, path.string() uses system codepage (not UTF-8), which breaks Unicode
    // On Linux/Mac, path.string() is already UTF-8
    inline std::string path_to_utf8(const std::filesystem::path& p) {
#ifdef _WIN32
        // On Windows, convert wide string to UTF-8
        const std::wstring wstr = p.wstring();
        if (wstr.empty()) {
            return std::string();
        }

        const int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(),
                                                   static_cast<int>(wstr.size()),
                                                   nullptr, 0, nullptr, nullptr);
        if (size_needed <= 0) {
            LOG_ERROR("Wide string to UTF-8 conversion failed: size calculation returned {}", size_needed);
            return std::string();
        }

        std::string utf8_str(size_needed, 0);
        const int converted = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(),
                                                  static_cast<int>(wstr.size()),
                                                  &utf8_str[0], size_needed, nullptr, nullptr);
        if (converted <= 0) {
            LOG_ERROR("Wide string to UTF-8 conversion failed during write");
            return std::string();
        }
        utf8_str.resize(converted);
        return utf8_str;
#else
        // On Linux/Mac, native encoding is UTF-8
        return p.string();
#endif
    }

    // Run once: set global OIIO attributes (threading, etc.)
    std::once_flag g_oiio_once;
    inline void init_oiio() {
        std::call_once(g_oiio_once, [] {
            int n = (int)std::max(1u, std::thread::hardware_concurrency());
            OIIO::attribute("threads", n);
        });
    }

    // Downscale (resample) to (nw, nh). Returns newly malloc’ed RGB buffer.
    static inline unsigned char* downscale_resample_direct(const unsigned char* src_rgb,
                                                           int w, int h, int nw, int nh,
                                                           int nthreads /* 0=auto, 1=single */) {
        // Allocate destination first
        size_t outbytes = (size_t)nw * nh * 3;
        auto* out = static_cast<unsigned char*>(std::malloc(outbytes));
        if (!out)
            throw std::bad_alloc();

        // Wrap src & dst without extra allocations/copies
        OIIO::ImageBuf srcbuf(OIIO::ImageSpec(w, h, 3, OIIO::TypeDesc::UINT8),
                              const_cast<unsigned char*>(src_rgb));
        OIIO::ImageBuf dstbuf(OIIO::ImageSpec(nw, nh, 3, OIIO::TypeDesc::UINT8), out);

        OIIO::ROI roi(0, nw, 0, nh, 0, 1, 0, 3);
        if (!OIIO::ImageBufAlgo::resample(dstbuf, srcbuf, /*interpolate=*/true, roi, nthreads)) {
            std::string err = dstbuf.geterror();
            std::free(out);
            throw std::runtime_error(std::string("Resample failed: ") + (err.empty() ? "unknown" : err));
        }
        return out; // already filled
    }

} // namespace

namespace lfs::core {

    std::tuple<int, int, int> get_image_info(std::filesystem::path p) {
        init_oiio();

        const std::string path_utf8 = path_to_utf8(p);
        auto in = OIIO::ImageInput::open(path_utf8);
        if (!in) {
            throw std::runtime_error("OIIO open failed: " + path_utf8 + " : " + OIIO::geterror());
        }
        const OIIO::ImageSpec& spec = in->spec();
        const int w = spec.width;
        const int h = spec.height;
        const int c = spec.nchannels;
        in->close();
        return {w, h, c};
    }

    std::tuple<unsigned char*, int, int, int>
    load_image_with_alpha(std::filesystem::path p) {
        init_oiio();

        const std::string path_utf8 = path_to_utf8(p);
        std::unique_ptr<OIIO::ImageInput> in(OIIO::ImageInput::open(path_utf8));
        if (!in)
            throw std::runtime_error("Load failed: " + path_utf8 + " : " + OIIO::geterror());

        const OIIO::ImageSpec& spec = in->spec();
        int w = spec.width, h = spec.height, file_c = spec.nchannels;

        auto finish = [&](unsigned char* data, int W, int H, int C) {
            in->close();
            return std::make_tuple(data, W, H, C);
        };

        // Fast path: read 4 channels directly
        if (file_c == 4) {
            // allocate and read directly into final RGB buffer
            auto* out = static_cast<unsigned char*>(std::malloc((size_t)w * h * 4));
            if (!out) {
                in->close();
                throw std::bad_alloc();
            }

            if (!in->read_image(/*subimage*/ 0, /*miplevel*/ 0,
                                /*chbegin*/ 0, /*chend*/ 4,
                                OIIO::TypeDesc::UINT8, out)) {
                std::string e = in->geterror();
                std::free(out);
                in->close();
                throw std::runtime_error("Read failed: " + path_utf8 + (e.empty() ? "" : (" : " + e)));
            }
            return finish(out, w, h, 4);
        } else {
            LOG_ERROR("load_image_with_alpha: image does not contain alpha channel -  alpha channels found: {}", file_c);
            in->close();
            return std::make_tuple(nullptr, 0, 0, 0);
        }
    }

    std::tuple<unsigned char*, int, int, int>
    load_image_from_memory(const uint8_t* const data, const size_t size) {
        init_oiio();

        OIIO::Filesystem::IOMemReader mem_reader(data, size);
        auto in = OIIO::ImageInput::open("memory.jpg", nullptr, &mem_reader);
        if (!in)
            throw std::runtime_error("Load from memory failed: " + OIIO::geterror());

        const auto& spec = in->spec();
        const int w = spec.width, h = spec.height, channels = spec.nchannels;

        auto* out = static_cast<unsigned char*>(std::malloc(static_cast<size_t>(w) * h * 3));
        if (!out) {
            in->close();
            throw std::bad_alloc();
        }

        if (!in->read_image(0, 0, 0, std::min(channels, 3), OIIO::TypeDesc::UINT8, out)) {
            std::free(out);
            in->close();
            throw std::runtime_error("Read from memory failed: " + in->geterror());
        }

        in->close();
        return {out, w, h, 3};
    }

    std::tuple<unsigned char*, int, int, int>
    load_image(std::filesystem::path p, int res_div, int max_width) {
        LOG_TIMER("load_image total");

        {
            LOG_TIMER("init_oiio");
            init_oiio();
        }

        const std::string path_utf8 = path_to_utf8(p);
        std::unique_ptr<OIIO::ImageInput> in;
        {
            LOG_TIMER("OIIO::ImageInput::open");
            in = std::unique_ptr<OIIO::ImageInput>(OIIO::ImageInput::open(path_utf8));
            if (!in)
                throw std::runtime_error("Load failed: " + path_utf8 + " : " + OIIO::geterror());
        }

        const OIIO::ImageSpec& spec = in->spec();
        int w = spec.width, h = spec.height, file_c = spec.nchannels;

        // Decide threading for the resample (see notes below)
        const int nthreads = 0; // set to 1 if you call this from multiple worker threads

        // Fast path: read 3 channels directly (drop alpha if present)
        if (file_c >= 3) {
            if (res_div <= 1) {
                LOG_PERF("Fast path: reading 3 channels directly");
                // allocate and read directly into final RGB buffer
                auto* out = [&]() {
                    LOG_TIMER("malloc RGB buffer");
                    return static_cast<unsigned char*>(std::malloc((size_t)w * h * 3));
                }();
                if (!out) {
                    in->close();
                    throw std::bad_alloc();
                }

                {
                    LOG_TIMER("OIIO read_image");
                    if (!in->read_image(/*subimage*/ 0, /*miplevel*/ 0,
                                        /*chbegin*/ 0, /*chend*/ 3,
                                        OIIO::TypeDesc::UINT8, out)) {
                        std::string e = in->geterror();
                        std::free(out);
                        in->close();
                        throw std::runtime_error("Read failed: " + path_utf8 + (e.empty() ? "" : (" : " + e)));
                    }
                }

                {
                    in->close();
                }

                if (max_width > 0 && (w > max_width || h > max_width)) {
                    LOG_PERF("Need downscaling: {}x{} -> max_width {}", w, h, max_width);
                    int scale_w;
                    int scale_h;
                    if (w > h) {
                        scale_h = std::max(1, max_width * h / w);
                        scale_w = std::max(1, max_width);
                    } else {
                        scale_w = std::max(1, max_width * w / h);
                        scale_h = std::max(1, max_width);
                    }
                    unsigned char* ret = nullptr;
                    try {
                        LOG_TIMER("downscale_resample_direct");
                        ret = downscale_resample_direct(out, w, h, scale_w, scale_h, nthreads);
                    } catch (...) {
                        std::free(out);
                        throw;
                    }
                    std::free(out);
                    LOG_PERF("Downscaled to {}x{}", scale_w, scale_h);
                    return {ret, scale_w, scale_h, 3};
                } else {
                    return {out, w, h, 3};
                }

            } else if (res_div == 2 || res_div == 4 || res_div == 8) {
                LOG_PERF("res_div path: res_div={}", res_div);
                // read full, then downscale in-place into a new buffer without extra copy
                auto* full = [&]() {
                    LOG_TIMER("malloc full buffer for res_div");
                    return static_cast<unsigned char*>(std::malloc((size_t)w * h * 3));
                }();
                if (!full) {
                    in->close();
                    throw std::bad_alloc();
                }

                {
                    LOG_TIMER("OIIO read_image (res_div)");
                    if (!in->read_image(0, 0, 0, 3, OIIO::TypeDesc::UINT8, full)) {
                        std::string e = in->geterror();
                        std::free(full);
                        in->close();
                        throw std::runtime_error("Read failed: " + path_utf8 + (e.empty() ? "" : (" : " + e)));
                    }
                }

                {
                    LOG_TIMER("OIIO close (res_div)");
                    in->close();
                }

                const int nw = std::max(1, w / res_div);
                const int nh = std::max(1, h / res_div);
                LOG_PERF("Target size after res_div: {}x{}", nw, nh);
                int scale_w = nw;
                int scale_h = nh;
                if (max_width > 0 && (nw > max_width || nh > max_width)) {
                    if (nw > nh) {
                        scale_h = std::max(1, max_width * nh / nw);
                        scale_w = std::max(1, max_width);
                    } else {
                        scale_w = std::max(1, max_width * nw / nh);
                        scale_h = std::max(1, max_width);
                    }
                }

                unsigned char* out = nullptr;
                try {
                    LOG_TIMER("downscale_resample_direct (res_div)");
                    out = downscale_resample_direct(full, w, h, scale_w, scale_h, nthreads);
                } catch (...) {
                    std::free(full);
                    throw;
                }
                std::free(full);
                LOG_PERF("Final size: {}x{}", scale_w, scale_h);
                return {out, scale_w, scale_h, 3};
            } else {
                LOG_ERROR("load_image: unsupported resize factor {}", res_div);
                // fall through
            }
        }

        // 1–2 channel inputs -> read native, then expand to RGB
        {
            LOG_PERF("Grayscale/2-channel path: file_c={}", file_c);
            const int in_c = std::min(2, std::max(1, file_c));
            std::vector<unsigned char> tmp;
            {
                LOG_TIMER("allocate temp buffer");
                tmp.resize((size_t)w * h * in_c);
            }

            {
                LOG_TIMER("OIIO read_image (grayscale)");
                if (!in->read_image(0, 0, 0, in_c, OIIO::TypeDesc::UINT8, tmp.data())) {
                    auto e = in->geterror();
                    in->close();
                    throw std::runtime_error("Read failed: " + path_utf8 + (e.empty() ? "" : (" : " + e)));
                }
            }

            {
                LOG_TIMER("OIIO close (grayscale)");
                in->close();
            }

            auto* base = [&]() {
                LOG_TIMER("malloc RGB base buffer");
                return static_cast<unsigned char*>(std::malloc((size_t)w * h * 3));
            }();
            if (!base)
                throw std::bad_alloc();

            {
                LOG_TIMER("expand to RGB");
                if (in_c == 1) {
                    const unsigned char* g = tmp.data();
                    for (size_t i = 0, N = (size_t)w * h; i < N; ++i) {
                        unsigned char v = g[i];
                        base[3 * i + 0] = v;
                        base[3 * i + 1] = v;
                        base[3 * i + 2] = v;
                    }
                } else { // 2 channels -> (R,G,avg)
                    const unsigned char* src = tmp.data();
                    for (size_t i = 0, N = (size_t)w * h; i < N; ++i) {
                        unsigned char r = src[2 * i + 0];
                        unsigned char g = src[2 * i + 1];
                        base[3 * i + 0] = r;
                        base[3 * i + 1] = g;
                        base[3 * i + 2] = (unsigned char)(((int)r + (int)g) / 2);
                    }
                }
            }

            if (res_div == 2 || res_div == 4 || res_div == 8) {
                const int nw = std::max(1, w / res_div);
                const int nh = std::max(1, h / res_div);
                unsigned char* out = nullptr;
                try {
                    out = downscale_resample_direct(base, w, h, nw, nh, nthreads);
                } catch (...) {
                    std::free(base);
                    throw;
                }
                std::free(base);
                return {out, nw, nh, 3};
            }

            return {base, w, h, 3};
        }
    }

    void save_image(const std::filesystem::path& path, lfs::core::Tensor image) {
        init_oiio();

        // Normalize to HxWxC, uint8 on CPU
        image = image.clone().to(lfs::core::Device::CPU).to(lfs::core::DataType::Float32);
        if (image.ndim() == 4)
            image = image.squeeze(0); // [B,C,H,W] -> [C,H,W]
        if (image.ndim() == 3 && image.shape()[0] <= 4)
            image = image.permute({1, 2, 0}); // [C,H,W]->[H,W,C]
        image = image.contiguous();

        const int height = (int)image.shape()[0];
        const int width = (int)image.shape()[1];
        int channels = (int)image.shape()[2];
        if (channels < 1 || channels > 4)
            throw std::runtime_error("save_image: channels must be in [1..4]");

        const std::string path_utf8 = path_to_utf8(path);
        LOG_INFO("Saving image: {} shape: [{}, {}, {}]", path_utf8, height, width, channels);

        auto img_uint8 = (image.clamp(0, 1) * 255.0f).to(lfs::core::DataType::UInt8).contiguous();

        // Prepare OIIO output
        auto out = OIIO::ImageOutput::create(path_utf8);
        if (!out) {
            throw std::runtime_error("ImageOutput::create failed for " + path_utf8 + " : " + OIIO::geterror());
        }

        OIIO::ImageSpec spec(width, height, channels, OIIO::TypeDesc::UINT8);

        // Set JPEG quality if needed
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        if (ext == ".jpg" || ext == ".jpeg")
            spec.attribute("CompressionQuality", 95);

        if (!out->open(path_utf8, spec)) {
            auto e = out->geterror();
            throw std::runtime_error("open('" + path_utf8 + "') failed: " + (e.empty() ? OIIO::geterror() : e));
        }

        if (!out->write_image(OIIO::TypeDesc::UINT8, img_uint8.ptr<uint8_t>())) {
            auto e = out->geterror();
            out->close();
            throw std::runtime_error("write_image failed: " + (e.empty() ? OIIO::geterror() : e));
        }
        out->close();
    }

    void save_image(const std::filesystem::path& path,
                    const std::vector<lfs::core::Tensor>& images,
                    bool horizontal,
                    int separator_width) {
        if (images.empty())
            throw std::runtime_error("No images provided");
        if (images.size() == 1) {
            lfs::core::save_image(path, images[0]);
            return;
        }

        // Prepare all to HWC float on CPU
        std::vector<lfs::core::Tensor> xs;
        xs.reserve(images.size());
        for (size_t idx = 0; idx < images.size(); ++idx) {
            auto img = images[idx];
            LOG_INFO("save_image: Input image {} shape before conversion: [{}, {}, {}]",
                     idx, img.shape()[0], img.shape()[1], img.shape()[2]);

            img = img.clone().to(lfs::core::Device::CPU).to(lfs::core::DataType::Float32);
            if (img.ndim() == 4)
                img = img.squeeze(0);
            if (img.ndim() == 3 && img.shape()[0] <= 4)
                img = img.permute({1, 2, 0});
            img = img.contiguous();

            LOG_INFO("save_image: Image {} shape after conversion to HWC: [{}, {}, {}]",
                     idx, img.shape()[0], img.shape()[1], img.shape()[2]);
            xs.push_back(img);
        }

        // Separator (white)
        lfs::core::Tensor sep;
        if (separator_width > 0) {
            const auto& ref = xs[0];
            sep = horizontal
                      ? lfs::core::Tensor::ones(lfs::core::TensorShape({ref.shape()[0], (size_t)separator_width, ref.shape()[2]}), ref.device(), ref.dtype())
                      : lfs::core::Tensor::ones(lfs::core::TensorShape({(size_t)separator_width, ref.shape()[1], ref.shape()[2]}), ref.device(), ref.dtype());
        }

        // Concatenate
        lfs::core::Tensor combo = xs[0];
        LOG_INFO("save_image: Starting combo shape: [{}, {}, {}]",
                 combo.shape()[0], combo.shape()[1], combo.shape()[2]);

        for (size_t i = 1; i < xs.size(); ++i) {
            LOG_INFO("save_image: Concatenating image {} (shape [{}, {}, {}]) along axis {}",
                     i, xs[i].shape()[0], xs[i].shape()[1], xs[i].shape()[2], horizontal ? 1 : 0);

            combo = (separator_width > 0)
                        ? lfs::core::Tensor::cat({combo, sep, xs[i]}, horizontal ? 1 : 0)
                        : lfs::core::Tensor::cat({combo, xs[i]}, horizontal ? 1 : 0);

            LOG_INFO("save_image: After concatenation, combo shape: [{}, {}, {}]",
                     combo.shape()[0], combo.shape()[1], combo.shape()[2]);
        }

        LOG_INFO("save_image: Final combo shape before saving: [{}, {}, {}]",
                 combo.shape()[0], combo.shape()[1], combo.shape()[2]);

        // Save
        lfs::core::save_image(path, combo);
    }

    void free_image(unsigned char* img) { std::free(img); }

    bool save_img_data(const std::filesystem::path& p, const std::tuple<unsigned char*, int, int, int>& image_data) {
        init_oiio(); // Assuming this initializes OIIO like in your load_image

        auto [data, width, height, channels] = image_data;

        if (!data || width <= 0 || height <= 0 || channels <= 0) {
            return false;
        }

        // Get file extension to determine format
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        // Check if format is supported
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".tif" && ext != ".tiff") {
            return false;
        }

        const std::string path_utf8 = path_to_utf8(p);
        std::unique_ptr<OIIO::ImageOutput> out(OIIO::ImageOutput::create(path_utf8));
        if (!out) {
            return false;
        }

        // Create image specification
        OIIO::ImageSpec spec(width, height, channels, OIIO::TypeDesc::UINT8);

        // Set format-specific attributes
        if (ext == ".jpg" || ext == ".jpeg") {
            spec.attribute("CompressionQuality", 95);
            // JPEG doesn't support alpha channel, so force to 3 channels if we have 4
            if (channels == 4) {
                spec.nchannels = 3;
            }
        } else if (ext == ".png") {
            // PNG supports alpha, no special handling needed
        } else if (ext == ".tif" || ext == ".tiff") {
            spec.attribute("Compression", "lzw");
        }

        if (!out->open(path_utf8, spec)) {
            return false;
        }

        bool success;
        if (ext == ".jpg" || ext == ".jpeg") {
            if (channels == 4) {
                // Convert RGBA to RGB for JPEG
                std::vector<unsigned char> rgb_data(width * height * 3);
                for (int i = 0; i < width * height; ++i) {
                    rgb_data[i * 3 + 0] = data[i * 4 + 0]; // R
                    rgb_data[i * 3 + 1] = data[i * 4 + 1]; // G
                    rgb_data[i * 3 + 2] = data[i * 4 + 2]; // B
                    // Skip alpha channel
                }
                success = out->write_image(OIIO::TypeDesc::UINT8, rgb_data.data());
            } else {
                success = out->write_image(OIIO::TypeDesc::UINT8, data);
            }
        } else {
            // PNG and TIFF can handle all channel counts
            success = out->write_image(OIIO::TypeDesc::UINT8, data);
        }

        out->close();
        return success;
    }

} // namespace lfs::core

namespace lfs::core::image_io {

    BatchImageSaver::BatchImageSaver(size_t num_workers)
        : num_workers_(std::min(num_workers, std::min(size_t(8), size_t(std::thread::hardware_concurrency())))) {

        LOG_INFO("[BatchImageSaver] Starting with {} worker threads", num_workers_);
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&BatchImageSaver::worker_thread, this);
        }
    }

    BatchImageSaver::~BatchImageSaver() { shutdown(); }

    void BatchImageSaver::shutdown() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_)
                return;
            stop_ = true;
            LOG_INFO("[BatchImageSaver] Shutting down...");
        }
        cv_.notify_all();

        for (auto& w : workers_)
            if (w.joinable())
                w.join();

        while (!task_queue_.empty()) {
            process_task(task_queue_.front());
            task_queue_.pop();
        }
        LOG_INFO("[BatchImageSaver] Shutdown complete");
    }

    void BatchImageSaver::queue_save(const std::filesystem::path& path, lfs::core::Tensor image) {
        if (!enabled_) {
            lfs::core::save_image(path, image);
            return;
        }
        SaveTask t;
        t.path = path;
        t.image = image.clone();
        t.is_multi = false;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                lfs::core::save_image(path, image);
                return;
            }
            task_queue_.push(std::move(t));
            active_tasks_++;
        }
        cv_.notify_one();
    }

    void BatchImageSaver::queue_save_multiple(const std::filesystem::path& path,
                                              const std::vector<lfs::core::Tensor>& images,
                                              bool horizontal,
                                              int separator_width) {
        if (!enabled_) {
            lfs::core::save_image(path, images, horizontal, separator_width);
            return;
        }
        SaveTask t;
        t.path = path;
        t.images.reserve(images.size());
        for (const auto& img : images)
            t.images.push_back(img.clone());
        t.is_multi = true;
        t.horizontal = horizontal;
        t.separator_width = separator_width;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                lfs::core::save_image(path, images, horizontal, separator_width);
                return;
            }
            task_queue_.push(std::move(t));
            active_tasks_++;
        }
        cv_.notify_one();
    }

    void BatchImageSaver::wait_all() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv_finished_.wait(lock, [this] { return task_queue_.empty() && active_tasks_ == 0; });
    }

    size_t BatchImageSaver::pending_count() const {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return task_queue_.size() + active_tasks_;
    }

    void BatchImageSaver::worker_thread() {
        while (true) {
            SaveTask t;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] { return stop_ || !task_queue_.empty(); });
                if (stop_ && task_queue_.empty())
                    break;
                if (task_queue_.empty())
                    continue;
                t = std::move(task_queue_.front());
                task_queue_.pop();
            }
            process_task(t);
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                active_tasks_--;
            }
            cv_finished_.notify_all();
        }
    }

    void BatchImageSaver::process_task(const SaveTask& t) {
        try {
            if (t.is_multi) {
                lfs::core::save_image(t.path, t.images, t.horizontal, t.separator_width);
            } else {
                lfs::core::save_image(t.path, t.image);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("[BatchImageSaver] Error saving {}: {}", t.path.string(), e.what());
        }
    }
} // namespace lfs::core::image_io
