/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <condition_variable>
#include <core/logger.hpp>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

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

std::tuple<int, int, int> get_image_info(std::filesystem::path p) {
    init_oiio();

    auto in = OIIO::ImageInput::open(p.string());
    if (!in) {
        throw std::runtime_error("OIIO open failed: " + p.string() + " : " + OIIO::geterror());
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

    std::unique_ptr<OIIO::ImageInput> in(OIIO::ImageInput::open(p.string()));
    if (!in)
        throw std::runtime_error("Load failed: " + p.string() + " : " + OIIO::geterror());

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
            throw std::runtime_error("Read failed: " + p.string() + (e.empty() ? "" : (" : " + e)));
        }
        return finish(out, w, h, 4);
    } else {
        LOG_ERROR("load_image_with_alpha: image does not contain alpha channel -  alpha channels found: {}", file_c);
    }
}

std::tuple<unsigned char*, int, int, int>
load_image(std::filesystem::path p, int res_div, int max_width) {
    init_oiio();

    std::unique_ptr<OIIO::ImageInput> in(OIIO::ImageInput::open(p.string()));
    if (!in)
        throw std::runtime_error("Load failed: " + p.string() + " : " + OIIO::geterror());

    const OIIO::ImageSpec& spec = in->spec();
    int w = spec.width, h = spec.height, file_c = spec.nchannels;

    auto finish = [&](unsigned char* data, int W, int H, int C) {
        in->close();
        return std::make_tuple(data, W, H, C);
    };

    // Decide threading for the resample (see notes below)
    const int nthreads = 0; // set to 1 if you call this from multiple worker threads

    // Fast path: read 3 channels directly (drop alpha if present)
    if (file_c >= 3) {
        if (res_div <= 1) {
            // allocate and read directly into final RGB buffer
            auto* out = static_cast<unsigned char*>(std::malloc((size_t)w * h * 3));
            if (!out) {
                in->close();
                throw std::bad_alloc();
            }

            if (!in->read_image(/*subimage*/ 0, /*miplevel*/ 0,
                                /*chbegin*/ 0, /*chend*/ 3,
                                OIIO::TypeDesc::UINT8, out)) {
                std::string e = in->geterror();
                std::free(out);
                in->close();
                throw std::runtime_error("Read failed: " + p.string() + (e.empty() ? "" : (" : " + e)));
            }
            in->close();

            if (max_width > 0 && (w > max_width || h > max_width)) {
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
                    ret = downscale_resample_direct(out, w, h, scale_w, scale_h, nthreads);
                } catch (...) {
                    std::free(out);
                    throw;
                }
                std::free(out);
                return {ret, scale_w, scale_h, 3};
            } else {
                return {out, w, h, 3};
            }

        } else if (res_div == 2 || res_div == 4 || res_div == 8) {
            // read full, then downscale in-place into a new buffer without extra copy
            auto* full = static_cast<unsigned char*>(std::malloc((size_t)w * h * 3));
            if (!full) {
                in->close();
                throw std::bad_alloc();
            }

            if (!in->read_image(0, 0, 0, 3, OIIO::TypeDesc::UINT8, full)) {
                std::string e = in->geterror();
                std::free(full);
                in->close();
                throw std::runtime_error("Read failed: " + p.string() + (e.empty() ? "" : (" : " + e)));
            }
            in->close();

            const int nw = std::max(1, w / res_div);
            const int nh = std::max(1, h / res_div);
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
                out = downscale_resample_direct(full, w, h, scale_w, scale_h, nthreads);
            } catch (...) {
                std::free(full);
                throw;
            }
            std::free(full);
            return {out, scale_w, scale_h, 3};
        } else {
            LOG_ERROR("load_image: unsupported resize factor {}", res_div);
            // fall through
        }
    }

    // 1–2 channel inputs -> read native, then expand to RGB
    {
        const int in_c = std::min(2, std::max(1, file_c));
        std::vector<unsigned char> tmp((size_t)w * h * in_c);
        if (!in->read_image(0, 0, 0, in_c, OIIO::TypeDesc::UINT8, tmp.data())) {
            auto e = in->geterror();
            in->close();
            throw std::runtime_error("Read failed: " + p.string() + (e.empty() ? "" : (" : " + e)));
        }
        in->close();

        auto* base = static_cast<unsigned char*>(std::malloc((size_t)w * h * 3));
        if (!base)
            throw std::bad_alloc();

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

void save_image(const std::filesystem::path& path, torch::Tensor image) {
    init_oiio();

    // Normalize to HxWxC, uint8 on CPU
    image = image.clone().to(torch::kCPU).to(torch::kFloat32);
    if (image.dim() == 4)
        image = image.squeeze(0); // [B,C,H,W] -> [C,H,W]
    if (image.dim() == 3 && image.size(0) <= 4)
        image = image.permute({1, 2, 0}); // [C,H,W]->[H,W,C]
    image = image.contiguous();

    const int height = (int)image.size(0);
    const int width = (int)image.size(1);
    int channels = (int)image.size(2);
    if (channels < 1 || channels > 4)
        throw std::runtime_error("save_image: channels must be in [1..4]");

    LOG_INFO("Saving image: {} shape: [{}, {}, {}]", path.string(), height, width, channels);

    auto img_uint8 = (image.clamp(0, 1) * 255.0f).to(torch::kUInt8).contiguous();

    // Prepare OIIO output
    const std::string fname = path.string();
    auto out = OIIO::ImageOutput::create(fname);
    if (!out) {
        throw std::runtime_error("ImageOutput::create failed for " + fname + " : " + OIIO::geterror());
    }

    OIIO::ImageSpec spec(width, height, channels, OIIO::TypeDesc::UINT8);

    // Set JPEG quality if needed
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    if (ext == ".jpg" || ext == ".jpeg")
        spec.attribute("CompressionQuality", 95);

    if (!out->open(fname, spec)) {
        auto e = out->geterror();
        throw std::runtime_error("open('" + fname + "') failed: " + (e.empty() ? OIIO::geterror() : e));
    }

    if (!out->write_image(OIIO::TypeDesc::UINT8, img_uint8.data_ptr<uint8_t>())) {
        auto e = out->geterror();
        out->close();
        throw std::runtime_error("write_image failed: " + (e.empty() ? OIIO::geterror() : e));
    }
    out->close();
}

void save_image(const std::filesystem::path& path,
                const std::vector<torch::Tensor>& images,
                bool horizontal,
                int separator_width) {
    if (images.empty())
        throw std::runtime_error("No images provided");
    if (images.size() == 1) {
        save_image(path, images[0]);
        return;
    }

    // Prepare all to HWC float on CPU
    std::vector<torch::Tensor> xs;
    xs.reserve(images.size());
    for (auto img : images) {
        img = img.clone().to(torch::kCPU).to(torch::kFloat32);
        if (img.dim() == 4)
            img = img.squeeze(0);
        if (img.dim() == 3 && img.size(0) <= 4)
            img = img.permute({1, 2, 0});
        xs.push_back(img.contiguous());
    }

    // Separator (white)
    torch::Tensor sep;
    if (separator_width > 0) {
        const auto ref = xs[0];
        sep = horizontal
                  ? torch::ones({ref.size(0), separator_width, ref.size(2)}, ref.options())
                  : torch::ones({separator_width, ref.size(1), ref.size(2)}, ref.options());
    }

    // Concatenate
    torch::Tensor combo = xs[0];
    for (size_t i = 1; i < xs.size(); ++i) {
        combo = (separator_width > 0)
                    ? torch::cat({combo, sep, xs[i]}, horizontal ? 1 : 0)
                    : torch::cat({combo, xs[i]}, horizontal ? 1 : 0);
    }

    // Save
    save_image(path, combo);
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

    std::unique_ptr<OIIO::ImageOutput> out(OIIO::ImageOutput::create(p.string()));
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

    if (!out->open(p.string(), spec)) {
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

namespace image_io {

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

    void BatchImageSaver::queue_save(const std::filesystem::path& path, torch::Tensor image) {
        if (!enabled_) {
            save_image(path, image);
            return;
        }
        SaveTask t;
        t.path = path;
        t.image = image.clone();
        t.is_multi = false;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                save_image(path, image);
                return;
            }
            task_queue_.push(std::move(t));
            active_tasks_++;
        }
        cv_.notify_one();
    }

    void BatchImageSaver::queue_save_multiple(const std::filesystem::path& path,
                                              const std::vector<torch::Tensor>& images,
                                              bool horizontal,
                                              int separator_width) {
        if (!enabled_) {
            save_image(path, images, horizontal, separator_width);
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
                save_image(path, images, horizontal, separator_width);
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
                save_image(t.path, t.images, t.horizontal, t.separator_width);
            } else {
                save_image(t.path, t.image);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("[BatchImageSaver] Error saving {}: {}", t.path.string(), e.what());
        }
    }
} // namespace image_io
