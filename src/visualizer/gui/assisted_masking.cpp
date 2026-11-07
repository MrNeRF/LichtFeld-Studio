/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/assisted_masking.hpp"
#include "core/cuda_error.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/image_codecs.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/nn/models/sam2.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "io/pipelined_image_loader.hpp"
#include "preprocessing/preprocess.hpp"
#include "preprocessing/video_masking.hpp"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace lfs::vis::gui {
    namespace fs = core::native_fs;
    using core::DataType;
    using core::Device;
    using core::Tensor;
    using core::nn::models::Sam2;
    namespace {
        struct FrameDrain {
            cudaStream_t stream;
            ~FrameDrain() { LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "assisted mask frame lease drain"); }
        };
    } // namespace

    struct AssistedMasking::Impl {
        JobRegistry& jobs;
        std::function<bool()> admission;
        JobHandle job;
        struct Work {
            uint64_t id;
            JobHandle job;
            AssistedMaskRequest input;
        };
        std::mutex mutex;
        std::condition_variable cv;
        std::optional<Work> pending;
        JobHandle closing_job;
        AssistedMaskSnapshot result;
        uint64_t next = 0;
        bool release = false, stopping = false;
        std::thread worker;

        Impl(JobRegistry& registry, std::function<bool()> check)
            : jobs(registry), admission(std::move(check)) {}

        void run() {
            std::optional<Sam2> model;
            std::filesystem::path image, model_key;
            std::filesystem::file_time_type stamp{};
            uintmax_t bytes = 0;
            Tensor previous;
            uint64_t previous_id = 0;
            cudaStream_t stream = nullptr;
            JobHandle active;
            const auto retire = [&] {
                if (stream)
                    LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "assisted mask drain");
                previous = {};
                model.reset();
                image.clear();
                if (stream) {
                    LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "assisted mask retirement");
                    core::CudaMemoryPool::instance().release_stream(stream);
                    LFS_CUDA_LOG_TEARDOWN(cudaStreamDestroy(stream), stream, "assisted mask stream teardown");
                    stream = nullptr;
                }
            };
            for (;;) {
                std::optional<Work> work;
                {
                    std::unique_lock lock(mutex);
                    cv.wait(lock, [&] { return stopping || release || pending.has_value(); });
                    if (release || stopping) {
                        if (!active)
                            active = closing_job;
                        lock.unlock();
                        retire();
                        lock.lock();
                        release = false;
                        closing_job = {};
                        // Keep Closing visible until poll() finalizes the old job.
                        // A new request must never inherit its retiring handle.
                        const bool stop = stopping;
                        const auto finished = active;
                        active = {};
                        lock.unlock();
                        if (finished)
                            jobs.finishWork(finished, true);
                        if (stop)
                            return;
                        continue;
                    }
                    work = std::move(pending);
                    pending.reset();
                    result.phase = "loading";
                }
                auto& w = *work;
                if (!active) {
                    active = w.job;
                    jobs.work(active);
                }
                const auto obsolete = [&] {
                    std::lock_guard lock(mutex);
                    return release || stopping || result.request != w.id;
                };
                const auto stage = [&](const char* value, float progress) {
                    jobs.report(active, progress, value);
                    std::lock_guard lock(mutex);
                    if (result.request == w.id)
                        result.phase = value;
                };
                bool model_failure = false;
                try {
                    const auto current_stamp = fs::last_write_time(w.input.image);
                    const auto current_bytes = fs::file_size(w.input.image);
#if defined(_MSC_VER)
                    const auto system_stamp = std::chrono::clock_cast<std::chrono::system_clock>(current_stamp);
#else
                    const auto system_stamp = std::filesystem::file_time_type::clock::to_sys(current_stamp);
#endif
                    const auto mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(system_stamp.time_since_epoch()).count();
                    if (mtime_ns != w.input.mtime_ns || current_bytes != w.input.size_bytes)
                        throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                    if (!stream)
                        LFS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                    const core::CUDAStreamGuard guard(stream);
                    if (model && model_key != w.input.weights) {
                        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                        previous = {};
                        model.reset();
                        image.clear();
                    }
                    if (!model || image != w.input.image || stamp != current_stamp || bytes != current_bytes) {
                        // Release the old encoder before admitting the next source's buffers.
                        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                        previous = {};
                        previous_id = 0;
                        size_t available = 0, total = 0;
                        LFS_CUDA_CHECK(cudaMemGetInfo(&available, &total));
                        auto [width, height, channels] = core::probe_image_info(w.input.image);
                        preprocessing::VideoMaskFrame frame{};
                        frame.image_path = w.input.image;
                        frame.width = width;
                        frame.height = height;
                        // Credit only allocations already owned by this model, not a second estimated model reserve.
                        const auto resident = model ? model->weights_bytes() + model->arena_bytes() : 0;
                        const auto plan = preprocessing::plan_video_mask_resources({&frame, 1}, {},
                                                                                   std::min(total, available + resident));
                        if (!plan)
                            throw std::runtime_error(std::string(plan.error().user_message()));
                        if (!model) {
                            auto weights = w.input.weights;
                            if (weights.empty()) {
                                const auto cached = preprocessing::ensure_sam2_weights(true, obsolete);
                                if (!cached)
                                    throw std::runtime_error(std::string(cached.error().user_message()));
                                weights = *cached;
                            }
                            if (obsolete())
                                continue;
                            model_failure = true;
                            auto loaded = Sam2::load(weights, Device::CUDA, DataType::Float32);
                            if (!loaded)
                                throw std::runtime_error(std::string(loaded.error().user_message()));
                            model.emplace(std::move(*loaded));
                            model->set_encoder_dtype(DataType::Float16);
                            model_key = w.input.weights;
                            model_failure = false;
                        }
                        io::PipelinedLoaderConfig config;
                        config.jpeg_batch_size = config.prefetch_count = config.output_queue_size = config.decoder_pool_size = 1;
                        config.decode_frame_ring_capacity = 2;
                        io::PipelinedImageLoader loader(config);
                        io::LoadParams params{};
                        params.skip_blob_cache = true;
                        loader.prefetch(0, w.input.image, params);
                        auto decoded = loader.get();
                        const FrameDrain frame_drain{stream};
                        if (!decoded.error.empty())
                            throw std::runtime_error(decoded.error);
                        core::waitForCUDAStream(stream, decoded.stream);
                        decoded.tensor.set_stream(stream);
                        stage("encoding", 0.3f);
                        model_failure = true;
                        const auto encoded = model->set_image(decoded.tensor.unsqueeze(0));
                        LFS_CUDA_CHECK(cudaStreamSynchronize(stream)); // Decoder owns frame leases until this point.
                        if (!encoded)
                            throw std::runtime_error(std::string(encoded.error().user_message()));
                        model_failure = false;
                        image = w.input.image;
                        stamp = current_stamp;
                        bytes = current_bytes;
                    }
                    if (obsolete())
                        continue;
                    auto [width, height, channels] = core::probe_image_info(image);
                    if (w.input.export_candidate >= 0) {
                        if (w.input.refine_request != previous_id || !previous.is_valid() ||
                            w.input.export_candidate >= int(previous.size(0)))
                            throw std::runtime_error(LOC("image_preview.sam_proposal_expired"));
                        stage("rendering", 0.9f);
                        auto selected = core::nn::resize2d(
                                            previous.slice(0, w.input.export_candidate, w.input.export_candidate + 1).unsqueeze(0),
                                            height, width, core::nn::ResizeMode::Bilinear, core::nn::CoordTransform::HalfPixel)
                                            .gt(0)
                                            .to(DataType::UInt8)
                                            .to(Device::CPU, stream)
                                            .contiguous();
                        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                        if (obsolete())
                            continue;
                        if (fs::last_write_time(image) != stamp || fs::file_size(image) != bytes)
                            throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                        std::vector<uint8_t> pixels(size_t(width) * height);
                        for (size_t p = 0; p < pixels.size(); ++p)
                            pixels[p] = ((selected.ptr<uint8_t>()[p] != 0) != w.input.exclude) ? 255 : 0;
                        auto encoded = core::image_codecs::encode_png(pixels, width, height, 1);
                        if (encoded.empty())
                            throw std::runtime_error(LOC("image_preview.sam_write_failed"));
                        auto png = std::make_shared<const std::vector<std::byte>>(std::move(encoded));
                        std::lock_guard lock(mutex);
                        if (!release && result.request == w.id) {
                            result.mask = std::move(png);
                            result.width = width;
                            result.height = height;
                            result.image = image;
                            result.source_time = stamp;
                            result.source_bytes = bytes;
                            result.phase = "mask_ready";
                        }
                        continue;
                    }
                    std::vector<core::nn::models::Sam2PointPrompt> points;
                    for (const auto& point : w.input.points)
                        points.push_back({point[0] * (width - 1), point[1] * (height - 1), int(point[2])});
                    auto box = w.input.box;
                    if (box) {
                        (*box)[0] *= width - 1;
                        (*box)[2] *= width - 1;
                        (*box)[1] *= height - 1;
                        (*box)[3] *= height - 1;
                    }
                    Tensor refinement;
                    if (w.input.refine_candidate >= 0) {
                        if (w.input.refine_request != previous_id || !previous.is_valid() ||
                            w.input.refine_candidate >= int(previous.size(0)))
                            throw std::runtime_error(LOC("image_preview.sam_proposal_expired"));
                        refinement = previous.slice(0, w.input.refine_candidate, w.input.refine_candidate + 1).reshape({1, 1, 256, 256});
                    }
                    stage("predicting", 0.7f);
                    model_failure = true;
                    auto prediction = model->predict(points, box, true, refinement.is_valid() ? &refinement : nullptr,
                                                     core::nn::models::Sam2MaskOutput::LowResolutionOnly);
                    if (!prediction)
                        throw std::runtime_error(std::string(prediction.error().user_message()));
                    model_failure = false;
                    previous = prediction->low_res_logits;
                    previous_id = w.id;
                    const int ow = std::max(1, int(width * std::min(1.0, 1536.0 / std::max(width, height))));
                    const int oh = std::max(1, int(height * std::min(1.0, 1536.0 / std::max(width, height))));
                    auto masks = core::nn::resize2d(previous.unsqueeze(0), oh, ow, core::nn::ResizeMode::Bilinear,
                                                    core::nn::CoordTransform::HalfPixel)
                                     .gt(0)
                                     .to(DataType::Float32);
                    auto scores = prediction->iou.to(Device::CPU, stream).contiguous();
                    LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                    if (obsolete())
                        continue;
                    if (fs::last_write_time(image) != stamp || fs::file_size(image) != bytes)
                        throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                    std::vector<Tensor> overlays;
                    for (int c = 0; c < int(previous.size(0)); ++c)
                        overlays.push_back(masks.slice(1, c, c + 1).reshape({oh, ow, 1}).expand({oh, ow, 4}));
                    stage("rendering", 0.95f);
                    std::lock_guard lock(mutex);
                    if (!release && result.request == w.id) {
                        result.overlays = std::move(overlays);
                        result.scores.assign(scores.ptr<float>(), scores.ptr<float>() + previous.size(0));
                        result.phase = "ready";
                    }
                } catch (const std::exception& error) {
                    if (stream)
                        LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "assisted mask error drain");
                    const auto* typed = dynamic_cast<const lfs::Exception*>(&error);
                    if (model_failure || (typed && typed->error().domain() == lfs::ErrorDomain::CUDA)) {
                        previous = {};
                        previous_id = 0;
                        model.reset();
                        image.clear();
                    }
                    std::lock_guard lock(mutex);
                    if (result.request == w.id) {
                        result.phase = "error";
                        result.error = error.what();
                    }
                }
            }
        }
    };

    AssistedMasking::AssistedMasking(JobRegistry& jobs, std::function<bool()> admission)
        : impl_(std::make_unique<Impl>(jobs, std::move(admission))) {}
    AssistedMasking::~AssistedMasking() {
        try {
            shutdown();
        } catch (const std::exception& error) {
            LOG_ERROR("Assisted mask shutdown failed: {}", error.what());
        }
    }
    uint64_t AssistedMasking::submit(AssistedMaskRequest input) {
        {
            std::lock_guard lock(impl_->mutex);
            if (impl_->stopping)
                throw std::runtime_error(LOC("image_preview.sam_shutting_down"));
        }
        const auto valid = [](float v) { return std::isfinite(v) && v >= 0 && v <= 1; };
        if (input.image.empty() || input.revision.empty() || input.points.size() > 256 ||
            (input.points.empty() && !input.box))
            throw std::invalid_argument(LOC("image_preview.sam_invalid_input"));
        for (const auto& p : input.points)
            if (!valid(p[0]) || !valid(p[1]) || (p[2] != 0 && p[2] != 1))
                throw std::invalid_argument(LOC("image_preview.sam_invalid_point"));
        if (input.box && (!std::ranges::all_of(*input.box, valid) || (*input.box)[0] >= (*input.box)[2] || (*input.box)[1] >= (*input.box)[3]))
            throw std::invalid_argument(LOC("image_preview.sam_invalid_box"));
        if (!impl_->admission())
            throw std::runtime_error(LOC("image_preview.sam_unavailable"));
        const auto prior = poll();
        if (prior.phase == "closing")
            return 0;
        if (!impl_->worker.joinable())
            impl_->worker = std::thread([this] { impl_->run(); });
        if (!impl_->job) {
            const auto job = impl_->jobs.init(JobType::AssistedMask, "loading");
            if (!job)
                return 0;
            impl_->job = *job;
        }
        std::lock_guard lock(impl_->mutex);
        const auto id = ++impl_->next;
        impl_->result = {};
        impl_->result.request = id;
        impl_->result.revision = input.revision;
        impl_->result.phase = "queued";
        impl_->pending = Impl::Work{id, impl_->job, std::move(input)};
        impl_->cv.notify_one();
        return id;
    }
    AssistedMaskSnapshot AssistedMasking::poll() {
        if (impl_->job) {
            const auto job = impl_->jobs.peek(impl_->job);
            if (job && job->status == JobStatus::CompletionPending) {
                impl_->jobs.canceled(impl_->job);
                impl_->jobs.free(impl_->job);
                impl_->job = {};
                std::lock_guard lock(impl_->mutex);
                impl_->result = {};
            }
        }
        std::lock_guard lock(impl_->mutex);
        return impl_->result;
    }
    void AssistedMasking::close() {
        (void)poll();
        if (!impl_->job)
            return;
        impl_->jobs.requestCancel(impl_->job);
        std::lock_guard lock(impl_->mutex);
        impl_->release = true;
        impl_->closing_job = impl_->job;
        impl_->pending.reset();
        impl_->result = {};
        impl_->result.phase = "closing";
        impl_->cv.notify_one();
    }
    void AssistedMasking::shutdown() {
        {
            std::lock_guard lock(impl_->mutex);
            impl_->stopping = true;
            impl_->closing_job = impl_->job;
            impl_->pending.reset();
            impl_->cv.notify_one();
        }
        // Stop and join before main-thread completion/snapshot operations, so
        // a failure there cannot leave a live thread owned by the destructor.
        if (impl_->worker.joinable())
            impl_->worker.join();
        (void)poll();
    }
} // namespace lfs::vis::gui
