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
#include "core/nn/models/llmdet.hpp"
#include "core/nn/models/sam2.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/utc_time.hpp"
#include "io/atomic_output.hpp"
#include "io/pipelined_image_loader.hpp"
#include "preprocessing/preprocess.hpp"
#include "preprocessing/video_mask_extraction.hpp"
#include "preprocessing/video_masking.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

namespace lfs::vis::gui {
    namespace fs = core::native_fs;
    using core::DataType;
    using core::Device;
    using core::Tensor;
    using core::nn::models::LlmDet;
    using core::nn::models::Sam2;
    namespace {
        struct FrameDrain {
            cudaStream_t stream;
            ~FrameDrain() { LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "assisted mask frame lease drain"); }
        };

        struct SemanticBoxes {
            std::vector<std::array<float, 4>> boxes;
            bool limited = false;
        };

        // Consume the decoder's lease on this worker/stream; no second image
        // cache, decoder or CPU inference path. Only bounded CPU boxes escape.
        SemanticBoxes find_semantic_boxes(const LlmDet& detector, const Tensor& rgb,
                                          const core::nn::models::LlmDetText& text, int width, int height,
                                          float confidence, cudaStream_t stream) {
            auto predicted = detector.forward(rgb.to(DataType::Float32), text);
            if (!predicted)
                throw lfs::Exception(predicted.error());
            const auto probabilities = core::nn::sigmoid(predicted->logits).to(Device::CPU, stream).contiguous();
            const auto boxes = predicted->boxes.to(Device::CPU, stream).contiguous();
            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
            struct Candidate {
                std::array<float, 4> box;
                float score;
            };
            std::vector<Candidate> candidates;
            const auto finite = [](float value) { return std::isfinite(value); };
            if (!std::ranges::all_of(std::span(probabilities.ptr<float>(), probabilities.numel()), finite) ||
                !std::ranges::all_of(std::span(boxes.ptr<float>(), boxes.numel()), finite))
                throw std::runtime_error(LOC("image_preview.sam_semantic_failed"));
            const auto tokens = probabilities.size(2);
            for (size_t query = 0; query < boxes.size(1); ++query) {
                float best = 0;
                for (const auto& category : text.category_tokens)
                    for (const int token : category)
                        best = std::max(best, probabilities.ptr<float>()[query * tokens + size_t(token)]);
                if (best <= confidence)
                    continue;
                const auto* b = boxes.ptr<float>() + query * 4;
                std::array<float, 4> box{(b[0] - b[2] * .5f) * width, (b[1] - b[3] * .5f) * height,
                                         (b[0] + b[2] * .5f) * width, (b[1] + b[3] * .5f) * height};
                for (int axis = 0; axis < 4; ++axis)
                    box[axis] = std::clamp(box[axis], 0.f, float(axis % 2 ? height : width));
                if (box[2] > box[0] && box[3] > box[1])
                    candidates.push_back({box, best});
            }
            std::ranges::stable_sort(candidates, std::greater{}, &Candidate::score);
            SemanticBoxes result;
            constexpr size_t max_instances = 256;
            const auto overlap = [](const auto& a, const auto& b) {
                const float intersection = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0])) *
                                           std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
                const float area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]);
                return intersection / std::max(area - intersection, 1e-9f);
            };
            for (const auto& candidate : candidates) {
                if (std::ranges::any_of(result.boxes, [&](const auto& box) { return overlap(box, candidate.box) > 0.7f; }))
                    continue;
                if (result.boxes.size() == max_instances) {
                    result.limited = true;
                    break;
                }
                result.boxes.push_back(candidate.box);
            }
            return result;
        }

        Tensor decode_mask(const std::vector<std::byte>& bytes, int width, int height) {
            auto [raw, w, h, channels] = core::load_image_from_memory(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
            const std::unique_ptr<unsigned char, decltype(&core::free_image)> pixels(raw, core::free_image);
            if (!raw || w != width || h != height || channels != 3)
                throw std::runtime_error(LOC("image_preview.sam_proposal_expired"));
            auto mask = Tensor::empty({size_t(height), size_t(width)}, Device::CPU, DataType::UInt8);
            for (size_t i = 0; i < mask.numel(); ++i)
                mask.ptr<uint8_t>()[i] = raw[i * 3];
            return mask;
        }

        std::shared_ptr<const std::vector<std::byte>> encode_mask(std::span<const uint8_t> pixels, int width, int height) {
            auto encoded = core::image_codecs::encode_png(pixels, width, height, 1);
            if (encoded.empty())
                throw lfs::Exception(lfs::make_error({.code = lfs::ErrorCode::Unavailable, .domain = lfs::ErrorDomain::Core, .user_message = LOC("image_preview.sam_write_failed"), .detection = LFS_SOURCE_SITE_CURRENT()}));
            return std::make_shared<const std::vector<std::byte>>(std::move(encoded));
        }
    } // namespace

    struct AssistedMasking::Impl {
        JobRegistry& jobs;
        std::function<bool()> admission;
        RevisionReader document_revision;
        std::optional<std::array<uint64_t, 2>> apply_revision;
        bool document_apply = false;
        JobHandle job;
        struct Sequence {
            struct Frame {
                AssistedMaskFrameSummary summary;
                std::shared_ptr<const std::vector<std::byte>> png;
                std::filesystem::file_time_type time{};
                uint64_t bytes = 0;
            };
            core::Uuid run{};
            std::vector<core::ExtractedFrame> sources;
            std::vector<Frame> frames;
            std::filesystem::path output, weights;
            size_t retained_bytes = 0;
            size_t approved = 0;
            size_t empty_approved = 0;
            uint64_t revision = 0;
        };
        std::shared_ptr<Sequence> sequence;
        struct Work {
            uint64_t id;
            JobHandle job;
            AssistedMaskRequest input;
            std::shared_ptr<Sequence> sequence;
        };
        std::mutex mutex;
        std::condition_variable cv;
        std::optional<Work> pending;
        JobHandle closing_job;
        AssistedMaskSnapshot result;
        uint64_t next = 0;
        bool release = false, stopping = false;
        std::thread worker;

        Impl(JobRegistry& registry, std::function<bool()> check, RevisionReader revision)
            : jobs(registry), admission(std::move(check)), document_revision(std::move(revision)) {}

        bool editable() const {
            const auto& phase = result.phase;
            return !release && !stopping && !pending &&
                   (phase == "idle" || phase == "editing" || phase == "ready" || phase == "tracked" ||
                    phase == "mask_ready" || phase == "saved" || phase == "error" || phase == "expired");
        }

        void retain(const Work& work, size_t index, std::shared_ptr<const std::vector<std::byte>> png,
                    std::filesystem::file_time_type time, uint64_t bytes, float score, bool reliable, bool keyframe = false,
                    int matched_objects = -1, bool semantic_limited = false) {
            std::lock_guard lock(mutex);
            if (release || stopping || result.request != work.id || sequence != work.sequence)
                return;
            auto& frame = sequence->frames.at(index);
            // Approved revisions are immutable during propagation. Explicit edits
            // revoke approval through the same owner before replacing a proposal.
            if (!keyframe && (frame.summary.approved || frame.summary.keyframe))
                return;
            const auto previous = frame.png ? frame.png->size() : 0;
            constexpr size_t budget = size_t(256) << 20;
            if (png->size() > budget || sequence->retained_bytes - previous > budget - png->size())
                throw lfs::Exception(lfs::make_error({.code = lfs::ErrorCode::ResourceExhausted, .domain = lfs::ErrorDomain::Core, .user_message = LOC("image_preview.sam_sequence_budget"), .detection = LFS_SOURCE_SITE_CURRENT()}));
            sequence->retained_bytes = sequence->retained_bytes - previous + png->size();
            if (frame.summary.approved) {
                --sequence->approved;
                if (frame.summary.matched_objects == 0)
                    --sequence->empty_approved;
            }
            auto saved_path = std::move(frame.summary.mask_path);
            frame = {{work.id, score, reliable, false, keyframe, std::move(saved_path), matched_objects, semantic_limited}, std::move(png), time, bytes};
            ++sequence->revision;
            ++result.completed;
        }

        void run() {
            std::optional<Sam2> model;
            std::optional<LlmDet> detector;
            std::filesystem::path image, model_key;
            std::filesystem::file_time_type stamp{};
            uintmax_t bytes = 0;
            Tensor previous;
            std::shared_ptr<const std::vector<std::byte>> previous_png;
            std::vector<float> previous_scores;
            std::filesystem::path previous_image;
            std::filesystem::file_time_type previous_time{};
            uint64_t previous_bytes = 0;
            uint64_t previous_id = 0;
            int previous_matches = -1;
            bool previous_limited = false;
            cudaStream_t stream = nullptr;
            JobHandle active;
            const auto clear_previous = [&] {
                previous = {};
                previous_png.reset();
                previous_scores.clear();
                previous_image.clear();
                previous_time = {};
                previous_bytes = previous_id = 0;
                previous_matches = -1;
                previous_limited = false;
            };
            const auto retire = [&] {
                if (stream)
                    LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "assisted mask drain");
                clear_previous();
                detector.reset();
                model.reset();
                image.clear();
                if (stream) {
                    LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "assisted mask retirement");
                    core::CudaMemoryPool::instance().release_stream(stream);
                    LFS_CUDA_LOG_STREAM_DESTROY(stream, "assisted mask stream teardown");
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
                bool preparation_failure = false;
                try {
                    if (w.input.publish_approved) {
                        std::vector<std::pair<size_t, Sequence::Frame>> approved;
                        {
                            std::lock_guard lock(mutex);
                            if (!w.input.exclude && w.sequence->empty_approved)
                                throw std::runtime_error(LOC("image_preview.sam_empty_keep"));
                            for (size_t i = 0; i < w.sequence->frames.size(); ++i)
                                if (w.sequence->frames[i].summary.approved)
                                    approved.emplace_back(i, w.sequence->frames[i]);
                            result.total = approved.size();
                        }
                        size_t written = 0;
                        for (const auto& [index, frame] : approved) {
                            if (obsolete())
                                break;
                            const auto& source = w.sequence->sources[index];
                            if (fs::last_write_time(source.path) != frame.time || fs::file_size(source.path) != frame.bytes)
                                throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                            auto png = frame.png;
                            if (w.input.exclude) {
                                auto mask = decode_mask(*png, source.width, source.height);
                                for (size_t p = 0; p < mask.numel(); ++p)
                                    mask.ptr<uint8_t>()[p] = mask.ptr<uint8_t>()[p] ? 0 : 255;
                                png = encode_mask({mask.ptr<uint8_t>(), mask.numel()}, source.width, source.height);
                            }
                            const auto path = preprocessing::video_mask_path(w.sequence->output, source.path.filename());
                            const bool existed = fs::exists(path);
                            const auto old_time = existed ? fs::last_write_time(path) : std::filesystem::file_time_type{};
                            const auto old_bytes = existed ? fs::file_size(path) : 0;
                            fs::create_directories(path.parent_path());
                            io::ScopedAtomicOutputFile output(path);
                            std::ofstream file(lfs::core::native_fs::native_path(output.temp_path()), std::ios::binary | std::ios::trunc);
                            file.write(reinterpret_cast<const char*>(png->data()), std::streamsize(png->size()));
                            file.close();
                            if (!file)
                                throw std::runtime_error(LOC("image_preview.sam_write_failed"));
                            if (fs::last_write_time(source.path) != frame.time || fs::file_size(source.path) != frame.bytes)
                                throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                            if (fs::exists(path) != existed ||
                                (existed && (fs::last_write_time(path) != old_time || fs::file_size(path) != old_bytes)))
                                throw std::runtime_error(LOC("image_preview.sam_output_changed"));
                            {
                                std::lock_guard lock(mutex);
                                if (release || stopping || result.request != w.id || sequence != w.sequence)
                                    break;
                                auto& current = sequence->frames[index];
                                if (!current.summary.approved || current.summary.revision != frame.summary.revision)
                                    throw std::runtime_error(LOC("image_preview.sam_proposal_expired"));
                            }
                            // The admitted atomic replacement may finish during cancellation.
                            // Approval and proposal edits are refused until this request finishes;
                            // only this worker can publish a new revision in the meantime.
                            // Never hold the UI snapshot mutex over filesystem I/O.
                            if (auto committed = output.commit(); !committed)
                                throw std::runtime_error(std::string(LOC("image_preview.sam_write_failed")) + ": " + committed.error().message);
                            {
                                std::lock_guard lock(mutex);
                                auto& current = w.sequence->frames[index];
                                if (current.summary.revision == frame.summary.revision)
                                    current.summary.mask_path = path;
                                ++w.sequence->revision;
                                if (result.request == w.id)
                                    ++result.completed;
                            }
                            stage("saving", float(++written) / float(approved.size()));
                        }
                        if (!obsolete())
                            stage("saved", 1.0f);
                        continue;
                    }
                    const auto current_stamp = fs::last_write_time(w.input.image);
                    const auto current_bytes = fs::file_size(w.input.image);
                    const auto system_stamp = core::file_time_to_system_clock(current_stamp);
                    const auto mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(system_stamp.time_since_epoch()).count();
                    if (mtime_ns != w.input.mtime_ns || current_bytes != w.input.size_bytes)
                        throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                    if (!stream)
                        LFS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                    const core::CUDAStreamGuard guard(stream);
                    SemanticBoxes detected;
                    if (w.input.review_frame >= 0) {
                        Sequence::Frame frame;
                        const auto index = size_t(w.input.review_frame);
                        {
                            std::lock_guard lock(mutex);
                            frame = w.sequence->frames[index];
                        }
                        if (!frame.png || frame.time != current_stamp || frame.bytes != current_bytes)
                            throw std::runtime_error(LOC("image_preview.sam_proposal_expired"));
                        const auto& source = w.sequence->sources[index];
                        auto cpu = decode_mask(*frame.png, source.width, source.height);
                        const FrameDrain drain{stream};
                        auto full = cpu.to(Device::CUDA, stream).to(DataType::Float32).div(255).reshape({1, 1, source.height, source.width});
                        previous = core::nn::resize2d(full, 256, 256, core::nn::ResizeMode::Bilinear,
                                                      core::nn::CoordTransform::HalfPixel)
                                       .mul(20)
                                       .sub(10)
                                       .squeeze(0);
                        previous_id = w.id;
                        previous_png = frame.png;
                        previous_scores = {frame.summary.score};
                        previous_matches = frame.summary.matched_objects;
                        previous_limited = frame.summary.semantic_limited;
                        previous_image = w.input.image;
                        previous_time = current_stamp;
                        previous_bytes = current_bytes;
                        const double scale = std::min(1.0, 1536.0 / std::max(source.width, source.height));
                        const int ow = std::max(1, int(source.width * scale)), oh = std::max(1, int(source.height * scale));
                        auto overlay = core::nn::resize2d(full, oh, ow, core::nn::ResizeMode::Bilinear,
                                                          core::nn::CoordTransform::HalfPixel)
                                           .gt(0.5f)
                                           .to(DataType::Float32)
                                           .reshape({oh, ow, 1})
                                           .expand({oh, ow, 4});
                        image.clear(); // Inspection does not install encoder features.
                        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                        std::lock_guard lock(mutex);
                        if (!release && result.request == w.id) {
                            result.overlays = {std::move(overlay)};
                            result.scores = {frame.summary.score};
                            result.matched_objects = previous_matches;
                            result.semantic_limited = previous_limited;
                            result.phase = "ready";
                        }
                        continue;
                    }
                    if (w.input.export_candidate < 0) {
                        if (model && model_key != w.input.weights) {
                            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                            clear_previous();
                            model.reset();
                            image.clear();
                        }
                        const bool encode_image = !model || image != w.input.image || stamp != current_stamp || bytes != current_bytes;
                        if (encode_image || w.input.semantic) {
                            // Drain prior work before admitting the next source buffers.
                            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                            if (previous_image != w.input.image || previous_time != current_stamp || previous_bytes != current_bytes) {
                                clear_previous();
                            }
                            size_t available = 0, total = 0;
                            LFS_CUDA_CHECK(cudaMemGetInfo(&available, &total));
                            auto [width, height, channels] = core::probe_image_info(w.input.image);
                            if (w.input.semantic && !LlmDet::input_size(size_t(height), size_t(width)))
                                throw std::runtime_error(LOC("image_preview.sam_semantic_geometry"));
                            preprocessing::VideoMaskFrame frame{};
                            frame.image_path = w.input.image;
                            frame.width = width;
                            frame.height = height;
                            // Credit only allocations already owned by this model, not a second estimated model reserve.
                            const auto resident = (model ? model->weights_bytes() + model->arena_bytes() : 0) +
                                                  (w.input.semantic && detector ? detector->weights_bytes() : 0);
                            const auto accounted = std::min(total, available + resident);
                            // LLMDet weights and bounded 800/1333 detector activations. The
                            // canonical planner already accounts for source-sized buffers.
                            const auto extra = w.input.semantic ? size_t(2) << 30 : 0;
                            const auto plan = preprocessing::plan_video_mask_resources({&frame, 1}, {},
                                                                                       accounted > extra ? accounted - extra : 0);
                            if (!plan)
                                throw std::runtime_error(w.input.semantic ? LOC("image_preview.sam_semantic_memory") : std::string(plan.error().user_message()));
                            if (!model) {
                                std::optional<core::nn::WeightFile> prepared_weights;
                                auto weights = w.input.weights;
                                if (weights.empty()) {
                                    auto cached = preprocessing::ensure_sam2_weights(true, obsolete);
                                    if (!cached)
                                        throw std::runtime_error(std::string(cached.error().user_message()));
                                    prepared_weights = std::move(*cached);
                                }
                                if (obsolete())
                                    continue;
                                model_failure = true;
                                auto loaded = prepared_weights ? Sam2::load(*prepared_weights, Device::CUDA, DataType::Float32)
                                                               : Sam2::load(weights, Device::CUDA, DataType::Float32);
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
                            if (encode_image) {
                                // Cancellation during detection must not leave the
                                // previous source associated with new encoder features.
                                image.clear();
                                stage("encoding", 0.3f);
                                model_failure = true;
                                const auto encoded = model->set_image(decoded.tensor.unsqueeze(0));
                                if (!encoded)
                                    throw std::runtime_error(std::string(encoded.error().user_message()));
                                model_failure = false;
                            }
                            if (w.input.semantic) {
                                const auto& semantic = *w.input.semantic;
                                if (decoded.tensor.ndim() != 3 || decoded.tensor.size(0) != 3 ||
                                    decoded.tensor.size(1) != size_t(height) || decoded.tensor.size(2) != size_t(width))
                                    throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                                if (!detector) {
                                    stage("loading", 0.35f);
                                    preparation_failure = true;
                                    auto prepared = semantic.prepare_weights(obsolete);
                                    if (!prepared)
                                        throw lfs::Exception(prepared.error());
                                    preparation_failure = false;
                                    model_failure = true;
                                    LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                                    const FrameDrain transfer_drain{stream};
                                    auto loaded = LlmDet::load(*prepared, Device::CUDA);
                                    if (!loaded)
                                        throw lfs::Exception(loaded.error());
                                    detector.emplace(std::move(*loaded));
                                }
                                model_failure = true;
                                if (obsolete())
                                    continue;
                                auto text = detector->encode(semantic.descriptions);
                                if (!text)
                                    throw lfs::Exception(text.error());
                                stage("predicting", 0.5f);
                                detected = find_semantic_boxes(*detector, decoded.tensor.unsqueeze(0), *text,
                                                               width, height, semantic.confidence, stream);
                                model_failure = false;
                            }
                            LFS_CUDA_CHECK(cudaStreamSynchronize(stream)); // Decoder owns frame leases until this point.
                            image = w.input.image;
                            stamp = current_stamp;
                            bytes = current_bytes;
                        }
                    }
                    if (obsolete())
                        continue;
                    auto [width, height, channels] = core::probe_image_info(w.input.image);
                    if (w.sequence) {
                        const auto source = std::ranges::find(w.sequence->sources, w.input.image, &core::ExtractedFrame::path);
                        if (source == w.sequence->sources.end() || source->width != width || source->height != height)
                            throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                    }
                    if (w.input.tracking) {
                        const auto& track = *w.input.tracking;
                        const auto session = w.sequence;
                        if (w.input.refine_request != previous_id || !previous.is_valid() ||
                            w.input.refine_candidate < 0 || w.input.refine_candidate >= int(previous.size(0)) ||
                            size_t(w.input.refine_candidate) >= previous_scores.size() ||
                            previous_image != w.input.image || previous_time != current_stamp || previous_bytes != current_bytes)
                            throw std::runtime_error(LOC("image_preview.sam_proposal_expired"));
                        if (!model->supports_video()) {
                            model_failure = true;
                            throw std::runtime_error(LOC("image_preview.sam_video_weights_required"));
                        }
                        const int effective_first = track.direction == preprocessing::VideoMaskDirection::Forward ? track.seed : track.first;
                        const int effective_last = track.direction == preprocessing::VideoMaskDirection::Reverse ? track.seed : track.last;
                        std::vector<std::pair<std::filesystem::file_time_type, uint64_t>> proofs(size_t(effective_last - effective_first + 1));
                        const auto proof = [&](size_t index) -> const auto& { return proofs[index - size_t(effective_first)]; };
                        for (int i = effective_first; i <= effective_last; ++i) {
                            if (obsolete())
                                break;
                            const auto& frame = session->sources[size_t(i)];
                            proofs[size_t(i - effective_first)] = {fs::last_write_time(frame.path), fs::file_size(frame.path)};
                        }
                        if (obsolete())
                            continue;
                        if (proof(size_t(track.seed)) != std::pair{current_stamp, current_bytes})
                            throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                        std::map<int, std::shared_ptr<const std::vector<std::byte>>> anchors;
                        std::vector<std::pair<int, Sequence::Frame>> protected_frames;
                        {
                            std::lock_guard lock(mutex);
                            result.total = 1;
                            for (int i = effective_first; i <= effective_last; ++i) {
                                const auto& frame = session->frames[size_t(i)];
                                if (i != track.seed && (frame.summary.approved || frame.summary.keyframe)) {
                                    protected_frames.emplace_back(i, frame);
                                } else if (i != track.seed) {
                                    ++result.total;
                                }
                            }
                        }
                        for (const auto& [index, frame] : protected_frames) {
                            const auto [time, frame_bytes] = proof(size_t(index));
                            if (frame.time != time || frame.bytes != frame_bytes)
                                throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                            anchors.emplace(index, frame.png);
                        }
                        Tensor seed;
                        if (previous_png) {
                            seed = decode_mask(*previous_png, width, height);
                        } else {
                            seed = core::nn::resize2d(previous.slice(0, w.input.refine_candidate, w.input.refine_candidate + 1).unsqueeze(0),
                                                      height, width, core::nn::ResizeMode::Bilinear, core::nn::CoordTransform::HalfPixel)
                                       .gt(0)
                                       .to(DataType::Float32)
                                       .mul(255)
                                       .to(DataType::UInt8)
                                       .to(Device::CPU, stream)
                                       .contiguous();
                            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                        }
                        auto seed_png = encode_mask({seed.ptr<uint8_t>(), seed.numel()}, width, height);
                        const float seed_score = previous_scores.at(size_t(w.input.refine_candidate));
                        bool unchanged = false;
                        {
                            std::lock_guard lock(mutex);
                            const auto& frame = session->frames[size_t(track.seed)];
                            unchanged = frame.png && *frame.png == *seed_png;
                            if (unchanged && result.request == w.id)
                                ++result.completed;
                        }
                        if (!unchanged)
                            retain(w, size_t(track.seed), seed_png, current_stamp, current_bytes, seed_score, true, true);
                        anchors.insert_or_assign(track.seed, seed_png);
                        const auto run_pass = [&](int from, int to, bool reverse) {
                            const int first = std::min(from, to), last = std::max(from, to);
                            std::vector<preprocessing::VideoMaskFrame> frames;
                            for (int i = first; i <= last; ++i) {
                                const auto& frame = session->sources[size_t(i)];
                                frames.push_back({.image_path = frame.path, .mask_path = {}, .width = frame.width, .height = frame.height, .time_seconds = frame.timestamp});
                            }
                            const auto& source = session->sources[size_t(from)];
                            auto mask = decode_mask(*anchors.at(from), source.width, source.height);
                            preprocessing::VideoMaskOptions options;
                            options.profile = preprocessing::VideoMaskProfile::Subject;
                            options.direction = reverse ? preprocessing::VideoMaskDirection::Reverse : preprocessing::VideoMaskDirection::Forward;
                            options.keep_subject = true;
                            options.dilation_radius = 0;
                            options.prompt.emplace();
                            options.prompt->frame = from - first;
                            options.prompt->mask = mask.to(DataType::Float32).div(255).reshape({1, 1, source.height, source.width});
                            options.cancel_requested = obsolete;
                            options.progress = [&](const auto&) {
                                float fraction = 0;
                                {
                                    std::lock_guard lock(mutex);
                                    if (result.total)
                                        fraction = float(result.completed) / float(result.total);
                                }
                                stage("tracking", fraction);
                            };
                            options.mask_ready = [&, first](size_t index, const Tensor& output, float score, bool reliable) {
                                const size_t original = size_t(first) + index;
                                if (original == size_t(track.seed) || anchors.contains(int(original)) || obsolete())
                                    return;
                                const auto& source = session->sources[original];
                                if (fs::last_write_time(source.path) != proof(original).first || fs::file_size(source.path) != proof(original).second)
                                    throw lfs::Exception(lfs::make_error({.code = lfs::ErrorCode::FailedPrecondition, .domain = lfs::ErrorDomain::Core, .user_message = LOC("image_preview.sam_source_changed"), .detection = LFS_SOURCE_SITE_CURRENT()}));
                                retain(w, original, encode_mask({output.ptr<uint8_t>(), output.numel()}, source.width, source.height),
                                       proof(original).first, proof(original).second, score, reliable);
                            };
                            const auto tracked = preprocessing::mask_video_sequence(frames, {}, options, stream, &*model);
                            model->reset_video();
                            image.clear();
                            if (!tracked && !obsolete()) {
                                model_failure = tracked.error().domain() == lfs::ErrorDomain::CUDA;
                                throw lfs::Exception(tracked.error());
                            }
                        };
                        for (const bool reverse : {false, true}) {
                            if ((reverse && track.direction == preprocessing::VideoMaskDirection::Forward) ||
                                (!reverse && track.direction == preprocessing::VideoMaskDirection::Reverse))
                                continue;
                            std::vector<int> checkpoints{track.seed};
                            for (const auto& [index, unused] : anchors)
                                if (reverse ? index < track.seed && index >= track.first : index > track.seed && index <= track.last)
                                    checkpoints.push_back(index);
                            checkpoints.push_back(reverse ? track.first : track.last);
                            std::ranges::sort(checkpoints);
                            checkpoints.erase(std::unique(checkpoints.begin(), checkpoints.end()), checkpoints.end());
                            if (reverse)
                                std::ranges::reverse(checkpoints);
                            for (size_t i = 1; i < checkpoints.size() && !obsolete(); ++i)
                                run_pass(checkpoints[i - 1], checkpoints[i], reverse);
                        }
                        clear_previous();
                        if (!obsolete())
                            stage("tracked", 1.0f);
                        continue;
                    }
                    if (w.input.export_candidate >= 0) {
                        if (w.input.refine_request != previous_id || !previous.is_valid() ||
                            w.input.export_candidate >= int(previous.size(0)) ||
                            w.input.export_candidate >= int(previous_scores.size()) || previous_image != w.input.image ||
                            previous_time != current_stamp || previous_bytes != current_bytes)
                            throw std::runtime_error(LOC("image_preview.sam_proposal_expired"));
                        stage("rendering", 0.9f);
                        Tensor selected;
                        if (previous_png) {
                            selected = decode_mask(*previous_png, width, height);
                        } else {
                            selected = core::nn::resize2d(
                                           previous.slice(0, w.input.export_candidate, w.input.export_candidate + 1).unsqueeze(0),
                                           height, width, core::nn::ResizeMode::Bilinear, core::nn::CoordTransform::HalfPixel)
                                           .gt(0)
                                           .to(DataType::UInt8)
                                           .to(Device::CPU, stream)
                                           .contiguous();
                            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                        }
                        if (obsolete())
                            continue;
                        if (fs::last_write_time(w.input.image) != current_stamp || fs::file_size(w.input.image) != current_bytes)
                            throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                        std::vector<uint8_t> pixels(size_t(width) * height);
                        for (size_t p = 0; p < pixels.size(); ++p)
                            pixels[p] = ((selected.ptr<uint8_t>()[p] != 0) != w.input.exclude) ? 255 : 0;
                        auto png = encode_mask(pixels, width, height);
                        if (w.input.retain_frame) {
                            const auto found = std::ranges::find(w.sequence->sources, w.input.image, &core::ExtractedFrame::path);
                            if (found == w.sequence->sources.end())
                                throw std::runtime_error(LOC("image_preview.sam_proposal_expired"));
                            const float score = previous_scores[size_t(w.input.export_candidate)];
                            retain(w, size_t(found - w.sequence->sources.begin()), png, current_stamp, current_bytes, score,
                                   score >= core::nn::models::kSam2MinIou && !previous_limited, true, previous_matches, previous_limited);
                        }
                        std::lock_guard lock(mutex);
                        if (!release && result.request == w.id) {
                            result.mask = std::move(png);
                            result.width = width;
                            result.height = height;
                            result.image = w.input.image;
                            result.source_time = current_stamp;
                            result.source_bytes = current_bytes;
                            result.matched_objects = previous_matches;
                            result.semantic_limited = previous_limited;
                            result.phase = "mask_ready";
                        }
                        continue;
                    }
                    Tensor semantic_preview;
                    if (w.input.semantic) {
                        model_failure = true;
                        auto low_union = Tensor::full({1, 256, 256}, -10.f, Device::CUDA);
                        auto combined = Tensor::full({1u, size_t(height), size_t(width)}, -10.f, Device::CUDA);
                        float quality = detected.boxes.empty() ? 0.f : 1.f;
                        const auto prediction_quality = [stream](const core::nn::models::Sam2Prediction& prediction) {
                            auto score = prediction.iou.to(Device::CPU, stream).contiguous();
                            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                            if (!score.numel() || !std::isfinite(score.ptr<float>()[0]))
                                throw std::runtime_error(LOC("image_preview.sam_semantic_failed"));
                            return score.ptr<float>()[0];
                        };
                        for (size_t i = 0; i < detected.boxes.size(); ++i) {
                            if (obsolete())
                                break;
                            auto prediction = model->predict({}, detected.boxes[i], false, nullptr,
                                                             core::nn::models::Sam2MaskOutput::RawLowResolution);
                            if (!prediction)
                                throw lfs::Exception(prediction.error());
                            const float initial_quality = prediction_quality(*prediction);
                            if (obsolete())
                                break;
                            // Refine each detector proposal before combining objects. A
                            // union is not a valid single-object SAM refinement prompt.
                            const auto mask_input = prediction->low_res_logits.clamp(-32.f, 32.f).unsqueeze(1);
                            auto refined = model->predict({}, detected.boxes[i], false, &mask_input,
                                                          core::nn::models::Sam2MaskOutput::RawLowResolution);
                            if (!refined)
                                throw lfs::Exception(refined.error());
                            const float refined_quality = prediction_quality(*refined);
                            const bool improved = refined_quality > initial_quality;
                            const auto& selected = improved ? *refined : *prediction;
                            auto original = model->postprocess_masks(selected.low_res_logits);
                            if (!original)
                                throw lfs::Exception(original.error());
                            combined = combined.maximum(*original);
                            low_union = low_union.maximum(selected.low_res_logits);
                            quality = std::min(quality, improved ? refined_quality : initial_quality);
                            stage("predicting", 0.6f + 0.3f * float(i + 1) / float(detected.boxes.size()));
                        }
                        if (obsolete())
                            continue;
                        // Keep the exact source-resolution union for Apply/export;
                        // the low-resolution union is only a future refinement seed.
                        semantic_preview = combined.gt(0).to(DataType::Float32);
                        auto selected = semantic_preview.mul(255).to(DataType::UInt8).to(Device::CPU, stream).contiguous();
                        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                        auto png = encode_mask({selected.ptr<uint8_t>(), selected.numel()}, width, height);
                        if (obsolete())
                            continue;
                        previous = low_union.clamp(-32.f, 32.f);
                        previous_png = std::move(png);
                        previous_scores = {quality};
                        model_failure = false;
                    } else {
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
                            if (previous_matches > 1)
                                throw lfs::Exception(lfs::make_error({
                                    .code = lfs::ErrorCode::Unsupported,
                                    .domain = lfs::ErrorDomain::Preprocess,
                                    .user_message = LOC("image_preview.sam_semantic_refine_combined"),
                                    .detection = LFS_SOURCE_SITE_CURRENT(),
                                }));
                            if (previous_matches != 0)
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
                        previous_png.reset();
                        auto scores = prediction->iou.to(Device::CPU, stream).contiguous();
                        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                        previous_scores.assign(scores.ptr<float>(), scores.ptr<float>() + previous.size(0));
                    }
                    previous_id = w.id;
                    previous_matches = w.input.semantic ? int(detected.boxes.size()) : -1;
                    previous_limited = detected.limited;
                    previous_image = w.input.image;
                    previous_time = current_stamp;
                    previous_bytes = current_bytes;
                    const int ow = std::max(1, int(width * std::min(1.0, 1536.0 / std::max(width, height))));
                    const int oh = std::max(1, int(height * std::min(1.0, 1536.0 / std::max(width, height))));
                    auto masks = core::nn::resize2d((semantic_preview.is_valid() ? semantic_preview : previous).unsqueeze(0), oh, ow,
                                                    core::nn::ResizeMode::Bilinear, core::nn::CoordTransform::HalfPixel)
                                     .gt(semantic_preview.is_valid() ? 0.5f : 0.f)
                                     .to(DataType::Float32);
                    if (obsolete())
                        continue;
                    if (fs::last_write_time(image) != stamp || fs::file_size(image) != bytes)
                        throw std::runtime_error(LOC("image_preview.sam_source_changed"));
                    if (w.sequence) {
                        const auto found = std::ranges::find(w.sequence->sources, w.input.image, &core::ExtractedFrame::path);
                        if (found != w.sequence->sources.end()) {
                            auto png = previous_png;
                            if (!png) {
                                auto selected = core::nn::resize2d(previous.slice(0, 0, 1).unsqueeze(0), height, width,
                                                                   core::nn::ResizeMode::Bilinear, core::nn::CoordTransform::HalfPixel)
                                                    .gt(0)
                                                    .to(DataType::Float32)
                                                    .mul(255)
                                                    .to(DataType::UInt8)
                                                    .to(Device::CPU, stream)
                                                    .contiguous();
                                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                                png = encode_mask({selected.ptr<uint8_t>(), selected.numel()}, width, height);
                            }
                            const float score = previous_scores.front();
                            retain(w, size_t(found - w.sequence->sources.begin()), std::move(png), current_stamp,
                                   current_bytes, score, score >= core::nn::models::kSam2MinIou && !previous_limited,
                                   true, previous_matches, previous_limited);
                        }
                    }
                    std::vector<Tensor> overlays;
                    for (int c = 0; c < int(previous.size(0)); ++c)
                        overlays.push_back(masks.slice(1, c, c + 1).reshape({oh, ow, 1}).expand({oh, ow, 4}));
                    stage("rendering", 0.95f);
                    std::lock_guard lock(mutex);
                    if (!release && result.request == w.id) {
                        result.overlays = std::move(overlays);
                        result.scores = previous_scores;
                        result.matched_objects = previous_matches;
                        result.semantic_limited = previous_limited;
                        result.phase = "ready";
                    }
                } catch (const std::exception& error) {
                    if (stream)
                        LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "assisted mask error drain");
                    // An obsolete preparation is a normal cancellation, not a
                    // failed model. Atomic staging has already unwound here.
                    if (preparation_failure && obsolete())
                        continue;
                    const auto* typed = dynamic_cast<const lfs::Exception*>(&error);
                    if (w.input.tracking && model) {
                        model->reset_video();
                        image.clear();
                    }
                    if (model_failure || (typed && typed->error().domain() == lfs::ErrorDomain::CUDA)) {
                        clear_previous();
                        detector.reset();
                        model.reset();
                        image.clear();
                    }
                    const bool semantic_failure = w.input.semantic &&
                                                  (model_failure || (typed && typed->error().domain() == lfs::ErrorDomain::CUDA));
                    if (semantic_failure || preparation_failure)
                        LOG_ERROR("Semantic mask generation failed: {}", error.what());
                    std::lock_guard lock(mutex);
                    if (result.request == w.id) {
                        result.phase = "error";
                        result.error = preparation_failure ? LOC("image_preview.sam_semantic_prepare_failed") : semantic_failure ? LOC("image_preview.sam_semantic_failed")
                                                                                                            : typed              ? std::string(typed->error().user_message())
                                                                                                                                 : error.what();
                    }
                }
            }
        }
    };

    AssistedMasking::AssistedMasking(JobRegistry& jobs, std::function<bool()> admission, RevisionReader revision)
        : impl_(std::make_unique<Impl>(jobs, std::move(admission), std::move(revision))) {}
    AssistedMasking::~AssistedMasking() {
        try {
            shutdown();
        } catch (const std::exception& error) {
            LOG_ERROR("Assisted mask shutdown failed: {}", error.what());
        }
    }
    uint64_t AssistedMasking::submit(AssistedMaskRequest input) {
        input.image = input.image.lexically_normal();
        std::shared_ptr<Impl::Sequence> sequence;
        {
            std::lock_guard lock(impl_->mutex);
            if (impl_->stopping)
                throw std::runtime_error(LOC("image_preview.sam_shutting_down"));
            if (!input.run.is_nil()) {
                sequence = impl_->sequence;
                if (!sequence || sequence->run != input.run)
                    throw std::invalid_argument(LOC("image_preview.sam_proposal_expired"));
                if (input.weights.empty())
                    input.weights = sequence->weights;
            }
        }
        const auto valid = [](float v) { return std::isfinite(v) && v >= 0 && v <= 1; };
        // Retaining is a destination for export_candidate, not another request mode.
        const int modes = int(input.tracking.has_value()) + int(input.review_frame >= 0) + int(input.publish_approved) + int(input.export_candidate >= 0) + int(input.semantic.has_value());
        if (modes > 1 || input.review_frame < -1 || (sequence && std::ranges::none_of(sequence->sources, [&](const auto& source) {
                                                         return source.path.lexically_normal() == input.image.lexically_normal();
                                                     })))
            throw std::invalid_argument(LOC("image_preview.sam_sequence_invalid"));
        if (input.image.empty() || input.revision.empty() || input.points.size() > 256 ||
            (input.points.empty() && !input.box && !input.tracking && !input.semantic && !input.publish_approved && input.review_frame < 0 && input.export_candidate < 0))
            throw std::invalid_argument(LOC("image_preview.sam_invalid_input"));
        if (input.semantic) {
            auto& semantic = *input.semantic;
            if (!std::isfinite(semantic.confidence) || semantic.confidence <= 0 || semantic.confidence >= 1)
                throw std::invalid_argument(LOC("image_preview.sam_semantic_threshold_invalid"));
            if (!semantic.prepare_weights || !LlmDet::valid_phrases(semantic.descriptions) || !input.points.empty() || input.box ||
                input.refine_request || input.refine_candidate != -1)
                throw std::invalid_argument(LOC("image_preview.sam_semantic_invalid"));
        }
        for (const auto& p : input.points)
            if (!valid(p[0]) || !valid(p[1]) || (p[2] != 0 && p[2] != 1))
                throw std::invalid_argument(LOC("image_preview.sam_invalid_point"));
        if (input.box && (!std::ranges::all_of(*input.box, valid) || (*input.box)[0] >= (*input.box)[2] || (*input.box)[1] >= (*input.box)[3]))
            throw std::invalid_argument(LOC("image_preview.sam_invalid_box"));
        if (input.tracking) {
            const auto& track = *input.tracking;
            if (!sequence || track.first < 0 || track.last < track.first ||
                track.seed < track.first || track.seed > track.last || track.last >= int(sequence->sources.size()) ||
                sequence->sources[size_t(track.seed)].path.lexically_normal() != input.image.lexically_normal())
                throw std::invalid_argument(LOC("image_preview.sam_sequence_invalid"));
        }
        if (input.review_frame >= 0 && (!sequence || input.review_frame >= int(sequence->sources.size()) ||
                                        sequence->sources[size_t(input.review_frame)].path.lexically_normal() != input.image.lexically_normal()))
            throw std::invalid_argument(LOC("image_preview.sam_sequence_invalid"));
        if (input.retain_frame && (!sequence || input.export_candidate < 0))
            throw std::invalid_argument(LOC("image_preview.sam_sequence_invalid"));
        if (input.publish_approved && (!sequence || sequence->output.empty()))
            throw std::invalid_argument(LOC("image_preview.sam_sequence_invalid"));
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
        impl_->apply_revision.reset();
        impl_->document_apply = input.export_candidate >= 0 && !input.retain_frame;
        if (impl_->document_apply && impl_->document_revision)
            impl_->apply_revision = impl_->document_revision();
        impl_->result = {};
        impl_->result.request = id;
        impl_->result.image = input.image;
        impl_->result.revision = input.revision;
        impl_->result.phase = "queued";
        impl_->pending = Impl::Work{id, impl_->job, std::move(input), std::move(sequence)};
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
        if (impl_->result.phase == "mask_ready" && impl_->document_apply && impl_->document_revision &&
            (!impl_->apply_revision || impl_->apply_revision != impl_->document_revision())) {
            impl_->result.phase = "expired";
            impl_->result.mask.reset();
        }
        auto result = impl_->result;
        result.sequence_revision = impl_->sequence ? impl_->sequence->revision : 0;
        result.approved = impl_->sequence ? impl_->sequence->approved : 0;
        result.empty_approved = impl_->sequence ? impl_->sequence->empty_approved : 0;
        return result;
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

    void AssistedMasking::beginSequence(core::Uuid run, std::vector<core::ExtractedFrame> frames,
                                        std::filesystem::path output, std::filesystem::path weights) {
        if (run.is_nil() || frames.empty() || frames.size() > size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument(LOC("image_preview.sam_sequence_invalid"));
        std::set<std::filesystem::path> paths;
        double time = -std::numeric_limits<double>::infinity();
        for (auto& frame : frames) {
            if (frame.path.empty() || frame.width <= 0 || frame.height <= 0 || !std::isfinite(frame.timestamp) ||
                frame.timestamp < time || !paths.insert(frame.path.lexically_normal()).second)
                throw std::invalid_argument(LOC("image_preview.sam_sequence_invalid"));
            frame.path = frame.path.lexically_normal();
            time = frame.timestamp;
        }
        {
            std::lock_guard lock(impl_->mutex);
            if (impl_->sequence && impl_->sequence->run == run) {
                if (impl_->sequence->sources != frames || impl_->sequence->output != output || impl_->sequence->weights != weights)
                    throw std::invalid_argument(LOC("image_preview.sam_sequence_invalid"));
                return;
            }
        }
        close();
        auto sequence = std::make_shared<Impl::Sequence>();
        sequence->run = run;
        sequence->frames.resize(frames.size());
        sequence->sources = std::move(frames);
        sequence->output = std::move(output);
        sequence->weights = std::move(weights);
        std::lock_guard lock(impl_->mutex);
        impl_->sequence = std::move(sequence);
    }

    std::vector<AssistedMaskFrameSummary> AssistedMasking::sequenceFrames(core::Uuid run, size_t first, size_t count) {
        std::lock_guard lock(impl_->mutex);
        std::vector<AssistedMaskFrameSummary> result;
        if (impl_->sequence && impl_->sequence->run == run && first < impl_->sequence->frames.size()) {
            count = std::min({count, size_t(256), impl_->sequence->frames.size() - first});
            result.reserve(count);
            for (size_t i = first; i < first + count; ++i)
                result.push_back(impl_->sequence->frames[i].summary);
        }
        return result;
    }

    bool AssistedMasking::approveFrames(core::Uuid run, std::span<const AssistedMaskApproval> approvals) {
        std::lock_guard lock(impl_->mutex);
        auto sequence = impl_->sequence;
        if (!sequence || sequence->run != run || approvals.empty() || approvals.size() > sequence->frames.size())
            return false;
        if (!impl_->editable())
            return false;
        std::set<size_t> indices;
        for (const auto& update : approvals) {
            if (update.index >= sequence->frames.size() || !indices.insert(update.index).second)
                return false;
            const auto& frame = sequence->frames[update.index];
            if (!frame.png || frame.summary.revision != update.revision)
                return false;
        }
        bool changed = false;
        for (const auto& update : approvals) {
            auto& frame = sequence->frames[update.index];
            if (frame.summary.approved == update.approved)
                continue;
            if (update.approved)
                ++sequence->approved;
            else
                --sequence->approved;
            if (frame.summary.matched_objects == 0) {
                if (update.approved)
                    ++sequence->empty_approved;
                else
                    --sequence->empty_approved;
            }
            frame.summary.approved = update.approved;
            changed = true;
        }
        if (changed)
            ++sequence->revision;
        return true;
    }

    std::weak_ptr<const std::vector<std::byte>> AssistedMasking::sequencePreview(std::string_view key) {
        if (!key.starts_with("sam:"))
            return {};
        key.remove_prefix(4);
        const auto split = key.find(':');
        if (split == std::string_view::npos)
            return {};
        const auto run = core::Uuid::from_string(key.substr(0, split));
        key.remove_prefix(split + 1);
        const auto next = key.find(':');
        if (!run || next == std::string_view::npos)
            return {};
        size_t index = 0;
        uint64_t revision = 0;
        const auto [index_end, index_error] = std::from_chars(key.data(), key.data() + next, index);
        const auto [revision_end, revision_error] = std::from_chars(key.data() + next + 1, key.data() + key.size(), revision);
        if (index_error != std::errc{} || index_end != key.data() + next || revision_error != std::errc{} || revision_end != key.data() + key.size())
            return {};
        std::lock_guard lock(impl_->mutex);
        const auto& sequence = impl_->sequence;
        if (!sequence || sequence->run != *run || index >= sequence->frames.size() || sequence->frames[index].summary.revision != revision)
            return {};
        return sequence->frames[index].png;
    }

    void AssistedMasking::endSequence(core::Uuid run) {
        {
            std::lock_guard lock(impl_->mutex);
            if (!impl_->sequence || impl_->sequence->run != run)
                return;
            impl_->sequence.reset();
        }
        close();
    }

    void AssistedMasking::invalidateFrame(core::Uuid run, size_t index) {
        std::lock_guard lock(impl_->mutex);
        auto sequence = impl_->sequence;
        if (!sequence || sequence->run != run || index >= sequence->frames.size())
            return;
        if (!impl_->editable())
            throw std::runtime_error(LOC("projects.gallery.error.busy"));
        auto& frame = sequence->frames[index];
        if (frame.png)
            sequence->retained_bytes -= frame.png->size();
        if (frame.summary.approved) {
            --sequence->approved;
            if (frame.summary.matched_objects == 0)
                --sequence->empty_approved;
        }
        auto saved_path = std::move(frame.summary.mask_path);
        frame = {};
        frame.summary.mask_path = std::move(saved_path);
        ++sequence->revision;
        if (impl_->result.image == sequence->sources[index].path) {
            impl_->result = {};
            impl_->result.phase = impl_->job ? "editing" : "idle";
        }
    }
} // namespace lfs::vis::gui
