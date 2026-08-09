/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#define NOMINMAX

#include "default_executor.h"
#include "icode_stream.h"
#include "icodec.h"
#include "icodec_registry.h"
#include "iexecutor.h"
#include "iimage.h"
#include "image.h"
#include "imgproc/device_buffer.h"
#include "imgproc/device_guard.h"
#include "imgproc/exception.h"
#include "imgproc/pinned_buffer.h"
#include "log.h"
#include "processing_results.h"
#include "user_executor.h"
#include <algorithm>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <nvimgcodec.h>
#include <nvtx3/nvtx3.hpp>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nvimgcodec {

    class IImage;
    class ICodeStream;
    class ICodecRegistry;
    class ICodec;
    class ILogger;

    template <typename ProcessorEntry>
    struct SampleEntry : public IImage {
        SampleEntry(const nvimgcodecExecutionParams_t* exec_params)
            : image_desc{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_DESC, sizeof(nvimgcodecImageDesc_t), nullptr, this, static_get_image_info, static_image_ready} {
        }

        SampleEntry(SampleEntry<ProcessorEntry>&& oth) = default;

        void setIndex(int index) override { sample_idx = index; }
        int getIndex() override { return sample_idx; }

        void setImageInfo(const nvimgcodecImageInfo_t* new_image_info) override { image_info = *new_image_info; }
        void getImageInfo(nvimgcodecImageInfo_t* out_image_info) override { *out_image_info = image_info; }

        void setPromise(std::shared_ptr<ProcessingResultsPromise> new_promise) override { /** not used */ }
        std::shared_ptr<ProcessingResultsPromise> getPromise() override { /** not used */ return nullptr; }

        nvimgcodecStatus_t imageReady(nvimgcodecProcessingStatus_t processing_status) {
            status = processing_status;
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        static nvimgcodecStatus_t static_get_image_info(void* instance, nvimgcodecImageInfo_t* result) {
            assert(instance);
            SampleEntry* handle = reinterpret_cast<SampleEntry*>(instance);
            handle->getImageInfo(result);
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        static nvimgcodecStatus_t static_image_ready(void* instance, nvimgcodecProcessingStatus_t processing_status) {
            assert(instance);
            SampleEntry* handle = reinterpret_cast<SampleEntry*>(instance);
            handle->imageReady(processing_status);
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        nvimgcodecImageDesc_t* getImageDesc() override {
            return &image_desc;
        }

        nvimgcodecImageDesc_t image_desc{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_DESC, sizeof(nvimgcodecImageDesc_t), nullptr};
        int sample_idx = -1;
        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        nvimgcodecProcessingStatus_t fallback_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        ICodeStream* code_stream = nullptr;
        ICodec* codec = nullptr;
        IImage* orig_image = nullptr; // image descriptor from the user
        nvimgcodecImageInfo_t orig_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        ProcessorEntry* processor = nullptr;
        bool should_copy = false;
        // Per-submission params, set in initState() AFTER executor->wait() drains
        // the previous submission. Read by workers via
        // static_cast<const nvimgcodec{Encode,Decode}Params_t*>(params).
        //
        // Aliases caller-owned storage. Per the public C API contract, the caller
        // of encode()/decode() must keep their params struct alive and unmodified
        // until the returned future resolves.
        //
        // Written only in initState(); never read after the future resolves.
        const void* params = nullptr;
    };

    struct PerThread {
        explicit PerThread(int device_id, int stream_idx, nvimgcodecPinnedAllocator_t* pinned_allocator = nullptr, nvimgcodecDeviceAllocator_t* device_allocator = nullptr)
            : stream_idx(stream_idx), event(nullptr) {
            if (device_id != NVIMGCODEC_DEVICE_CPU_ONLY) {
                CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }

            pinned_buffers.reserve(2);
            for (int i = 0; i < 2; ++i) {
                pinned_buffers.emplace_back(pinned_allocator);
            }

            device_buffers.reserve(2);
            for (int i = 0; i < 2; ++i) {
                device_buffers.emplace_back(device_allocator);
            }
        }
        PerThread(const PerThread& oth) = delete;
        PerThread& operator=(const PerThread& oth) = delete;

        PerThread(PerThread&& other) noexcept
            : stream_idx(std::exchange(other.stream_idx, -1)), event(std::exchange(other.event, nullptr)), pinned_buffers(std::move(other.pinned_buffers)), device_buffers(std::move(other.device_buffers)), user_streams(std::move(other.user_streams)) {
        }

        PerThread& operator=(PerThread&& other) noexcept {
            if (this != &other) {
                if (event) {
                    LOG_CUDA_ERROR(cudaEventDestroy(event));
                }

                stream_idx = std::exchange(other.stream_idx, -1);
                event = std::exchange(other.event, nullptr);
                pinned_buffers = std::move(other.pinned_buffers);
                device_buffers = std::move(other.device_buffers);
                user_streams = std::move(other.user_streams);
                elligible_samples_snapshot_ = std::move(other.elligible_samples_snapshot_);
            }
            return *this;
        }

        ~PerThread() {
            pinned_buffers.clear();
            device_buffers.clear();
            if (event) {
                LOG_CUDA_ERROR(cudaEventDestroy(event));
                event = nullptr;
            }
        }

        int stream_idx;
        cudaEvent_t event;
        std::vector<std::pair<int, nvimgcodecProcessingStatus_t>> processed_samples;
        // reusable temporary buffers
        std::vector<PinnedBuffer> pinned_buffers;
        std::vector<DeviceBuffer> device_buffers;
        size_t pinned_buffer_idx = 0;
        size_t device_buffer_idx = 0;

        std::set<cudaStream_t> user_streams;

        // snapshot of the elligible samples per processor (atomic counters)
        // By getting a per-thread snapshot before fetching a sample idx for processing
        // we make sure we process enough samples to fill each processor (may check a few more)
        std::map<void*, size_t> elligible_samples_snapshot_;
    };

    // Synchronization Model:
    // - Multiple worker threads can share one CUDA stream to reduce overhead and improve throughput
    // - Each stream has a barrier that all its assigned threads must reach before post-synchronization
    // - The last thread to arrive at the barrier triggers completePostSync via the completion function
    // - completePostSync handles CUDA event recording and user stream synchronization once per stream
    // - executor->wait() at the start of each iteration ensures no overlap between batches
    // - An extra thread (num_threads_) gets its own dedicated stream for special processing
    template <typename CompletionFunc>
    struct PerStream {
        explicit PerStream(int device_id, std::set<int> _thread_ids, CompletionFunc completion)
            : thread_ids(std::move(_thread_ids)), barrier(nullptr) {
            if (thread_ids.size() > 1) {
                barrier = std::make_unique<std::barrier<CompletionFunc>>(thread_ids.size(), completion);
            }
            if (device_id == NVIMGCODEC_DEVICE_CPU_ONLY) {
                stream = nullptr;
                event = nullptr;
            } else {
                CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
        }

        PerStream(const PerStream& oth) = delete;
        PerStream& operator=(const PerStream& oth) = delete;

        // Move constructor
        PerStream(PerStream&& other) noexcept
            : stream(other.stream), event(other.event), thread_ids(std::move(other.thread_ids)), barrier(std::move(other.barrier)), user_streams(std::move(other.user_streams)), processed_samples(std::move(other.processed_samples)) {
            other.stream = nullptr;
            other.event = nullptr;
        }

        PerStream& operator=(PerStream&& other) noexcept {
            if (this != &other) {
                // Safe to omit destruction here; the resources from moved-from object will be cleaned up by its destructor
                std::swap(stream, other.stream);
                std::swap(event, other.event);
                std::swap(thread_ids, other.thread_ids);
                std::swap(barrier, other.barrier);
                std::swap(user_streams, other.user_streams);
                std::swap(processed_samples, other.processed_samples);
                // mutex_ is intentionally not swapped (not movable or copiable)
            }
            return *this;
        }

        ~PerStream() {
            if (stream) {
                LOG_CUDA_ERROR(cudaStreamDestroy(stream));
                stream = nullptr;
            }
            if (event) {
                LOG_CUDA_ERROR(cudaEventDestroy(event));
                event = nullptr;
            }
        }

        cudaStream_t stream = nullptr;
        cudaEvent_t event = nullptr;
        std::set<int> thread_ids;

        std::unique_ptr<std::barrier<CompletionFunc>> barrier;
        std::mutex mutex_;
        std::set<cudaStream_t> user_streams;
        std::vector<std::pair<int, nvimgcodecProcessingStatus_t>> processed_samples;
    };

    // CRTP pattern
    template <typename Impl, typename Factory, typename Processor>
    class ImageGenericCodec {
        // Completion function for barrier - triggers completePostSync when all threads arrive
        struct PostSyncCompletionFunction {
            ImageGenericCodec* codec_ptr;
            int stream_idx;
            void operator()() noexcept {
                // completePostSync calls into ProcessingResultsPromise::set, which can
                // throw std::runtime_error.  This functor is declared noexcept and is
                // invoked from a thread-pool barrier callback, so an escaping
                // exception would call std::terminate.
                try {
                    codec_ptr->completePostSync(stream_idx);
                } catch (const std::exception& e) {
                    if (codec_ptr && codec_ptr->logger_)
                        NVIMGCODEC_LOG_ERROR(codec_ptr->logger_,
                                             "completePostSync threw: " << e.what());
                } catch (...) {
                    if (codec_ptr && codec_ptr->logger_)
                        NVIMGCODEC_LOG_ERROR(codec_ptr->logger_,
                                             "completePostSync threw an unknown exception");
                }
            }
        };

        // RAII guard to ensure postSync is always called, even on exception
        struct PostSyncGuard {
            ImageGenericCodec* codec;
            int tid;

            PostSyncGuard(ImageGenericCodec* c, int thread_id) : codec(c), tid(thread_id) {}
            ~PostSyncGuard() {
                if (!codec)
                    return;
                // postSync's call chain reaches ProcessingResultsPromise::set, which
                // can throw std::runtime_error; a destructor must not propagate
                // exceptions.
                try {
                    codec->postSync(tid);
                } catch (const std::exception& e) {
                    if (codec->logger_)
                        NVIMGCODEC_LOG_ERROR(codec->logger_,
                                             "postSync threw during cleanup: " << e.what());
                } catch (...) {
                    if (codec->logger_)
                        NVIMGCODEC_LOG_ERROR(codec->logger_,
                                             "postSync threw an unknown exception during cleanup");
                }
            }

            PostSyncGuard(const PostSyncGuard&) = delete;
            PostSyncGuard& operator=(const PostSyncGuard&) = delete;
        };

    public:
        struct ProcessorEntry {
            const Factory* factory_;
            std::unique_ptr<Processor> instance_;
            std::string id_;
            nvimgcodecBackendKind_t backend_kind_;
            nvimgcodecBackendParams_t backend_params_;
            size_t sample_count_hint_ = 0;
            size_t assigned_batch_size_ = 0;
            std::atomic<size_t> elligible_samples_{0};
            size_t sample_count_ = 0;
            ProcessorEntry* fallback_ = nullptr;
            bool has_batched_api = false;

            ProcessorEntry() = default;
            ProcessorEntry(const ProcessorEntry&) = delete;
            ProcessorEntry& operator=(const ProcessorEntry&) = delete;
            ProcessorEntry(ProcessorEntry&& o) noexcept
                : factory_(o.factory_), instance_(std::move(o.instance_)), id_(std::move(o.id_)), backend_kind_(o.backend_kind_), backend_params_(o.backend_params_), sample_count_hint_(o.sample_count_hint_), assigned_batch_size_(o.assigned_batch_size_), elligible_samples_(o.elligible_samples_.load(std::memory_order_relaxed)), sample_count_(o.sample_count_), fallback_(o.fallback_), has_batched_api(o.has_batched_api) {}
            ProcessorEntry& operator=(ProcessorEntry&& o) noexcept {
                factory_ = o.factory_;
                instance_ = std::move(o.instance_);
                id_ = std::move(o.id_);
                backend_kind_ = o.backend_kind_;
                backend_params_ = o.backend_params_;
                sample_count_hint_ = o.sample_count_hint_;
                assigned_batch_size_ = o.assigned_batch_size_;
                elligible_samples_.store(o.elligible_samples_.load(std::memory_order_relaxed), std::memory_order_relaxed);
                sample_count_ = o.sample_count_;
                fallback_ = o.fallback_;
                has_batched_api = o.has_batched_api;
                return *this;
            }
        };

        explicit ImageGenericCodec(ILogger* logger,
                                   ICodecRegistry* codec_registry,
                                   const nvimgcodecExecutionParams_t* exec_params,
                                   const char* options = nullptr)
            : logger_(logger), codec_registry_(codec_registry), exec_params_(*exec_params), backends_(exec_params->num_backends), options_(options ? options : ""), executor_(std::move(GetExecutor(exec_params, logger))), num_threads_(executor_->getExecutorDesc()->getNumThreads(executor_->getExecutorDesc()->instance)), num_streams_(0), setup_barrier_(num_threads_ + 1, SetupCompletionFunction{this}) {

            if (exec_params_.device_id == NVIMGCODEC_DEVICE_CURRENT) {
                CHECK_CUDA(cudaGetDevice(&exec_params_.device_id));
            }

            DeviceGuard device_guard(exec_params_.device_id);

            auto backend = exec_params->backends;
            for (int i = 0; i < exec_params->num_backends; ++i) {
                backends_[i] = *backend;
                ++backend;
            }
            exec_params_.backends = backends_.data();
            exec_params_.executor = executor_->getExecutorDesc();

            // Deciding number of CUDA streams:
            // Environment variable takes precedence, otherwise default value is the same as num_threads_
            num_streams_ = num_threads_;
            if (const char* env_val = std::getenv("NVIMGCODEC_DEFAULT_NUM_CUDA_STREAMS")) {
                char* endptr = nullptr;
                long val = std::strtol(env_val, &endptr, 10);
                if (endptr != env_val) { // Only use if a valid number was parsed
                    if (val > 0 && val <= static_cast<long>(num_threads_)) {
                        num_streams_ = static_cast<size_t>(val);
                    }
                }
            }

            std::optional<size_t> opt_num_streams = std::nullopt;
            auto parseOptions = [&opt_num_streams](const char* options) {
                std::istringstream iss(options ? options : "");
                std::string token;
                while (std::getline(iss, token, ' ')) {
                    // Look for a token matching ":<option>=<value>" (global options only)
                    if (!token.starts_with(':'))
                        continue;
                    auto equal = token.find('=');
                    if (equal == std::string::npos)
                        continue;
                    std::string option = token.substr(1, equal - 1); // skip the ':'
                    std::string value_str = token.substr(equal + 1);

                    std::istringstream value(value_str);
                    if (option == "num_cuda_streams") {
                        size_t temp;
                        if (value >> temp) {
                            opt_num_streams = temp;
                        }
                    }
                }
            };
            parseOptions(options);

            if (opt_num_streams.has_value()) {
                if (opt_num_streams.value() > 0 && opt_num_streams.value() <= num_threads_) {
                    NVIMGCODEC_LOG_INFO(logger_, "Using " << opt_num_streams.value() << " CUDA streams (from options)");
                    num_streams_ = opt_num_streams.value();
                } else {
                    NVIMGCODEC_LOG_WARNING(
                        logger_, "Invalid value for num_cuda_streams: " << opt_num_streams.value()
                                                                        << " (should be between 1 and num_threads(" << num_threads_ << ")). Using default of " << num_streams_);
                }
            } else {
                NVIMGCODEC_LOG_INFO(logger_, "Using default of " << num_streams_ << " CUDA streams");
            }

            // Worker threads round-robin across num_streams_ streams; extra thread gets dedicated stream
            per_thread_.reserve(num_threads_ + 1);
            // Assign threads to streams using a temporary array to avoid duplicate logic
            std::vector<std::set<int>> stream_to_thread_ids(num_streams_ + 1);
            for (size_t i = 0; i < num_threads_ + 1; i++) {
                int stream_idx = (i == num_threads_) ? num_streams_ : static_cast<int>(i % num_streams_);
                per_thread_.emplace_back(exec_params_.device_id, stream_idx, exec_params_.pinned_allocator, exec_params_.device_allocator);
                stream_to_thread_ids[stream_idx].insert(static_cast<int>(i));
            }

            for (size_t i = 0; i < num_streams_ + 1; i++) {
                PostSyncCompletionFunction completion{this, static_cast<int>(i)};
                per_stream_.emplace_back(exec_params_.device_id, std::move(stream_to_thread_ids[i]), completion);
            }

            size_t total_num_processors = 0;
            for (size_t codec_idx = 0; codec_idx < codec_registry_->getCodecsCount(); codec_idx++) {
                auto* codec = codec_registry_->getCodecByIndex(codec_idx);
                total_num_processors += Impl::getProcessorsNum(codec);
            }
            processors_.resize(total_num_processors);

            size_t processor_count = 0;
            NVIMGCODEC_LOG_TRACE(logger_, "initializing " << codec_registry_->getCodecsCount() << " codecs");
            for (size_t codec_idx = 0; codec_idx < codec_registry_->getCodecsCount(); codec_idx++) {
                auto* codec = codec_registry_->getCodecByIndex(codec_idx);
                NVIMGCODEC_LOG_TRACE(logger_, "initializing " << codec->name());

                ProcessorEntry*& first_processor = codec_to_first_processor_[codec];
                size_t num_processors = Impl::getProcessorsNum(codec);
                NVIMGCODEC_LOG_TRACE(logger_, "initializing " << num_processors << " processors for " << codec->name());
                for (size_t i = 0; i < num_processors; i++) {
                    auto* factory = Impl::getFactory(codec, i);
                    if (!factory) {
                        continue;
                    }
                    NVIMGCODEC_LOG_TRACE(logger_, "initializing " << Impl::getId(factory));

                    auto backend_kind = factory->getBackendKind();
                    auto backend_params = nvimgcodecBackendParams_t{NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t),
                                                                    nullptr, 1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED};
                    auto backend = exec_params_.backends;

                    bool backend_allowed = true;
                    if (exec_params_.num_backends != 0) {
                        backend_allowed = false;
                        for (auto b = 0; b < exec_params_.num_backends; ++b) {
                            if (backend[b].kind == backend_kind) {
                                backend_allowed = true;
                                backend_params = backend[b].params;
                                break;
                            }
                        }
                    }
                    if (!backend_allowed) {
                        NVIMGCODEC_LOG_DEBUG(logger_, "backend " << backend_kind << " not allowed. Skipping " << Impl::getId(factory));
                        continue;
                    }

                    assert(processor_count < total_num_processors);
                    auto& processor = processors_[processor_count++];
                    processor.backend_kind_ = backend_kind;
                    processor.backend_params_ = backend_params;
                    processor.id_ = Impl::getId(factory);
                    processor.factory_ = factory;
                    processor.sample_count_ = 0;
                    processor.elligible_samples_.store(0, std::memory_order_relaxed);

                    if (exec_params->pre_init) {
                        NVIMGCODEC_LOG_INFO(logger_, "create " << processor.id_ << " load_hint " << processor.backend_params_.load_hint
                                                               << " load_hint_policy " << processor.backend_params_.load_hint_policy);
                        processor.instance_ = Impl::createInstance(processor.factory_, &exec_params_, options_.c_str());
                        processor.has_batched_api = Impl::hasBatchedAPI(processor.instance_.get());
                        if (processor.has_batched_api)
                            batched_processors_.insert(&processor);
                    }
                    if (!first_processor) {
                        first_processor = &processor;
                    } else {
                        auto* curr_processor = first_processor;
                        while (curr_processor->fallback_) {
                            curr_processor = curr_processor->fallback_;
                        }
                        curr_processor->fallback_ = &processor;
                    }

                    NVIMGCODEC_LOG_TRACE(logger_, "initializing " << Impl::getId(factory) << " DONE");
                }
            }
            assert(processor_count <= total_num_processors);
            if (total_num_processors > 0 && processor_count == 0) {
                throw Exception(ARCH_MISMATCH, "Requested backends not among available processors.");
            }
            processors_.resize(processor_count); // actual number of processors instantiated
            for (auto& processor : processors_) {
                if (processor.fallback_ == nullptr &&
                    (processor.backend_params_.load_hint < 1.0f ||
                     processor.backend_params_.load_hint_policy == NVIMGCODEC_LOAD_HINT_POLICY_ADAPTIVE_MINIMIZE_IDLE_TIME)) {
                    NVIMGCODEC_LOG_INFO(
                        logger_, processor.id_ << ": adjusting load hint to 1.0f (non-adaptive), as there's no fallback processor available");
                    processor.backend_params_.load_hint = 1.0f;
                    processor.backend_params_.load_hint_policy = NVIMGCODEC_LOAD_HINT_POLICY_FIXED;
                }
            }
        }

        virtual ~ImageGenericCodec() {
            // drain executor first
            executor_.reset();

            std::optional<DeviceGuard> device_guard;
            try {
                device_guard.emplace(exec_params_.device_id);
            } catch (const std::exception& e) {
                NVIMGCODEC_LOG_ERROR(logger_, "Could not set CUDA device during " << Impl::process_name() << " teardown: " << e.what());
            }

            // deallocate temp buffers before we destroy thread resources
            samples_.clear();
            // destroy processors
            batched_processors_.clear();
            processors_.clear();
            // per thread and per stream resources
            per_thread_.clear();
            per_stream_.clear();
        }

    protected:
        ILogger* logger_;
        ICodecRegistry* codec_registry_;
        nvimgcodecExecutionParams_t exec_params_;
        std::vector<nvimgcodecBackend_t> backends_;
        std::string options_;
        std::unique_ptr<IExecutor> executor_;
        size_t num_threads_;
        size_t num_streams_;

        struct SetupCompletionFunction {
            ImageGenericCodec* this_ptr;
            void operator()() noexcept { this_ptr->completeSetup(); }
        };
        std::barrier<SetupCompletionFunction> setup_barrier_;

        std::vector<ICodeStream*> code_streams_;
        std::vector<IImage*> images_;

        int num_samples_ = 0;
        std::vector<SampleEntry<ProcessorEntry>> samples_;
        std::atomic<int> atomic_idx_{0};

        float adaptive_delta_ = 0.1f;
        std::vector<ProcessorEntry> processors_;
        std::set<ProcessorEntry*> batched_processors_;
        std::unordered_map<const ICodec*, ProcessorEntry*> codec_to_first_processor_;

        std::vector<const nvimgcodecCodeStreamDesc_t*> batched_code_stream_descs_;
        std::vector<const nvimgcodecImageDesc_t*> batched_image_descs_;
        std::vector<SampleEntry<ProcessorEntry>*> batched_processed_;

        std::vector<PerThread> per_thread_;
        std::vector<PerStream<PostSyncCompletionFunction>> per_stream_;

        std::vector<int> curr_order_;
        std::shared_ptr<ProcessingResultsPromise> curr_promise_;

        std::chrono::time_point<std::chrono::high_resolution_clock> start_iteration_time_ =
            std::chrono::time_point<std::chrono::high_resolution_clock>::min();
        std::chrono::time_point<std::chrono::high_resolution_clock> last_activity_main_thread_ =
            std::chrono::time_point<std::chrono::high_resolution_clock>::min();
        std::chrono::time_point<std::chrono::high_resolution_clock> last_activity_ =
            std::chrono::time_point<std::chrono::high_resolution_clock>::min();

        void preSync(SampleEntry<ProcessorEntry>& sample, int tid) {
            if (exec_params_.device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
                return;
            auto& t = per_thread_[tid];
            auto& s = per_stream_[t.stream_idx];
            sample.image_info.cuda_stream = s.stream;
            auto user_cuda_stream = sample.orig_image_info.cuda_stream;
            if (t.user_streams.find(user_cuda_stream) == t.user_streams.end()) {
                if (!exec_params_.skip_pre_sync) {
                    nvtx3::scoped_range marker{"sync"};
                    NVIMGCODEC_LOG_TRACE(logger_, "cudaEventRecord(" << t.event << ", " << user_cuda_stream << ")");
                    CHECK_CUDA(cudaEventRecord(t.event, user_cuda_stream));
                    NVIMGCODEC_LOG_TRACE(logger_, "cudaStreamWaitEvent(" << s.stream << ", " << t.event << ")");
                    CHECK_CUDA(cudaStreamWaitEvent(s.stream, t.event));
                }
                t.user_streams.insert(user_cuda_stream);
            }
        }

        // Collect per-thread results into shared stream storage; last thread triggers completePostSync
        void postSync(int tid) {
            NVIMGCODEC_LOG_DEBUG(logger_, tid << ": postSync");
            auto& t = per_thread_[tid];
            auto& s = per_stream_[t.stream_idx];
            {
                std::lock_guard<std::mutex> lock(s.mutex_);
                s.user_streams.insert(t.user_streams.begin(), t.user_streams.end());
                s.processed_samples.insert(s.processed_samples.end(), t.processed_samples.begin(), t.processed_samples.end());
            }
            t.processed_samples.clear();
            t.user_streams.clear();

            if (s.barrier) {
                // Arrive at barrier; the last thread triggers completePostSync automatically
                (void)s.barrier->arrive();
            } else {
                // (1 thread per stream) No barrier, so we need to trigger completePostSync manually
                completePostSync(t.stream_idx);
            }
            NVIMGCODEC_LOG_DEBUG(logger_, tid << ": postSync DONE");
        }

        // Barrier completion: synchronize processing stream with user streams once per stream
        // If CUDA sync fails, mark all samples from this stream as failed
        void completePostSync(int stream_idx) {
            nvtx3::scoped_range marker{"sync"};
            auto& s = per_stream_[stream_idx];

            std::set<cudaStream_t> user_streams_copy;
            std::vector<std::pair<int, nvimgcodecProcessingStatus_t>> processed_samples_copy;
            std::swap(user_streams_copy, s.user_streams);
            std::swap(processed_samples_copy, s.processed_samples);
            try {
                if (!user_streams_copy.empty()) {
                    // Record the current state of our processing stream once
                    NVIMGCODEC_LOG_DEBUG(logger_, "stream " << stream_idx << ": cudaEventRecord(" << s.event << ", " << s.stream << ")");
                    CHECK_CUDA(cudaEventRecord(s.event, s.stream));

                    // Make all user streams wait for this event
                    for (auto user_stream : user_streams_copy) {
                        NVIMGCODEC_LOG_DEBUG(logger_, "stream " << stream_idx << ": cudaStreamWaitEvent(" << user_stream << ", " << s.event << ")");
                        CHECK_CUDA(cudaStreamWaitEvent(user_stream, s.event));
                    }
                }
            } catch (const std::exception& e) {
                NVIMGCODEC_LOG_ERROR(logger_, "Exception during completePostSync: " << e.what());
                for (auto& [sample_idx, status] : processed_samples_copy) {
                    if (status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                        status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
                    }
                }
            }
            curr_promise_->set(processed_samples_copy);
        }

        void initSample(int sample_idx) {
            auto& sample = samples_[sample_idx];
            sample.image_desc.instance = &sample;
            sample.should_copy = false;
            sample.sample_idx = sample_idx;
            sample.code_stream = code_streams_[sample_idx];
            sample.orig_image = images_[sample_idx];
            sample.orig_image_info = nvimgcodecImageInfo_t{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            sample.orig_image->getImageInfo(&sample.orig_image_info);
            sample.codec = sample.code_stream->getCodec();
            sample.image_info = sample.orig_image_info;
        }

        void setupSample(int sample_idx, int tid) {
            NVIMGCODEC_LOG_TRACE(logger_, tid << ": setupSample #" << sample_idx);
            auto& t = per_thread_[tid];

            // If we know the processors were already initialized, we can init the sample in parallel here
            // otherwise it needs to be done earlier
            if (exec_params_.pre_init)
                initSample(sample_idx);

            auto& sample = samples_[sample_idx];

            if (!sample.codec) {
                sample.processor = nullptr;
                sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                NVIMGCODEC_LOG_DEBUG(logger_, tid << ": failure #" << sample_idx << " : NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED");
                curr_promise_->set(sample_idx, ProcessingResult::failure(sample.status));
                return;
            }

            auto it = codec_to_first_processor_.find(sample.codec);
            assert(it != codec_to_first_processor_.end());
            sample.processor = it->second;
            sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            sample.fallback_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;

            if (!sample.processor) {
                sample.processor = nullptr;
                sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                NVIMGCODEC_LOG_DEBUG(logger_, tid << ": failure #" << sample_idx << " : NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED");
                curr_promise_->set(sample_idx, ProcessingResult::failure(sample.status));
                return;
            }

            for (; sample.processor; sample.processor = sample.processor->fallback_) {
                sample.fallback_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
                // if we have enough samples for this processor, we can skip it if the next processor is also elligible
                if (t.elligible_samples_snapshot_[sample.processor] >= sample.processor->sample_count_hint_ && sample.processor->fallback_ &&
                    static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor->fallback_, tid) ==
                        NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                    sample.processor = sample.processor->fallback_;
                    sample.status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
                    break;
                } else {
                    sample.status = static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor, tid);
                    if (sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                        sample.processor->elligible_samples_.fetch_add(1, std::memory_order_relaxed);
                        // if there's a chance we'd like to fallback, query the next decoder just in case.
                        if (static_cast<int>(sample.processor->sample_count_hint_) < num_samples_ &&
                            sample_idx >= static_cast<int>(sample.processor->sample_count_hint_) && sample.processor->fallback_) {
                            sample.fallback_status = static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor->fallback_, tid);
                        }
                        break;
                    }
                }
            }
            if (sample.status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                curr_promise_->set(sample_idx, ProcessingResult::failure(sample.status));
            }
        }

        void cooperativeSetup(int tid) {
            nvtx3::scoped_range marker{"cooperativeSetup"};
            int ordered_sample_idx;
            auto& t = per_thread_[tid];
            t.user_streams.clear();
            t.processed_samples.clear();
            size_t num_processors = processors_.size();
            do {
                // it is important to get the snapshot before fetching a sample idx for processing
                // so that we process enough samples to fill each processor (may check a few more)
                for (size_t i = 0; i < num_processors; i++) {
                    auto& processor = processors_[i];
                    t.elligible_samples_snapshot_[&processor] = processor.elligible_samples_.load(std::memory_order_relaxed);
                }
                ordered_sample_idx = atomic_idx_.fetch_add(1, std::memory_order_relaxed);
                if (ordered_sample_idx >= num_samples_) {
                    break;
                }
                int sample_idx = curr_order_[ordered_sample_idx];
                setupSample(sample_idx, tid);
            } while (true);
        }

        void completeSetup() noexcept {
            nvtx3::scoped_range marker{"completeSetup"};
            for (int sample_idx : curr_order_) {
                auto& sample = samples_[sample_idx];
                if (sample.status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                    NVIMGCODEC_LOG_INFO(logger_, "Sample #" << sample.sample_idx << " can't be processed");
                    continue;
                }
                auto* first_processor = sample.processor;
                while (sample.processor->sample_count_ >= sample.processor->sample_count_hint_ && sample.processor->fallback_) {
                    if (sample.fallback_status == NVIMGCODEC_PROCESSING_STATUS_UNKNOWN) {
                        sample.fallback_status = static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor->fallback_, 0);
                    }
                    assert(sample.fallback_status != NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
                    if (sample.fallback_status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                        NVIMGCODEC_LOG_DEBUG(logger_, "Sample #" << sample.sample_idx << ", moving to next processor: " << sample.processor->fallback_->id_);
                        sample.processor = sample.processor->fallback_;
                        sample.status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
                        sample.fallback_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
                    } else {
                        NVIMGCODEC_LOG_DEBUG(logger_, "Sample #" << sample.sample_idx << ", using first processor: " << first_processor->id_);
                        sample.processor = first_processor;
                        break; // stop search
                    }
                }
                assert(sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
                NVIMGCODEC_LOG_DEBUG(logger_, "Sample #" << sample.sample_idx << " assigned to " << sample.processor->id_);
                sample.processor->sample_count_++;
            }
            atomic_idx_.store(0, std::memory_order_relaxed);
        }

        ProcessorEntry* initProcessorsAndGetFirstForCodec(const ICodec* codec) {
            auto it = codec_to_first_processor_.find(codec);
            assert(it != codec_to_first_processor_.end());
            auto* firstProcessor = it->second;
            auto* processor = firstProcessor;
            while (processor) {
                if (processor->instance_)
                    break;

                NVIMGCODEC_LOG_INFO(logger_, "create " << processor->id_ << " load_hint " << processor->backend_params_.load_hint
                                                       << " load_hint_policy " << processor->backend_params_.load_hint_policy);
                processor->instance_ = Impl::createInstance(processor->factory_, &exec_params_, options_.c_str());
                processor->has_batched_api = Impl::hasBatchedAPI(processor->instance_.get());
                if (processor->has_batched_api)
                    batched_processors_.insert(processor);
                processor = processor->fallback_;
            }
            return firstProcessor;
        }

        void initState(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
                       const void* params) {
            int N = images.size();
            assert(static_cast<int>(code_streams.size()) == N);

            curr_promise_ = std::make_shared<ProcessingResultsPromise>(N);
            code_streams_ = code_streams;
            images_ = images;

            num_samples_ = N;
            while (static_cast<int>(samples_.size()) < N)
                samples_.emplace_back(&exec_params_);
            while (static_cast<int>(samples_.size()) > N)
                samples_.pop_back();

            for (auto& sample : samples_)
                sample.params = params;

            assert(code_streams.size() == static_cast<size_t>(num_samples_));
            curr_order_.resize(num_samples_);
            std::iota(curr_order_.begin(), curr_order_.end(), 0);

            if (!exec_params_.pre_init) {
                for (int sample_idx : curr_order_) {
                    initSample(sample_idx);
                    auto& sample = samples_[sample_idx];
                    if (!sample.codec)
                        continue;
                    sample.processor = initProcessorsAndGetFirstForCodec(sample.codec);
                }
            }

            for (auto& processor : processors_) {
                processor.sample_count_hint_ = num_samples_;
                processor.sample_count_ = 0;
                processor.elligible_samples_.store(0, std::memory_order_relaxed);
            }
        }

        void canProcess(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
                        nvimgcodecProcessingStatus_t* processing_status, int force_format, const void* params) {
            DeviceGuard device_guard(exec_params_.device_id);
            initState(code_streams, images, params);
            for (int sample_idx : curr_order_) {
                auto& sample = samples_[sample_idx];
                processing_status[sample_idx] = sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                if (!sample.codec) {
                    continue;
                }
                while (sample.processor) {
                    sample.status = static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor, 0);
                    bool can_decode_with_other_format_or_params = (static_cast<unsigned int>(sample.status) & 0b11) == 0b01;
                    // A corrupted codestream cannot become valid by trying another backend.
                    if (sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS ||
                        sample.status == NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED ||
                        (!force_format && can_decode_with_other_format_or_params)) {
                        break;
                    }
                    sample.processor = sample.processor->fallback_;
                }
                processing_status[sample_idx] = sample.status;
            }
        }

        ProcessorEntry* nextParallelProcessor(ProcessorEntry* processor) {
            if (processor && processor->has_batched_api)
                return nextParallelProcessor(processor->fallback_);
            else
                return processor;
        }

        bool shouldProcessSingleImage(int sample_idx) {
            auto& sample = samples_[sample_idx];
            if (curr_promise_->isSet(sample_idx))
                return false;
            else if (sample.processor->has_batched_api)
                return false;
            else
                return true;
        }

        void processSample(int sample_idx, int tid) {
            auto& t = per_thread_[tid];
            NVIMGCODEC_LOG_TRACE(logger_, tid << ": processSample");

            auto& sample = samples_[sample_idx];
            auto sample_idx_str = std::to_string(sample_idx);
            nvtx3::scoped_range marker{Impl::process_name() + " #" + sample_idx_str};

            sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;

            preSync(sample, tid);

            bool failed = false;
            for (auto*& processor = sample.processor; processor; processor = nextParallelProcessor(processor->fallback_)) {
                if (failed) {
                    NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << processor->id_ << " canProcess #" << sample.sample_idx);
                    assert(processor->instance_);

                    assert(!processor->has_batched_api);
                    sample.status = static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor, tid);
                    failed = (sample.status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
                    if (failed)
                        continue;
                }

                nvtx3::scoped_range marker{processor->id_ + " process #" + sample_idx_str};
                NVIMGCODEC_LOG_INFO(logger_, tid << ": " << processor->id_ << " process #" << sample_idx);

                assert(!curr_promise_->isSet(sample_idx));
                sample.status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
                failed = !static_cast<Impl*>(this)->processImpl(sample, tid);
                if (!failed)
                    break;
            }

            if (sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << sample.processor->id_ << " set success #" << sample_idx << " done");
            } else {
                assert(sample.status != NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
                NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << (sample.processor ? sample.processor->id_ : "") << " set failure #" << sample_idx
                                                  << " status=" << sample.status);
            }
            t.processed_samples.emplace_back(sample_idx, sample.status);
            NVIMGCODEC_LOG_DEBUG(logger_, tid << ": processed_samples size=" << t.processed_samples.size());

            NVIMGCODEC_LOG_TRACE(logger_, tid << ": processSample DONE");
        }

        ProcessingResultsPromise::FutureImpl process(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
                                                     const void* params) {
            NVIMGCODEC_LOG_INFO(logger_, Impl::process_name() << " num_samples=" << code_streams.size());
            DeviceGuard device_guard(exec_params_.device_id);
            auto executor = executor_->getExecutorDesc();

            NVIMGCODEC_LOG_TRACE(logger_, "waiting for previous tasks to finish");
            executor->wait(executor->instance, exec_params_.device_id);
            NVIMGCODEC_LOG_TRACE(logger_, "previous tasks finished");

            assert(static_cast<int>(per_thread_.size()) >= executor->getNumThreads(executor->instance));
            std::chrono::milliseconds last_iter_duration(0), last_iter_duration_main(0);
            if (start_iteration_time_ > std::chrono::time_point<std::chrono::high_resolution_clock>::min()) {
                last_iter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(last_activity_ - start_iteration_time_);
                last_iter_duration_main =
                    std::chrono::duration_cast<std::chrono::milliseconds>(last_activity_main_thread_ - start_iteration_time_);
                NVIMGCODEC_LOG_INFO(logger_, "Last iter time : " << last_iter_duration.count() << "ms, Last iter time main thread : "
                                                                 << last_iter_duration_main.count() << "ms");
            }
            start_iteration_time_ = std::chrono::high_resolution_clock::now();

            initState(code_streams, images, params);

            assert(curr_promise_);
            auto future = curr_promise_->getFuture();

            static_cast<Impl*>(this)->sortSamples();

            adjustBatchSizes(last_iter_duration_main, last_iter_duration);

            atomic_idx_.store(0, std::memory_order_relaxed);
            if (num_samples_ <= 1 || num_threads_ <= 1) {
                PostSyncGuard post_sync_guard(this, 0); // postSync on scope exit
                cooperativeSetup(0);
                completeSetup();
                for (int sample_idx : curr_order_) {
                    if (shouldProcessSingleImage(sample_idx)) {
                        processSample(sample_idx, 0);
                    }
                }
            } else {
                auto setup_task = [](int tid, int sample_idx, void* context) -> void {
                    auto* this_ptr = reinterpret_cast<ImageGenericCodec<Impl, Factory, Processor>*>(context);
                    DeviceGuard device_guard(this_ptr->exec_params_.device_id);
                    PostSyncGuard post_sync_guard(this_ptr, tid); // postSync on scope exit
                    this_ptr->cooperativeSetup(tid);
                    this_ptr->setup_barrier_.arrive_and_wait();

                    int ordered_sample_idx;
                    while ((ordered_sample_idx = this_ptr->atomic_idx_.fetch_add(1, std::memory_order_relaxed)) < this_ptr->num_samples_) {
                        int sample_idx = this_ptr->curr_order_[ordered_sample_idx];
                        if (this_ptr->shouldProcessSingleImage(sample_idx))
                            this_ptr->processSample(sample_idx, tid);
                    }
                };
                for (size_t task_idx = 0; task_idx < num_threads_; task_idx++) {
                    executor->schedule(executor->instance, exec_params_.device_id, task_idx, this, setup_task);
                }
                executor->run(executor->instance, exec_params_.device_id);
                cooperativeSetup(num_threads_); // one past the last thread idx
                setup_barrier_.arrive_and_wait();
                // completeSetup() is automatically executed by the last thread to arrive at the barrier
            }

            batchProcess();
            last_activity_main_thread_ = std::chrono::high_resolution_clock::now();
            last_activity_ = std::chrono::high_resolution_clock::now();
            return future;
        }

        void batchProcess() {
            const int tid = num_threads_; // the last entry in per_thread_ is the main thread
            PostSyncGuard post_sync_guard(this, tid);
            NVIMGCODEC_LOG_TRACE(logger_, "batchProcess");
            auto executor = executor_->getExecutorDesc();
            // Batch processors run on the main thread
            NVIMGCODEC_LOG_TRACE(logger_, "batched_processors_ size=" << batched_processors_.size());
            for (auto* processor_ptr : batched_processors_) {
                auto& processor = *processor_ptr;
                assert(processor.instance_ && processor.has_batched_api);
                NVIMGCODEC_LOG_DEBUG(logger_, processor.id_ + " start");
                batched_code_stream_descs_.clear();
                batched_image_descs_.clear();
                batched_processed_.clear();
                for (int sample_idx : curr_order_) {
                    auto& sample = samples_[sample_idx];
                    assert(sample_idx == sample.sample_idx);
                    if (sample.processor == processor_ptr && sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                        batched_processed_.push_back(&sample);
                        batched_code_stream_descs_.push_back(sample.code_stream->getCodeStreamDesc());
                        batched_image_descs_.push_back(sample.getImageDesc());
                        preSync(sample, tid);
                    }
                }

                if (batched_code_stream_descs_.empty()) {
                    NVIMGCODEC_LOG_DEBUG(logger_, processor.id_ + ": no samples for this processor. Skip");
                    continue;
                }

                auto ret = static_cast<Impl*>(this)->processBatchImpl(processor);
                size_t fallback_count = 0;
                for (auto* sample : batched_processed_) {
                    const bool decoded_ok = ret && sample->status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
                    // Next single-sample (non-batched) processor in the fallback chain. Batch-only
                    // decoders (decode == nullptr, e.g. the nvjpeg lossless decoder) implement only
                    // decodeBatch and must never be dispatched to the single-sample
                    // processSample()/decode() path, or that NULL decode callback would be called.
                    auto* next_processor = decoded_ok ? nullptr : nextParallelProcessor(sample->processor->fallback_);
                    if (decoded_ok) {
                        NVIMGCODEC_LOG_DEBUG(logger_, processor.id_ << " set success #" << sample->sample_idx);
                        curr_promise_->set(sample->sample_idx, ProcessingResult::success());
                    } else if (next_processor == nullptr) {
                        NVIMGCODEC_LOG_INFO(logger_, processor.id_ + " set failure #" << sample->sample_idx << " status=" << sample->status);
                        curr_promise_->set(sample->sample_idx, ProcessingResult::failure(sample->status));
                    } else {
                        NVIMGCODEC_LOG_INFO(
                            logger_, processor.id_ << " failed #" << sample->sample_idx << ". Trying next: " << next_processor->id_);
                        sample->processor = next_processor;
                        auto parallel_process_task = [](int tid, int sample_idx, void* context) -> void {
                            auto* this_ptr = reinterpret_cast<ImageGenericCodec<Impl, Factory, Processor>*>(context);
                            DeviceGuard device_guard(this_ptr->exec_params_.device_id);
                            this_ptr->setupSample(sample_idx, tid);
                            assert(this_ptr->shouldProcessSingleImage(sample_idx));
                            this_ptr->processSample(sample_idx, tid);
                            // We don't have a way to know if this thread has already executed postSync, so we need to add sync with user stream here.
                            auto& t = this_ptr->per_thread_[tid];
                            auto& s = this_ptr->per_stream_[t.stream_idx];
                            auto& sample = this_ptr->samples_[sample_idx];
                            assert(sample_idx == sample.sample_idx);
                            auto user_stream = sample.orig_image_info.cuda_stream;
                            NVIMGCODEC_LOG_TRACE(this_ptr->logger_, "cudaEventRecord(" << t.event << ", " << s.stream << ")");
                            CHECK_CUDA(cudaEventRecord(t.event, s.stream));
                            NVIMGCODEC_LOG_TRACE(this_ptr->logger_, "cudaStreamWaitEvent(" << user_stream << ", " << t.event << ")");
                            CHECK_CUDA(cudaStreamWaitEvent(user_stream, t.event));
                            NVIMGCODEC_LOG_INFO(this_ptr->logger_, "curr_promise_->set(" << sample_idx << ", " << sample.status << ")");
                            this_ptr->curr_promise_->set(sample_idx, ProcessingResult{sample.status, nullptr});
                        };
                        executor->schedule(executor->instance, exec_params_.device_id, sample->sample_idx, this, parallel_process_task);
                        fallback_count++;
                    }
                }
                if (fallback_count > 0) {
                    executor->run(executor->instance, exec_params_.device_id);
                }

                NVIMGCODEC_LOG_DEBUG(logger_, processor.id_ + " DONE");
            }
            NVIMGCODEC_LOG_TRACE(logger_, "batchProcess DONE");
        }

        static std::unique_ptr<IExecutor> GetExecutor(const nvimgcodecExecutionParams_t* exec_params, ILogger* logger) {
            if (exec_params->executor)
                return std::make_unique<UserExecutor>(exec_params->executor);
            else
                return std::make_unique<DefaultExecutor>(logger, exec_params->max_num_cpu_threads);
        }

        void adjustBatchSizes(std::chrono::milliseconds last_iter_duration_main, std::chrono::milliseconds last_iter_duration) {
            bool adaptive_load_update_done = false;
            float multiplier = 1.0f;
            auto adaptive_load_param_update = [&]() {
                if (adaptive_load_update_done)
                    return;
                const float decay_rate = 0.1f;
                const float target = 0.005f; // 0.5% increments at the end
                adaptive_delta_ = adaptive_delta_ + decay_rate * (target - adaptive_delta_);
                if ((last_iter_duration_main + std::chrono::milliseconds(2)) < last_iter_duration) {
                    multiplier = 1.0f;
                } else if ((last_iter_duration + std::chrono::milliseconds(2)) < last_iter_duration_main) {
                    multiplier = -1.0f;
                } else {
                    multiplier = 0.0f;
                }
            };

            auto selectBatchSize = [](float load_hint, int max_batch_size, int preferred_mini_batch) {
                if (load_hint == 0.0f)
                    return 0;
                auto batch_size = static_cast<int>(std::round(load_hint * max_batch_size));
                if (preferred_mini_batch > 0) {
                    int tail = batch_size % preferred_mini_batch;
                    if (tail > 0) {
                        batch_size = batch_size + preferred_mini_batch - tail;
                    }
                }
                return batch_size;
            };

            for (auto* processor : batched_processors_) {
                if (!processor->has_batched_api || !processor->fallback_) {
                    processor->sample_count_hint_ = num_samples_;
                } else if (processor->backend_params_.load_hint_policy == NVIMGCODEC_LOAD_HINT_POLICY_ADAPTIVE_MINIMIZE_IDLE_TIME) {
                    if (!adaptive_load_update_done) {
                        adaptive_load_param_update();
                        adaptive_load_update_done = true;
                    }
                    auto new_load_hint = std::clamp(processor->backend_params_.load_hint + multiplier * adaptive_delta_, 0.0f, 1.0f);
                    NVIMGCODEC_LOG_INFO(logger_,
                                        processor->id_ << " adjust load hint from " << processor->backend_params_.load_hint << " to " << new_load_hint);
                    processor->backend_params_.load_hint = new_load_hint;
                }

                if ((processor->instance_ != nullptr) && (processor->assigned_batch_size_ == 0)) {
                    int minibatch = Impl::getMiniBatchSize(processor->instance_.get());
                    processor->assigned_batch_size_ =
                        selectBatchSize(processor->backend_params_.load_hint, num_samples_, minibatch);
                    NVIMGCODEC_LOG_INFO(
                        logger_, processor->id_ << " selecting max batch size to " << processor->assigned_batch_size_ << " mini_bs=" << minibatch);
                } else {
                    NVIMGCODEC_LOG_INFO(
                        logger_, processor->id_ << " Using previous max batch size of " << processor->assigned_batch_size_);
                }
                processor->sample_count_hint_ = std::min<size_t>(processor->assigned_batch_size_, num_samples_);
            }
        }
    };

} // namespace nvimgcodec
