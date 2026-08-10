/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define NOMINMAX

#include "hw_decoder.h"
#include "nvjpeg_utils.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <nvimgcodec.h>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "errors_handling.h"
#include "log.h"
#include "type_convert.h"
#include <nvtx3/nvtx3.hpp>

#include "imgproc/convert_kernel_gpu.h"
#include "imgproc/device_buffer.h"
#include "imgproc/device_guard.h"
#include "imgproc/image_info_checks.h"
#include "imgproc/out_of_bound_roi_fill.h"
#include "imgproc/region_orientation.h"
#include "imgproc/type_utils.h"

#if WITH_DYNAMIC_NVJPEG_ENABLED
#include "dynlink/dynlink_nvjpeg.h"
#else
#define nvjpegIsSymbolAvailable(T) (true)
#endif

namespace nvjpeg {

    namespace {

        // Decode dispatch.  decodeBatch() picks DecodePath::Legacy or
        // DecodePath::BatchSingle via chooseDecodePath(batch_size, code_streams).
        // The BatchSingle branch uses decodeSingleImageImpl for batch_size == 1 and
        // decodeBatchSingle for larger batches.
        //   1. Single-image: one sample through the per-image 3-function pipeline.
        //   2. Batch-single: many samples through that same pipeline over a leased
        //      worker pool.
        //   3. Legacy batched API: nvjpegDecodeBatchedEx.
        // The per-image paths share decodeSingleHostStage.  The normal single-image
        // path can reuse one parsed stream from canDecode via a single-slot handoff.

        // State pool shared by DecoderImpl instances on the same (device, nvjpeg_flags).
        //   - One SharedContext per key, created on first use by a DecoderImpl
        //     (see getSharedHwContext).  Each DecoderImpl holds the shared_ptr; the
        //     registry holds only weak_ptrs, so the context (and its CUDA streams /
        //     events / nvJPEG handle / worker pool) is torn down when the last
        //     DecoderImpl on that key is destroyed.  Re-creating a Decoder afterwards
        //     rebuilds the context from scratch.
        //   - Callers borrow workers via acquireWorker*/acquireWorkers and return them
        //     via releaseWorker*/releaseWorkers.  Workers are reused across calls;
        //     their nvJPEG resources are released in ~SharedContext.
        struct SharedContext {
            // Per-worker scratch state for one page of the double-buffered decode
            // pipeline. These slots are permanently owned by DecodeWorker objects.
            struct DecodeSlot {
                nvjpegJpegStream_t nvjpeg_stream_ = nullptr;
                nvjpegDecodeParams_t decode_params_ = nullptr;
                nvjpegBufferPinned_t pinned_buffer_ = nullptr;
                const nvimgcodecImageDesc_t* image_ = nullptr;
                nvimgcodecImageInfo_t image_info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                nvimgcodecImageInfo_t cs_image_info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                nvimgcodecCodeStreamInfo_t codestream_info_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
                nvimgcodecSampleFormat_t sample_format_ = NVIMGCODEC_SAMPLEFORMAT_UNKNOWN;
                uint32_t roi_image_width_ = 0;
                uint32_t roi_image_height_ = 0;
                bool has_oob_roi_ = false;
            };

            // One entry in the shared worker pool.  A DecodeWorker holds all the nvJPEG
            // and CUDA objects needed to run the 3-function per-image decode pipeline
            // (nvjpegDecodeJpegHost / TransferToDevice / Device) for one image at a time.
            // Workers are reused across decode calls to amortize nvJPEG state allocation.
            struct DecodeWorker {
                explicit DecodeWorker(nvimgcodecDeviceAllocator_t* device_allocator = nullptr)
                    : helper_device_buffer_(device_allocator) {
                }

                // Permanent indices into SharedContext::decode_slots_ for the two
                // double-buffer pages.  The host stage alternates between them via
                // (current_slot_idx_+1)%2, allowing the CPU to parse image N+1 on
                // slot_ids_[1] while the GPU is still executing the device stage for
                // image N on slot_ids_[0].
                std::array<int, 2> slot_ids_{-1, -1};
                int current_slot_idx_ = 0; // index into slot_ids_ last handed to the host stage

                int advanceDecodeSlot() {
                    current_slot_idx_ = (current_slot_idx_ + 1) % static_cast<int>(slot_ids_.size());
                    return slot_ids_[current_slot_idx_];
                }

                int currentDecodeSlot() const {
                    return slot_ids_[current_slot_idx_];
                }
                // Recorded on the worker's stream after the device stage completes.
                // The caller's stream waits on this event to know when the decoded output
                // buffer is safe to read.  Direction: worker → caller.
                cudaEvent_t decode_done_event_ = nullptr;
                // Records the caller's CUDA stream position at the start of a decode call.
                // The worker's stream waits on this before doing any GPU writes so that
                // any prior GPU work the caller submitted to the same buffer has finished.
                // Direction: caller → worker.
                cudaEvent_t caller_ready_event_ = nullptr;
                cudaStream_t stream_ = nullptr;                 // private non-blocking CUDA stream for this worker
                nvjpegBufferDevice_t device_buffer_ = nullptr;  // device-side intermediate buffer owned by nvJPEG
                nvjpegJpegDecoder_t decoder_ = nullptr;         // alias to SharedContext::decoder_ (shared among workers)
                nvjpegJpegState_t state_ = nullptr;             // per-worker decode state (not thread-safe, hence one per worker)
                nvimgcodec::DeviceBuffer helper_device_buffer_; // temporary device buffer for I_YUV→P_YUV conversion
                bool in_use_ = false;                           // true → currently leased by an acquire call
            };

            // Constructs the shared context: creates the nvJPEG handle, queries HW
            // engine info, and probes for optional API symbols.  Worker objects are NOT
            // created here; they are added lazily by ensureWorkers() (called from
            // preallocateDecodeStates / acquireWorker* on the first decode).  The
            // shared nvJPEG decoder object is also lazy-created on the first
            // BatchSingle support probe or worker allocation.
            SharedContext(
                int device_id,
                const char* plugin_id,
                const nvimgcodecFrameworkDesc_t* framework,
                unsigned int nvjpeg_flags,
                const nvjpegDevAllocatorV2_t& device_allocator,
                const nvjpegPinnedAllocatorV2_t& pinned_allocator,
                const nvimgcodecDeviceAllocator_t* device_allocator_desc, // used by DeviceBuffer inside each worker
                size_t device_mem_padding,
                size_t pinned_mem_padding)
                : device_id_(device_id), plugin_id_(plugin_id), framework_(framework), device_allocator_(device_allocator), pinned_allocator_(pinned_allocator)
                  // Store a copy of the nvimgcodec allocator desc so DecodeWorker's
                  // DeviceBuffer can use it even after the original exec_params are gone.
                  ,
                  device_allocator_desc_(device_allocator_desc ? *device_allocator_desc : nvimgcodecDeviceAllocator_t{NVIMGCODEC_STRUCTURE_TYPE_DEVICE_ALLOCATOR, sizeof(nvimgcodecDeviceAllocator_t), nullptr, nullptr, nullptr, nullptr, 0}) {
                nvimgcodec::DeviceGuard device_guard(device_id_);
                // Use the V2 allocator API when either allocator is fully populated;
                // nvJPEG uses its default allocator for any nullptr side.
                bool has_device_allocator = device_allocator_.dev_malloc && device_allocator_.dev_free;
                bool has_pinned_allocator = pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free;
                if ((has_device_allocator || has_pinned_allocator) && nvjpegIsSymbolAvailable("nvjpegCreateExV2")) {
                    XM_CHECK_NVJPEG(nvjpegCreateExV2(
                        NVJPEG_BACKEND_HARDWARE,
                        has_device_allocator ? &device_allocator_ : nullptr,
                        has_pinned_allocator ? &pinned_allocator_ : nullptr,
                        nvjpeg_flags,
                        &handle_));
                } else {
                    XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, nvjpeg_flags, &handle_));
                }

                if (device_mem_padding != 0) {
                    XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(device_mem_padding, handle_));
                }
                if (pinned_mem_padding != 0) {
                    XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(pinned_mem_padding, handle_));
                }

                // Query the number of HW JPEG engine instances on this GPU.  The
                // extension's nvJPEG minimum is >= 12.1 so this symbol is expected
                // to exist; fail decoder creation if it is unavailable because the
                // dispatch heuristic requires an actual engine count.
                if (!nvjpegIsSymbolAvailable("nvjpegGetHardwareDecoderInfo")) {
                    throw NvJpegException::FromNvJpegError(
                        NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED,
                        WHERE("nvjpegGetHardwareDecoderInfo unavailable"));
                }
                XM_CHECK_NVJPEG(nvjpegGetHardwareDecoderInfo(handle_, &num_hw_engines_, &num_cores_per_hw_engine_));
                if (num_hw_engines_ == 0) {
                    throw NvJpegException::FromNvJpegError(
                        NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED,
                        WHERE("nvjpegGetHardwareDecoderInfo reported zero HW engines"));
                }
                hw_dec_info_status_ = NVJPEG_STATUS_SUCCESS;

                // Probe for the extended batched API; used by canDecode's legacy
                // support probe and by decodeBatch's legacy path.
                has_batched_ex_api_ = nvjpegIsSymbolAvailable("nvjpegDecodeBatchedEx") && nvjpegIsSymbolAvailable("nvjpegDecodeBatchedSupportedEx");
            }

            // Destroy all worker resources then the shared decoder and handle.
            // Wrapped in try/catch because DeviceGuard's constructor can throw
            // (cudaSetDevice failure, std::out_of_range from invalid device_id_) and a
            // destructor must not propagate exceptions; std::terminate is the
            // alternative.
            ~SharedContext() {
                try {
                    nvimgcodec::DeviceGuard device_guard(device_id_);
                    for (auto& worker : workers_) {
                        // Release helper buffer before stream teardown: the
                        // buffer's cudaFreeAsync deleter references worker.stream_, so
                        // it must run before the stream is destroyed.
                        releaseHelperDeviceBuffer(worker);
                        if (worker.state_) {
                            XM_NVJPEG_LOG_DESTROY(nvjpegJpegStateDestroy(worker.state_));
                        }
                        if (worker.device_buffer_) {
                            XM_NVJPEG_LOG_DESTROY(nvjpegBufferDeviceDestroy(worker.device_buffer_));
                        }
                        if (worker.decode_done_event_) {
                            XM_CUDA_LOG_DESTROY(cudaEventDestroy(worker.decode_done_event_));
                        }
                        if (worker.caller_ready_event_) {
                            XM_CUDA_LOG_DESTROY(cudaEventDestroy(worker.caller_ready_event_));
                        }
                        if (worker.stream_) {
                            XM_CUDA_LOG_DESTROY(cudaStreamDestroy(worker.stream_));
                        }
                    }
                    // decoder_ is shared among all workers (each worker holds an alias, not
                    // a copy), so it is destroyed here after all workers are cleaned up.
                    if (decoder_) {
                        XM_NVJPEG_LOG_DESTROY(nvjpegDecoderDestroy(decoder_));
                    }
                    for (auto& decode_slot : decode_slots_) {
                        if (decode_slot.pinned_buffer_) {
                            XM_NVJPEG_LOG_DESTROY(nvjpegBufferPinnedDestroy(decode_slot.pinned_buffer_));
                        }
                        if (decode_slot.decode_params_) {
                            XM_NVJPEG_LOG_DESTROY(nvjpegDecodeParamsDestroy(decode_slot.decode_params_));
                        }
                        if (decode_slot.nvjpeg_stream_) {
                            XM_NVJPEG_LOG_DESTROY(nvjpegJpegStreamDestroy(decode_slot.nvjpeg_stream_));
                        }
                    }
                    if (handle_) {
                        XM_NVJPEG_LOG_DESTROY(nvjpegDestroy(handle_));
                    }
                } catch (const std::exception& e) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_,
                                         "Exception in ~SharedContext: " << e.what());
                } catch (...) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_,
                                         "Unknown exception in ~SharedContext");
                }
            }

            // Eagerly initialize at least `count` workers in the pool at construction
            // time so the first batch doesn't pay the cost of allocating nvJPEG states
            // and CUDA streams on the critical path.
            void preallocateDecodeStates(int count) {
                // make sure at least `count` reusable decode
                // states exist in the shared pool. This does not lease them.
                std::lock_guard<std::mutex> lock(workers_mutex_);
                ensureWorkers(count);
            }

            // Ensure the shared nvJPEG decoder exists for BatchSingle support probes.
            // This is cheaper than allocating workers and keeps legacy-only decoders
            // from paying BatchSingle setup cost.
            void ensureDecoder() {
                std::lock_guard<std::mutex> lock(workers_mutex_);
                ensureDecoderLocked();
            }

            // Used by decodeSingleImageImpl: blocks until at least one worker is free,
            // then leases one via a *random* first-free
            // scan.  Random selection spreads load across HW engine cores under the
            // multi-thread customer pattern (N Python threads each owning a Decoder)
            // where a deterministic stride would synchronise with the per-thread
            // arrival rhythm and pin each thread to a fixed worker.  acquireWorkersUpTo
            // keeps rotation because it leases a full engine-count slice every call.
            int acquireWorker() {
                std::unique_lock<std::mutex> lock(workers_mutex_);
                ensureWorkers(num_hw_engines_);
                single_workers_cv_.wait(lock, [&]() {
                    for (size_t i = 0; i < workers_.size(); ++i) {
                        if (!workers_[i].in_use_)
                            return true;
                    }
                    return false;
                });

                size_t preferred_count = std::min(workers_.size(), static_cast<size_t>(num_hw_engines_));

                // Thread-local RNG: no shared-rng contention; the std::random_device
                // seed runs once per thread and the mt19937 sequence afterwards is
                // sufficient for load-spreading.
                thread_local std::mt19937 rng{std::random_device{}()};

                auto find_first_free_random = [&](size_t begin, size_t end) {
                    size_t count = end - begin;
                    if (count == 0)
                        return -1;
                    std::uniform_int_distribution<size_t> dist(0, count - 1);
                    size_t start = begin + dist(rng);
                    for (size_t offset = 0; offset < count; ++offset) {
                        size_t i = begin + ((start - begin + offset) % count);
                        if (!workers_[i].in_use_)
                            return static_cast<int>(i);
                    }
                    return -1;
                };

                // The CV wait above only releases this thread once at least one
                // workers_[i] has in_use_ == false. The two find_first_free_random
                // ranges below together cover the entire workers_ vector, so at
                // least one of them must locate the free worker and worker_id will
                // be >= 0 by the time we reach the array index. Coverity cannot
                // track that cross-call invariant and conservatively assumes the
                // negative return survives.
                // coverity[negative_returns : FALSE]
                int worker_id = find_first_free_random(0, preferred_count);
                if (worker_id < 0) {
                    // coverity[negative_returns : FALSE]
                    worker_id = find_first_free_random(preferred_count, workers_.size());
                }
                assert(worker_id >= 0);
                workers_[worker_id].in_use_ = true;
                return worker_id;
            }

            // Blocks until >= 1 worker is free, then atomically leases up to
            // `max_count` of the currently free workers.  Atomic set acquisition
            // prevents hold-and-wait deadlock between concurrent decodeBatchSingle
            // calls; the "up to" relaxation keeps batch-single from starving when
            // some workers are busy with single-image.
            std::vector<int> acquireWorkersUpTo(int max_count) {
                std::unique_lock<std::mutex> lock(workers_mutex_);
                ensureWorkers(max_count);
                single_workers_cv_.wait(lock, [&]() {
                    for (size_t i = 0; i < workers_.size(); ++i) {
                        if (!workers_[i].in_use_) {
                            return true;
                        }
                    }
                    return false;
                });

                std::vector<int> worker_ids;
                worker_ids.reserve(max_count);
                // Rotating scan start avoids repeatedly biasing small batches toward
                // low worker indices.  After each call, the next acquisition
                // starts one slot past the last leased worker, spreading load across
                // the pool over time.
                size_t start = next_worker_acquire_start_ % workers_.size();
                for (size_t offset = 0; offset < workers_.size() && static_cast<int>(worker_ids.size()) < max_count; ++offset) {
                    size_t i = (start + offset) % workers_.size();
                    if (!workers_[i].in_use_) {
                        workers_[i].in_use_ = true;
                        worker_ids.push_back(static_cast<int>(i));
                    }
                }
                assert(!worker_ids.empty());
                next_worker_acquire_start_ = (static_cast<size_t>(worker_ids.back()) + 1) % workers_.size();
                return worker_ids;
            }

            // Return a set of workers leased by decodeBatchSingle back to the free pool
            // and wake any threads blocked in acquireWorker* waiting for capacity.
            void releaseWorkers(const std::vector<int>& worker_ids) {
                {
                    std::lock_guard<std::mutex> lock(workers_mutex_);
                    for (int worker_id : worker_ids) {
                        bool valid_worker_id = worker_id >= 0 && worker_id < static_cast<int>(workers_.size());
                        assert(valid_worker_id);
                        if (valid_worker_id) {
                            workers_[worker_id].in_use_ = false;
                        }
                    }
                }
                single_workers_cv_.notify_all();
            }

            // Clear the worker's helper device buffer (used for I_YUV→P_YUV conversion).
            // Releases d_ptr first so its cudaFreeAsync deleter (which uses worker.stream_)
            // runs before the worker stream is destroyed.
            void releaseHelperDeviceBuffer(DecodeWorker& worker) noexcept {
                worker.helper_device_buffer_.d_ptr.reset();
                worker.helper_device_buffer_.data = nullptr;
                worker.helper_device_buffer_.size = 0;
                worker.helper_device_buffer_.capacity = 0;
                worker.helper_device_buffer_.stream = nullptr;
                worker.helper_device_buffer_.deleter_strategy.reset();
            }

            // Return a single worker leased by decodeSingleImageImpl back to the pool.
            void releaseWorker(int worker_id) {
                {
                    std::lock_guard<std::mutex> lock(workers_mutex_);
                    bool valid_worker_id = worker_id >= 0 && worker_id < static_cast<int>(workers_.size());
                    assert(valid_worker_id);
                    if (valid_worker_id) {
                        workers_[worker_id].in_use_ = false;
                    }
                }
                single_workers_cv_.notify_all();
            }

            // Fully initialize one new DecodeWorker: create its private CUDA stream and
            // events, allocate device and pinned buffers, then create the nvJPEG state
            // and attach the device buffer to it.  Also initializes both double-buffer
            // slots with their pinned buffers and nvjpegJpegStream objects.
            // Called only from ensureWorkers() under workers_mutex_.
            void initializeDecodeWorker(DecodeWorker& worker) {
                nvimgcodec::DeviceGuard device_guard(device_id_);
                ensureDecoderLocked();

                XM_CHECK_CUDA(cudaStreamCreateWithFlags(&worker.stream_, cudaStreamNonBlocking));
                // Disable timing on per-worker events.  They are used only for
                // stream-to-stream ordering (cudaStreamWaitEvent), never for
                // cudaEventElapsedTime.  B200 traces showed
                // cudaEventRecord on a timing-enabled event had median 3.8 µs but
                // mean 696 µs and tail max 14 ms — accounting for ~52% of
                // decodeSingleDeviceStage wall.  cudaEventDisableTiming flattens it.
                XM_CHECK_CUDA(cudaEventCreateWithFlags(&worker.decode_done_event_, cudaEventDisableTiming));
                XM_CHECK_CUDA(cudaEventCreateWithFlags(&worker.caller_ready_event_, cudaEventDisableTiming));
                // Pre-record decode_done_event_ so the first cudaStreamWaitEvent on it
                // is a no-op (avoids waiting on an uninitialized event).
                XM_CHECK_CUDA(cudaEventRecord(worker.decode_done_event_, worker.stream_));

                if (device_allocator_.dev_malloc && device_allocator_.dev_free) {
                    XM_CHECK_NVJPEG(nvjpegBufferDeviceCreateV2(handle_, &device_allocator_, &worker.device_buffer_));
                } else {
                    XM_CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle_, nullptr, &worker.device_buffer_));
                }

                // Each worker gets its own nvjpegJpegState_t (not thread-safe) backed
                // by the shared decoder and the worker's private device buffer.
                worker.decoder_ = decoder_;
                XM_CHECK_NVJPEG(nvjpegDecoderStateCreate(handle_, worker.decoder_, &worker.state_));
                XM_CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(worker.state_, worker.device_buffer_));

                // Initialize both double-buffer decode slots.  Each gets its own pinned
                // buffer (for host→GPU staging).  Having two slots lets the CPU parse
                // image N+1 while the GPU finishes image N without the two stages racing
                // on the same nvjpegJpegStream_t or pinned buffer.
                for (auto& slot_id : worker.slot_ids_) {
                    slot_id = createDecodeSlot();
                    auto& slot = decode_slots_.back();
                    if (pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free) {
                        XM_CHECK_NVJPEG(nvjpegBufferPinnedCreateV2(handle_, &pinned_allocator_, &slot.pinned_buffer_));
                    } else {
                        XM_CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle_, nullptr, &slot.pinned_buffer_));
                    }
                }
            }

            // --- Identity / logging ---
            int device_id_ = NVIMGCODEC_DEVICE_CURRENT;
            const char* plugin_id_ = nullptr;
            const nvimgcodecFrameworkDesc_t* framework_ = nullptr;

            // --- nvJPEG objects shared by all workers ---
            nvjpegHandle_t handle_ = nullptr;       // single nvJPEG library handle for this context
            nvjpegJpegDecoder_t decoder_ = nullptr; // shared HW decoder object; each worker aliases this

            // --- Allocator descriptors stored for worker creation ---
            nvjpegDevAllocatorV2_t device_allocator_{nullptr, nullptr, nullptr};
            nvjpegPinnedAllocatorV2_t pinned_allocator_{nullptr, nullptr, nullptr};
            // nvimgcodec-flavored allocator desc kept for DeviceBuffer inside each worker
            nvimgcodecDeviceAllocator_t device_allocator_desc_{NVIMGCODEC_STRUCTURE_TYPE_DEVICE_ALLOCATOR, sizeof(nvimgcodecDeviceAllocator_t), nullptr, nullptr, nullptr, nullptr, 0};

            // --- HW engine info (queried once at construction) ---
            unsigned int num_hw_engines_ = 0;          // number of independent HW JPEG engines on this GPU
            unsigned int num_cores_per_hw_engine_ = 0; // parallel decode cores per engine
            bool has_batched_ex_api_ = false;          // nvjpegDecodeBatchedEx / SupportedEx available
            nvjpegStatus_t hw_dec_info_status_ = NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;

            // --- Worker / slot pools ---
            // These pools only grow. Use deque so lazy push_back growth does not move
            // existing objects while decode holds references after dropping workers_mutex_.
            std::mutex workers_mutex_;
            std::condition_variable single_workers_cv_; // notified by releaseWorker* to wake blocked acquires
            std::deque<DecodeWorker> workers_;
            std::deque<DecodeSlot> decode_slots_;
            size_t next_worker_acquire_start_ = 0; // rotating start for acquireWorkersUpTo

        private:
            // Release the nvjpeg objects owned by a DecodeSlot.  Used by the
            // exception path of createDecodeSlot and ensureWorkers so a mid-init
            // failure doesn't leave half-constructed slots in decode_slots_.
            void destroyDecodeSlotResources(DecodeSlot& slot) {
                if (slot.pinned_buffer_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegBufferPinnedDestroy(slot.pinned_buffer_));
                    slot.pinned_buffer_ = nullptr;
                }
                if (slot.decode_params_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegDecodeParamsDestroy(slot.decode_params_));
                    slot.decode_params_ = nullptr;
                }
                if (slot.nvjpeg_stream_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegJpegStreamDestroy(slot.nvjpeg_stream_));
                    slot.nvjpeg_stream_ = nullptr;
                }
            }

            // Release the CUDA + nvjpeg objects owned by a DecodeWorker.  When
            // remove_slots is true, also tears down any DecodeSlots the worker
            // had managed to claim (used when initializeDecodeWorker throws
            // mid-way through worker construction).  Leaves the shared decoder_
            // alone — it belongs to the SharedContext.
            void destroyDecodeWorkerResources(DecodeWorker& worker, bool remove_slots) {
                if (remove_slots) {
                    for (auto it = worker.slot_ids_.rbegin(); it != worker.slot_ids_.rend(); ++it) {
                        int slot_id = *it;
                        if (slot_id < 0 || slot_id >= static_cast<int>(decode_slots_.size())) {
                            *it = -1;
                            continue;
                        }
                        destroyDecodeSlotResources(decode_slots_[slot_id]);
                        if (slot_id == static_cast<int>(decode_slots_.size()) - 1) {
                            decode_slots_.pop_back();
                        }
                        *it = -1;
                    }
                }

                if (worker.state_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegJpegStateDestroy(worker.state_));
                    worker.state_ = nullptr;
                }
                if (worker.device_buffer_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegBufferDeviceDestroy(worker.device_buffer_));
                    worker.device_buffer_ = nullptr;
                }
                if (worker.decode_done_event_) {
                    XM_CUDA_LOG_DESTROY(cudaEventDestroy(worker.decode_done_event_));
                    worker.decode_done_event_ = nullptr;
                }
                if (worker.caller_ready_event_) {
                    XM_CUDA_LOG_DESTROY(cudaEventDestroy(worker.caller_ready_event_));
                    worker.caller_ready_event_ = nullptr;
                }
                if (worker.stream_) {
                    XM_CUDA_LOG_DESTROY(cudaStreamDestroy(worker.stream_));
                    worker.stream_ = nullptr;
                }
                worker.decoder_ = nullptr;
                worker.in_use_ = false;
            }

            // Allocate one new DecodeSlot with its nvJPEG objects and append it to
            // decode_slots_.  pinned_buffer_ is left null and filled in by
            // initializeDecodeWorker after the slot is assigned to a worker.
            // Must be called with workers_mutex_ held.
            int createDecodeSlot() {
                assert(decoder_ != nullptr);
                DecodeSlot decode_slot;
                try {
                    XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &decode_slot.decode_params_));
                    XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &decode_slot.nvjpeg_stream_));
                    decode_slots_.push_back(decode_slot);
                } catch (...) {
                    destroyDecodeSlotResources(decode_slot);
                    throw;
                }
                return static_cast<int>(decode_slots_.size()) - 1;
            }

            // Grow the worker pool to at least `count` entries, creating and fully
            // initializing each new worker.  Must be called with workers_mutex_ held.
            void ensureWorkers(int count) {
                while (static_cast<int>(workers_.size()) < count) {
                    auto& worker = workers_.emplace_back(device_allocator_desc_.device_malloc ? &device_allocator_desc_ : nullptr);
                    try {
                        initializeDecodeWorker(worker);
                    } catch (...) {
                        destroyDecodeWorkerResources(worker, true);
                        workers_.pop_back();
                        throw;
                    }
                }
            }

            // Create the shared decoder object used by BatchSingle workers and support
            // probes. Must be called with workers_mutex_ held.
            void ensureDecoderLocked() {
                if (!decoder_) {
                    XM_CHECK_NVJPEG(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_HARDWARE, &decoder_));
                }
            }
        };

        // Function pointers from C APIs are compared by identity, not by call
        // semantics.  Cast them to uintptr_t via memcpy so the map ordering is
        // well-defined (reinterpret_cast<uintptr_t>(funcptr) is UB on hosts where
        // function pointers have a different size than void*).
        template <typename FunctionPtr>
        std::uintptr_t functionPointerKey(FunctionPtr ptr) {
            static_assert(sizeof(ptr) <= sizeof(std::uintptr_t), "Function pointer does not fit in uintptr_t");
            std::uintptr_t key = 0;
            std::memcpy(&key, &ptr, sizeof(ptr));
            return key;
        }

        std::uintptr_t objectPointerKey(const void* ptr) {
            return reinterpret_cast<std::uintptr_t>(ptr);
        }

        // SharedContextKey identifies a compatible SharedContext.  Two DecoderImpls
        // share a context only when their allocator callbacks + contexts + padding
        // also match — otherwise the second decoder's allocator could outlive the
        // nvjpeg handle that's calling into it.
        struct SharedContextKey {
            int device_id;
            unsigned int nvjpeg_flags;
            std::uintptr_t nvjpeg_device_ctx;
            std::uintptr_t nvjpeg_device_malloc;
            std::uintptr_t nvjpeg_device_free;
            std::uintptr_t nvjpeg_pinned_ctx;
            std::uintptr_t nvjpeg_pinned_malloc;
            std::uintptr_t nvjpeg_pinned_free;
            std::uintptr_t device_buffer_ctx;
            std::uintptr_t device_buffer_malloc;
            std::uintptr_t device_buffer_free;
            size_t device_mem_padding;
            size_t pinned_mem_padding;

            bool operator<(const SharedContextKey& other) const {
                return std::tie(device_id, nvjpeg_flags,
                                nvjpeg_device_ctx, nvjpeg_device_malloc, nvjpeg_device_free,
                                nvjpeg_pinned_ctx, nvjpeg_pinned_malloc, nvjpeg_pinned_free,
                                device_buffer_ctx, device_buffer_malloc, device_buffer_free,
                                device_mem_padding, pinned_mem_padding) <
                       std::tie(other.device_id, other.nvjpeg_flags,
                                other.nvjpeg_device_ctx, other.nvjpeg_device_malloc, other.nvjpeg_device_free,
                                other.nvjpeg_pinned_ctx, other.nvjpeg_pinned_malloc, other.nvjpeg_pinned_free,
                                other.device_buffer_ctx, other.device_buffer_malloc, other.device_buffer_free,
                                other.device_mem_padding, other.pinned_mem_padding);
            }
        };

        SharedContextKey makeSharedContextKey(
            int device_id,
            unsigned int nvjpeg_flags,
            const nvjpegDevAllocatorV2_t& device_allocator,
            const nvjpegPinnedAllocatorV2_t& pinned_allocator,
            const nvimgcodecDeviceAllocator_t* device_allocator_desc,
            size_t device_mem_padding,
            size_t pinned_mem_padding) {
            return SharedContextKey{
                device_id,
                nvjpeg_flags,
                objectPointerKey(device_allocator.dev_ctx),
                functionPointerKey(device_allocator.dev_malloc),
                functionPointerKey(device_allocator.dev_free),
                objectPointerKey(pinned_allocator.pinned_ctx),
                functionPointerKey(pinned_allocator.pinned_malloc),
                functionPointerKey(pinned_allocator.pinned_free),
                device_allocator_desc ? objectPointerKey(device_allocator_desc->device_ctx) : 0,
                device_allocator_desc ? functionPointerKey(device_allocator_desc->device_malloc) : 0,
                device_allocator_desc ? functionPointerKey(device_allocator_desc->device_free) : 0,
                device_mem_padding,
                pinned_mem_padding,
            };
        }

        // Registry of compatible SharedContexts, keyed as above.  Holds weak_ptrs;
        // the only shared_ptr lives in each DecoderImpl, so ~SharedContext runs when
        // the last DecoderImpl on that key is destroyed.  Stale weak_ptrs are pruned
        // lazily during lookup.
        //
        // Process-exit safety: ~DecoderImpl skips shared_context_.reset() once the
        // atexit handler has fired, so CUDA/nvJPEG teardown is not invoked while the
        // driver may already be shutting down.
        std::mutex& getSharedHwContextMutex() {
            static auto* mutex = new std::mutex();
            return *mutex;
        }

        std::map<SharedContextKey, std::weak_ptr<SharedContext>>& getSharedHwContexts() {
            static auto* contexts = new std::map<SharedContextKey, std::weak_ptr<SharedContext>>();
            return *contexts;
        }

        // True once std::atexit has fired.  ~DecoderImpl consults this to decide
        // whether dropping the last shared_ptr<SharedContext> is safe while CUDA may
        // already be tearing down; during process exit the context is retained.
        std::atomic<bool>& getInAtexit() {
            static auto* flag = new std::atomic<bool>(false);
            static std::once_flag register_once;
            std::call_once(register_once, []() {
                if (std::atexit([]() { getInAtexit().store(true, std::memory_order_relaxed); }) != 0) {
                    throw std::runtime_error("Failed to register nvJPEG HW decoder atexit handler");
                }
            });
            return *flag;
        }

        std::shared_ptr<SharedContext> getSharedHwContext(
            int device_id, const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, unsigned int nvjpeg_flags,
            const nvjpegDevAllocatorV2_t& device_allocator, const nvjpegPinnedAllocatorV2_t& pinned_allocator,
            const nvimgcodecDeviceAllocator_t* device_allocator_desc,
            size_t device_mem_padding, size_t pinned_mem_padding) {
            (void)getInAtexit(); // ensure atexit handler is registered before any decoder lives
            std::lock_guard<std::mutex> lock(getSharedHwContextMutex());
            auto& shared_hw_contexts = getSharedHwContexts();
            SharedContextKey key = makeSharedContextKey(
                device_id, nvjpeg_flags, device_allocator, pinned_allocator,
                device_allocator_desc, device_mem_padding, pinned_mem_padding);
            auto it = shared_hw_contexts.find(key);
            if (it != shared_hw_contexts.end()) {
                if (auto shared = it->second.lock())
                    return shared;
                shared_hw_contexts.erase(it);
            }

            auto shared = std::make_shared<SharedContext>(
                device_id, plugin_id, framework, nvjpeg_flags, device_allocator, pinned_allocator,
                device_allocator_desc, device_mem_padding, pinned_mem_padding);
            shared_hw_contexts.emplace(key, shared);
            return shared;
        }

        // Per-decoder-instance state.  Each DecoderImpl is created by NvJpegHwDecoderPlugin::create
        // (one per nvimgcodec decoder handle) and holds per-instance parameters such as
        // execution params, batched-API state, and a reference to the shared context for
        // its GPU.  The nvJPEG handle and decode worker pool live in SharedContext;
        // parser slots are per DecoderImpl so independent callers do not share parse state.
        struct DecoderImpl {
            DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
                        const char* options = nullptr);
            ~DecoderImpl();

            // C-callable thunk: cast the opaque decoder handle to DecoderImpl* and
            // forward to the typed member function.  All static_* methods below follow
            // this same pattern.
            static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);

            // Returns NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED — the HW JPEG decoder
            // does not expose per-image metadata beyond what the codec info API provides.
            nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const;
            static nvimgcodecStatus_t static_get_metadata(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) {
                try {
                    XM_CHECK_NULL(decoder);
                    auto handle = reinterpret_cast<DecoderImpl*>(decoder);
                    return handle->getMetadata(code_stream, metadata, metadata_count);
                } catch (const std::runtime_error& e) {
                    return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
                }
            }

            // Check whether the HW JPEG engine can decode this image with the given params.
            // Uses a header-only nvJPEG support probe for most samples.  The normal
            // single-image setup path may full-parse and hand one parsed stream to the
            // following decode stage (see tryAdoptParsedHandoff).
            nvimgcodecProcessingStatus_t canDecode(
                const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
            static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
                                                                  const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx) {
                try {
                    XM_CHECK_NULL(decoder);
                    auto handle = reinterpret_cast<DecoderImpl*>(decoder);
                    return handle->canDecode(image, code_stream, params, thread_idx);
                } catch (const std::runtime_error& e) {
                    return NVIMGCODEC_PROCESSING_STATUS_FAIL;
                }
            }

            // Per-sample entry point. Delegates through decodeBatch with batch_size=1
            // so chooseDecodePath remains the single source of truth.
            nvimgcodecStatus_t decode(
                const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
            static nvimgcodecStatus_t static_decode(
                nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
                const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx) {
                try {
                    XM_CHECK_NULL(decoder);
                    auto handle = reinterpret_cast<DecoderImpl*>(decoder);
                    return handle->decode(image, code_stream, params, thread_idx);
                } catch (const std::runtime_error& e) {
                    return NVIMGCODEC_STATUS_INTERNAL_ERROR;
                }
            }

            // Batch entry point.  Dispatches via chooseDecodePath(batch_size, code_streams).
            nvimgcodecStatus_t decodeBatch(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams, int batch_size,
                                           const nvimgcodecDecodeParams_t* params, int thread_idx);
            static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t** images,
                                                          const nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecDecodeParams_t* params, int thread_idx) {
                try {
                    XM_CHECK_NULL(decoder);
                    auto handle = reinterpret_cast<DecoderImpl*>(decoder);
                    return handle->decodeBatch(images, code_streams, batch_size, params, thread_idx);
                } catch (const std::runtime_error& e) {
                    return NVIMGCODEC_STATUS_INTERNAL_ERROR;
                }
            }

            // Returns the preferred minibatch size: num_hw_engines_ * num_cores_per_hw_engine_.
            // The framework uses this to pack batches that saturate the HW engine.
            nvimgcodecStatus_t getMiniBatchSize(int* batch_size);
            static nvimgcodecStatus_t static_get_mini_batch_size(nvimgcodecDecoder_t decoder, int* batch_size) {
                try {
                    XM_CHECK_NULL(decoder);
                    auto handle = reinterpret_cast<DecoderImpl*>(decoder);
                    return handle->getMiniBatchSize(batch_size);
                } catch (const std::runtime_error& e) {
                    return NVIMGCODEC_STATUS_INTERNAL_ERROR;
                }
            }

            // Scratch state for one parse of a JPEG bitstream, used only by canDecode().
            //
            // Parser state 0 can also serve as a single parsed-stream
            // handoff from canDecode() to the immediately following single-image
            // decode.  decodeSingleHostStage() swaps that stream into the worker's own
            // decode slot and skips nvjpegJpegStreamParse when (image*, code_stream*)
            // match.
            struct ParserSlot {
                nvjpegJpegStream_t nvjpeg_stream_ = nullptr;
                nvjpegDecodeParams_t decode_params_ = nullptr;
                const nvimgcodecImageDesc_t* image_ = nullptr;
                const nvimgcodecCodeStreamDesc_t* code_stream_ = nullptr;
            };

            // One persistent scratch parser slot per executor thread. canDecode() runs
            // on a known thread_idx and only touches that thread's state, so
            // cooperativeSetup takes no shared mutex and does not allocate after the
            // first use.
            //
            // State 0 owns one additional persistent handoff slot for the normal
            // single-image fast path. Keeping that slot separate is required: later
            // canDecode calls in a sequential setup may reuse the scratch slot, but
            // must not overwrite the parsed stream handed to decodeSingleHostStage().
            struct PerThreadParserState {
                ParserSlot scratch_slot_;
                ParserSlot handoff_slot_;
                bool handoff_claimed_ = false;
                uint64_t can_decode_stats_epoch_ = UINT64_MAX;
                int can_decode_success_count_ = 0;
                unsigned int can_decode_css_mask_ = 0;
            };

            // Size the per-thread parser state count: one state per executor thread plus
            // one for the caller thread. nvJPEG objects inside each state are created on
            // first use by ensureParserSlot().
            void preallocateParserSlots(int count);
            ParserSlot& ensureParserSlot(int thread_idx, bool use_handoff_slot);
            bool claimParsedHandoff(int thread_idx);
            void cacheParsedHandoff(int thread_idx, const nvimgcodecImageDesc_t* image,
                                    const nvimgcodecCodeStreamDesc_t* code_stream);
            bool shouldPrepareBatchSingleInCanDecode(bool use_parsed_handoff) const;
            void recordCanDecodeSuccess(int thread_idx, unsigned int css_bit);
            bool getCanDecodeCssMaskFromSetup(int batch_size, unsigned int& css_mask) const;
            bool tryAdoptParsedHandoff(
                const nvimgcodecImageDesc_t* image,
                const nvimgcodecCodeStreamDesc_t* code_stream,
                int decode_slot_id);
            void resetParsedHandoff();
            void finishDecodeBatchSetupState();
            void destroyParserSlots();

            // Parse the "module:key=value" option string (see definition for keys).
            void parseOptions(const char* options);

            // Acquires one worker from the shared pool, runs the full host+device
            // single-image pipeline on it, then returns the worker to the pool.
            nvimgcodecStatus_t decodeSingleImageImpl(
                const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params);

            // Decodes a whole minibatch using the 3-function per-image API:
            // nvjpegDecodeJpegHost, nvjpegDecodeJpegTransferToDevice, and
            // nvjpegDecodeJpegDevice.  Leases a pool of workers, dispatches
            // one executor task per host thread, and each task processes its strided
            // share of samples through its subset of workers.
            nvimgcodecStatus_t decodeBatchSingle(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams,
                                                 int batch_size, const nvimgcodecDecodeParams_t* params);
            // Stage 1 of the 3-function pipeline (CPU side): parse the JPEG bitstream,
            // configure decode parameters, and call nvjpegDecodeJpegHost.
            // Returns false (and signals imageReady with FAIL) if any step fails.
            bool decodeSingleHostStage(
                const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params,
                SharedContext::DecodeWorker& worker);

            // Stage 2 of the 3-function pipeline (GPU side): transfer the
            // host-decoded data to the device and run nvjpegDecodeJpegDevice, then
            // handle any I_YUV conversion or out-of-bounds ROI fill.
            void decodeSingleDeviceStage(SharedContext::DecodeWorker& worker, bool host_stage_succeeded);

            const char* plugin_id_;
            // Shared context for this GPU: owns the nvJPEG handle, decode worker pool,
            // and decode slots used by all DecoderImpl instances with the same key.
            std::shared_ptr<SharedContext> shared_context_;
            nvjpegHandle_t handle_ = nullptr; // alias into shared_context_->handle_
            nvjpegDevAllocatorV2_t device_allocator_;
            nvjpegPinnedAllocatorV2_t pinned_allocator_;
            const nvimgcodecFrameworkDesc_t* framework_;

            nvjpegJpegState_t state_;
            cudaEvent_t event_;

            const nvimgcodecExecutionParams_t* exec_params_;
            NvjpegVersion nvjpeg_version_;
            unsigned int num_hw_engines_;
            unsigned int num_cores_per_hw_engine_;
            int preallocate_batch_size_ = 1;
            int preallocate_width_ = 1;
            int preallocate_height_ = 1;

            // Scratch owned by DecodePath::Legacy (nvjpegDecodeBatched*).  These
            // vectors are cleared between batches but keep their capacity.
            std::vector<const unsigned char*> legacy_batch_bitstreams_;
            std::vector<size_t> legacy_batch_bitstream_sizes_;
            std::vector<nvjpegImage_t> legacy_batch_outputs_;
            std::vector<nvjpegDecodeParams_t> legacy_batch_nvjpeg_params_;
            std::vector<int> legacy_batch_valid_sample_idxs_;

            std::deque<PerThreadParserState> per_thread_parser_states_;
            uint64_t can_decode_stats_epoch_ = 0;

            // Additional DecodePath::Legacy scratch used for I_YUV decode.
            std::vector<nvimgcodecImageInfo_t> legacy_batch_decoded_image_info_;
            std::vector<nvimgcodec::DeviceBuffer> legacy_batch_helper_buffers_;

            // Successful sample indices for DecodePath::BatchSingle; clear() keeps
            // capacity between batches while preserving the original batch order.
            std::vector<int> batch_single_valid_sample_idxs_;
            // Per-batch imageReady bookkeeping for DecodePath::BatchSingle.  Worker
            // tasks report samples independently; if a later task fails, cleanup must
            // fail only samples that have not already reported SUCCESS/FAIL.
            std::vector<uint8_t> batch_single_image_ready_;

            bool has_batched_ex_api_;
            nvjpegStatus_t hw_dec_info_status_ = NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;

            // Force-path opt-out flags parsed from the "options" string.
            bool force_legacy_batch_ = false;
            bool force_batch_single_ = false;

            // Default true: matches libjpeg's chroma upsampling behaviour.
            bool fancy_upsampling_ = true;
            bool enable_roi_fancy_upsampling_ = true;
            unsigned int nvjpeg_extra_flags_ = 0;

            // DecodePath::Legacy uses nvjpegDecodeBatchedEx; BatchSingle uses the per-image pipeline.
            enum class DecodePath { Legacy,
                                    BatchSingle };

            static unsigned int chromaSubsamplingMaskBit(nvimgcodecChromaSubsampling_t chroma_subsampling) {
                return 1u << (static_cast<unsigned int>(chroma_subsampling) & 0x1Fu);
            }

            static bool hasMixedChromaSubsampling(unsigned int css_mask) {
                return (css_mask & (css_mask - 1)) != 0;
            }

            // Batch-aware heuristic used by decodeBatch dispatch.
            DecodePath chooseDecodePath(int batch_size, const nvimgcodecCodeStreamDesc_t** code_streams) const {
                if (force_legacy_batch_)
                    return DecodePath::Legacy;
                if (force_batch_single_)
                    return DecodePath::BatchSingle;

                // On single-engine GPUs (e.g. A100) the BatchSingle setup overhead
                // doesn't pay for itself when there is only one engine to feed; route
                // to the legacy batched path instead.  Multi-engine GPUs benefit from
                // BatchSingle's parallelism even at batch_size == 1.  This check must
                // run before the max_num_cpu_threads guard: the Python
                // single_python_threads path intentionally uses one C++ thread per
                // Decoder while relying on concurrent single-image BatchSingle work.
                if (batch_size <= 1) {
                    return (num_hw_engines_ <= 1) ? DecodePath::Legacy : DecodePath::BatchSingle;
                }

                if (exec_params_->max_num_cpu_threads < 2)
                    return DecodePath::Legacy;

                // Mixed CSS + fancy upsampling + non-trivial batches are routed to
                // BatchSingle on multi-engine GPUs.
                if (fancy_upsampling_ && batch_size >= 13) {
                    unsigned int css_mask = 0;
                    if (!getCanDecodeCssMaskFromSetup(batch_size, css_mask)) {
                        // Fallback for direct decodeBatch calls or cases where setup
                        // probed more samples than were assigned to this backend.
                        for (int i = 0; i < batch_size; ++i) {
                            if (code_streams[i] == nullptr)
                                continue;
                            nvimgcodecImageInfo_t cs_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                            if (code_streams[i]->getImageInfo(code_streams[i]->instance, &cs_info) != NVIMGCODEC_STATUS_SUCCESS)
                                continue;
                            css_mask |= chromaSubsamplingMaskBit(cs_info.chroma_subsampling);
                            if (hasMixedChromaSubsampling(css_mask))
                                break;
                        }
                    }
                    if (hasMixedChromaSubsampling(css_mask)) {
                        if (num_hw_engines_ <= 1)
                            return DecodePath::Legacy;
                        return DecodePath::BatchSingle;
                    }
                }

                return DecodePath::Legacy;
            }

            // Configure nvjpeg_params for this decode request: sets EXIF orientation (with
            // a version-specific rotation-direction fix) and ROI if the code stream view
            // specifies a region.  Sets need_params = true when any non-default setting is
            // applied before nvjpegDecodeBatchedSupportedEx.
            // Returns false and fills processing_status on any unsupported combination.
            bool setDecodeParams(bool& need_params, nvimgcodecProcessingStatus_t& processing_status, nvjpegDecodeParams_t& nvjpeg_params,
                                 const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageInfo_t& image_info, const nvimgcodecImageInfo_t& cs_image_info,
                                 const nvimgcodecDecodeParams_t* user_params) {
                nvjpegDecodeParamsSetExifOrientation(nvjpeg_params, NVJPEG_ORIENTATION_NORMAL);
                if (user_params->apply_exif_orientation) {
                    nvjpegExifOrientation_t orientation = nvimgcodec_to_nvjpeg_orientation(image_info.orientation);

                    // nvJPEG versions before 12.2 report 90- and 270-degree EXIF
                    // rotation in the opposite direction.
                    if (nvjpeg_version_ < NvjpegVersion(12, 2, 0)) {
                        if (orientation == NVJPEG_ORIENTATION_ROTATE_90)
                            orientation = NVJPEG_ORIENTATION_ROTATE_270;
                        else if (orientation == NVJPEG_ORIENTATION_ROTATE_270)
                            orientation = NVJPEG_ORIENTATION_ROTATE_90;
                    }

                    if (orientation == NVJPEG_ORIENTATION_UNKNOWN) {
                        processing_status = NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
                        return false;
                    }

                    if (orientation != NVJPEG_ORIENTATION_NORMAL) {
                        need_params = true;
                        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Setting up EXIF orientation " << orientation);
                        if (NVJPEG_STATUS_SUCCESS != nvjpegDecodeParamsSetExifOrientation(nvjpeg_params, orientation)) {
                            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpegDecodeParamsSetExifOrientation failed");
                            processing_status = NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
                            return false;
                        }
                    }
                }
                nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
                auto ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
                if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream information");
                    processing_status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
                    return false;
                }

                if (codestream_info.code_stream_view) {
                    // nvjpeg's ROI primitive operates in the same coordinate space as the decode
                    // output: when apply_exif_orientation is on, that is display coords, so we pass
                    // the user-supplied region as-is and bound-check it against the effective dims.
                    const auto& region = codestream_info.code_stream_view->region;
                    const auto [eff_w, eff_h] = nvimgcodec::oriented_dims(
                        cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height,
                        user_params->apply_exif_orientation ? image_info.orientation
                                                            : nvimgcodec::kIdentityOrientation);
                    if (region.ndim != 0) {
                        assert(region.ndim == 2);
                        need_params = true;
                        if (fancy_upsampling_ && !enable_roi_fancy_upsampling_ &&
                            cs_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_444 &&
                            cs_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY) {
                            NVIMGCODEC_LOG_WARNING(framework_, plugin_id_,
                                                   "ROI decode with fancy upsampling may produce wrong results on image edge (1 pixel wide) with nvJPEG < 13.2 (available in CTK 13.3)."
                                                   " To enable anyway, set enable_roi_fancy_upsampling=1");
                            processing_status = NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                            return false;
                        }

                        if (eff_h > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
                            eff_w > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Image dimensions exceeds int32, nvjpeg ROI decode is not supported.");
                            processing_status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
                            return false;
                        }

                        int roi_y_begin = region.start[0];
                        int roi_x_begin = region.start[1];
                        int roi_y_end = region.end[0];
                        int roi_x_end = region.end[1];

                        NVIMGCODEC_LOG_DEBUG(
                            framework_,
                            plugin_id_,
                            "Input ROI: y=[" << roi_y_begin << ", " << roi_y_end << "); x=[" << roi_x_begin << ", " << roi_x_end << ")");

                        roi_y_begin = std::max(0, roi_y_begin);
                        roi_x_begin = std::max(0, roi_x_begin);
                        roi_y_end = std::min(static_cast<int>(eff_h), roi_y_end);
                        roi_x_end = std::min(static_cast<int>(eff_w), roi_x_end);

                        NVIMGCODEC_LOG_DEBUG(
                            framework_,
                            plugin_id_,
                            "ROI After clipping: y=[" << roi_y_begin << ", " << roi_y_end << "); x=[" << roi_x_begin << ", " << roi_x_end << ")");

                        if (roi_x_end < roi_x_begin || roi_y_end < roi_y_begin) {
                            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "invalid ROI");
                            processing_status = NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                            return false;
                        }

                        int decoded_image_width = roi_x_end - roi_x_begin;
                        int decoded_image_height = roi_y_end - roi_y_begin;
                        if (NVJPEG_STATUS_SUCCESS != nvjpegDecodeParamsSetROI(nvjpeg_params, roi_x_begin, roi_y_begin, decoded_image_width, decoded_image_height)) {
                            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpegDecodeParamsSetROI failed");
                            processing_status = NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                            return false;
                        }
                    } else {
                        XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params, 0, 0, -1, -1));
                    }
                } else {
                    XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params, 0, 0, -1, -1));
                }
                return true;
            }
        };

    } // namespace

    // Construct the plugin descriptor and populate decoder_desc_ with the C function
    // pointers the framework will call to create/query/destroy decoder instances.
    NvJpegHwDecoderPlugin::NvJpegHwDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework) {
        decoder_desc_ = nvimgcodecDecoderDesc_t{
            NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC,
            sizeof(nvimgcodecDecoderDesc_t),
            NULL,
            this,
            plugin_id_,
            "jpeg",
            NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY,
            static_create,
            DecoderImpl::static_destroy,
            DecoderImpl::static_get_metadata,
            DecoderImpl::static_can_decode,
            DecoderImpl::static_decode,
            DecoderImpl::static_decode_batch,
            DecoderImpl::static_get_mini_batch_size};
    }

    // Probe whether NVJPEG_BACKEND_HARDWARE is usable on the current system by
    // attempting to create a temporary handle.  Returns false so the framework can
    // silently skip the plugin on unsupported GPUs.
    bool NvJpegHwDecoderPlugin::isPlatformSupported() {
        nvjpegHandle_t handle;
        nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, 0, &handle);
        if (status == NVJPEG_STATUS_SUCCESS) {
            XM_CHECK_NVJPEG(nvjpegDestroy(handle));
            return true;
        } else {
            return false;
        }
    }

    // Return the address of the framework-facing descriptor populated in the constructor.
    nvimgcodecDecoderDesc_t* NvJpegHwDecoderPlugin::getDecoderDesc() {
        return &decoder_desc_;
    }

    // The HW JPEG decoder does not expose metadata beyond what the codec provides.
    nvimgcodecStatus_t DecoderImpl::getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

    void DecoderImpl::destroyParserSlots() {
        for (auto& state : per_thread_parser_states_) {
            auto destroy_slot = [this](ParserSlot& parser_slot) {
                if (parser_slot.decode_params_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegDecodeParamsDestroy(parser_slot.decode_params_));
                    parser_slot.decode_params_ = nullptr;
                }
                if (parser_slot.nvjpeg_stream_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegJpegStreamDestroy(parser_slot.nvjpeg_stream_));
                    parser_slot.nvjpeg_stream_ = nullptr;
                }
                parser_slot.image_ = nullptr;
                parser_slot.code_stream_ = nullptr;
            };
            destroy_slot(state.scratch_slot_);
            destroy_slot(state.handoff_slot_);
            state.handoff_claimed_ = false;
        }
    }

    void DecoderImpl::preallocateParserSlots(int count) {
        while (static_cast<int>(per_thread_parser_states_.size()) < count) {
            per_thread_parser_states_.emplace_back();
        }
    }

    DecoderImpl::ParserSlot& DecoderImpl::ensureParserSlot(int thread_idx, bool use_handoff_slot) {
        assert(thread_idx >= 0 && thread_idx < static_cast<int>(per_thread_parser_states_.size()));
        auto& state = per_thread_parser_states_[thread_idx];
        auto& slot = use_handoff_slot ? state.handoff_slot_ : state.scratch_slot_;
        if (!slot.decode_params_ || !slot.nvjpeg_stream_) {
            try {
                if (!slot.decode_params_) {
                    XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &slot.decode_params_));
                }
                if (!slot.nvjpeg_stream_) {
                    XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &slot.nvjpeg_stream_));
                }
            } catch (...) {
                if (slot.decode_params_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegDecodeParamsDestroy(slot.decode_params_));
                    slot.decode_params_ = nullptr;
                }
                if (slot.nvjpeg_stream_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegJpegStreamDestroy(slot.nvjpeg_stream_));
                    slot.nvjpeg_stream_ = nullptr;
                }
                throw;
            }
        }
        return slot;
    }

    bool DecoderImpl::claimParsedHandoff(int thread_idx) {
        if (force_legacy_batch_ || force_batch_single_ || num_hw_engines_ <= 1 ||
            exec_params_->max_num_cpu_threads >= 2 || thread_idx != 0)
            return false;
        if (per_thread_parser_states_.empty())
            return false;
        assert(!per_thread_parser_states_.empty());
        auto& state = per_thread_parser_states_[0];
        if (state.handoff_claimed_)
            return false;
        state.handoff_claimed_ = true;
        return true;
    }

    void DecoderImpl::cacheParsedHandoff(int thread_idx, const nvimgcodecImageDesc_t* image,
                                         const nvimgcodecCodeStreamDesc_t* code_stream) {
        assert(thread_idx == 0 && !per_thread_parser_states_.empty());
        auto& s = per_thread_parser_states_[0].handoff_slot_;
        s.image_ = image;
        s.code_stream_ = code_stream;
    }

    bool DecoderImpl::shouldPrepareBatchSingleInCanDecode(bool use_parsed_handoff) const {
        if (force_legacy_batch_)
            return false;
        if (force_batch_single_)
            return true;
        if (num_hw_engines_ <= 1)
            return false;

        // Preserve the Python-threaded single-image path: each decode() call runs a
        // one-sample setup/decode cycle, so parser slot 0 is a cheap single-slot
        // handoff to the immediately following decode.
        //
        // Parallel multi-sample canDecode calls claim work via the framework's
        // atomic sample index.  Without exact batch context here, trying to infer
        // the mixed-CSS BatchSingle heuristic would require shared atomics or a
        // lock.  Keep canDecode cheap for those batches; decodeBatch() has the
        // exact batch and the host stage will full-parse before nvjpegDecodeJpegHost.
        return use_parsed_handoff;
    }

    void DecoderImpl::recordCanDecodeSuccess(int thread_idx, unsigned int css_bit) {
        assert(thread_idx >= 0 && thread_idx < static_cast<int>(per_thread_parser_states_.size()));
        auto& state = per_thread_parser_states_[thread_idx];
        if (state.can_decode_stats_epoch_ != can_decode_stats_epoch_) {
            state.can_decode_stats_epoch_ = can_decode_stats_epoch_;
            state.can_decode_success_count_ = 0;
            state.can_decode_css_mask_ = 0;
        }
        state.can_decode_success_count_++;
        state.can_decode_css_mask_ |= css_bit;
    }

    bool DecoderImpl::getCanDecodeCssMaskFromSetup(int batch_size, unsigned int& css_mask) const {
        int success_count = 0;
        css_mask = 0;
        for (const auto& state : per_thread_parser_states_) {
            if (state.can_decode_stats_epoch_ != can_decode_stats_epoch_)
                continue;
            success_count += state.can_decode_success_count_;
            css_mask |= state.can_decode_css_mask_;
        }
        return success_count == batch_size;
    }

    bool DecoderImpl::tryAdoptParsedHandoff(
        const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream,
        int decode_slot_id) {
        assert(decode_slot_id >= 0 && decode_slot_id < static_cast<int>(shared_context_->decode_slots_.size()));
        if (per_thread_parser_states_.empty())
            return false;

        auto& slot = per_thread_parser_states_[0].handoff_slot_;
        if (!slot.nvjpeg_stream_)
            return false;
        if (slot.image_ != image || slot.code_stream_ != code_stream)
            return false;

        std::swap(shared_context_->decode_slots_[decode_slot_id].nvjpeg_stream_, slot.nvjpeg_stream_);
        slot.image_ = nullptr;
        slot.code_stream_ = nullptr;
        return true;
    }

    void DecoderImpl::resetParsedHandoff() {
        if (per_thread_parser_states_.empty())
            return;
        auto& state = per_thread_parser_states_[0];
        state.handoff_claimed_ = false;
        state.handoff_slot_.image_ = nullptr;
        state.handoff_slot_.code_stream_ = nullptr;
    }

    void DecoderImpl::finishDecodeBatchSetupState() {
        resetParsedHandoff();
        ++can_decode_stats_epoch_;
    }

    nvimgcodecProcessingStatus_t DecoderImpl::canDecode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
                                                        const nvimgcodecDecodeParams_t* params, int thread_idx) {
        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;

        bool use_parsed_handoff = claimParsedHandoff(thread_idx);
        bool prepare_batch_single = shouldPrepareBatchSingleInCanDecode(use_parsed_handoff);
        // Use this thread's persistent parser state. No mutex: the framework never
        // runs two cooperativeSetup tasks with the same thread_idx at the same time.
        auto& slot = ensureParserSlot(thread_idx, use_parsed_handoff);
        auto decode_params = slot.decode_params_;
        auto jpeg_stream = slot.nvjpeg_stream_;
        unsigned int can_decode_css_bit = 0;
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode ");
            XM_CHECK_NULL(code_stream);
            XM_CHECK_NULL(params);

            // Reject non-JPEG streams up front before touching nvjpeg.
            nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            auto ret = code_stream->getImageInfo(code_stream->instance, &cs_image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

            bool is_jpeg = strcmp(cs_image_info.codec_name, "jpeg") == 0;
            if (!is_jpeg) {
                return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            }

            // Fetch the caller's requested image_info (output layout / dtype).
            XM_CHECK_NULL(image);
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

            if (!nvimgcodec::check_planes_consistency(framework_, plugin_id_, image_info)) {
                return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }

            // Chroma-subsampling early reject.  The HW path supports a fixed
            // subset; 4:1:0 / 4:1:1 are reported unsupported.
            static const std::set<nvimgcodecChromaSubsampling_t> supported_css{NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422,
                                                                               NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_GRAY};
            if (supported_css.find(cs_image_info.chroma_subsampling) == supported_css.end()) {
                return NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            can_decode_css_bit = chromaSubsamplingMaskBit(cs_image_info.chroma_subsampling);

            // The BatchSingle support probe needs the exact output format / CMYK
            // policy. The legacy batched support probe intentionally matches main:
            // keep it header-only plus setDecodeParams() so canDecode stays cheap.
            if (prepare_batch_single) {
                int num_channels = std::max(image_info.num_planes, image_info.plane_info[0].num_channels);
                auto sample_format = num_channels == 1 ? NVIMGCODEC_SAMPLEFORMAT_P_Y : image_info.sample_format;
                auto nvjpeg_format = nvimgcodec_to_nvjpeg_format(sample_format);
                XM_CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(decode_params, nvjpeg_format));
                int allow_cmyk = (image_info.color_spec != NVIMGCODEC_COLORSPEC_UNCHANGED) &&
                                 (image_info.color_spec != NVIMGCODEC_COLORSPEC_CMYK) &&
                                 (image_info.color_spec != NVIMGCODEC_COLORSPEC_YCCK);
                XM_CHECK_NVJPEG(nvjpegDecodeParamsSetAllowCMYK(decode_params, allow_cmyk));
            }

            bool need_params = false;
            if (!setDecodeParams(need_params, status, decode_params, code_stream, image_info, cs_image_info, params)) {
                return status;
            }

            // YUV/YCC sanity checks: nvjpeg HW can only emit YUV when the
            // bitstream is YCC-encoded, and I_YUV (interleaved) further requires
            // 4:4:4 (no subsampling) because we don't run a chroma upsampler.
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_YUV || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_YUV) {
                if (cs_image_info.color_spec != NVIMGCODEC_COLORSPEC_SYCC) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "YUV/YCC decoding is only supported if image is encoded into YCC.");
                    return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                }
            }

            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_YUV) {
                if (cs_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_444) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "YUV/YCC decoding is only supported if image is encoded without subsampling (444).");
                    return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                }
            }

            // Map the encoded JPEG bitstream from the caller's io_stream.
            // Failure here means a broken/unreadable source; surface it as
            // IMAGE_CORRUPTED for the current sample.
            assert(code_stream->io_stream);
            void* encoded_stream_data = nullptr;
            size_t encoded_stream_data_size = 0;
            if (code_stream->io_stream->size(code_stream->io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
            if (code_stream->io_stream->map(code_stream->io_stream->instance, &encoded_stream_data, 0, encoded_stream_data_size) !=
                NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
            assert(encoded_stream_data != nullptr);
            assert(encoded_stream_data_size > 0);

            // Parse the JPEG bitstream into jpeg_stream.
            //
            // nvjpegDecoderJpegSupported for BatchSingle requires the full
            // nvjpegJpegStreamParse state.  However canDecode has no batch-size
            // argument, while decodeBatch() may still route the final batch to
            // Legacy.  Only pay the full-parse cost for single-image and forced
            // BatchSingle requests; only the normal single-image case is cached for
            // handoff.  Otherwise keep the baseline header-only batched support
            // probe.
            if (prepare_batch_single) {
                XM_CHECK_NVJPEG(nvjpegJpegStreamParse(
                    handle_, static_cast<const unsigned char*>(encoded_stream_data),
                    encoded_stream_data_size, false, false, jpeg_stream));
            } else {
                // Try the cheap header-only parse first.  For most JPEGs that's
                // enough for nvjpegDecodeBatchedSupportedEx and avoids the
                // entropy-marker walk that the full nvjpegJpegStreamParse does
                // (~tens of us/image; canDecode runs per image at all batch sizes).
                //
                // Some nvJPEG versions reject valid EXIF-rotated JPEGs from
                // ParseHeader while full Parse accepts them. Retry with a full
                // parse for those samples.
                //
                // On affected nvJPEG versions, a failed ParseHeader can leave the
                // stream object unusable for another ParseHeader call. Recreate it
                // before the full-parse retry.
                nvjpegStatus_t parse_status = nvjpegJpegStreamParseHeader(
                    handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size, jpeg_stream);
                if (parse_status != NVJPEG_STATUS_SUCCESS) {
                    if (nvjpeg_version_ <= NvjpegVersion(13, 2, 0)) {
                        XM_NVJPEG_LOG_DESTROY(nvjpegJpegStreamDestroy(jpeg_stream));
                        XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &slot.nvjpeg_stream_));
                        jpeg_stream = slot.nvjpeg_stream_;
                    }
                    XM_CHECK_NVJPEG(nvjpegJpegStreamParse(
                        handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size,
                        false, false, jpeg_stream));
                }
            }

            // Ask nvjpeg whether the HW backend can decode this exact
            // (stream, decode_params) tuple.  The Supported flavor depends on
            // which dispatch path canDecode primed for.
            // isSupported semantics are inverted: 0 = supported, !=0 = not.
            int isSupported = -1;
            if (prepare_batch_single) {
                shared_context_->ensureDecoder();
                XM_CHECK_NVJPEG(nvjpegDecoderJpegSupported(shared_context_->decoder_, jpeg_stream, decode_params, &isSupported));
            } else if (has_batched_ex_api_) {
                XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupportedEx(handle_, jpeg_stream, decode_params, &isSupported));
            } else {
                if (need_params) {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpegDecodeBatchedSupportedEx API is not supported");
                    return NVIMGCODEC_PROCESSING_STATUS_FAIL;
                }
                XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupported(handle_, jpeg_stream, &isSupported));
            }

            // Record the verdict, and for the single-image path cache the parsed
            // stream for decodeSingleHostStage adoption.
            if (isSupported == 0) {
                status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "decoding image on HW is supported");
                if (prepare_batch_single && use_parsed_handoff) {
                    // Park the parsed stream in the single-slot handoff for decodeSingleHostStage.
                    cacheParsedHandoff(thread_idx, image, code_stream);
                }
            } else {
                status = NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "decoding image on HW is NOT supported");
            }

            // ROI / multi-image checks.  These OR additional unsupported
            // flags into status without preventing the decode itself — the
            // framework will pick a different backend when ROI is required
            // but the HW path can't honor it.
            nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
            ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

            if (codestream_info.code_stream_view) {
                // Multi-image bitstreams (e.g. MPO): the HW path only emits
                // image index 0.  Higher indices get marked unsupported.
                if (codestream_info.code_stream_view->image_idx != 0) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED;
                }

                // ROI: ndim==0 means no ROI (most common).  ndim==2 is the
                // standard rectangular region — out-of-bounds regions need
                // verify_region_fill_support to confirm the requested fill
                // mode is implementable on the HW path.  Other ndim are
                // rejected outright (we don't support non-2D regions).
                const auto& region = codestream_info.code_stream_view->region;
                if (region.ndim == 0) {
                    // no ROI, okay
                } else if (region.ndim == 2) {
                    if (nvimgcodec::is_region_out_of_bounds_effective(region, image_info.orientation,
                                                                      cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height,
                                                                      params->apply_exif_orientation)) {
                        if (auto err_message = nvimgcodec::verify_region_fill_support(region, image_info); !err_message.empty()) {
                            status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                            NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, err_message);
                        }
                    }
                } else {
                    status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Region decoding is supported only for 2 dimensions.");
                }
            }
        } catch (const NvJpegException& e) {
            // XM_CHECK_NVJPEG / XM_CHECK_CUDA throw on failure.  Any nvjpeg
            // call above that surfaces an error lands here; report PROCESSING_FAIL
            // so the framework can try a different backend.
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if hw nvjpeg can decode - " << e.info());
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
        if (status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS && can_decode_css_bit != 0) {
            recordCanDecodeSuccess(thread_idx, can_decode_css_bit);
        }
        return status;
    }

    // Per-sample framework entry point. Wraps the request in a one-element
    // batch and delegates to decodeBatch so chooseDecodePath is the single
    // source of truth for path selection (BatchSingle vs. legacy batched).
    // On single-engine GPUs this routes through the legacy path, avoiding
    // BatchSingle's per-image overhead.
    nvimgcodecStatus_t DecoderImpl::decode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx) {
        (void)thread_idx;
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);

        const nvimgcodecImageDesc_t* images_arr[1] = {image};
        const nvimgcodecCodeStreamDesc_t* streams_arr[1] = {code_stream};
        return decodeBatch(images_arr, streams_arr, 1, params, /*thread_idx=*/0);
    }

    // Decode a single image using the 3-stage API
    nvimgcodecStatus_t DecoderImpl::decodeSingleImageImpl(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params) {
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);

        bool image_ready_reported = false;
        try {
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            auto ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            int worker_id = shared_context_->acquireWorker();
            struct WorkerLease {
                std::shared_ptr<SharedContext> ctx;
                int worker_id;
                // RAII for decode_api=single_image: always return the borrowed worker
                // to the shared pool, even if host or device decode throws.
                ~WorkerLease() {
                    if (ctx) {
                        ctx->releaseWorker(worker_id);
                    }
                }
            } lease{shared_context_, worker_id};

            auto& worker = shared_context_->workers_[worker_id];
            // Capture the caller's CUDA stream position into caller_ready_event_, then
            // make the worker's private stream wait on it so any GPU work the caller
            // submitted before this call completes before we touch the output buffer.
            XM_CHECK_CUDA(cudaEventRecord(worker.caller_ready_event_, image_info.cuda_stream));
            XM_CHECK_CUDA(cudaStreamWaitEvent(worker.stream_, worker.caller_ready_event_));

            bool host_ok = decodeSingleHostStage(image, code_stream, params, worker);
            if (!host_ok) {
                image_ready_reported = true;
            }
            decodeSingleDeviceStage(worker, host_ok);
            if (!host_ok) {
                // The sample failure was already reported through imageReady().
                // Return SUCCESS for the plugin call itself so the framework uses
                // the per-sample status for fallback/reporting, matching the other
                // nvJPEG decoders.
                return NVIMGCODEC_STATUS_SUCCESS;
            }
            image_ready_reported = true;
            // Wait for the decode to finish before the caller's stream reads the output.
            XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, worker.decode_done_event_));
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg code stream - " << e.info());
            if (!image_ready_reported) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            }
            // imageReady() carries the sample result.  A successful return here
            // means the decoder reported that result without an API-level failure.
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg code stream - " << e.what());
            if (!image_ready_reported) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            }
            // imageReady() carries the sample result.  A successful return here
            // means the decoder reported that result without an API-level failure.
            return NVIMGCODEC_STATUS_SUCCESS;
        }
    }

    // Parse the space-separated "module:key=value" option string.
    // Recognised keys: preallocate_width_hint, preallocate_height_hint,
    // preallocate_batch_size, force_legacy_batch, force_batch_single,
    // fancy_upsampling, enable_roi_fancy_upsampling (module prefix "nvjpeg_hw_decoder" or empty).
    void DecoderImpl::parseOptions(const char* options) {
        enable_roi_fancy_upsampling_ = nvjpeg_version_ >= ROI_FANCY_UPSAMPLING_FIX_VERSION;
        std::istringstream iss(options ? options : "");
        std::string token;
        while (std::getline(iss, token, ' ')) {
            std::string::size_type colon = token.find(':');
            std::string::size_type equal = token.find('=');
            if (colon == std::string::npos || equal == std::string::npos || colon > equal)
                continue;
            std::string module = token.substr(0, colon);
            if (module != "" && module != "nvjpeg_hw_decoder")
                continue;
            std::string option = token.substr(colon + 1, equal - colon - 1);
            std::string value_str = token.substr(equal + 1);

            std::istringstream value(value_str);
            if (option == "preallocate_width_hint") {
                value >> preallocate_width_;
            } else if (option == "preallocate_height_hint") {
                value >> preallocate_height_;
            } else if (option == "preallocate_batch_size") {
                value >> preallocate_batch_size_;
            } else if (option == "force_legacy_batch") {
                int v = 0;
                value >> v;
                force_legacy_batch_ = (v != 0);
            } else if (option == "force_batch_single") {
                int v = 0;
                value >> v;
                force_batch_single_ = (v != 0);
            } else if (option == "fancy_upsampling") {
                value >> fancy_upsampling_;
            } else if (option == "enable_roi_fancy_upsampling") {
                value >> enable_roi_fancy_upsampling_;
            } else if (option == "extra_flags") {
                value >> nvjpeg_extra_flags_;
            }
        }
    }

    // Construct a DecoderImpl: parse options, resolve or create the shared context for
    // exec_params->device_id, preallocate parser/worker slots, and call
    // nvjpegDecodeBatchedPreAllocate to warm up the HW memory pool.
    DecoderImpl::DecoderImpl(
        const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
        : plugin_id_(plugin_id), device_allocator_{nullptr, nullptr, nullptr}, pinned_allocator_{nullptr, nullptr, nullptr}, framework_(framework), exec_params_(exec_params) {
        nvimgcodec::DeviceGuard device_guard(exec_params_->device_id);
        nvjpeg_version_ = get_nvjpeg_version();
        if (!nvjpeg_version_) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to get nvJPEG version");
            throw std::runtime_error("Failed to get nvJPEG version");
        }
        parseOptions(options);

        bool has_device_allocator = false;
        bool has_pinned_allocator = false;
        if (nvjpegIsSymbolAvailable("nvjpegCreateExV2")) {
            if (exec_params->device_allocator && exec_params->device_allocator->device_malloc && exec_params->device_allocator->device_free) {
                device_allocator_.dev_ctx = exec_params->device_allocator->device_ctx;
                device_allocator_.dev_malloc = exec_params->device_allocator->device_malloc;
                device_allocator_.dev_free = exec_params->device_allocator->device_free;
                has_device_allocator = true;
            }

            if (exec_params->pinned_allocator && exec_params->pinned_allocator->pinned_malloc && exec_params->pinned_allocator->pinned_free) {
                pinned_allocator_.pinned_ctx = exec_params->pinned_allocator->pinned_ctx;
                pinned_allocator_.pinned_malloc = exec_params->pinned_allocator->pinned_malloc;
                pinned_allocator_.pinned_free = exec_params->pinned_allocator->pinned_free;
                has_pinned_allocator = true;
            }
        }

        unsigned int nvjpeg_flags = get_nvjpeg_flags(nvjpeg_version_, fancy_upsampling_, nvjpeg_extra_flags_);

        if (fancy_upsampling_) {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "Fancy upsampling enabled; performance may be worse compared to simple upsampling");
        }

        size_t device_mem_padding = exec_params->device_allocator ? exec_params->device_allocator->device_mem_padding : 0;
        size_t pinned_mem_padding = exec_params->pinned_allocator ? exec_params->pinned_allocator->pinned_mem_padding : 0;
        // Obtain (or create) the shared context for this device.  All DecoderImpl
        // instances on the same GPU share one nvJPEG handle and worker pool.
        shared_context_ = getSharedHwContext(
            exec_params_->device_id,
            plugin_id_,
            framework_,
            nvjpeg_flags,
            has_device_allocator ? device_allocator_ : nvjpegDevAllocatorV2_t{nullptr, nullptr, nullptr},
            has_pinned_allocator ? pinned_allocator_ : nvjpegPinnedAllocatorV2_t{nullptr, nullptr, nullptr},
            exec_params->device_allocator,
            device_mem_padding,
            pinned_mem_padding);

        // Copy the shared fields into local members so the rest of the class does
        // not need to go through shared_context_ for every access.
        handle_ = shared_context_->handle_;
        num_hw_engines_ = shared_context_->num_hw_engines_;
        num_cores_per_hw_engine_ = shared_context_->num_cores_per_hw_engine_;
        has_batched_ex_api_ = shared_context_->has_batched_ex_api_;
        hw_dec_info_status_ = shared_context_->hw_dec_info_status_;

        auto executor = exec_params_->executor;
        assert(executor);
        int executor_threads = executor->getNumThreads(executor->instance);
        // Pre-create one parser state per executor thread, plus one for the caller
        // thread (thread_idx = num_threads in the framework). Each state's nvJPEG
        // objects are created lazily on first use.
        preallocateParserSlots(executor_threads + 1);

        NVIMGCODEC_LOG_INFO(framework_, plugin_id_,
                            "HW decoder available num_hw_engines=" << num_hw_engines_ << " num_cores_per_hw_engine=" << num_cores_per_hw_engine_);

        if (hw_dec_info_status_ == NVJPEG_STATUS_SUCCESS) {
            float hw_load = exec_params->num_backends == 0 ? 1.0f : 0.0f;
            for (int b = 0; b < exec_params->num_backends; b++) {
                auto& backend = exec_params->backends[b];
                if (backend.kind == NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY) {
                    hw_load = backend.params.load_hint;
                    break;
                }
            }
            int preferred_mini_batch = 0;
            getMiniBatchSize(&preferred_mini_batch);

            int full_batch_size = preallocate_batch_size_;
            preallocate_batch_size_ = static_cast<int>(std::round(hw_load * full_batch_size));
            if (preferred_mini_batch > 0) {
                int tail = preallocate_batch_size_ % preferred_mini_batch;
                if (tail > 0) {
                    preallocate_batch_size_ = preallocate_batch_size_ + preferred_mini_batch - tail;
                }
            }
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "adjust preallocate_batch_size " << full_batch_size << " to " << preallocate_batch_size_);
        } else {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "adjust preallocate_batch_size " << preallocate_batch_size_ << " to 0");
            preallocate_batch_size_ = 0;
        }

        XM_CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_));
        XM_CHECK_CUDA(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));

        // call nvjpegDecodeBatchedPreAllocate to use memory pool for HW decoder even if hint is 0
        // due to considerable performance benefit - >20% for 8GPU training
        if (hw_dec_info_status_ == NVJPEG_STATUS_SUCCESS && nvjpegIsSymbolAvailable("nvjpegDecodeBatchedPreAllocate")) {
            if (preallocate_width_ < 1)
                preallocate_width_ = 1;
            if (preallocate_height_ < 1)
                preallocate_height_ = 1;
            nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_444;
            nvjpegOutputFormat_t format = NVJPEG_OUTPUT_RGBI;
            std::stringstream ss;
            ss << "nvjpegDecodeBatchedPreAllocate batch_size=" << preallocate_batch_size_ << " width=" << preallocate_width_
               << " height=" << preallocate_height_;
            auto msg = ss.str();
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, msg);
            nvtx3::scoped_range marker{msg};
            XM_CHECK_NVJPEG(nvjpegDecodeBatchedPreAllocate(handle_, state_, preallocate_batch_size_,
                                                           preallocate_width_, preallocate_height_, subsampling, format));
        }
    }

    // Allocate and initialise a new DecoderImpl, returning it as an opaque handle.
    nvimgcodecStatus_t NvJpegHwDecoderPlugin::create(
        nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_create");
            XM_CHECK_NULL(decoder);
            XM_CHECK_NULL(exec_params);
            if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;

            *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new DecoderImpl(plugin_id_, framework_, exec_params, options));
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg decoder:" << e.info());
            return e.nvimgcodecStatus();
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg decoder:" << e.what());
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    // C-callable thunk for NvJpegHwDecoderPlugin::create.
    nvimgcodecStatus_t NvJpegHwDecoderPlugin::static_create(
        void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        if (!instance) {
            return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
        }

        NvJpegHwDecoderPlugin* handle = reinterpret_cast<NvJpegHwDecoderPlugin*>(instance);
        return handle->create(decoder, exec_params, options);
    }

    // Destroy per-instance state (batched decode params, event, state_).
    // The shared handle_, worker pool, and slot pool are owned by shared_context_
    // and are released only when the last shared_ptr for this device is destroyed.
    DecoderImpl::~DecoderImpl() {
        // After atexit, the CUDA driver may already be tearing down — any
        // cuda*Destroy / nvjpeg*Destroy call could segfault.  Leak both
        // per-instance state AND our shared_context_ reference; the registry's
        // weak_ptr will simply find an expired entry next time. The leak is
        // intentional and bounded (one shared_ptr per decoder at process exit).
        if (getInAtexit().load(std::memory_order_relaxed)) {
            // `new` would throw std::bad_alloc on OOM; this destructor is implicitly
            // noexcept, and at atexit there's nothing useful left to do. Use the
            // nothrow form so a (highly unlikely) allocation failure cannot escape.
            // coverity[RESOURCE_LEAK]
            // coverity[leaked_storage : INTENTIONAL]
            (void)new (std::nothrow) std::shared_ptr<SharedContext>(std::move(shared_context_));
            return;
        }
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_destroy");
            nvimgcodec::DeviceGuard device_guard(exec_params_->device_id);
            destroyParserSlots();
            for (auto& nvjpeg_param : legacy_batch_nvjpeg_params_)
                XM_NVJPEG_LOG_DESTROY(nvjpegDecodeParamsDestroy(nvjpeg_param));
            if (event_)
                XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
            if (state_)
                XM_NVJPEG_LOG_DESTROY(nvjpegJpegStateDestroy(state_));
            // shared_context_ drops here; if we're the last holder,
            // ~SharedContext tears down the worker pool + nvjpeg handle.
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg decoder - " << e.info());
        } catch (const std::exception& e) {
            // DeviceGuard's constructor can throw std::out_of_range / std::runtime_error;
            // a destructor must not propagate exceptions.
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg decoder - " << e.what());
        } catch (...) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg decoder - unknown exception");
        }
    }

    // C-callable thunk: cast the opaque handle and delete the DecoderImpl.
    nvimgcodecStatus_t DecoderImpl::static_destroy(nvimgcodecDecoder_t decoder) {
        try {
            XM_CHECK_NULL(decoder);
            DecoderImpl* handle = reinterpret_cast<DecoderImpl*>(decoder);
            delete handle;
        } catch (const NvJpegException& e) {
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    // Run the CPU-side host stage of the nvJPEG 3-function HW decode pipeline:
    //   nvjpegJpegStreamParse → nvjpegDecodeJpegHost
    //
    // tryAdoptParsedHandoff() checks whether canDecode() left a parsed
    // nvjpegJpegStream_t in the single-image handoff slot for this exact
    // (image*, code_stream*) pair.  A hit moves that stream into the worker's
    // decode slot; a miss parses the JPEG in this host stage.
    //
    // The worker has two decode slots (slot 0 and slot 1); advancing here chooses
    // the one NOT currently in use by the GPU device stage, so its
    // nvjpegJpegStream_t and pinned buffer can be written safely.
    bool DecoderImpl::decodeSingleHostStage(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params,
        SharedContext::DecodeWorker& worker) {
        int slot_id = worker.advanceDecodeSlot();
        auto& slot = shared_context_->decode_slots_[slot_id];
        // Attempt to reuse the parsed stream that canDecode() handed off for this image.
        // On success, slot now holds the cached parsed nvjpegJpegStream_t.
        // On miss, the slot remains available for the parse below.
        bool reused_parsed_stream = tryAdoptParsedHandoff(image, code_stream, slot_id);

        try {
            XM_CHECK_NULL(image);
            XM_CHECK_NULL(code_stream);
            XM_CHECK_NULL(params);
            assert(code_stream->io_stream);

            void* jpeg_raw = nullptr;
            size_t jpeg_size = 0;
            auto status = code_stream->io_stream->size(code_stream->io_stream->instance, &jpeg_size);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return false;
            }
            status = code_stream->io_stream->map(code_stream->io_stream->instance, &jpeg_raw, 0, jpeg_size);
            if (status != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return false;
            }
            auto ret = image->getImageInfo(image->instance, &slot.image_info_);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return false;
            }
            ret = code_stream->getImageInfo(code_stream->instance, &slot.cs_image_info_);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return false;
            }
            ret = code_stream->getCodeStreamInfo(code_stream->instance, &slot.codestream_info_);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return false;
            }

            slot.image_ = image;

            int num_channels = std::max(slot.image_info_.num_planes, slot.image_info_.plane_info[0].num_channels);
            slot.sample_format_ = num_channels == 1 ? NVIMGCODEC_SAMPLEFORMAT_P_Y : slot.image_info_.sample_format;
            auto nvjpeg_format = nvimgcodec_to_nvjpeg_format(slot.sample_format_);
            auto decode_params = slot.decode_params_;
            XM_CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(decode_params, nvjpeg_format));
            int allow_cmyk = (slot.image_info_.color_spec != NVIMGCODEC_COLORSPEC_UNCHANGED) &&
                             (slot.image_info_.color_spec != NVIMGCODEC_COLORSPEC_CMYK) &&
                             (slot.image_info_.color_spec != NVIMGCODEC_COLORSPEC_YCCK);
            XM_CHECK_NVJPEG(nvjpegDecodeParamsSetAllowCMYK(decode_params, allow_cmyk));

            nvimgcodecProcessingStatus_t processing_status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
            bool need_params = false;
            if (!setDecodeParams(need_params, processing_status, decode_params, code_stream, slot.image_info_, slot.cs_image_info_, params)) {
                image->imageReady(image->instance, processing_status);
                return false;
            }

            const auto [roi_image_width, roi_image_height] = nvimgcodec::oriented_dims(
                slot.cs_image_info_.plane_info[0].width, slot.cs_image_info_.plane_info[0].height,
                params->apply_exif_orientation ? slot.image_info_.orientation
                                               : nvimgcodec::kIdentityOrientation);
            slot.roi_image_width_ = roi_image_width;
            slot.roi_image_height_ = roi_image_height;
            slot.has_oob_roi_ = slot.codestream_info_.code_stream_view &&
                                slot.codestream_info_.code_stream_view->region.ndim != 0 &&
                                nvimgcodec::is_region_out_of_bounds(
                                    slot.codestream_info_.code_stream_view->region,
                                    slot.roi_image_width_,
                                    slot.roi_image_height_);

            if (!reused_parsed_stream) {
                // No cached parse available: parse the full JPEG bitstream now.
                // Multi-sample BatchSingle and forced BatchSingle requests usually
                // get here: canDecode stays header-only or does not use the
                // single-slot handoff for those samples.
                XM_CHECK_NVJPEG(nvjpegJpegStreamParse(handle_, static_cast<const unsigned char*>(jpeg_raw), jpeg_size, false, false, slot.nvjpeg_stream_));
            }
            // slot.nvjpeg_stream_ now holds a valid parsed JPEG (either reused
            // from canDecode or freshly parsed above).  Run the CPU host stage.
            XM_CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(worker.state_, slot.pinned_buffer_));
            XM_CHECK_NVJPEG(nvjpegDecodeJpegHost(handle_, worker.decoder_, worker.state_, decode_params, slot.nvjpeg_stream_));
            return true;
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not run host stage for HW per-image decode path - " << e.info());
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not run host stage for HW per-image decode path - " << e.what());
        }
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return false;
    }

    // Run the GPU-side stages of the 3-function pipeline on the worker's stream:
    //   nvjpegDecodeJpegTransferToDevice → nvjpegDecodeJpegDevice
    // then optionally convert P_YUV → I_YUV and fill any out-of-bounds ROI region.
    // Records decode_done_event_ after all GPU work (even on failure) so that the
    // caller's stream can safely order buffer frees after this decode completes.
    void DecoderImpl::decodeSingleDeviceStage(SharedContext::DecodeWorker& t, bool host_stage_succeeded) {
        // Wait for prior GPU work on this worker before reusing its stream,
        // device buffer, or either double-buffer slot. The host stage advances the
        // slot before it can fail, so even a host-stage failure must drain the
        // worker stream before this worker is reused.
        XM_CHECK_CUDA(cudaStreamSynchronize(t.stream_));
        if (!host_stage_succeeded)
            return;

        auto& slot = shared_context_->decode_slots_[t.currentDecodeSlot()];

        try {
            auto working_image_info = slot.image_info_;
            working_image_info.cuda_stream = t.stream_;
            auto decoded_image_info = working_image_info;
            auto nvjpeg_format = nvimgcodec_to_nvjpeg_format(slot.sample_format_);

            // nvJPEG always outputs YUV in planar format (NVJPEG_OUTPUT_YUV).
            // If the caller requested interleaved I_YUV we decode into a temporary
            // planar buffer first, then convert with LaunchConvertNormKernel below.
            if (slot.sample_format_ == NVIMGCODEC_SAMPLEFORMAT_I_YUV) {
                assert(nvjpeg_format == NVJPEG_OUTPUT_YUV);
                decoded_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_YUV;
                decoded_image_info.num_planes = slot.image_info_.plane_info[0].num_channels;
                for (unsigned int c = 0; c < decoded_image_info.num_planes; ++c) {
                    auto& plane_info = decoded_image_info.plane_info[c];
                    plane_info = slot.image_info_.plane_info[0];
                    plane_info.num_channels = 1;
                    plane_info.row_stride = static_cast<size_t>(plane_info.width) * nvimgcodec::TypeSize(plane_info.sample_type);
                }
                assert(nvimgcodec::GetBufferSize(decoded_image_info) == nvimgcodec::GetImageSize(decoded_image_info));
                t.helper_device_buffer_.resize(nvimgcodec::GetBufferSize(decoded_image_info), t.stream_);
                decoded_image_info.buffer = t.helper_device_buffer_.data;
            }

            nvjpegImage_t nvjpeg_image{};
            auto* ptr = reinterpret_cast<unsigned char*>(decoded_image_info.buffer);
            for (uint32_t c = 0; c < decoded_image_info.num_planes; ++c) {
                nvjpeg_image.channel[c] = ptr;
                nvjpeg_image.pitch[c] = decoded_image_info.plane_info[c].row_stride;
                ptr += nvjpeg_image.pitch[c] * decoded_image_info.plane_info[c].height;

                if (slot.codestream_info_.code_stream_view && slot.codestream_info_.code_stream_view->region.ndim != 0) {
                    int roi_y_begin = slot.codestream_info_.code_stream_view->region.start[0];
                    int roi_x_begin = slot.codestream_info_.code_stream_view->region.start[1];

                    if (roi_y_begin < 0) {
                        nvjpeg_image.channel[c] += (-roi_y_begin) * nvjpeg_image.pitch[c];
                    }
                    if (roi_x_begin < 0) {
                        size_t bytes_per_sample = decoded_image_info.plane_info[c].sample_type >> 11;
                        nvjpeg_image.channel[c] += (-roi_x_begin) * bytes_per_sample * decoded_image_info.plane_info[c].num_channels;
                    }
                }
            }

            XM_CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle_, t.decoder_, t.state_, slot.nvjpeg_stream_, t.stream_));
            XM_CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle_, t.decoder_, t.state_, &nvjpeg_image, t.stream_));

            if (slot.sample_format_ == NVIMGCODEC_SAMPLEFORMAT_I_YUV) {
                nvimgcodec::LaunchConvertNormKernel(working_image_info, decoded_image_info, t.stream_);
            }

            if (slot.has_oob_roi_) {
                // has_oob_roi_ is computed in decodeSingleHostStage as
                //   code_stream_view && code_stream_view->region.ndim != 0 && is_region_out_of_bounds(...)
                // so taking this branch implies code_stream_view != nullptr.  Coverity
                // can't track that invariant across the host/device-stage boundary.
                // coverity[forward_null : FALSE]
                nvimgcodec::fill_out_of_bounds_region(
                    working_image_info,
                    slot.roi_image_width_,
                    slot.roi_image_height_,
                    slot.codestream_info_.code_stream_view->region);
            }

            XM_CHECK_CUDA(cudaEventRecord(t.decode_done_event_, t.stream_));
            slot.image_->imageReady(slot.image_->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
            return;
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not run device stage for HW per-image decode path - " << e.info());
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not run device stage for HW per-image decode path - " << e.what());
        }
        // Record decode_done_event_ even on failure.  Any GPU kernels that were
        // submitted to t.stream_ before the error was detected (e.g. a failing
        // nvjpegDecodeJpegDevice) are still in-flight.  Recording the event here
        // places it AFTER those kernels so that the cudaStreamWaitEvent in
        // decodeBatchSingle correctly orders cudaFreeAsync (for this image's buffer)
        // after the GPU finishes — preventing a use-after-free / CUDA #700.
        XM_CUDA_LOG_DESTROY(cudaEventRecord(t.decode_done_event_, t.stream_));
        slot.image_->imageReady(slot.image_->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
    }

    // Decode a whole batch using the per-image 3-function pipeline (decode path 2).
    // Leases up to num_hw_engines_ workers from the shared pool, then submits
    // host_threads executor tasks; each task owns a strided slice of the leased
    // workers and drains sample indices from a shared atomic counter.
    // Double-buffering within each worker allows CPU and GPU stages to overlap.
    nvimgcodecStatus_t DecoderImpl::decodeBatchSingle(
        const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecDecodeParams_t* params) {
        struct WorkerLease {
            std::shared_ptr<SharedContext> ctx;
            std::vector<int> worker_ids;
            // RAII for decode_api=single on a batch: lease a fixed worker subset for
            // the lifetime of the minibatch and release it only after all tasks finish.
            ~WorkerLease() {
                if (ctx && !worker_ids.empty()) {
                    ctx->releaseWorkers(worker_ids);
                }
            }
        };

        // Immutable batch-scoped state shared (by pointer) across all executor tasks.
        struct BatchContext {
            DecoderImpl* decoder;
            const nvimgcodecImageDesc_t** images;
            const nvimgcodecCodeStreamDesc_t** code_streams;
            const nvimgcodecDecodeParams_t* params;
            const std::vector<int>* sample_idxs;
            int sample_count;
            int host_threads;
            int total_states;
            cudaEvent_t batch_ready_event;
            const std::vector<int>* worker_ids;
            std::vector<uint8_t>* image_ready;
            std::atomic<int> next_sample_idx{0};
        };

        // Mirror the batched path's event contract: block the CPU until the previous
        // decodeBatchSingle call's GPU writes are complete, so the caller can safely
        // reuse output buffers across consecutive calls.  On the very first call,
        // event_ is freshly created (pre-signaled), so this returns immediately.
        XM_CHECK_CUDA(cudaEventSynchronize(event_));

        batch_single_valid_sample_idxs_.clear();
        batch_single_image_ready_.resize(batch_size);
        std::fill(batch_single_image_ready_.begin(), batch_single_image_ready_.end(), 0);

        struct BatchSingleImageReadyGuard {
            const nvimgcodecImageDesc_t** images;
            std::vector<uint8_t>* image_ready;
            bool active = true;

            ~BatchSingleImageReadyGuard() {
                if (!active)
                    return;
                for (size_t sample_idx = 0; sample_idx < image_ready->size(); ++sample_idx) {
                    if ((*image_ready)[sample_idx])
                        continue;
                    if (images[sample_idx]) {
                        images[sample_idx]->imageReady(images[sample_idx]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    }
                }
            }
        } image_ready_guard{images, &batch_single_image_ready_};

        cudaStream_t batch_stream = 0;
        bool batch_stream_initialized = false;
        for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            auto ret = images[sample_idx]->getImageInfo(images[sample_idx]->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                images[sample_idx]->imageReady(images[sample_idx]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                batch_single_image_ready_[sample_idx] = 1;
                continue;
            }
            batch_single_valid_sample_idxs_.push_back(sample_idx);
            if (!batch_stream_initialized) {
                batch_stream = image_info.cuda_stream;
                batch_stream_initialized = true;
            } else if (batch_stream != image_info.cuda_stream) {
                throw std::logic_error("Expected the same CUDA stream for all samples in the HW batch-single decode path");
            }
        }
        if (!batch_stream_initialized) {
            image_ready_guard.active = false;
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        // Determine the target amount of host-side and worker-side parallelism.
        // We cap both at the valid sample count so we don't create more parallelism
        // than decodable images.
        int valid_sample_count = static_cast<int>(batch_single_valid_sample_idxs_.size());
        auto executor = exec_params_->executor;
        assert(executor);
        int host_threads = executor->getNumThreads(executor->instance);
        host_threads = std::max(1, std::min(valid_sample_count, host_threads));
        int requested_states = std::min(valid_sample_count, static_cast<int>(num_hw_engines_));
        WorkerLease lease{shared_context_, shared_context_->acquireWorkersUpTo(requested_states)};
        int total_states = static_cast<int>(lease.worker_ids.size());
        assert(total_states > 0);
        host_threads = std::min(host_threads, total_states);
        BatchContext context{
            this,
            images,
            code_streams,
            params,
            &batch_single_valid_sample_idxs_,
            valid_sample_count,
            host_threads,
            total_states,
            event_,
            &lease.worker_ids,
            &batch_single_image_ready_};

        // Record the caller's stream position so every worker stream can wait on it
        // before touching output buffers (same pattern as decodeSingleImageImpl).
        XM_CHECK_CUDA(cudaEventRecord(event_, batch_stream));

        // Each executor task (identified by task_idx in [0, host_threads)) owns a
        // strided subset of the leased workers: task 0 gets workers 0, host_threads,
        // 2*host_threads, …; task 1 gets workers 1, 1+host_threads, …; etc.
        //
        // Samples are assigned dynamically through a shared atomic counter, which
        // spreads work across tasks when image sizes are clustered in the input
        // batch and lets faster tasks keep pulling work until the batch is exhausted.
        auto worker = [](int, int task_idx, void* task_context) {
            auto& ctx = *reinterpret_cast<BatchContext*>(task_context);
            if (task_idx >= ctx.total_states) {
                return;
            }

            auto for_each_task_worker = [&](auto&& fn) {
                for (int state_idx = task_idx; state_idx < ctx.total_states; state_idx += ctx.host_threads) {
                    fn((*ctx.worker_ids)[state_idx]);
                }
            };

            // Scope guard: on abnormal exit (exception inside the loop below), record
            // decode_done_event_ on every leased worker stream so the caller-side join
            // in decodeBatchSingle still sees a completed event and doesn't deadlock.
            // Disarmed at the normal exit point below.
            struct DecodeDoneEventGuard {
                BatchContext& ctx;
                int task_idx;
                bool active = true;
                ~DecodeDoneEventGuard() {
                    if (!active)
                        return;
                    // Aliases so XM_CUDA_LOG_DESTROY can resolve framework_/plugin_id_.
                    auto* framework_ = ctx.decoder->framework_;
                    auto* plugin_id_ = ctx.decoder->plugin_id_;
                    for (int state_idx = task_idx; state_idx < ctx.total_states; state_idx += ctx.host_threads) {
                        int worker_id = (*ctx.worker_ids)[state_idx];
                        auto& t = ctx.decoder->shared_context_->workers_[worker_id];
                        // Best-effort during abnormal exit; a destructor must not throw
                        // and the CUDA context may already be in an error state.
                        XM_CUDA_LOG_DESTROY(cudaEventRecord(t.decode_done_event_, t.stream_));
                    }
                }
            } decode_done_guard{ctx, task_idx};

            // decode_api=single is still one batch submission from nvimgcodec's
            // point of view. Each task gets a subset of leased states and pulls the
            // next sample index from the shared counter until the batch is drained.
            for_each_task_worker([&](int worker_id) {
                auto& t = ctx.decoder->shared_context_->workers_[worker_id];
                XM_CHECK_CUDA(cudaStreamWaitEvent(t.stream_, ctx.batch_ready_event));
            });

            int state_idx = task_idx;
            while (true) {
                int sample_pos = ctx.next_sample_idx.fetch_add(1, std::memory_order_relaxed);
                if (sample_pos >= ctx.sample_count) {
                    break;
                }
                int sample_idx = (*ctx.sample_idxs)[sample_pos];
                int worker_id = (*ctx.worker_ids)[state_idx];
                state_idx += ctx.host_threads;
                if (state_idx >= ctx.total_states) {
                    state_idx = task_idx;
                }
                auto& t = ctx.decoder->shared_context_->workers_[worker_id];
                bool host_ok = ctx.decoder->decodeSingleHostStage(
                    ctx.images[sample_idx], ctx.code_streams[sample_idx], ctx.params, t);
                if (!host_ok) {
                    // decodeSingleHostStage reports imageReady(FAIL) before
                    // returning false. Mark before the required stream drain below,
                    // because that drain can itself observe a CUDA error.
                    (*ctx.image_ready)[sample_idx] = 1;
                }

                // Synchronizes before reusing the worker stream/device buffer, even
                // when host stage failed and already signaled imageReady(FAIL).
                ctx.decoder->decodeSingleDeviceStage(t, host_ok);
                if (host_ok) {
                    // On the host-success path, decodeSingleDeviceStage is the
                    // imageReady(SUCCESS/FAIL) reporting site.
                    (*ctx.image_ready)[sample_idx] = 1;
                }
            }

            for_each_task_worker([&](int worker_id) {
                auto& t = ctx.decoder->shared_context_->workers_[worker_id];
                XM_CHECK_CUDA(cudaEventRecord(t.decode_done_event_, t.stream_));
            });
            decode_done_guard.active = false;
        };

        bool executor_failed = false;
        std::exception_ptr executor_exception;
        int scheduled_tasks = 0;
        for (int task_idx = 0; task_idx < host_threads; ++task_idx) {
            auto ret = executor->schedule(executor->instance, exec_params_->device_id, task_idx, &context, worker);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                break;
            }
            ++scheduled_tasks;
        }
        if (scheduled_tasks == 0) {
            throw std::runtime_error("Failed to schedule HW batch-single decode task");
        }
        if (scheduled_tasks != host_threads) {
            // Keep already-queued tasks safe by running them while BatchContext and
            // WorkerLease are still alive.  Tasks pull samples from next_sample_idx,
            // so fewer tasks can still drain the whole batch, just with less CPU
            // parallelism.
            NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Scheduled only " << scheduled_tasks << " of " << host_threads << " HW batch-single decode tasks; continuing with fewer host tasks");
            context.host_threads = scheduled_tasks;
        }
        auto ret = executor->run(executor->instance, exec_params_->device_id);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to run HW batch-single decode tasks");
        }

        ret = executor->wait(executor->instance, exec_params_->device_id);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            executor_failed = true;
            executor_exception = std::make_exception_ptr(
                std::runtime_error("Failed to wait for HW batch-single decode tasks"));
        }

        // Always submit the stream-wait and event-record on batch_stream, even when
        // the executor failed.  If a worker threw (e.g. cudaStreamSynchronize returned
        // a CUDA error after a failing nvjpegDecodeJpegDevice), some worker streams may
        // still have in-flight GPU kernels that write to image buffers.  The
        // cudaStreamWaitEvent calls below order those writes before any cudaFreeAsync
        // that the caller will submit on batch_stream when it destroys the failed images,
        // preventing a use-after-free / CUDA #700.  On failure the event records are
        // best-effort: we ignore their return codes since the context may already be
        // in an error state.
        for (int worker_id : *context.worker_ids) {
            XM_CUDA_LOG_DESTROY(cudaStreamWaitEvent(batch_stream, shared_context_->workers_[worker_id].decode_done_event_));
        }
        // Record event_ after all decode-done waits so the next call's
        // cudaEventSynchronize(event_) fires only after this batch's GPU writes complete.
        XM_CUDA_LOG_DESTROY(cudaEventRecord(event_, batch_stream));

        if (executor_failed) {
            std::rethrow_exception(executor_exception);
        }
        image_ready_guard.active = false;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    // Report the preferred minibatch size to the framework so it can pack batches
    // that keep all HW engine cores busy: engines * cores_per_engine.
    nvimgcodecStatus_t DecoderImpl::getMiniBatchSize(int* batch_size) {
        *batch_size = num_hw_engines_ * num_cores_per_hw_engine_;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    // Framework batch entry point.  Dispatches via chooseDecodePath(batch_size, code_streams).
    nvimgcodecStatus_t DecoderImpl::decodeBatch(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams,
                                                int batch_size, const nvimgcodecDecodeParams_t* params, int thread_idx) {
        if (thread_idx != 0) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Logic error: Implementation not multithreaded");
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
        // Release the single-image parsed-stream handoff and advance the setup
        // stats epoch when decodeBatch returns. Per-thread parser states are
        // persistent scratch for canDecode() and are not batch-owned.
        struct ResetGuard {
            DecoderImpl* decoder;
            ~ResetGuard() { decoder->finishDecodeBatchSetupState(); }
        } reset_guard{this};
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_hw_decode_batch, " << batch_size << " samples");
            XM_CHECK_NULL(code_streams);
            XM_CHECK_NULL(images)
            XM_CHECK_NULL(params)
            if (batch_size < 1) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }

            // Both single- and multi-sample batches dispatch through
            // chooseDecodePath so it stays the single source of truth. On
            // single-engine GPUs send batch_size==1 down the legacy path;
            // multi-engine GPUs keep the BatchSingle fast path.
            switch (chooseDecodePath(batch_size, code_streams)) {
            case DecodePath::BatchSingle:
                if (batch_size == 1) {
                    return decodeSingleImageImpl(images[0], code_streams[0], params);
                }
                NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "Using HW batch-single decode path for batch of " << batch_size << " samples");
                try {
                    return decodeBatchSingle(images, code_streams, batch_size, params);
                } catch (const NvJpegException& e) {
                    // decodeBatchSingle reports/cleans up per-sample imageReady
                    // state itself.  Catch here so the generic batch catch below
                    // does not send a second FAIL to samples that already reported.
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg batch-single batch - " << e.info());
                    return e.nvimgcodecStatus();
                } catch (const std::exception& e) {
                    // Same as above: BatchSingle owns per-sample reporting once
                    // it starts, including exceptional exits from executor tasks.
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg batch-single batch - " << e.what());
                    return NVIMGCODEC_STATUS_INTERNAL_ERROR;
                }
            case DecodePath::Legacy:
                break; // Continue to nvjpegDecodeBatched* below.
            }

            legacy_batch_bitstreams_.clear();
            legacy_batch_bitstream_sizes_.clear();
            legacy_batch_outputs_.clear();
            legacy_batch_valid_sample_idxs_.clear();
            legacy_batch_decoded_image_info_.clear();

            nvjpegOutputFormat_t nvjpeg_format = NVJPEG_OUTPUT_UNCHANGED;
            bool need_params = false;
            bool need_convert_from_planar = false;

            legacy_batch_nvjpeg_params_.reserve(batch_size);
            while (static_cast<int>(legacy_batch_nvjpeg_params_.size()) < batch_size) {
                legacy_batch_nvjpeg_params_.emplace_back();
                nvjpegDecodeParamsCreate(handle_, &legacy_batch_nvjpeg_params_.back());
            }

            legacy_batch_helper_buffers_.reserve(batch_size);
            while (static_cast<int>(legacy_batch_helper_buffers_.size()) < batch_size) {
                legacy_batch_helper_buffers_.emplace_back(exec_params_ ? exec_params_->device_allocator : nullptr);
            }

            // bool pageable = false;
            cudaStream_t stream{0};
            // The batch-level stream/output-format invariants are initialized from
            // the first sample accepted for nvjpegDecodeBatched*, not raw sample 0.
            bool batch_properties_initialized = false;
            for (int sample_idx = 0, i = 0; sample_idx < batch_size; sample_idx++) {
                auto* image = images[sample_idx];
                XM_CHECK_NULL(image);

                nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                auto ret = image->getImageInfo(image->instance, &image_info);
                if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    continue;
                }

                XM_CHECK_NULL(code_streams[sample_idx]);
                const auto* code_stream = code_streams[sample_idx];
                assert(code_stream->io_stream);
                void* encoded_stream_data = nullptr;
                size_t encoded_stream_data_size = 0;
                if (code_stream->io_stream->size(code_stream->io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                    continue;
                }
                if (code_stream->io_stream->map(code_stream->io_stream->instance, &encoded_stream_data, 0, encoded_stream_data_size) !=
                    NVIMGCODEC_STATUS_SUCCESS) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                    continue;
                }
                assert(encoded_stream_data != nullptr);
                assert(encoded_stream_data_size > 0);

                nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                ret = code_stream->getImageInfo(code_stream->instance, &cs_image_info);
                if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    continue;
                }

                nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
                ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
                if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    continue;
                }

                // cudaPointerAttributes attributes;
                // pageable |= cudaPointerGetAttributes(&attributes, encoded_stream_data) == cudaErrorInvalidValue ||
                //             attributes.type == cudaMemoryTypeUnregistered;

                nvimgcodecProcessingStatus_t tmp_status;
                bool need_params_tmp = false;
                if (!setDecodeParams(need_params_tmp, tmp_status, legacy_batch_nvjpeg_params_[i], code_stream, image_info, cs_image_info, params)) {
                    image->imageReady(image->instance, tmp_status);
                    continue;
                }
                need_params |= need_params_tmp;

                auto sample_nvjpeg_format = nvimgcodec_to_nvjpeg_format(image_info.sample_format);
                if (!batch_properties_initialized) {
                    nvjpeg_format = sample_nvjpeg_format;
                    stream = image_info.cuda_stream;
                    batch_properties_initialized = true;
                } else {
                    if (stream != image_info.cuda_stream) {
                        throw std::logic_error("Expected the same CUDA stream for all the samples in the minibatch (" + std::to_string((uint64_t)stream) + "!=" + std::to_string((uint64_t)image_info.cuda_stream) + ")");
                    }
                    if (nvjpeg_format != sample_nvjpeg_format) {
                        throw std::logic_error("Expected the same format for all the samples in the minibatch");
                    }
                }

                // Image info that will be used for nvjpeg decoding.
                // nvJPEG can only decode yuv to planar, we need to decode to helper buffer and then convert to interleaved
                // for other cases it will be just equal to image info provided by user
                auto decoded_image_info = image_info;
                if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_YUV) {
                    assert(nvjpeg_format == NVJPEG_OUTPUT_YUV);
                    need_convert_from_planar = true;

                    decoded_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_YUV;
                    decoded_image_info.num_planes = image_info.plane_info[0].num_channels;
                    for (unsigned int c = 0; c < decoded_image_info.num_planes; ++c) {
                        auto& plane_info = decoded_image_info.plane_info[c];
                        plane_info = image_info.plane_info[0];
                        plane_info.num_channels = 1;
                        plane_info.row_stride = static_cast<size_t>(plane_info.width) * nvimgcodec::TypeSize(plane_info.sample_type);
                    }
                    assert(nvimgcodec::GetBufferSize(decoded_image_info) == nvimgcodec::GetImageSize(decoded_image_info));
                    legacy_batch_helper_buffers_[i].resize(nvimgcodec::GetBufferSize(decoded_image_info), image_info.cuda_stream);
                    decoded_image_info.buffer = legacy_batch_helper_buffers_[i].data;
                    legacy_batch_decoded_image_info_.push_back(decoded_image_info);
                }

                // get output image
                nvjpegImage_t nvjpeg_image;
                unsigned char* ptr = reinterpret_cast<unsigned char*>(decoded_image_info.buffer);
                ;
                for (uint32_t c = 0; c < decoded_image_info.num_planes; ++c) {
                    nvjpeg_image.channel[c] = ptr;
                    nvjpeg_image.pitch[c] = decoded_image_info.plane_info[c].row_stride;
                    ptr += nvjpeg_image.pitch[c] * decoded_image_info.plane_info[c].height;

                    if (codestream_info.code_stream_view && codestream_info.code_stream_view->region.ndim != 0) {
                        int roi_y_begin = codestream_info.code_stream_view->region.start[0];
                        int roi_x_begin = codestream_info.code_stream_view->region.start[1];

                        if (roi_y_begin < 0) {
                            nvjpeg_image.channel[c] += (-roi_y_begin) * nvjpeg_image.pitch[c];
                        }

                        if (roi_x_begin < 0) {
                            size_t bytes_per_sample = decoded_image_info.plane_info[c].sample_type >> 11;
                            nvjpeg_image.channel[c] += (-roi_x_begin) * bytes_per_sample * decoded_image_info.plane_info[c].num_channels;
                        }
                    }
                }

                legacy_batch_bitstreams_.push_back(static_cast<const unsigned char*>(encoded_stream_data));
                legacy_batch_bitstream_sizes_.push_back(encoded_stream_data_size);
                legacy_batch_outputs_.push_back(nvjpeg_image);
                legacy_batch_valid_sample_idxs_.push_back(sample_idx);
                i++;
            }
            if (legacy_batch_bitstreams_.size() > 0) {
                // Synchronize with previous iteration
                XM_CHECK_CUDA(cudaEventSynchronize(event_));
                XM_CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(handle_, state_, legacy_batch_bitstreams_.size(), 1, nvjpeg_format));

                if (has_batched_ex_api_) {
                    nvtx3::scoped_range marker{"nvjpegDecodeBatchedEx"};
                    XM_CHECK_NVJPEG(nvjpegDecodeBatchedEx(handle_, state_, legacy_batch_bitstreams_.data(), legacy_batch_bitstream_sizes_.data(),
                                                          legacy_batch_outputs_.data(), legacy_batch_nvjpeg_params_.data(), stream));
                } else {
                    if (need_params)
                        throw std::logic_error("Unexpected error");
                    nvtx3::scoped_range marker{"nvjpegDecodeBatched"};
                    XM_CHECK_NVJPEG(nvjpegDecodeBatched(
                        handle_, state_, legacy_batch_bitstreams_.data(), legacy_batch_bitstream_sizes_.data(), legacy_batch_outputs_.data(), stream));
                }

                // set status to success, it may be overwritten to fail if post processing (OOB ROI or YUV planar conversion) fails
                for (int sample_idx : legacy_batch_valid_sample_idxs_) {
                    images[sample_idx]->imageReady(images[sample_idx]->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
                }

                if (need_convert_from_planar) {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "convert from planar to interleaved");

                    for (size_t i = 0; i < legacy_batch_decoded_image_info_.size(); ++i) {
                        auto sample_idx = legacy_batch_valid_sample_idxs_[i];
                        auto* image = images[sample_idx];
                        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                        auto ret = image->getImageInfo(image->instance, &image_info);
                        assert(ret == NVIMGCODEC_STATUS_SUCCESS);

                        try {
                            nvimgcodec::LaunchConvertNormKernel(image_info, legacy_batch_decoded_image_info_[i], image_info.cuda_stream);
                        } catch (std::runtime_error& e) {
                            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not convert from planar to interleaved for sample " << sample_idx << " - " << e.what());
                            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                        }
                    }
                }

                for (auto sample_idx : legacy_batch_valid_sample_idxs_) {
                    const auto* code_stream = code_streams[sample_idx];
                    nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
                    auto ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
                    assert(ret == NVIMGCODEC_STATUS_SUCCESS);

                    nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                    ret = code_stream->getImageInfo(code_stream->instance, &cs_image_info);
                    assert(ret == NVIMGCODEC_STATUS_SUCCESS);

                    // Look up the orientation for this sample so we can interpret the region in
                    // the same coord system as the output buffer (display when apply_exif=true).
                    nvimgcodecImageInfo_t image_info_for_orient{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                    ret = images[sample_idx]->getImageInfo(images[sample_idx]->instance, &image_info_for_orient);
                    assert(ret == NVIMGCODEC_STATUS_SUCCESS);
                    const auto [eff_w, eff_h] = nvimgcodec::oriented_dims(
                        cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height,
                        params->apply_exif_orientation ? image_info_for_orient.orientation
                                                       : nvimgcodec::kIdentityOrientation);

                    bool has_oob_roi = codestream_info.code_stream_view && codestream_info.code_stream_view->region.ndim != 0 &&
                                       nvimgcodec::is_region_out_of_bounds(codestream_info.code_stream_view->region, eff_w, eff_h);

                    if (has_oob_roi) {
                        nvtx3::scoped_range marker{"fill_out_of_bounds_region"};
                        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "fill_out_of_bounds_region for sample: " << sample_idx);

                        auto* image = images[sample_idx];
                        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                        ret = image->getImageInfo(image->instance, &image_info);
                        assert(ret == NVIMGCODEC_STATUS_SUCCESS);

                        try {
                            nvimgcodec::fill_out_of_bounds_region(
                                image_info,
                                eff_w, eff_h,
                                codestream_info.code_stream_view->region);
                        } catch (std::runtime_error& e) {
                            NVIMGCODEC_LOG_ERROR(
                                framework_, plugin_id_,
                                "Could not fill out of bounds ROI for sample " << sample_idx << " - " << e.what());
                            images[sample_idx]->imageReady(images[sample_idx]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                        }
                    }
                }

                XM_CHECK_CUDA(cudaEventRecord(event_, stream));
            }
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg batch - " << e.info());
            for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
                images[sample_idx]->imageReady(images[sample_idx]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return e.nvimgcodecStatus();
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg batch - " << e.what());
            for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
                images[sample_idx]->imageReady(images[sample_idx]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
    }

} // namespace nvjpeg
