/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "Intersect.h"
#include "Projection.h"
#include "Rasterization.h"
#include "SphericalHarmonics.h"
#include "hip_check.hpp"
#include "photometric.hpp"
#include "training_api.h"
#include "application_dispatch_api.h"
#ifdef LFS_HIP_FAST_RUNTIME
#include "backward.h"
#include "forward.h"
#endif
#include <hipcub/hipcub.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <vector>
#ifdef _WIN32
extern "C" int lfs_hip_adapter_pci(const uint8_t luid[8], uint32_t address[3]);
#endif

namespace gsplat_lfs {
    void compute_view_dirs(const float*, const float*, uint32_t, uint32_t, float*, hipStream_t);
}

namespace {
    LfsHipApplicationDispatch dispatch_callbacks{};
    std::atomic<const LfsHipApplicationDispatch*> installed_dispatch{nullptr};
    std::mutex dispatch_install_mutex;
    struct DispatchCall {
        const LfsHipApplicationDispatch* callbacks=nullptr;
        void begin(LfsHipError* error) {
            auto* value=installed_dispatch.load(std::memory_order_acquire);
            if(!value)return;
            if(value->begin(value->user,error))
                throw std::runtime_error(error?error->message:"HIP shared access rejected submission");
            callbacks=value;
        }
        ~DispatchCall(){if(callbacks)callbacks->end(callbacks->user);}
    };
    constexpr float sh_c0 = 0.28209479177387814f;
    constexpr std::array<unsigned, 5> attributes{3, 4, 3, 3, 1};

    struct Failure : std::runtime_error {
        int status;
        Failure(int status, const char* message) : std::runtime_error(message), status(status) {}
    };
    void require(bool condition, const char* message) {
        if (!condition)
            throw Failure(LFS_HIP_TRAIN_INVALID, message);
    }
    size_t multiply(size_t a, size_t b) {
        require(!b || a <= SIZE_MAX / b, "HIP training size overflow");
        return a * b;
    }
    struct DeviceScope {
        int previous = 0;
        explicit DeviceScope(int selected) {
            hip_check(hipGetDevice(&previous), "current device");
            hip_check(hipSetDevice(selected), "select training device");
        }
        ~DeviceScope() { (void)hipSetDevice(previous); }
    };
    struct Stream {
        hipStream_t value = nullptr;
        Stream() { hip_check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking), "training stream"); }
        ~Stream() { (void)hipStreamDestroy(value); }
    };
    struct Budget {
        uint64_t limit, live = 0, peak = 0;
    };
    struct Buffer {
        Budget& budget;
        void* pointer = nullptr;
        size_t bytes = 0;
        size_t charged_bytes = 0;
        hipExternalMemory_t external = nullptr;
        bool owns_pointer = true;
        explicit Buffer(Budget& budget, size_t bytes, const LfsHipTrainSharedModel* shared = nullptr, const LfsHipTrainDeviceModel* device = nullptr)
            : budget(budget), bytes(bytes), charged_bytes(device ? device->allocation_bytes : shared ? shared->allocation_bytes
                                                                                                     : bytes) {
            if (charged_bytes > budget.limit - budget.live)
                throw Failure(LFS_HIP_TRAIN_RESOURCE_EXHAUSTED, "HIP session allocation budget exhausted");
            if (device) {
                pointer = reinterpret_cast<void*>(static_cast<uintptr_t>(device->address));
                owns_pointer = false;
            } else if (shared) {
#ifdef _WIN32
                hipExternalMemoryHandleDesc descriptor{};
                descriptor.type = hipExternalMemoryHandleTypeOpaqueWin32;
                descriptor.handle.win32.handle = reinterpret_cast<void*>(static_cast<uintptr_t>(shared->win32_handle));
                descriptor.size = shared->allocation_bytes;
                descriptor.flags = hipExternalMemoryDedicated;
                hip_check(hipImportExternalMemory(&external, &descriptor), "HIP shared model import");
                hipExternalMemoryBufferDesc mapping{};
                mapping.size = shared->allocation_bytes;
                const auto result = hipExternalMemoryGetMappedBuffer(&pointer, external, &mapping);
                if (result != hipSuccess) {
                    (void)hipDestroyExternalMemory(external);
                    external = nullptr;
                    hip_check(result, "HIP shared model mapping");
                }
#else
                throw Failure(LFS_HIP_TRAIN_INVALID, "Shared HIP model import currently requires Windows");
#endif
            } else if (bytes)
                hip_check(hipMalloc(&pointer, bytes), "training allocation");
            budget.live += charged_bytes;
            budget.peak = std::max(budget.peak, budget.live);
        }
        ~Buffer() {
            if (pointer && owns_pointer)
                (void)hipFree(pointer);
            if (external)
                (void)hipDestroyExternalMemory(external);
            budget.live -= charged_bytes;
        }
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        template <class T = float>
        T* as(size_t offset = 0) const { return static_cast<T*>(pointer) + offset; }
        void zero(hipStream_t stream) { hip_check(hipMemsetAsync(pointer, 0, bytes, stream), "training clear"); }
        void swap_owned(Buffer& other) {
            require(&budget == &other.budget && owns_pointer && other.owns_pointer && !external && !other.external,
                    "Cannot swap foreign HIP workspace storage");
            std::swap(pointer, other.pointer);
            std::swap(bytes, other.bytes);
            std::swap(charged_bytes, other.charged_bytes);
        }
        void release_owned() {
            require(owns_pointer && !external, "Cannot release borrowed HIP storage");
            if (pointer)
                hip_check(hipFree(pointer), "release replaced HIP optimizer storage");
            budget.live -= charged_bytes;
            pointer = nullptr;
            bytes = charged_bytes = 0;
        }
    };
    struct Layout {
        size_t offsets[5]{};
        size_t floats = 0;
        explicit Layout(size_t count, size_t alignment_floats = 64) {
            for (size_t i = 0; i < attributes.size(); ++i) {
                floats = (floats + alignment_floats - 1) / alignment_floats * alignment_floats;
                offsets[i] = floats;
                floats += multiply(count, attributes[i]);
            }
        }
    };
    Layout external_layout(size_t count, const LfsHipTrainModelLayout& source) {
        require(source.struct_bytes == sizeof(source) && source.required_bytes > 0 &&
                    source.required_bytes <= SIZE_MAX && source.required_bytes % 4 == 0,
                "Invalid HIP external model layout extent");
        Layout result(count);
        for (size_t i = 0; i < attributes.size(); ++i) {
            const uint64_t bytes = multiply(count, attributes[i] * sizeof(float));
            require(source.float_counts[i] == count * attributes[i] && source.offsets_bytes[i] % 256 == 0 &&
                        source.offsets_bytes[i] <= source.required_bytes && bytes <= source.required_bytes - source.offsets_bytes[i],
                    "Invalid HIP external model region");
            for (size_t j = 0; j < i; ++j) {
                const uint64_t previous_end = source.offsets_bytes[j] + source.float_counts[j] * sizeof(float);
                require(source.offsets_bytes[i] >= previous_end || source.offsets_bytes[i] + bytes <= source.offsets_bytes[j],
                        "Overlapping HIP external model regions");
            }
            result.offsets[i] = source.offsets_bytes[i] / sizeof(float);
        }
        result.floats = source.required_bytes / sizeof(float);
        return result;
    }
    struct Sh {
        unsigned maximum, active;
        size_t bases, slots, storage_floats;
        float rate;
        Buffer values, gradient, first, second, directions, mask;
        Sh(Budget& budget, size_t count, const LfsHipTrainSh& config, const LfsHipTrainDeviceModel* resident = nullptr)
            : maximum(config.max_degree), active(config.active_degree), bases((maximum + 1) * (maximum + 1)), slots(((bases - 1) * 3 + 3) / 4), storage_floats(multiply((count + 31) / 32 * 32, slots * 4)), rate(config.rest_learning_rate), values(budget, multiply(storage_floats, 4), nullptr, resident), gradient(budget, multiply(count, bases * 12)), first(budget, values.bytes), second(budget, values.bytes), directions(budget, multiply(count, 12)), mask(budget, count) {}
    };
    struct Photo {
        Buffer partials, map;
        Photo(Budget& budget, size_t pixels) : partials(budget, multiply(pixels, 18)), map(budget, multiply(pixels, 4)) {}
    };
    struct Frame {
        unsigned width, height, tile_width, tile_height;
        size_t pixels;
        LfsHipTrainFrame descriptor;
        Buffer target, image, alpha, image_gradient, alpha_gradient, last, camera, offsets;
        std::unique_ptr<Photo> photo;
        Frame(Budget& budget, const LfsHipTrainFrame& input)
            : width(input.width), height(input.height), tile_width((width + 15) / 16), tile_height((height + 15) / 16), pixels(multiply(width, height)), descriptor(input), target(budget, multiply(pixels, 12)), image(budget, multiply(pixels, 12)), alpha(budget, multiply(pixels, 4)), image_gradient(budget, multiply(pixels, 12)), alpha_gradient(budget, multiply(pixels, 4)), last(budget, multiply(pixels, 4)), camera(budget, 25 * sizeof(float)), offsets(budget, multiply(size_t(tile_width) * tile_height + 1, 4)) {
            descriptor.target_chw = nullptr;
        }
    };
    struct Bins {
        size_t capacity;
        Buffer keys, sorted_keys, ids, sorted_ids;
        Bins(Budget& budget, size_t capacity)
            : capacity(capacity), keys(budget, multiply(capacity, 8)), sorted_keys(budget, multiply(capacity, 8)), ids(budget, multiply(capacity, 4)), sorted_ids(budget, multiply(capacity, 4)) {}
    };
#ifdef LFS_HIP_FAST_RUNTIME
    struct JointAdam {
        std::array<std::unique_ptr<Buffer>, 5> packed, bounds;
        std::unique_ptr<Buffer> sh_packed, sh_bounds;
        JointAdam(Budget& budget, size_t count, hipStream_t queue) {
            for (unsigned family = 0; family < attributes.size(); ++family) {
                packed[family] = std::make_unique<Buffer>(budget, multiply(count, attributes[family] * 4));
                bounds[family] = std::make_unique<Buffer>(budget, multiply((count + 255) / 256, 16));
                packed[family]->zero(queue);
                bounds[family]->zero(queue);
            }
        }
        void prepare_sh(Budget& budget, const Sh& sh, size_t count, hipStream_t queue) {
            if (sh_packed && sh_packed->bytes == sh.storage_floats * 2)
                return;
            auto packed_replacement = std::make_unique<Buffer>(budget, multiply(sh.storage_floats, 2));
            auto bounds_replacement = std::make_unique<Buffer>(budget, multiply((count + 255) / 256, 16));
            packed_replacement->zero(queue);
            bounds_replacement->zero(queue);
            sh_packed.swap(packed_replacement);
            sh_bounds.swap(bounds_replacement);
        }
    };
    struct FastFrame {
        Budget& budget;
        hipStream_t queue;
        std::vector<std::unique_ptr<Buffer>> buffers;
        size_t cursor = 0;
        char* tiles = nullptr;
        fast_lfs::rasterization::ForwardResult result;
        std::unique_ptr<Buffer> helpers;
        std::unique_ptr<Buffer> sh_gradient;
        float3 camera_center{};
        bool reserve_growth = false;
        bool reuse_phase_helpers = false;
        bool has_growth_reserve = false;
        char* phase_candidate = nullptr;
        size_t phase_candidate_bytes = 0;
        bool depth_presort_allowed = true;
        bool depth_presort_attempted = false;
        FastFrame(Budget& budget, hipStream_t queue) : budget(budget), queue(queue) {
            const char* value = std::getenv("LFS_HIP_FAST_ARENA_GROWTH");
            reserve_growth = !value || std::string_view(value) != "0";
            value = std::getenv("LFS_HIP_FAST_PHASE_HELPERS");
            reuse_phase_helpers = !value || std::string_view(value) != "0";
        }
        char* allocate(size_t bytes) {
            bytes = std::max<size_t>(bytes, 256);
            if (cursor == buffers.size())
                buffers.push_back(std::make_unique<Buffer>(budget, bytes));
            else if (buffers[cursor]->bytes < bytes) {
                // Optional bounded headroom amortizes tiny changes in the
                // intersection count. The old allocation remains charged until
                // replacement succeeds; tight budgets retain exact allocation.
                size_t capacity = bytes;
                const size_t extra = std::min<size_t>(bytes / 16, 8 * 1024 * 1024);
                const uint64_t available = budget.limit - budget.live;
                if (reserve_growth && bytes >= 1024 * 1024 && bytes <= available && extra <= available - bytes)
                    capacity += extra;
                std::unique_ptr<Buffer> replacement;
                try {
                    replacement = std::make_unique<Buffer>(budget, capacity);
                } catch (const Failure& failure) {
                    if (capacity == bytes || failure.status != LFS_HIP_TRAIN_RESOURCE_EXHAUSTED)
                        throw;
                } catch (const LfsHipRasterFailure& failure) {
                    if (capacity == bytes || failure.code != hipErrorOutOfMemory)
                        throw;
                }
                if (!replacement) {
                    const auto pending = hipGetLastError();
                    if (pending != hipSuccess && pending != hipErrorOutOfMemory)
                        hip_check(pending, "HIP optional workspace growth pending error");
                    reserve_growth = false;
                    std::fprintf(stderr, "[warning] HIP optional workspace growth exceeded available memory; "
                                         "retrying exact capacity and disabling extra headroom for this workspace\n");
                    replacement = std::make_unique<Buffer>(budget, bytes);
                }
                has_growth_reserve |= replacement->bytes > bytes;
                buffers[cursor].swap(replacement);
            }
            return static_cast<char*>(buffers[cursor++]->pointer);
        }
        char* allocate_phase(size_t bytes) {
            char* output = allocate(bytes);
            const size_t capacity = buffers[cursor - 1]->bytes;
            if (capacity > phase_candidate_bytes) {
                phase_candidate = output;
                phase_candidate_bytes = capacity;
            }
            return output;
        }
        template<class Submit>
        void forward_with_depth_fallback(size_t model_count, Submit&& submit,
                                         size_t extra_backward_bytes = 0) {
            for (;;) {
                cursor = 0;
                phase_candidate = nullptr;
                phase_candidate_bytes = 0;
                depth_presort_attempted = false;
                try {
                    submit();
                    if (depth_presort_attempted || has_growth_reserve) {
                        // Keep room for the mandatory backward helpers as well.
                        // Otherwise optional forward storage could fit the budget
                        // yet make an otherwise valid training step fail later.
                        const size_t helper_count = reuse_phase_helpers
                                                        ? std::max<size_t>(result.n_visible, 1)
                                                        : model_count;
                        const size_t helper_bytes = multiply(helper_count, 16 * sizeof(float));
                        const bool reuses_phase = reuse_phase_helpers && phase_candidate &&
                                                  phase_candidate_bytes >= helper_bytes;
                        const size_t helper_charge = !reuses_phase &&
                                                             (!helpers || helpers->bytes < helper_bytes)
                                                         ? helper_bytes
                                                         : 0;
                        const uint64_t available = budget.limit - budget.live;
                        if (helper_charge > available || extra_backward_bytes > available - helper_charge)
                            throw Failure(LFS_HIP_TRAIN_RESOURCE_EXHAUSTED,
                                          "HIP optional forward storage leaves insufficient backward workspace budget");
                    }
                    return;
                } catch (const Failure& failure) {
                    if ((!depth_presort_attempted && !has_growth_reserve) || failure.status != LFS_HIP_TRAIN_RESOURCE_EXHAUSTED)
                        throw;
                } catch (const LfsHipRasterFailure& failure) {
                    if ((!depth_presort_attempted && !has_growth_reserve) || failure.code != hipErrorOutOfMemory)
                        throw;
                } catch (const std::bad_alloc&) {
                    if (!depth_presort_attempted && !has_growth_reserve)
                        throw;
                }
                // No optimizer has run in forward. Complete submitted readers
                // before releasing the partial workspace and replay once with
                // the original sort. Screen-share accumulation uses max, so
                // replay does not double its contribution.
                hip_check(hipStreamSynchronize(queue), "HIP depth sort fallback completion");
                const auto pending_error = hipGetLastError();
                if (pending_error != hipSuccess && pending_error != hipErrorOutOfMemory)
                    hip_check(pending_error, "HIP depth sort fallback pending error");
                buffers.clear();
                helpers.reset();
                sh_gradient.reset();
                tiles = nullptr;
                result = {};
                if (depth_presort_attempted) {
                    depth_presort_allowed = false;
                    std::fprintf(stderr, "[warning] HIP visible-depth presort exceeded available memory; "
                                         "retrying with the original sort and disabling presort for this workspace\n");
                }
                if (has_growth_reserve) {
                    reserve_growth = false;
                    has_growth_reserve = false;
                    std::fprintf(stderr, "[warning] HIP workspace headroom leaves insufficient training memory; "
                                         "retrying with exact workspace capacities\n");
                }
            }
        }
        float* backward_helpers(size_t bytes) {
            // Only phase allocations qualify. Visibility, retained indices,
            // tile state and persistent projected attributes survive forward.
            // Forward has completed on this queue before this storage is reused;
            // no phase arrays are read by the backward interface.
            if (reuse_phase_helpers && phase_candidate && phase_candidate_bytes >= bytes) {
                helpers.reset();
                hip_check(hipMemsetAsync(phase_candidate, 0, bytes, queue), "HIP phase helper clear");
                return reinterpret_cast<float*>(phase_candidate);
            }
            if (!helpers || helpers->bytes < bytes) {
                auto replacement = std::make_unique<Buffer>(budget, bytes);
                helpers.swap(replacement);
            }
            helpers->zero(queue);
            return helpers->as();
        }
    };
#endif
    struct ProfileEvent {
        hipEvent_t value = nullptr;
        ProfileEvent() { hip_check(hipEventCreate(&value), "HIP profiling event"); }
        ~ProfileEvent() {
            if (value)
                (void)hipEventDestroy(value);
        }
        ProfileEvent(const ProfileEvent&) = delete;
        ProfileEvent& operator=(const ProfileEvent&) = delete;
    };
    struct Profile {
        std::array<ProfileEvent, 9> events;
        void mark(unsigned i, hipStream_t stream) {
            // Diagnostic mode intentionally completes each boundary. This makes
            // attribution unambiguous even across runtime allocation fences.
            hip_check(hipStreamSynchronize(stream), "HIP synchronized profiling boundary");
            hip_check(hipEventRecord(events[i].value, stream), "HIP profiling marker");
            hip_check(hipEventSynchronize(events[i].value), "HIP profiling marker completion");
        }
        void print(uint64_t step) {
            constexpr const char* phases[]{"projection", "count_scan", "binning_sort", "forward", "gradient_clear", "loss_gradient", "backward", "adam"};
            std::fprintf(stderr, "[profile] HIP step=%llu", static_cast<unsigned long long>(step));
            for (unsigned i = 0; i < 8; ++i) {
                float ms = 0;
                hip_check(hipEventElapsedTime(&ms, events[i].value, events[i + 1].value), "HIP profiling elapsed time");
                std::fprintf(stderr, " %s_ms=%.6f", phases[i], ms);
            }
            std::fputc('\n', stderr);
        }
    };
    struct Widen {
        __host__ __device__ int64_t operator()(int32_t value) const { return value; }
    };
    using CountIterator = hipcub::TransformInputIterator<int64_t, Widen, const int32_t*>;

    unsigned blocks(size_t count) { return static_cast<unsigned>(std::min<size_t>((count - 1) / 256 + 1, 4096)); }
    __global__ void validate_finite(const float* values, size_t count, unsigned* invalid) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            if (!isfinite(values[i]))
                atomicOr(invalid, 1u);
    }
    __global__ void validate_resident_sh(const float* values, size_t count, size_t rest, size_t slots, unsigned* invalid) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count * rest * 3; i += size_t(gridDim.x) * blockDim.x) {
            const size_t gaussian = i / (rest * 3), attribute = i % (rest * 3);
            const size_t address = ((gaussian / 32 * slots + attribute / 4) * 32 + gaussian % 32) * 4 + attribute % 4;
            if (!isfinite(values[address]))
                atomicOr(invalid, 1u);
        }
    }
    __global__ void validate_model(const float* parameter, Layout layout, size_t count, unsigned* invalid) {
        for (size_t g = size_t(blockIdx.x) * blockDim.x + threadIdx.x; g < count; g += size_t(gridDim.x) * blockDim.x) {
            bool valid = true;
            const unsigned sizes[5] = {3, 4, 3, 3, 1};
            for (unsigned family = 0; family < 5; ++family)
                for (unsigned c = 0; c < sizes[family]; ++c)
                    valid = valid && isfinite(parameter[layout.offsets[family] + g * sizes[family] + c]);
            double norm = 0;
            for (unsigned c = 0; c < 4; ++c) {
                const double q = parameter[layout.offsets[1] + g * 4 + c];
                norm += q * q;
            }
            valid = valid && norm > 1e-20 && norm < 1e20;
            for (unsigned c = 0; c < 3; ++c) {
                const float scale = expf(parameter[layout.offsets[2] + g * 3 + c]);
                valid = valid && isfinite(scale) && scale > 0;
            }
            if (!valid)
                atomicOr(invalid, 1u);
        }
    }
    __global__ void color_kernel(const float* coefficient, float* color, size_t count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            color[i] = fmaxf(0, sh_c0 * coefficient[i] + 0.5f);
    }
    __device__ size_t sh_address(size_t gaussian, size_t attribute, size_t slots) {
        return ((gaussian / 32 * slots + attribute / 4) * 32 + gaussian % 32) * 4 + attribute % 4;
    }
    template <bool Upload>
    __global__ void sh_transfer(float* canonical, float* swizzled, size_t count, size_t rest, size_t slots) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count * rest * 3; i += size_t(gridDim.x) * blockDim.x) {
            const size_t address = sh_address(i / (rest * 3), i % (rest * 3), slots);
            if constexpr (Upload)
                swizzled[address] = canonical[i];
            else
                canonical[i] = swizzled[address];
        }
    }
    __global__ void sh_mask(const int32_t* radii, const float* dirs, bool* mask, float* color, size_t count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const float x = dirs[i * 3], y = dirs[i * 3 + 1], z = dirs[i * 3 + 2];
            mask[i] = radii[i * 2] > 0 && radii[i * 2 + 1] > 0 && isfinite(x) && isfinite(y) && isfinite(z) &&
                      (x != 0 || y != 0 || z != 0);
            if (!mask[i])
                color[i * 3] = color[i * 3 + 1] = color[i * 3 + 2] = 0;
        }
    }
    __global__ void sh_clamp(float* color, float* gradient, size_t count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            if (gradient)
                gradient[i] = color[i] > 0 ? gradient[i] : 0;
            else
                color[i] = fmaxf(0, color[i]);
        }
    }
    __global__ void sh0_scatter(const float* canonical, float* gradient, size_t count, size_t bases) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count * 3; i += size_t(gridDim.x) * blockDim.x)
            gradient[i] = canonical[(i / 3) * bases * 3 + i % 3];
    }
    template <bool Swizzled = false>
    __global__ void sh_adam(float* values, const float* gradient, float* first, float* second,
                            size_t size, size_t count, size_t bases, size_t slots, size_t active,
                            float rate, float correction1, float correction2, unsigned* invalid) {
        // Traverse physical swizzled storage so parameter and both moment
        // read/writes are coalesced. Only the canonical gradient is gathered.
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += size_t(gridDim.x) * blockDim.x) {
            const size_t gaussian = (i / 4 / (slots * 32)) * 32 + (i / 4) % 32;
            const size_t attribute = ((i / 4 / 32) % slots) * 4 + i % 4;
            if (gaussian >= count || attribute >= active * 3)
                continue;
            const float g = gradient[Swizzled ? i : gaussian * bases * 3 + attribute + 3];
            const float m = 0.9f * first[i] + 0.1f * g, v = 0.999f * second[i] + 0.001f * g * g;
            const float next = values[i] - rate * (m / correction1) / (sqrtf(v / correction2) + 1e-8f);
            if (!isfinite(g) || !isfinite(m) || !isfinite(v) || !isfinite(next)) {
                atomicOr(invalid, 1u);
                continue;
            }
            values[i] = next;
            first[i] = m;
            second[i] = v;
        }
    }
    __global__ void loss_gradient(const float* image, const float* target, float* gradient, size_t count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            gradient[i] = 2 * (image[i] - target[i]) / count;
    }
    __device__ bool loss_valid(size_t index, unsigned width, unsigned height, LfsHipTrainLoss loss) {
        const unsigned x = index % width, y = (index / width) % height;
        return loss.kind == LFS_HIP_LOSS_MSE || !loss.valid_padding ||
               ((width <= 10 || (x >= 5 && x < width - 5)) && (height <= 10 || (y >= 5 && y < height - 5)));
    }
    __global__ void l1_gradient(const float* image, const float* target, float* gradient, size_t count,
                                unsigned width, unsigned height, LfsHipTrainLoss loss, float normalized) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const float difference = image[i] - target[i];
            gradient[i] = loss_valid(i, width, height, loss) && difference != 0 ? copysignf(normalized, difference) : 0;
        }
    }
    __global__ void loss_reduce(const float* image, const float* target, const float* map, float* partials,
                                size_t count, unsigned width, unsigned height, LfsHipTrainLoss loss, float normalized) {
        float sum = 0;
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            if (!loss_valid(i, width, height, loss))
                continue;
            const float delta = image[i] - target[i];
            float value = delta * delta;
            if (loss.kind == LFS_HIP_LOSS_L1_SSIM) {
                value = (1 - loss.ssim_weight) * fabsf(delta);
                if (loss.ssim_weight > 0)
                    value += loss.ssim_weight * (1 - map[i % (size_t(width) * height)]);
            }
            sum += value * normalized;
        }
        __shared__ float shared[256];
        shared[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned stride = 128; stride; stride >>= 1) {
            if (threadIdx.x < stride)
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            partials[blockIdx.x] = shared[0];
    }
    __global__ void loss_finish(const float* partials, unsigned count, float* output) {
        float sum = 0;
        for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
            sum += partials[i];
        __shared__ float shared[256];
        shared[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned stride = 128; stride; stride >>= 1) {
            if (threadIdx.x < stride)
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            *output = shared[0];
    }
    __global__ void adam_kernel(float* parameter, const float* gradient, float* first, float* second,
                                const float* color, size_t count, Layout layout, Layout optimizer_layout, LfsHipTrainRates rates,
                                float correction1, float correction2, unsigned* invalid, bool sh_gradient) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count * 14; i += size_t(gridDim.x) * blockDim.x) {
            size_t local = i;
            unsigned family = 0;
            const unsigned sizes[5] = {3, 4, 3, 3, 1};
            while (local >= count * sizes[family]) {
                local -= count * sizes[family];
                ++family;
            }
            const float learning_rates[5] = {rates.means, rates.quaternions, rates.log_scales, rates.sh0, rates.raw_opacities};
            const size_t index = optimizer_layout.offsets[family] + local;
            const size_t parameter_index = layout.offsets[family] + local;
            float g = gradient[index];
            if (family == 3 && !sh_gradient)
                g *= color[local] > 0 ? sh_c0 : 0;
            const float m = 0.9f * first[index] + 0.1f * g;
            const float v = 0.999f * second[index] + 0.001f * g * g;
            const float next = parameter[parameter_index] - learning_rates[family] * (m / correction1) / (sqrtf(v / correction2) + 1e-8f);
            if (!isfinite(g) || !isfinite(m) || !isfinite(v) || !isfinite(next)) {
                atomicOr(invalid, 1u);
                continue;
            }
            parameter[parameter_index] = next;
            first[index] = m;
            second[index] = v;
        }
    }
} // namespace

struct LfsHipTraining {
    int device;
    size_t count;
    Stream stream;
    Budget budget;
    // Exported parameters keep 256-byte region alignment. Private optimizer
    // arrays need only float4 alignment; their offsets never depend on an
    // external model's reservation, padding, or family ordering.
    Layout layout, optimizer_layout;
    Buffer parameter, gradient, first, second, color, radii, means2d, depths, conics, counts, prefix, invalid;
    std::unique_ptr<Frame> frame;
    std::unique_ptr<Bins> bins;
    std::unique_ptr<Buffer> scratch;
    std::unique_ptr<Sh> sh;
    bool fast_raster = false;
    bool fused_optimizer = false;
#ifdef LFS_HIP_FAST_RUNTIME
    std::unique_ptr<FastFrame> fast;
    std::unique_ptr<JointAdam> joint;
#endif
    size_t scan_bytes = 0;
    uint64_t steps = 0, intersections = 0;
    std::unique_ptr<Profile> profile;
    void mark(unsigned i) {
        if (profile)
            profile->mark(i, stream.value);
    }
    bool poisoned = false;
    bool optimizer_pending = false;
    std::atomic_bool cancelled = false;
    LfsHipTrainLoss loss{sizeof(LfsHipTrainLoss), LFS_HIP_LOSS_MSE, 0, 0};

    explicit LfsHipTraining(const LfsHipTrainConfig& config, const LfsHipTrainSharedModel* shared = nullptr, const LfsHipTrainModelLayout* external = nullptr, const LfsHipTrainDeviceModel* borrowed = nullptr)
        : device(config.device_ordinal), count(config.gaussian_count), budget{config.allocation_budget_bytes}, layout(external ? external_layout(count, *external) : Layout(count)), optimizer_layout(count, 4), parameter(budget, multiply(layout.floats, 4), shared, borrowed), gradient(budget, multiply(optimizer_layout.floats, 4)), first(budget, gradient.bytes), second(budget, gradient.bytes), color(budget, multiply(count, 12)), radii(budget, multiply(count, 8)), means2d(budget, multiply(count, 8)), depths(budget, multiply(count, 4)), conics(budget, multiply(count, 12)), counts(budget, multiply(count, 4)), prefix(budget, multiply(count, 8)), invalid(budget, 4) {
        hip_check(hipcub::DeviceScan::InclusiveSum(nullptr, scan_bytes,
                                                   CountIterator(counts.as<int32_t>(), Widen{}), prefix.as<int64_t>(), count, stream.value),
                  "scan query");
        scratch = std::make_unique<Buffer>(budget, scan_bytes);
        first.zero(stream.value);
        second.zero(stream.value);
        const char* profiling = std::getenv("LFS_HIP_TRAIN_PROFILE");
        if (profiling && std::string_view(profiling) == "1")
            profile = std::make_unique<Profile>();
    }
    void healthy(bool needs_frame = true) {
        if (poisoned)
            throw Failure(LFS_HIP_TRAIN_POISONED, "HIP training session is poisoned by a previous GPU failure");
        require(!needs_frame || frame != nullptr, "HIP training frame is not prepared");
    }
    void ensure_geometry() {
        if (color.bytes && radii.bytes && means2d.bytes && depths.bytes && conics.bytes && counts.bytes && prefix.bytes)
            return;
        // Construct the entire replacement before committing any field. A
        // failed switch back to 3DGUT leaves the selected FastGS path usable.
        Buffer new_color(budget, multiply(count, 12)), new_radii(budget, multiply(count, 8)),
            new_means(budget, multiply(count, 8)), new_depths(budget, multiply(count, 4)),
            new_conics(budget, multiply(count, 12)), new_counts(budget, multiply(count, 4)),
            new_prefix(budget, multiply(count, 8));
        color.swap_owned(new_color);
        radii.swap_owned(new_radii);
        means2d.swap_owned(new_means);
        depths.swap_owned(new_depths);
        conics.swap_owned(new_conics);
        counts.swap_owned(new_counts);
        prefix.swap_owned(new_prefix);
    }
    void trim_geometry() {
        // Called only before a frame is bound. FastGS owns its own visibility,
        // projection and color storage; none of these buffers is used there.
        color.release_owned();
        radii.release_owned();
        means2d.release_owned();
        depths.release_owned();
        conics.release_owned();
        counts.release_owned();
        prefix.release_owned();
    }
    void check_cancel() const {
        if (cancelled.load(std::memory_order_relaxed))
            throw Failure(LFS_HIP_TRAIN_CANCELLED, "HIP training cancelled before optimizer update");
    }
    float loss_normalization() const {
        size_t width = frame->width, height = frame->height;
        if (loss.kind == LFS_HIP_LOSS_L1_SSIM && loss.valid_padding) {
            width = width > 10 ? width - 10 : width;
            height = height > 10 ? height - 10 : height;
        }
        return 1.0f / (width * height * 3);
    }
    void photo_forward() {
        auto& f = *frame;
        if (!f.photo)
            f.photo = std::make_unique<Photo>(budget, f.pixels);
        lfs_hip_photo::forward(f.height, f.width, f.image.as(), f.target.as(), f.photo->partials.as<__half>(), f.photo->map.as(), stream.value);
        hip_check(hipGetLastError(), "HIP fused photometric forward");
    }
    void image_gradient() {
        auto& f = *frame;
        if (loss.kind == LFS_HIP_LOSS_MSE) {
            hipLaunchKernelGGL(loss_gradient, dim3(blocks(f.pixels * 3)), dim3(256), 0, stream.value,
                               f.image.as(), f.target.as(), f.image_gradient.as(), f.pixels * 3);
        } else if (loss.ssim_weight == 0) {
            hipLaunchKernelGGL(l1_gradient, dim3(blocks(f.pixels * 3)), dim3(256), 0, stream.value,
                               f.image.as(), f.target.as(), f.image_gradient.as(), f.pixels * 3,
                               f.width, f.height, loss, loss_normalization());
        } else {
            photo_forward();
            lfs_hip_photo::backward(f.height, f.width, loss.ssim_weight, loss.valid_padding != 0, f.image.as(),
                                    f.target.as(), f.photo->partials.as<__half>(), f.image_gradient.as(), stream.value);
        }
        hip_check(hipGetLastError(), "HIP photometric image gradient");
    }
    void prepare() {
        ensure_geometry();
        auto& f = *frame;
        auto queue = stream.value;
        mark(0);
        gsplat_lfs::launch_projection_ut_3dgs_fused_kernel(parameter.as(layout.offsets[0]), parameter.as(layout.offsets[1]),
                                                           parameter.as(layout.offsets[2]), nullptr, f.camera.as(), nullptr, f.camera.as(16), count, 1, f.width, f.height,
                                                           0.3f, f.descriptor.near_plane, f.descriptor.far_plane, 0, PINHOLE, {}, ShutterType::GLOBAL,
                                                           nullptr, nullptr, nullptr, radii.as<int32_t>(), means2d.as(), depths.as(), conics.as(), nullptr, queue);
        mark(1);
        gsplat_lfs::launch_intersect_tile_kernel(means2d.as(), radii.as<int32_t>(), depths.as(), nullptr, nullptr,
                                                 1, count, 0, false, 16, f.tile_width, f.tile_height, nullptr, counts.as<int32_t>(), nullptr, nullptr, queue);
        auto bytes = scan_bytes;
        hip_check(hipcub::DeviceScan::InclusiveSum(scratch->pointer, bytes,
                                                   CountIterator(counts.as<int32_t>(), Widen{}), prefix.as<int64_t>(), count, queue),
                  "tile count scan");
        int64_t total = 0;
        hip_check(hipMemcpyAsync(&total, prefix.as<int64_t>() + count - 1, sizeof(total), hipMemcpyDeviceToHost, queue), "intersection count");
        hip_check(hipStreamSynchronize(queue), "intersection count completion");
        require(total >= 0 && total < INT32_MAX, "HIP training intersection count exceeds supported range");
        intersections = total;
        mark(2);
        if (!total) {
            hip_check(hipMemsetAsync(f.offsets.pointer, 0, f.offsets.bytes, queue), "empty tile offsets");
            mark(3);
            return;
        }
        const size_t needed = static_cast<size_t>(total) + 1;
        // Query workspace before choosing geometric headroom. The budget must
        // cover both old/new bins AND both old/new sort scratch until commit.
        // hipCUB's null-storage query never reads the input/output arrays.
        size_t sort_bytes = 0;
        hip_check(hipcub::DeviceRadixSort::SortPairs(nullptr, sort_bytes,
                                                     bins ? bins->keys.as<int64_t>() : nullptr,
                                                     bins ? bins->sorted_keys.as<int64_t>() : nullptr,
                                                     bins ? bins->ids.as<int32_t>() : nullptr,
                                                     bins ? bins->sorted_ids.as<int32_t>() : nullptr,
                                                     needed, 0, 64, queue),
                  "sort query");
        const size_t scratch_replacement_bytes = sort_bytes > scratch->bytes ? std::max(scan_bytes, sort_bytes) : 0;
        if (scratch_replacement_bytes > budget.limit - budget.live)
            throw Failure(LFS_HIP_TRAIN_RESOURCE_EXHAUSTED, "HIP sort workspace replacement exceeds allocation budget");
        std::unique_ptr<Bins> replacement;
        if (!bins || needed > bins->capacity) {
            const size_t proposed = bins ? std::max(needed, bins->capacity + bins->capacity / 2) : needed;
            const size_t available = (budget.limit - budget.live - scratch_replacement_bytes) / 24;
            const size_t capacity = std::min<size_t>(std::min<size_t>(INT32_MAX, proposed), available);
            if (capacity < needed)
                throw Failure(LFS_HIP_TRAIN_RESOURCE_EXHAUSTED, "HIP intersection capacity exceeds allocation budget");
            replacement = std::make_unique<Bins>(budget, capacity);
        }
        std::unique_ptr<Buffer> new_scratch;
        if (sort_bytes > scratch->bytes)
            new_scratch = std::make_unique<Buffer>(budget, std::max(scan_bytes, sort_bytes));
        if (replacement)
            bins.swap(replacement);
        if (new_scratch)
            scratch.swap(new_scratch);
        // Sort only the live intersections plus one sentinel, not all reserved capacity.
        gsplat_lfs::launch_fill_isect_sentinels_kernel(bins->keys.as<int64_t>(), bins->ids.as<int32_t>(), needed,
                                                       std::numeric_limits<int64_t>::max(), queue);
        gsplat_lfs::launch_intersect_tile_kernel(means2d.as(), radii.as<int32_t>(), depths.as(), nullptr, nullptr,
                                                 1, count, 0, false, 16, f.tile_width, f.tile_height, prefix.as<int64_t>(), nullptr,
                                                 bins->keys.as<int64_t>(), bins->ids.as<int32_t>(), queue, total);
        hip_check(hipcub::DeviceRadixSort::SortPairs(scratch->pointer, sort_bytes,
                                                     bins->keys.as<int64_t>(), bins->sorted_keys.as<int64_t>(), bins->ids.as<int32_t>(), bins->sorted_ids.as<int32_t>(),
                                                     needed, 0, 64, queue),
                  "tile depth sort");
        gsplat_lfs::launch_intersect_offset_kernel(bins->sorted_keys.as<int64_t>(), needed, 1,
                                                   f.tile_width, f.tile_height, f.offsets.as<int32_t>(), queue);
        mark(3);
    }
#ifdef LFS_HIP_FAST_RUNTIME
    void forward_fast() {
        require(frame != nullptr, "HIP training frame is not configured");
        auto& f = *frame;
        if (!fast)
            fast = std::make_unique<FastFrame>(budget, stream.value);
        fast->cursor = 0;
        fast->phase_candidate = nullptr;
        fast->phase_candidate_bytes = 0;
        const float* r = f.descriptor.world_to_camera;
        fast->camera_center = make_float3(
            -(r[0] * r[3] + r[4] * r[7] + r[8] * r[11]),
            -(r[1] * r[3] + r[5] * r[7] + r[9] * r[11]),
            -(r[2] * r[3] + r[6] * r[7] + r[10] * r[11]));
        size_t extra_backward_bytes = 0;
        if (sh) {
            if (!fused_optimizer && (!fast->sh_gradient || fast->sh_gradient->bytes < sh->values.bytes))
                extra_backward_bytes = sh->values.bytes;
            else if (fused_optimizer && (!joint->sh_packed || joint->sh_packed->bytes != sh->storage_floats * 2))
                extra_backward_bytes = multiply(sh->storage_floats, 2) + multiply((count + 255) / 256, 16);
        }
        fast->forward_with_depth_fallback(count, [&] {
            auto* camera = reinterpret_cast<float3*>(fast->allocate(sizeof(float3)));
            hip_check(hipMemcpyAsync(camera, &fast->camera_center, sizeof(fast->camera_center), hipMemcpyHostToDevice, stream.value), "FastGS camera center");
            auto allocate = [this](size_t bytes) { return fast->allocate(bytes); };
            fast->result = fast_lfs::rasterization::forward(allocate, [](size_t) {}, [this](size_t bytes) { return fast->allocate_phase(bytes); }, [this](const void* source, size_t bytes) {
                    char* output = fast->allocate(bytes);
                    hip_check(hipMemcpyAsync(output, source, bytes, hipMemcpyDeviceToDevice, stream.value), "FastGS retained indices");
                    return output; }, [this](size_t bytes) { return fast->tiles = fast->allocate(bytes); }, reinterpret_cast<float3*>(parameter.as(layout.offsets[0])), reinterpret_cast<float3*>(parameter.as(layout.offsets[2])), reinterpret_cast<float4*>(parameter.as(layout.offsets[1])), parameter.as(layout.offsets[4]), reinterpret_cast<float3*>(parameter.as(layout.offsets[3])), sh ? sh->values.as<float4>() : nullptr, nullptr, 0, 0, f.camera.as<float4>(), camera, f.image.as(), f.alpha.as(), nullptr, nullptr, nullptr, nullptr, count, sh ? (sh->active + 1) * (sh->active + 1) : 1, sh ? sh->bases : 1, f.width, f.height, f.descriptor.fx, f.descriptor.fy, f.descriptor.cx, f.descriptor.cy, f.descriptor.near_plane, f.descriptor.far_plane, false, stream.value, nullptr, 1.0f, profile ? std::function<void(unsigned)>([this](unsigned phase) { mark(phase); }) : std::function<void(unsigned)>{}, fast->depth_presort_allowed, &fast->depth_presort_attempted);
            intersections = fast->result.n_instances;
            // Own stack camera data until its asynchronous upload has completed.
            hip_check(hipStreamSynchronize(stream.value), "FastGS forward completion");
        }, extra_backward_bytes);
    }
    void backward_fast(const LfsHipTrainRates& rates) {
        auto& f = *frame;
        // Helper indices are compact workset indices, not model indices. Keep
        // a real row for the empty case so pointer construction stays valid;
        // preprocess still visits every model row for invisible Adam momentum.
        const size_t helper_count = fast->reuse_phase_helpers ? std::max<size_t>(fast->result.n_visible, 1) : count;
        const size_t helper_bytes = multiply(helper_count, 16 * sizeof(float));
        auto* helper = fast->backward_helpers(helper_bytes);
        fast_lfs::rasterization::FusedAdamSettings settings{};
        settings.enabled = fused_optimizer;
        settings.eps = 1e-8f; // Match the FP32 reference; codec precision is the only optimizer change.
        const float correction1 = 1 - std::pow(0.9f, static_cast<float>(steps + 1));
        const float correction2 = 1 - std::pow(0.999f, static_cast<float>(steps + 1));
        const float learning_rates[]{rates.means, rates.quaternions, rates.log_scales, rates.sh0, rates.raw_opacities};
        std::array<fast_lfs::rasterization::FusedAdamParam*, 5> families{
            &settings.means, &settings.rotation, &settings.scaling, &settings.sh0, &settings.opacity};
        for (unsigned family = 0; family < families.size(); ++family) {
            auto& p = *families[family];
            p.param = parameter.as(layout.offsets[family]);
            p.hip_gradient = fused_optimizer ? nullptr : gradient.as(optimizer_layout.offsets[family]);
            p.n_primitives = count;
            p.n_attributes = attributes[family];
            p.n_elements = count * attributes[family];
            if (fused_optimizer) {
                p.enabled = true;
                p.joint_bits = 16;
                p.joint_packed = joint->packed[family]->as<uint8_t>();
                p.joint_bounds = joint->bounds[family]->as();
                p.step_size = learning_rates[family] / correction1;
                p.bias_correction2_sqrt_rcp = 1 / std::sqrt(correction2);
            }
        }
        if (sh) {
            if (!fused_optimizer && (!fast->sh_gradient || fast->sh_gradient->bytes < sh->values.bytes)) {
                auto replacement = std::make_unique<Buffer>(budget, sh->values.bytes);
                fast->sh_gradient.swap(replacement);
            }
            if (!fused_optimizer) {
                fast->sh_gradient->zero(stream.value);
                settings.shN.hip_gradient = fast->sh_gradient->as();
            } else {
                joint->prepare_sh(budget, *sh, count, stream.value);
                settings.shN.param = sh->values.as();
                settings.shN.enabled = sh->active > 0;
                settings.shN.joint_bits = 8;
                settings.shN.joint_packed = joint->sh_packed->as<uint8_t>();
                settings.shN.joint_bounds = joint->sh_bounds->as();
                settings.shN.n_attributes = sh->slots * 4;
                settings.shN.step_size = sh->rate / correction1;
                settings.shN.bias_correction2_sqrt_rcp = 1 / std::sqrt(correction2);
            }
            settings.shN.n_elements = sh->storage_floats;
            settings.shN.n_primitives = count;
        }
        auto& result = fast->result;
        auto* camera = reinterpret_cast<float3*>(fast->buffers[0]->pointer);
        float* opacity_gradient = helper;
        auto* color_gradient = reinterpret_cast<float3*>(helper + helper_count);
        auto* mean_gradient = reinterpret_cast<float2*>(helper + helper_count * 4);
        auto* conic_gradient = reinterpret_cast<float3*>(helper + helper_count * 6);
        float* depth_gradient = helper + helper_count * 9;
#ifdef LFS_HIP_FAST_PACKED_GRADIENTS
        opacity_gradient = helper + 5;
        color_gradient = reinterpret_cast<float3*>(helper + 6);
        mean_gradient = reinterpret_cast<float2*>(helper);
        conic_gradient = reinterpret_cast<float3*>(helper + 2);
        depth_gradient = helper + 9;
#endif
        if (fused_optimizer) {
            check_cancel();
            optimizer_pending = true;
        }
        fast_lfs::rasterization::backward(nullptr, f.image_gradient.as(), {}, nullptr, nullptr,
                                          f.image.as(), f.alpha.as(), reinterpret_cast<float3*>(parameter.as(layout.offsets[0])),
                                          reinterpret_cast<float3*>(parameter.as(layout.offsets[2])), reinterpret_cast<float4*>(parameter.as(layout.offsets[1])),
                                          parameter.as(layout.offsets[4]), sh ? sh->values.as<float4>() : nullptr,
                                          f.camera.as<float4>(), camera, nullptr, result.per_primitive_buffers, fast->tiles,
                                          result.sorted_primitive_indices, result.primitive_work_indices,
                                          opacity_gradient, color_gradient, mean_gradient, conic_gradient, depth_gradient, nullptr, nullptr, nullptr,
                                          count, result.n_instances, result.n_visible,
                                          sh ? (sh->active + 1) * (sh->active + 1) : 1, sh ? sh->bases : 1,
                                          f.width, f.height, f.descriptor.fx, f.descriptor.fy, f.descriptor.cx, f.descriptor.cy,
                                          false, DensificationType::None, settings, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr, stream.value);
    }
#endif
    void forward() {
#ifdef LFS_HIP_FAST_RUNTIME
        if (fast_raster) {
            forward_fast();
            return;
        }
#endif
        prepare();
        auto& f = *frame;
        if (!sh || sh->active == 0) {
            hipLaunchKernelGGL(color_kernel, dim3(blocks(count * 3)), dim3(256), 0, stream.value,
                               parameter.as(layout.offsets[3]), color.as(), count * 3);
            hip_check(hipGetLastError(), "SH0 color activation");
        } else {
            gsplat_lfs::compute_view_dirs(parameter.as(layout.offsets[0]), f.camera.as(), 1, count, sh->directions.as(), stream.value);
            hipLaunchKernelGGL(sh_mask, dim3(blocks(count)), dim3(256), 0, stream.value,
                               radii.as<int32_t>(), sh->directions.as(), sh->mask.as<bool>(), color.as(), count);
            hip_check(hipGetLastError(), "SH visibility mask");
            gsplat_lfs::launch_spherical_harmonics_swizzled_fwd_kernel(sh->active, sh->maximum, 3, sh->directions.as(),
                                                                       parameter.as(layout.offsets[3]), sh->values.as(), sh->mask.as<bool>(), count, color.as(), stream.value);
            hipLaunchKernelGGL(sh_clamp, dim3(blocks(count * 3)), dim3(256), 0, stream.value, color.as(), nullptr, count * 3);
            hip_check(hipGetLastError(), "SH color clamp");
        }
        gsplat_lfs::launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<3>(
            parameter.as(layout.offsets[0]), parameter.as(layout.offsets[1]), parameter.as(layout.offsets[2]),
            color.as(), parameter.as(layout.offsets[4]), nullptr, nullptr, nullptr,
            1, count, intersections, f.width, f.height, 16, f.camera.as(), nullptr, f.camera.as(16), PINHOLE, {},
            ShutterType::GLOBAL, nullptr, nullptr, nullptr, f.offsets.as<int32_t>(),
            bins ? bins->sorted_ids.as<int32_t>() : nullptr, f.image.as(), f.alpha.as(), f.last.as<int32_t>(), stream.value);
        mark(4);
    }
};

namespace {
    template <class F>
    int boundary(LfsHipTraining* session, LfsHipError* error, F&& operation) noexcept {
        DispatchCall dispatch;
        if (error)
            *error = {};
        int status = LFS_HIP_TRAIN_GPU_ERROR;
        const auto report = [&](int code, const char* message) {
            if (error) {
                error->native_code = code;
                std::snprintf(error->message, sizeof(error->message), "%s", message);
            }
        };
        try {
            dispatch.begin(error);
            operation();
            return LFS_HIP_TRAIN_OK;
        } catch (const Failure& failure) {
            status = failure.status;
            report(-1, failure.what());
        } catch (const LfsHipRasterFailure& failure) {
            status = failure.code == hipErrorOutOfMemory ? LFS_HIP_TRAIN_RESOURCE_EXHAUSTED : LFS_HIP_TRAIN_GPU_ERROR;
            report(failure.code, failure.what());
        } catch (const std::bad_alloc&) {
            status = LFS_HIP_TRAIN_RESOURCE_EXHAUSTED;
            report(-1, "HIP training host allocation failed");
        } catch (const std::exception& failure) {
            report(-1, failure.what());
        } catch (...) {
            report(-1, "Unknown HIP training failure");
        }
        if (session && session->optimizer_pending && status == LFS_HIP_TRAIN_RESOURCE_EXHAUSTED) {
            status = LFS_HIP_TRAIN_POISONED;
            session->poisoned = true;
            report(-1, "HIP resource failure after optimizer dispatch; session must not continue");
        }
        if (session && status != LFS_HIP_TRAIN_GPU_ERROR && status != LFS_HIP_TRAIN_POISONED) {
            // Even a cancelled step may have queued raster/gradient work.
            // Complete it before exposing the session to its next host call.
            try {
                DeviceScope selected(session->device);
                hip_check(hipStreamSynchronize(session->stream.value), "failed HIP operation completion");
                if (status == LFS_HIP_TRAIN_RESOURCE_EXHAUSTED)
                    (void)hipGetLastError();
            } catch (const std::exception& failure) {
                status = LFS_HIP_TRAIN_GPU_ERROR;
                report(-1, failure.what());
            }
        }
        if (session && status == LFS_HIP_TRAIN_GPU_ERROR)
            session->poisoned = true;
        return status;
    }
    void finite_array(const float* values, size_t count) {
        require(values != nullptr, "Missing HIP training input array");
        static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);
        uint32_t invalid = 0;
        // Classify every IEEE-754 exponent with an integer reduction so the
        // host compiler can vectorize validation. Throw only after the scan;
        // callers still validate before publishing or mutating session state.
        for (size_t i = 0; i < count; ++i)
            invalid |= (std::bit_cast<uint32_t>(values[i]) & 0x7f800000u) == 0x7f800000u;
        require(invalid == 0, "Nonfinite HIP training input");
    }

    void validate_device_model(const LfsHipTrainDeviceModel& source, int ordinal, size_t required_bytes) {
        require(source.struct_bytes == sizeof(source) && source.device_ordinal == ordinal && source.address &&
                    source.address <= UINTPTR_MAX && source.address % 256 == 0 &&
                    source.allocation_bytes <= SIZE_MAX && source.allocation_bytes >= required_bytes,
                "Invalid native HIP model descriptor");
        auto* pointer = reinterpret_cast<void*>(static_cast<uintptr_t>(source.address));
        hipPointerAttribute_t attributes{};
        auto status = hipPointerGetAttributes(&attributes, pointer);
        if (status != hipSuccess) {
            (void)hipGetLastError();
            if (status != hipErrorInvalidValue && status != hipErrorInvalidDevicePointer && status != hipErrorNotSupported)
                hip_check(status, "Native HIP model pointer attributes");
            require(false, "Native model address is not a HIP allocation");
        }
        require(attributes.type == hipMemoryTypeDevice && !attributes.isManaged && attributes.device == ordinal,
                "Native HIP model belongs to a different GPU or memory type");
        hipDeviceptr_t base = nullptr;
        size_t size = 0;
        status = hipMemGetAddressRange(&base, &size, pointer);
        if (status != hipSuccess) {
            (void)hipGetLastError();
            if (status != hipErrorInvalidValue && status != hipErrorInvalidDevicePointer && status != hipErrorNotSupported)
                hip_check(status, "Native HIP model allocation bounds");
            require(false, "Cannot verify native HIP model allocation bounds");
        }
        require(base == pointer && source.allocation_bytes <= size,
                "Native HIP model descriptor is not an allocation base or exceeds allocation bounds");
    }
    void validate_shared_model(const LfsHipTrainSharedModel& shared, const hipDeviceProp_t& properties,
                               int ordinal, size_t required_bytes) {
        require(shared.struct_bytes == sizeof(shared) && shared.win32_handle && shared.device_node_mask &&
                    shared.allocation_bytes <= SIZE_MAX && shared.allocation_bytes >= required_bytes,
                "Invalid HIP shared model descriptor or allocation size");
#ifdef _WIN32
        const bool hip_luid_valid = properties.luidDeviceNodeMask &&
                                    std::any_of(std::begin(properties.luid), std::end(properties.luid), [](char value) { return value != 0; });
        if (hip_luid_valid) {
            require(std::memcmp(properties.luid, shared.device_luid, sizeof(shared.device_luid)) == 0 &&
                        (properties.luidDeviceNodeMask & shared.device_node_mask),
                    "HIP and Vulkan shared model adapters differ");
            return;
        }
        // On current Windows APUs HIP may not publish a LUID. Resolve the Vulkan
        // LUID through the OS and compare its PCI address; never match by name.
        uint32_t address[3]{};
        require(lfs_hip_adapter_pci(shared.device_luid, address) && shared.device_node_mask == 1,
                "Cannot uniquely resolve shared model adapter PCI address");
        char pci[64]{};
        hip_check(hipDeviceGetPCIBusId(pci, sizeof(pci), ordinal), "HIP shared model PCI identity");
        unsigned domain = 0, bus = 0, device = 0, function = 0;
        require(std::sscanf(pci, "%x:%x:%x.%x", &domain, &bus, &device, &function) == 4 && domain == 0 &&
                    address[0] == bus && address[1] == device && address[2] == function,
                "HIP and Vulkan shared model PCI adapters differ");
#else
        throw Failure(LFS_HIP_TRAIN_INVALID, "Shared HIP model import currently requires Windows");
#endif
    }
} // namespace

extern "C" {
int lfs_hip_application_dispatch_install(const LfsHipApplicationDispatch* hooks,LfsHipError* error) {
    if(error)*error={};
    try {
        require(hooks&&hooks->struct_bytes==sizeof(*hooks)&&hooks->begin&&hooks->end,"Invalid HIP dispatch hooks");
        std::lock_guard lock(dispatch_install_mutex);
        if(auto* existing=installed_dispatch.load(std::memory_order_acquire)) {
            require(existing->user==hooks->user&&existing->begin==hooks->begin&&existing->end==hooks->end,
                    "HIP dispatch hooks cannot change after installation");
        } else {
            dispatch_callbacks=*hooks;
            installed_dispatch.store(&dispatch_callbacks,std::memory_order_release);
        }
        return 0;
    } catch(const std::exception& failure) {
        if(error){error->native_code=-1;std::snprintf(error->message,sizeof(error->message),"%s",failure.what());}
        return LFS_HIP_TRAIN_INVALID;
    } catch(...) {
        if(error){error->native_code=-1;std::snprintf(error->message,sizeof(error->message),"HIP dispatch hook installation failed");}
        return LFS_HIP_TRAIN_INVALID;
    }
}
int lfs_hip_train_host_alloc(int device, size_t bytes, void** output, LfsHipError* error) {
    if (output) *output = nullptr;
    const int status = boundary(nullptr, error, [&] {
        require(output && bytes, "Invalid HIP host allocation");
        DeviceScope selected(device);
        hip_check(hipHostMalloc(output, bytes, hipHostMallocDefault), "HIP host target allocation");
    });
    // Optional allocation errors must not survive in HIP's per-thread launch
    // error slot and surface later as an unrelated training kernel failure.
    if (status != LFS_HIP_TRAIN_OK) (void)hipGetLastError();
    return status;
}
int lfs_hip_train_host_free(int device, void* pointer, LfsHipError* error) {
    return boundary(nullptr, error, [&] {
        require(pointer, "Invalid HIP host allocation release");
        DeviceScope selected(device);
        hip_check(hipHostFree(pointer), "HIP host target release");
    });
}
uint32_t lfs_hip_train_abi(void) { return 1; }

static int create_session(const LfsHipTrainConfig* config, const LfsHipTrainModel* model,
                          const LfsHipTrainSharedModel* shared, LfsHipTraining** output, LfsHipError* error, const LfsHipTrainModelLayout* external = nullptr, bool attach = false, const LfsHipTrainDeviceModel* borrowed = nullptr) {
    return boundary(nullptr, error, [&] {
        require(output, "Missing HIP session output");
        *output = nullptr;
        require(config && (model || (attach && (shared || borrowed) && external)) && config->struct_bytes == sizeof(*config), "Invalid HIP session descriptor");
        require(config->gaussian_count > 0 && config->gaussian_count <= 10000000 && config->allocation_budget_bytes > 0,
                "Invalid HIP Gaussian count or allocation budget");
        int devices = 0;
        hip_check(hipGetDeviceCount(&devices), "HIP device count");
        require(config->device_ordinal >= 0 && config->device_ordinal < devices, "HIP training device is unavailable");
        DeviceScope selected(config->device_ordinal);
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties, config->device_ordinal), "HIP training properties");
        require(properties.warpSize == 32, "HIP training currently requires wave32 hardware");
        const std::string_view architecture(properties.gcnArchName);
        require(architecture.substr(0, architecture.find(':')) == LFS_HIP_COMPILED_ARCH,
                "HIP training module was compiled for a different GPU architecture");
        const float* arrays[5]{};
        if (model) {
            arrays[0] = model->means;
            arrays[1] = model->quaternions;
            arrays[2] = model->log_scales;
            arrays[3] = model->sh0;
            arrays[4] = model->raw_opacities;
        }
        const size_t count = config->gaussian_count;
        if (shared)
            validate_shared_model(*shared, properties, config->device_ordinal,
                                  multiply((external ? external_layout(count, *external) : Layout(count)).floats, 4));
        if (borrowed) {
            require(!shared && external, "Native HIP model cannot also import external storage");
            validate_device_model(*borrowed, config->device_ordinal, multiply(external_layout(count, *external).floats, 4));
        }
        if (!attach) {
            for (size_t i = 0; i < attributes.size(); ++i)
                finite_array(arrays[i], count * attributes[i]);
            for (size_t i = 0; i < count; ++i) {
                double norm = 0;
                for (size_t j = 0; j < 4; ++j)
                    norm += double(model->quaternions[i * 4 + j]) * model->quaternions[i * 4 + j];
                require(norm > 1e-20 && norm < 1e20, "Invalid HIP training quaternion norm");
                for (size_t j = 0; j < 3; ++j) {
                    const float scale = std::exp(model->log_scales[i * 3 + j]);
                    require(std::isfinite(scale) && scale > 0, "HIP log scale cannot be represented after activation");
                }
            }
        }
        auto session = std::make_unique<LfsHipTraining>(*config, shared, external, borrowed);
        if (attach) {
            session->invalid.zero(session->stream.value);
            hipLaunchKernelGGL(validate_model, dim3(blocks(count)), dim3(256), 0, session->stream.value,
                               session->parameter.as(), session->layout, count, session->invalid.as<unsigned>());
            hip_check(hipGetLastError(), "HIP resident model validation");
            unsigned invalid = 0;
            hip_check(hipMemcpyAsync(&invalid, session->invalid.pointer, sizeof(invalid), hipMemcpyDeviceToHost, session->stream.value),
                      "HIP resident model validation status");
            hip_check(hipStreamSynchronize(session->stream.value), "HIP resident model validation completion");
            require(!invalid, "Invalid resident HIP model parameters");
        } else {
            for (size_t i = 0; i < attributes.size(); ++i)
                hip_check(hipMemcpyAsync(session->parameter.as(session->layout.offsets[i]), arrays[i], count * attributes[i] * sizeof(float),
                                         hipMemcpyHostToDevice, session->stream.value),
                          "HIP model upload");
            hip_check(hipStreamSynchronize(session->stream.value), "HIP model upload completion");
        }
        *output = session.release();
    });
}

int lfs_hip_train_create(const LfsHipTrainConfig* config, const LfsHipTrainModel* model, LfsHipTraining** output, LfsHipError* error) {
    return create_session(config, model, nullptr, output, error);
}

int lfs_hip_train_create_shared(const LfsHipTrainConfig* config, const LfsHipTrainModel* model,
                                const LfsHipTrainSharedModel* shared, LfsHipTraining** output, LfsHipError* error) {
    if (!shared)
        return boundary(nullptr, error, [&] {
            if (output)
                *output = nullptr;
            require(false, "Missing HIP shared model descriptor");
        });
    return create_session(config, model, shared, output, error);
}

int lfs_hip_train_create_shared_layout(const LfsHipTrainConfig* config, const LfsHipTrainModel* model,
                                       const LfsHipTrainSharedModel* shared, const LfsHipTrainModelLayout* layout,
                                       LfsHipTraining** output, LfsHipError* error) {
    if (!shared || !layout)
        return boundary(nullptr, error, [&] {
            if (output)
                *output = nullptr;
            require(false, "Missing HIP external model descriptor or layout");
        });
    return create_session(config, model, shared, output, error, layout);
}

int lfs_hip_train_attach_shared_layout(const LfsHipTrainConfig* config, const LfsHipTrainSharedModel* shared,
                                       const LfsHipTrainModelLayout* layout, LfsHipTraining** output, LfsHipError* error) {
    return create_session(config, nullptr, shared, output, error, layout, true);
}

int lfs_hip_train_attach_device_layout(const LfsHipTrainConfig* config, const LfsHipTrainDeviceModel* model,
                                       const LfsHipTrainModelLayout* layout, LfsHipTraining** output, LfsHipError* error) {
    return create_session(config, nullptr, nullptr, output, error, layout, true, model);
}

int lfs_hip_train_model_layout(uint64_t count, LfsHipTrainModelLayout* output, LfsHipError* error) {
    return boundary(nullptr, error, [&] {
        require(output && output->struct_bytes == sizeof(*output) && count > 0 && count <= 10000000,
                "Invalid HIP shared model layout request");
        const Layout layout(count);
        LfsHipTrainModelLayout result{};
        result.struct_bytes = sizeof(result);
        result.required_bytes = multiply(layout.floats, 4);
        for (size_t i = 0; i < attributes.size(); ++i) {
            result.offsets_bytes[i] = multiply(layout.offsets[i], 4);
            result.float_counts[i] = multiply(count, attributes[i]);
        }
        *output = result;
    });
}

int lfs_hip_train_destroy(LfsHipTraining* session, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session, "Missing HIP training session");
        DeviceScope selected(session->device);
        hip_check(hipStreamSynchronize(session->stream.value), "HIP session drain");
        delete session;
    });
}

int lfs_hip_train_set_rasterizer(LfsHipTraining* session, uint32_t kind, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(kind <= 1, "Unknown HIP rasterizer");
        require(session && session->steps == 0 && !session->frame, "Select HIP rasterizer before setting a frame or training");
#ifndef LFS_HIP_FAST_RUNTIME
        require(kind == 0, "FastGS HIP was not compiled; enable LFS_HIP_FAST_RUNTIME explicitly");
#endif
        DeviceScope selected(session->device);
        if (kind == 0)
            session->ensure_geometry();
        else {
            const char* trim = std::getenv("LFS_HIP_FAST_TRIM_UNUSED");
            if (!trim || std::string_view(trim) != "0")
                session->trim_geometry();
        }
        session->fast_raster = kind == 1;
    });
}

int lfs_hip_train_set_optimizer(LfsHipTraining* session, uint32_t kind, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && kind <= 1, "Unknown HIP optimizer");
        session->healthy(false);
        require(session->steps == 0 && !session->frame, "Select HIP optimizer before binding a frame or training");
        require(kind == 1 || !session->fused_optimizer, "Recreate HIP session to restore FP32 optimizer storage");
        if (kind == 0 || session->fused_optimizer)
            return;
#ifdef LFS_HIP_FAST_RUNTIME
        require(session->fast_raster, "Fused HIP optimizer requires explicit FastGS selection");
        DeviceScope selected(session->device);
        auto replacement = std::make_unique<JointAdam>(session->budget, session->count, session->stream.value);
        hip_check(hipStreamSynchronize(session->stream.value), "HIP optimizer replacement completion");
        session->gradient.release_owned();
        session->first.release_owned();
        session->second.release_owned();
        session->joint = std::move(replacement);
        session->fused_optimizer = true;
#else
        require(false, "Fused HIP optimizer was not compiled");
#endif
    });
}

int lfs_hip_train_set_frame(LfsHipTraining* session, const LfsHipTrainFrame* input, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && input && input->struct_bytes == sizeof(*input), "Invalid HIP frame descriptor");
        session->healthy(false);
        require(input->width > 0 && input->height > 0 && input->width <= 16384 && input->height <= 16384, "Invalid HIP frame dimensions");
        require(std::isfinite(input->fx) && std::isfinite(input->fy) && input->fx > 0 && input->fy > 0 &&
                    std::isfinite(input->cx) && std::isfinite(input->cy) && std::isfinite(input->near_plane) &&
                    std::isfinite(input->far_plane) && input->near_plane > 0 && input->far_plane > input->near_plane,
                "Invalid HIP pinhole camera");
        finite_array(input->world_to_camera, 16);
        require(std::abs(input->world_to_camera[12]) < 1e-4f && std::abs(input->world_to_camera[13]) < 1e-4f &&
                    std::abs(input->world_to_camera[14]) < 1e-4f && std::abs(input->world_to_camera[15] - 1) < 1e-4f,
                "HIP camera requires a rigid world-to-camera matrix");
        for (unsigned row = 0; row < 3; ++row)
            for (unsigned other = 0; other < 3; ++other) {
                double dot = 0;
                for (unsigned col = 0; col < 3; ++col)
                    dot += double(input->world_to_camera[row * 4 + col]) * input->world_to_camera[other * 4 + col];
                require(std::abs(dot - (row == other ? 1 : 0)) < 0.002, "HIP camera rotation is not orthonormal");
            }
        const float* r = input->world_to_camera;
        const double determinant = double(r[0]) * (double(r[5]) * r[10] - double(r[6]) * r[9]) -
                                   double(r[1]) * (double(r[4]) * r[10] - double(r[6]) * r[8]) +
                                   double(r[2]) * (double(r[4]) * r[9] - double(r[5]) * r[8]);
        require(std::abs(determinant - 1) < 0.003, "HIP camera requires a proper rotation");
        finite_array(input->target_chw, multiply(multiply(input->width, input->height), 3));
        DeviceScope selected(session->device);
        std::unique_ptr<Frame> replacement;
        if (!session->frame || session->frame->width != input->width || session->frame->height != input->height)
            replacement = std::make_unique<Frame>(session->budget, *input);
        auto& frame = replacement ? *replacement : *session->frame;
        float camera[25];
        std::copy_n(input->world_to_camera, 16, camera);
        const float intrinsics[9] = {input->fx, 0, input->cx, 0, input->fy, input->cy, 0, 0, 1};
        std::copy_n(intrinsics, 9, camera + 16);
        hip_check(hipMemcpyAsync(frame.camera.pointer, camera, sizeof(camera), hipMemcpyHostToDevice, session->stream.value), "camera upload");
        hip_check(hipMemcpyAsync(frame.target.pointer, input->target_chw, frame.target.bytes, hipMemcpyHostToDevice, session->stream.value), "target upload");
        hip_check(hipStreamSynchronize(session->stream.value), "frame upload completion");
        frame.descriptor = *input;
        frame.descriptor.target_chw = nullptr;
        if (replacement)
            session->frame.swap(replacement);
    });
}

int lfs_hip_train_step(LfsHipTraining* session, const LfsHipTrainRates* rates, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && rates, "Missing HIP training session or rates");
        session->healthy();
        const float values[] = {rates->means, rates->quaternions, rates->log_scales, rates->sh0, rates->raw_opacities};
        for (float value : values)
            require(std::isfinite(value) && value >= 0, "Invalid HIP learning rate");
        require(session->steps < UINT32_MAX, "HIP optimizer step exceeds supported range");
        session->check_cancel();
        DeviceScope selected(session->device);
        session->forward();
        session->check_cancel();
        auto& f = *session->frame;
        auto queue = session->stream.value;
        if (!session->fused_optimizer)
            session->gradient.zero(queue);
        f.alpha_gradient.zero(queue);
        session->mark(5);
        session->image_gradient();
        session->mark(6);
        const auto& offsets = session->layout.offsets;
        const auto& gradient_offsets = session->optimizer_layout.offsets;
#ifdef LFS_HIP_FAST_RUNTIME
        if (session->fast_raster)
            session->backward_fast(*rates);
        else
#endif
            gsplat_lfs::launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<3>(
                session->parameter.as(offsets[0]), session->parameter.as(offsets[1]), session->parameter.as(offsets[2]),
                session->color.as(), session->parameter.as(offsets[4]), nullptr, nullptr, nullptr,
                1, session->count, session->intersections, f.width, f.height, 16, f.camera.as(), nullptr, f.camera.as(16),
                PINHOLE, {}, ShutterType::GLOBAL, nullptr, nullptr, nullptr, f.offsets.as<int32_t>(),
                session->bins ? session->bins->sorted_ids.as<int32_t>() : nullptr, f.alpha.as(), f.last.as<int32_t>(),
                f.image_gradient.as(), f.alpha_gradient.as(), session->gradient.as(gradient_offsets[0]), session->gradient.as(gradient_offsets[1]),
                session->gradient.as(gradient_offsets[2]), session->gradient.as(gradient_offsets[3]), session->gradient.as(gradient_offsets[4]),
                nullptr, nullptr, nullptr, nullptr, queue);
        const bool higher_sh = session->sh && session->sh->active > 0;
        if (higher_sh && !session->fast_raster) {
            auto& sh = *session->sh;
            hipLaunchKernelGGL(sh_clamp, dim3(blocks(session->count * 3)), dim3(256), 0, queue,
                               session->color.as(), session->gradient.as(gradient_offsets[3]), session->count * 3);
            hip_check(hipGetLastError(), "SH color clamp gradient");
            sh.gradient.zero(queue);
            gsplat_lfs::launch_spherical_harmonics_swizzled_bwd_kernel(sh.active, sh.maximum, 3,
                                                                       sh.directions.as(), session->parameter.as(offsets[3]), sh.values.as(), sh.mask.as<bool>(),
                                                                       session->gradient.as(gradient_offsets[3]), session->count, sh.bases, false, sh.gradient.as(), nullptr, queue);
            hipLaunchKernelGGL(sh0_scatter, dim3(blocks(session->count * 3)), dim3(256), 0, queue,
                               sh.gradient.as(), session->gradient.as(gradient_offsets[3]), session->count, sh.bases);
            hip_check(hipGetLastError(), "SH0 coefficient gradient scatter");
        }
        session->mark(7);
        // A cancellation before Adam leaves all parameters and moments untouched.
        if (!session->fused_optimizer)
            session->check_cancel();
        session->invalid.zero(queue);
        session->optimizer_pending = true;
        unsigned invalid = 0;
        const float step = static_cast<float>(session->steps + 1);
        if (!session->fused_optimizer) {
            hipLaunchKernelGGL(adam_kernel, dim3(blocks(session->count * 14)), dim3(256), 0, queue,
                               session->parameter.as(), session->gradient.as(), session->first.as(), session->second.as(), session->fast_raster ? nullptr : session->color.as(),
                               session->count, session->layout, session->optimizer_layout, *rates, 1 - std::pow(0.9f, step), 1 - std::pow(0.999f, step), session->invalid.as<unsigned>(), higher_sh || session->fast_raster);
            hip_check(hipGetLastError(), "HIP Adam");
        }
        if (higher_sh && !session->fused_optimizer) {
            auto& sh = *session->sh;
            const size_t active_rest = (sh.active + 1) * (sh.active + 1) - 1;
#ifdef LFS_HIP_FAST_RUNTIME
            if (session->fast_raster) {
                hipLaunchKernelGGL(sh_adam<true>, dim3(blocks(sh.storage_floats)), dim3(256), 0, queue,
                                   sh.values.as(), session->fast->sh_gradient->as(), sh.first.as(), sh.second.as(), sh.storage_floats,
                                   session->count, sh.bases, sh.slots, active_rest, sh.rate,
                                   1 - std::pow(0.9f, step), 1 - std::pow(0.999f, step), session->invalid.as<unsigned>());
            } else
#endif
                hipLaunchKernelGGL(sh_adam<false>, dim3(blocks(sh.storage_floats)), dim3(256), 0, queue,
                                   sh.values.as(), sh.gradient.as(), sh.first.as(), sh.second.as(), sh.storage_floats,
                                   session->count, sh.bases, sh.slots, active_rest, sh.rate,
                                   1 - std::pow(0.9f, step), 1 - std::pow(0.999f, step), session->invalid.as<unsigned>());
            hip_check(hipGetLastError(), "HIP SH-rest Adam");
        }
        if (session->fused_optimizer) {
            hipLaunchKernelGGL(validate_model, dim3(blocks(session->count)), dim3(256), 0, queue,
                               session->parameter.as(), session->layout, session->count, session->invalid.as<unsigned>());
            hip_check(hipGetLastError(), "HIP fused Adam model validation");
            if (session->sh) {
                hipLaunchKernelGGL(validate_resident_sh, dim3(blocks(session->count * (session->sh->bases - 1) * 3)), dim3(256), 0, queue,
                                   session->sh->values.as(), session->count, session->sh->bases - 1, session->sh->slots, session->invalid.as<unsigned>());
                hip_check(hipGetLastError(), "HIP fused SH-rest validation");
            }
        }
        session->mark(8);
        hip_check(hipMemcpyAsync(&invalid, session->invalid.pointer, sizeof(invalid), hipMemcpyDeviceToHost, queue), "optimizer health readback");
        hip_check(hipStreamSynchronize(queue), "HIP training step completion");
        if (invalid) {
            session->poisoned = true;
            throw Failure(LFS_HIP_TRAIN_POISONED, "Nonfinite HIP optimizer result; session must not continue");
        }
        ++session->steps;
        if (session->profile)
            session->profile->print(session->steps);
        session->optimizer_pending = false;
    });
}

int lfs_hip_train_render(LfsHipTraining* session, float* output, size_t count, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && output, "Missing HIP render destination");
        session->healthy();
        require(count == session->frame->pixels * 3, "HIP render destination size mismatch");
        DeviceScope selected(session->device);
        session->forward();
        hip_check(hipMemcpyAsync(output, session->frame->image.pointer, count * sizeof(float), hipMemcpyDeviceToHost, session->stream.value), "HIP render download");
        hip_check(hipStreamSynchronize(session->stream.value), "HIP render completion");
    });
}

int lfs_hip_train_read_parameter(LfsHipTraining* session, uint32_t family, float* output, size_t count, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && output && family <= attributes.size(), "Invalid HIP parameter readback");
        session->healthy(false);
        if (family == 5) {
            require(session->sh && count == session->count * (session->sh->bases - 1) * 3, "HIP SH-rest destination size mismatch");
            DeviceScope selected(session->device);
            auto& sh = *session->sh;
            hipLaunchKernelGGL((sh_transfer<false>), dim3(blocks(count)), dim3(256), 0, session->stream.value,
                               sh.gradient.as(), sh.values.as(), session->count, sh.bases - 1, sh.slots);
            hip_check(hipGetLastError(), "HIP canonical SH-rest gather");
            hip_check(hipMemcpyAsync(output, sh.gradient.pointer, count * sizeof(float), hipMemcpyDeviceToHost, session->stream.value), "HIP SH-rest readback");
            hip_check(hipStreamSynchronize(session->stream.value), "HIP SH-rest readback completion");
            return;
        }
        require(count == session->count * attributes[family], "HIP parameter destination size mismatch");
        DeviceScope selected(session->device);
        hip_check(hipMemcpyAsync(output, session->parameter.as(session->layout.offsets[family]), count * sizeof(float),
                                 hipMemcpyDeviceToHost, session->stream.value),
                  "HIP parameter readback");
        hip_check(hipStreamSynchronize(session->stream.value), "HIP parameter readback completion");
    });
}

int lfs_hip_train_info(LfsHipTraining* session, LfsHipTrainInfo* output, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && output, "Missing HIP training info output");
        *output = {session->steps, session->budget.live, session->budget.peak, session->budget.limit,
                   session->intersections, session->bins ? session->bins->capacity : 0,
                   session->frame ? session->frame->width : 0, session->frame ? session->frame->height : 0, session->poisoned};
    });
}
int lfs_hip_train_set_budget(LfsHipTraining* session, uint64_t bytes, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session, "Missing HIP training session");
        session->healthy(false);
        require(bytes >= session->budget.live && bytes > 0, "HIP budget cannot be smaller than live allocations");
        session->budget.limit = bytes;
    });
}
int lfs_hip_train_set_loss(LfsHipTraining* session, const LfsHipTrainLoss* loss, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && loss && loss->struct_bytes == sizeof(*loss), "Invalid HIP loss descriptor");
        session->healthy(false);
        require(loss->kind <= LFS_HIP_LOSS_L1_SSIM && std::isfinite(loss->ssim_weight) &&
                    loss->ssim_weight >= 0 && loss->ssim_weight <= 1 && loss->valid_padding <= 1 &&
                    (loss->kind != LFS_HIP_LOSS_MSE || loss->ssim_weight == 0),
                "Unsupported HIP loss kind, SSIM weight or padding");
        DeviceScope selected(session->device);
        if ((loss->kind == LFS_HIP_LOSS_MSE || loss->ssim_weight == 0) && session->frame)
            session->frame->photo.reset();
        session->loss = *loss;
    });
}
int lfs_hip_train_set_sh(LfsHipTraining* session, const LfsHipTrainSh* config, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && config && config->struct_bytes == sizeof(*config), "Invalid HIP SH descriptor");
        session->healthy(false);
        require(session->steps == 0 || !session->sh, "HIP SH storage cannot be replaced after optimizer updates");
        require(config->max_degree <= 3 && config->active_degree <= config->max_degree &&
                    std::isfinite(config->rest_learning_rate) && config->rest_learning_rate >= 0,
                "Invalid HIP SH degree or learning rate");
        const size_t rest = (config->max_degree + 1) * (config->max_degree + 1) - 1;
        const size_t count = multiply(session->count, rest * 3);
        require(config->float_count == count && (rest || !config->canonical_rest), "HIP SH-rest source size mismatch");
        if (rest)
            finite_array(config->canonical_rest, count);
        DeviceScope selected(session->device);
        if (!rest) {
#ifdef LFS_HIP_FAST_RUNTIME
            if (session->joint) {
                session->joint->sh_packed.reset();
                session->joint->sh_bounds.reset();
            }
#endif
            session->sh.reset();
            return;
        }
        auto replacement = std::make_unique<Sh>(session->budget, session->count, *config);
        Buffer upload(session->budget, multiply(count, 4));
        replacement->values.zero(session->stream.value);
        replacement->first.zero(session->stream.value);
        replacement->second.zero(session->stream.value);
        hip_check(hipMemcpyAsync(upload.pointer, config->canonical_rest, upload.bytes, hipMemcpyHostToDevice, session->stream.value), "HIP initial SH-rest upload");
        hipLaunchKernelGGL((sh_transfer<true>), dim3(blocks(count)), dim3(256), 0, session->stream.value,
                           upload.as(), replacement->values.as(), session->count, rest, replacement->slots);
        hip_check(hipGetLastError(), "HIP initial SH-rest swizzle");
        hip_check(hipStreamSynchronize(session->stream.value), "HIP SH-rest initialization completion");
        session->sh.swap(replacement);
#ifdef LFS_HIP_FAST_RUNTIME
        if (session->joint) {
            session->joint->sh_packed.reset();
            session->joint->sh_bounds.reset();
            session->sh->first.release_owned();
            session->sh->second.release_owned();
        }
#endif
    });
}
int lfs_hip_train_attach_resident_sh(LfsHipTraining* session, const LfsHipTrainResidentSh* config, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && config && config->struct_bytes == sizeof(*config), "Invalid resident HIP SH descriptor");
        session->healthy(false);
        require(session->steps == 0, "Attach resident HIP SH before optimizer updates");
        require(!session->parameter.owns_pointer || session->parameter.external, "Resident SH requires a borrowed or imported model allocation");
        require(config->max_degree >= 1 && config->max_degree <= 3 && config->active_degree <= config->max_degree &&
                    std::isfinite(config->rest_learning_rate) && config->rest_learning_rate >= 0 && config->offset_bytes % 256 == 0,
                "Invalid resident HIP SH degree, rate or alignment");
        const size_t bases = (config->max_degree + 1) * (config->max_degree + 1);
        const size_t slots = ((bases - 1) * 3 + 3) / 4;
        const size_t bytes = multiply((session->count + 31) / 32 * 32, slots * 16);
        require(config->offset_bytes <= session->parameter.bytes && bytes <= session->parameter.bytes - config->offset_bytes,
                "Resident HIP SH region exceeds the model allocation");
        const uint64_t end = config->offset_bytes + bytes;
        for (unsigned family = 0; family < attributes.size(); ++family) {
            const uint64_t begin = session->layout.offsets[family] * sizeof(float);
            const uint64_t family_end = begin + session->count * attributes[family] * sizeof(float);
            require(end <= begin || config->offset_bytes >= family_end, "Resident HIP SH overlaps a live base parameter");
        }
        DeviceScope selected(session->device);
        LfsHipTrainDeviceModel resident{sizeof(resident), session->device,
                                        reinterpret_cast<uintptr_t>(session->parameter.pointer) + config->offset_bytes, 0};
        const LfsHipTrainSh descriptor{sizeof(descriptor), config->max_degree, config->active_degree, config->rest_learning_rate, nullptr, 0};
        auto replacement = std::make_unique<Sh>(session->budget, session->count, descriptor, &resident);
        replacement->first.zero(session->stream.value);
        replacement->second.zero(session->stream.value);
        session->invalid.zero(session->stream.value);
        hipLaunchKernelGGL(validate_resident_sh, dim3(blocks(session->count * (bases - 1) * 3)), dim3(256), 0, session->stream.value,
                           replacement->values.as(), session->count, bases - 1, slots, session->invalid.as<unsigned>());
        hip_check(hipGetLastError(), "HIP resident SH validation");
        unsigned invalid = 0;
        hip_check(hipMemcpyAsync(&invalid, session->invalid.pointer, sizeof(invalid), hipMemcpyDeviceToHost, session->stream.value), "HIP resident SH validation result");
        hip_check(hipStreamSynchronize(session->stream.value), "HIP resident SH attachment completion");
        require(!invalid, "Nonfinite resident HIP SH coefficient");
        session->sh.swap(replacement);
#ifdef LFS_HIP_FAST_RUNTIME
        if (session->joint) {
            session->joint->sh_packed.reset();
            session->joint->sh_bounds.reset();
            session->sh->first.release_owned();
            session->sh->second.release_owned();
        }
#endif
    });
}

int lfs_hip_train_set_sh_degree(LfsHipTraining* session, uint32_t degree, float rate, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && degree <= 3 && std::isfinite(rate) && rate >= 0, "Invalid HIP active SH degree or rate");
        session->healthy(false);
        require(session->sh ? degree <= session->sh->maximum : degree == 0 && rate == 0,
                "HIP active SH degree exceeds configured storage");
        if (session->sh) {
            session->sh->active = degree;
            session->sh->rate = rate;
        }
    });
}
int lfs_hip_train_loss(LfsHipTraining* session, float* output, LfsHipError* error) {
    return boundary(session, error, [&] {
        require(session && output, "Missing HIP loss session or output");
        session->healthy();
        DeviceScope selected(session->device);
        session->forward();
        auto& f = *session->frame;
        if (session->loss.kind == LFS_HIP_LOSS_L1_SSIM && session->loss.ssim_weight > 0)
            session->photo_forward();
        const unsigned chunks = blocks(f.pixels * 3);
        // Reuse the image-gradient allocation for diagnostics. No extra full
        // image readback or metric workspace; the next step overwrites it.
        hipLaunchKernelGGL(loss_reduce, dim3(chunks), dim3(256), 0, session->stream.value,
                           f.image.as(), f.target.as(), f.photo ? f.photo->map.as() : nullptr,
                           f.image_gradient.as(), f.pixels * 3, f.width, f.height, session->loss, session->loss_normalization());
        hip_check(hipGetLastError(), "HIP diagnostic loss partial reduction");
        hipLaunchKernelGGL(loss_finish, dim3(1), dim3(256), 0, session->stream.value,
                           f.image_gradient.as(), chunks, f.image_gradient.as(chunks));
        hip_check(hipGetLastError(), "HIP diagnostic loss reduction");
        float result = 0;
        hip_check(hipMemcpyAsync(&result, f.image_gradient.as(chunks), sizeof(result), hipMemcpyDeviceToHost, session->stream.value), "HIP diagnostic loss readback");
        hip_check(hipStreamSynchronize(session->stream.value), "HIP diagnostic loss completion");
        if (!std::isfinite(result)) {
            session->poisoned = true;
            throw Failure(LFS_HIP_TRAIN_POISONED, "Nonfinite HIP diagnostic loss");
        }
        *output = result;
    });
}
void lfs_hip_train_set_cancel(LfsHipTraining* session, int value) {
    if (session)
        session->cancelled.store(value != 0, std::memory_order_relaxed);
}
}

#include "application_fast.hpp"
#include "application_mcmc.hpp"
#include "application_refine.hpp"
#include "application_photo.hpp"
#include "application_adam.hpp"
#include "application_sh.hpp"
#include "application_image.hpp"

#include "application_extra.hpp"
#include "application_popspa.hpp"

#include "application_morton.hpp"
#include "application_lpips.hpp"
#include "application_structure.hpp"
#include "application_blob.hpp"
#include "application_controller.hpp"
#include "application_mrnf.hpp"
#include "application_bilateral.hpp"
#include "application_ppisp.hpp"
#include "application_geometry.hpp"
#include "application_masks.hpp"
#include "application_gsplat.hpp"
