/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "point_cloud_updates.hpp"
#include "core/tensor_execution.hpp"
#include "core/tensor_upload.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace lfs::vis {
    namespace {
        bool terminal(PointCloudUpdateState state) {
            return state != PointCloudUpdateState::Queued && state != PointCloudUpdateState::Uploading &&
                   state != PointCloudUpdateState::Publishing;
        }
        bool sameTarget(const PointCloudUpdateTarget& a, const PointCloudUpdateTarget& b) {
            return a.scene == b.scene && a.scene_generation == b.scene_generation && a.uuid == b.uuid;
        }
        void validateInput(const core::Tensor& tensor, bool colors) {
            if (!tensor.is_valid() || tensor.ndim() != 2 || tensor.size(1) != 3 || !tensor.is_contiguous())
                throw std::invalid_argument("Point-cloud inputs must be contiguous [N, 3] tensors");
            if (tensor.dtype() != core::DataType::Float32 && (!colors || tensor.dtype() != core::DataType::UInt8))
                throw std::invalid_argument(colors ? "Colors must have dtype float32 or uint8" : "Positions must have dtype float32");
        }
    } // namespace

    std::string PointCloudUpdateTicket::state() const {
        switch (state_.load(std::memory_order_acquire)) {
        case PointCloudUpdateState::Queued: return "queued";
        case PointCloudUpdateState::Uploading:
        case PointCloudUpdateState::Publishing: return "uploading";
        case PointCloudUpdateState::Published: return "published";
        case PointCloudUpdateState::Superseded: return "superseded";
        case PointCloudUpdateState::Cancelled: return "cancelled";
        case PointCloudUpdateState::Failed: return "failed";
        }
        return "failed";
    }
    std::string PointCloudUpdateTicket::error() const {
        // Written before the release-store of Failed and immutable afterwards.
        if (state_.load(std::memory_order_acquire) != PointCloudUpdateState::Failed)
            return {};
        const auto error = std::atomic_load_explicit(&error_, std::memory_order_acquire);
        return error ? *error : std::string{};
    }
    bool PointCloudUpdateTicket::cancel() {
        auto state = state_.load(std::memory_order_acquire);
        while (state == PointCloudUpdateState::Queued || state == PointCloudUpdateState::Uploading) {
            if (state_.compare_exchange_weak(state, PointCloudUpdateState::Cancelled, std::memory_order_acq_rel))
                return true;
        }
        return false;
    }
    void PointCloudUpdateTicket::supersede() {
        auto state = state_.load(std::memory_order_acquire);
        while (state == PointCloudUpdateState::Queued || state == PointCloudUpdateState::Uploading) {
            if (state_.compare_exchange_weak(state, PointCloudUpdateState::Superseded, std::memory_order_acq_rel))
                return;
        }
    }
    void PointCloudUpdateTicket::fail(std::string error) {
        auto state = state_.load(std::memory_order_acquire);
        if (terminal(state))
            return;
        // Only the first failure owns the immutable payload. Publish it before
        // Failed, so concurrent/repeated failures cannot race error() readers.
        std::shared_ptr<const std::string> expected;
        if (!std::atomic_compare_exchange_strong_explicit(&error_, &expected, std::make_shared<const std::string>(std::move(error)),
                                                          std::memory_order_acq_rel, std::memory_order_acquire))
            return;
        while (!terminal(state)) {
            if (state_.compare_exchange_weak(state, PointCloudUpdateState::Failed, std::memory_order_acq_rel))
                return;
        }
    }

    struct PointCloudUpdateManager::Request {
        PointCloudUpdateTarget target;
        PointCloudUpdateInput input;
        std::shared_ptr<PointCloudUpdateTicket> ticket = std::make_shared<PointCloudUpdateTicket>();
        Prepared prepared;
        core::Scene::PointCloudRetirement retired;
        size_t bytes = 0;
        bool resolved = false;
        bool active = false;
        bool ready = false;
        bool publishing = false;
    };

    PointCloudUpdateManager::PointCloudUpdateManager(Prepare prepare, std::function<void()> wake, bool resolve_on_scene)
        : prepare_(std::move(prepare)), wake_(std::move(wake)), resolve_on_scene_(resolve_on_scene), worker_([this] { run(); }) {}
    PointCloudUpdateManager::~PointCloudUpdateManager() { shutdown(); }
    void PointCloudUpdateManager::shutdown() {
        stop();
        if (worker_.joinable())
            worker_.join();
    }
    void PointCloudUpdateManager::stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cancelAll();
    }
    std::shared_ptr<PointCloudUpdateTicket> PointCloudUpdateManager::submit(
        PointCloudUpdateTarget target, PointCloudUpdateInput input) {
        validateInput(input.points, false);
        validateInput(input.colors, true);
        if (input.points.size(0) != input.colors.size(0))
            throw std::invalid_argument("Positions and colors must have the same point count");
        const auto backend = core::gpu_backend_of(input.points);
        const auto color_backend = core::gpu_backend_of(input.colors);
        if (backend && color_backend && backend != color_backend)
            throw std::invalid_argument("GPU inputs must use the same backend");
        if ((backend && backend != core::default_gpu_backend()) ||
            (color_backend && color_backend != core::default_gpu_backend()))
            throw std::invalid_argument("GPU source backend must match the viewer backend; convert on the producer first");
        if (input.centroid && (!std::isfinite(input.centroid->x) || !std::isfinite(input.centroid->y) || !std::isfinite(input.centroid->z)))
            throw std::invalid_argument("Centroid must be finite");
        if (!target.revision)
            throw std::invalid_argument("Async updates require a scene-owned point cloud");
        // Conservative reservation: source, staging, destination and conversion temporaries.
        const size_t n = input.points.size(0);
        if (n > kMaxBytes / kReservedBytesPerPoint)
            throw std::length_error("Point-cloud update exceeds the 1 GiB preparation budget");
        auto request = std::make_shared<Request>();
        request->bytes = n * kReservedBytesPerPoint;
        request->resolved = !resolve_on_scene_;
        request->target = std::move(target);
        request->input = std::move(input);
        {
            std::lock_guard lock(mutex_);
            if (stopping_)
                throw std::runtime_error("Point-cloud updates are shutting down");
            if (requests_.size() >= kMaxRequests || request->bytes > kMaxBytes - retained_bytes_)
                throw std::runtime_error("Point-cloud update queue is at its memory limit; retry after inputs are released");
            for (const auto& previous : requests_)
                if (sameTarget(previous->target, request->target))
                    previous->ticket->supersede();
            retained_bytes_ += request->bytes;
            requests_.push_back(request);
        }
        cv_.notify_one();
        if (wake_)
            wake_();
        return request->ticket;
    }
    void PointCloudUpdateManager::cancelAll() {
        {
            std::lock_guard lock(mutex_);
            for (const auto& request : requests_)
                request->ticket->cancel();
        }
        cv_.notify_one();
    }
    bool PointCloudUpdateManager::hasReady() const {
        std::lock_guard lock(mutex_);
        return std::any_of(requests_.begin(), requests_.end(), [](const auto& r) { return (r->ready || !r->resolved) && !r->publishing; });
    }
    size_t PointCloudUpdateManager::retainedBytes() const {
        std::lock_guard lock(mutex_);
        return retained_bytes_;
    }
    size_t PointCloudUpdateManager::retainedRequests() const {
        std::lock_guard lock(mutex_);
        return requests_.size();
    }

    void PointCloudUpdateManager::resolveQueued(const Resolve& resolve) {
        std::vector<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(mutex_);
            for (const auto& request : requests_) {
                if (!request->resolved && !request->active && !request->publishing && !terminal(request->ticket->state_.load())) {
                    request->publishing = true; // prevents worker retirement until the local refs are gone
                    pending.push_back(request);
                }
            }
        }
        for (const auto& request : pending) {
            try {
                resolve(request->target, request->input);
                size_t points = request->input.points.size(0);
                for (const auto& companion : request->input.companions) {
                    const auto count = static_cast<size_t>(companion.cloud->size());
                    if (count > kMaxBytes / kReservedBytesPerPoint || points > kMaxBytes / kReservedBytesPerPoint - count)
                        throw std::length_error("Merged point cloud exceeds the preparation budget");
                    points += count;
                }
                const auto bytes = points * kReservedBytesPerPoint;
                std::lock_guard lock(mutex_);
                if (bytes > request->bytes) {
                    const auto growth = bytes - request->bytes;
                    if (growth > kMaxBytes - retained_bytes_)
                        throw std::length_error("Merged point cloud exceeds the queue memory limit");
                    retained_bytes_ += growth;
                } else {
                    retained_bytes_ -= request->bytes - bytes;
                }
                request->bytes = bytes;
            } catch (const std::exception& e) {
                // LFS-CENSUS-OK(empty-catch): the ticket exposes the failure to its Python caller.
                request->ticket->fail(e.what());
            }
        }
        std::vector<Request*> resolved;
        for (const auto& request : pending)
            resolved.push_back(request.get());
        pending.clear();
        {
            std::lock_guard lock(mutex_);
            for (auto* request : resolved) {
                request->resolved = true;
                request->publishing = false;
            }
        }
        cv_.notify_one();
    }

    void PointCloudUpdateManager::publishReady(const Publish& publish) {
        std::vector<std::shared_ptr<Request>> ready;
        {
            std::lock_guard lock(mutex_);
            for (const auto& request : requests_) {
                if (!request->ready || request->publishing)
                    continue;
                auto state = PointCloudUpdateState::Uploading;
                if (request->ticket->state_.compare_exchange_strong(state, PointCloudUpdateState::Publishing, std::memory_order_acq_rel)) {
                    request->publishing = true;
                    ready.push_back(request);
                }
            }
        }
        for (const auto& request : ready) {
            try {
                publish(request->target, request->prepared, request->retired);
                request->ticket->state_.store(PointCloudUpdateState::Published, std::memory_order_release);
            } catch (const std::exception& e) {
                // LFS-CENSUS-OK(empty-catch): the ticket exposes the failure to its Python caller.
                request->ticket->fail(e.what());
            }
        }
        std::vector<Request*> retired;
        for (const auto& request : ready)
            retired.push_back(request.get());
        ready.clear();
        {
            std::lock_guard lock(mutex_);
            for (auto* request : retired) {
                request->publishing = false;
                request->ready = false;
            }
        }
        cv_.notify_one();
    }

    void PointCloudUpdateManager::run() {
        for (;;) {
            std::shared_ptr<Request> work;
            bool discard = false;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&] {
                    return (stopping_ && requests_.empty()) || std::any_of(requests_.begin(), requests_.end(), [](const auto& r) {
                               return !r->active && !r->publishing && (terminal(r->ticket->state_.load()) || (r->resolved && !r->ready));
                           });
                });
                const auto it = std::find_if(requests_.begin(), requests_.end(), [](const auto& r) {
                    return !r->active && !r->publishing && (terminal(r->ticket->state_.load()) || (r->resolved && !r->ready));
                });
                if (it == requests_.end()) {
                    if (stopping_ && requests_.empty())
                        return;
                    continue;
                }
                work = *it;
                auto state = PointCloudUpdateState::Queued;
                discard = !work->ticket->state_.compare_exchange_strong(state, PointCloudUpdateState::Uploading);
                if (discard) {
                    retained_bytes_ -= work->bytes;
                    requests_.erase(it);
                } else {
                    work->active = true;
                }
            }
            if (discard) {
                const auto ticket = work->ticket;
                work.reset(); // potentially expensive native/Python/GPU destruction stays here
                ticket->inputs_released_.store(true, std::memory_order_release);
                continue;
            }
            try {
                work->prepared = prepare_(work->input, [ticket = work->ticket] {
                    ticket->inputs_released_.store(true, std::memory_order_release);
                });
            } catch (const std::exception& e) {
                // LFS-CENSUS-OK(empty-catch): the ticket exposes the failure to its Python caller.
                work->ticket->fail(e.what());
            }
            // prepare must settle its own GPU work before returning, including on failure.
            work->input = {};
            work->ticket->inputs_released_.store(true, std::memory_order_release);
            {
                std::lock_guard lock(mutex_);
                work->active = false;
                work->ready = !terminal(work->ticket->state_.load());
            }
            if (wake_)
                wake_();
        }
    }

    PointCloudUpdateManager::Prepare preparePointCloudUpdate(
        core::SplatTensorAllocator allocator) {
        return [allocator = std::move(allocator)](
                   PointCloudUpdateInput& input, const std::function<void()>& release_inputs) {
            using namespace core;
            const auto backend = gpu_backend_of(input.points).value_or(gpu_backend_of(input.colors).value_or(default_gpu_backend()));
            if (backend != default_gpu_backend())
                throw std::invalid_argument("GPU source backend must match the viewer backend; convert on the producer first");
            TensorWorkQueue queue(backend, TensorWorkQueue::Mode::Independent);
            TensorWorkQueue::Scope scope(queue);
            const auto n = input.points.size(0);
            const TensorShape shape{n, size_t{3}};
            auto cloud = std::make_shared<PointCloud>();
            glm::vec3 centroid = input.centroid.value_or(glm::vec3(0.0f));
            TensorCompletion completion;
            if (n == 0) {
                cloud->means = Tensor::empty(shape, Device::CPU, DataType::Float32);
                cloud->colors = Tensor::empty(shape, Device::CPU, DataType::Float32);
                input.points = {};
                input.colors = {};
                input.source_owners.reset();
                release_inputs();
                centroid = glm::vec3(0.0f);
            } else {
                Tensor points_host, colors_host, colors_gpu, centroid_gpu;
                TensorUpload points_upload, colors_upload;
                // Request pinned CPU staging (the allocator may fall back to pageable
                // storage). Published tensors never retain producer pointers, and
                // these snapshot copies never run on the viewer.
                if (input.points.device() == Device::CPU) {
                    points_host = Tensor::empty(shape, Device::CPU, DataType::Float32, /*use_pinned=*/true);
                    std::memcpy(points_host.data_ptr(), input.points.data_ptr(), points_host.bytes());
                    const auto* points = points_host.ptr<float>();
                    glm::dvec3 sum(0.0);
                    for (size_t i = 0; i < n * 3; ++i) {
                        if (!std::isfinite(points[i]))
                            throw std::invalid_argument("Positions must be finite");
                        if (!input.centroid)
                            sum[i % 3] += points[i];
                    }
                    if (!input.centroid)
                        centroid = glm::vec3(sum / static_cast<double>(n));
                }
                if (input.colors.device() == Device::CPU) {
                    colors_host = Tensor::empty(shape, Device::CPU, DataType::Float32, /*use_pinned=*/true);
                    auto* colors = colors_host.ptr<float>();
                    const auto* f = input.colors.dtype() == DataType::Float32 ? input.colors.ptr<float>() : nullptr;
                    const auto* u = input.colors.dtype() == DataType::UInt8 ? input.colors.ptr<uint8_t>() : nullptr;
                    for (size_t i = 0; i < n * 3; ++i) {
                        colors[i] = f ? f[i] : static_cast<float>(u[i]) / 255.0f;
                        if (!std::isfinite(colors[i]) || colors[i] < 0 || colors[i] > 1)
                            throw std::invalid_argument("Float colors must be finite and in [0, 1]");
                    }
                }
                const bool cpu_inputs = input.points.device() == Device::CPU && input.colors.device() == Device::CPU;
                if (cpu_inputs) {
                    input.points = {};
                    input.colors = {};
                    input.source_owners.reset();
                    release_inputs();
                }
                // Queue and uploads own in-flight work even if any subsequent step throws.
                completion = queue.execute([&] {
                    cloud->means = allocator ? allocator(shape, n, DataType::Float32, "PointCloud.means")
                                             : Tensor::empty(shape, Device::GPU, DataType::Float32);
                    cloud->colors = allocator ? allocator(shape, n, DataType::Float32, "PointCloud.colors")
                                              : Tensor::empty(shape, Device::GPU, DataType::Float32);
                    if (points_host.is_valid())
                        points_upload.enqueue(cloud->means, points_host, TensorExecutionTarget(queue));
                    else {
                        cloud->means.copy_from(input.points);
                        if (!input.centroid)
                            centroid_gpu = cloud->means.mean(0);
                    }
                    if (colors_host.is_valid())
                        colors_upload.enqueue(cloud->colors, colors_host, TensorExecutionTarget(queue));
                    else {
                        if (input.colors.dtype() == DataType::UInt8) {
                            colors_gpu = input.colors.to(DataType::Float32) / 255.0f;
                            cloud->colors.copy_from(colors_gpu);
                        } else
                            cloud->colors.copy_from(input.colors);
                    }
                });
                completion.wait(); // only this independent queue; never a host-wide drain
                if (centroid_gpu.is_valid()) {
                    const auto host = centroid_gpu.cpu();
                    const auto* p = host.ptr<float>();
                    centroid = {p[0], p[1], p[2]};
                }
                if (!cpu_inputs) {
                    // Small scalar readbacks are worker-only and ordered on the upload queue.
                    if (!cloud->means.isfinite().all().item<bool>() || !cloud->colors.isfinite().all().item<bool>() ||
                        cloud->colors.min().item<float>() < 0 || cloud->colors.max().item<float>() > 1)
                        throw std::invalid_argument("Positions must be finite and colors must be in [0, 1]");
                    input.points = {};
                    input.colors = {};
                    input.source_owners.reset();
                    release_inputs();
                }
            }
            std::shared_ptr<PointCloud> merged;
            if (!input.companions.empty() && (input.target_visible || input.companions.size() > 1)) {
                std::vector<Tensor> positions, colors;
                const auto append = [&](const PointCloud& source, const glm::mat4& transform) {
                    if (source.size() == 0)
                        return;
                    auto means = source.means.device() == Device::GPU ? source.means : source.means.to(Device::GPU);
                    auto color = source.colors.device() == Device::GPU ? source.colors : source.colors.to(Device::GPU);
                    if (color.dtype() == DataType::UInt8)
                        color = color.to(DataType::Float32) / 255.0f;
                    std::vector<float> rotation, offset{transform[3][0], transform[3][1], transform[3][2]};
                    // GLM stores columns. Flattening them into the row-major tensor
                    // gives R^T, so row positions P * R^T + t match GLM's M * p.
                    for (int column = 0; column < 3; ++column)
                        for (int row = 0; row < 3; ++row)
                            rotation.push_back(transform[column][row]);
                    auto r = Tensor::from_vector(rotation, {size_t{3}, size_t{3}}, Device::CPU).to(Device::GPU);
                    auto t = Tensor::from_vector(offset, {size_t{1}, size_t{3}}, Device::CPU).to(Device::GPU);
                    positions.push_back(means.matmul(r).add(t));
                    colors.push_back(std::move(color));
                };
                completion = queue.execute([&] {
                    for (size_t i = 0; i <= input.companions.size(); ++i) {
                        if (input.target_visible && i == input.target_index)
                            append(*cloud, input.transform);
                        if (i < input.companions.size()) {
                            const auto& companion = input.companions[i];
                            append(*companion.cloud, companion.transform);
                        }
                    }
                    auto pos = Tensor::cat(positions, 0);
                    auto col = Tensor::cat(colors, 0);
                    const auto count = pos.size(0);
                    merged = std::make_shared<PointCloud>();
                    merged->means = allocator ? allocator(pos.shape(), count, DataType::Float32, "PointCloud.merged.means") : Tensor::empty_like(pos);
                    merged->colors = allocator ? allocator(col.shape(), count, DataType::Float32, "PointCloud.merged.colors") : Tensor::empty_like(col);
                    merged->means.copy_from(pos);
                    merged->colors.copy_from(col);
                });
                completion.wait();
            }
            // The independent queue retires on this worker. Published tensors
            // use a stable execution target; renderers acquire their own native
            // views and dependencies through the public tensor interop seam.
            const auto target = TensorExecutionTarget::default_queue(backend);
            cloud->means.set_stream(target);
            cloud->colors.set_stream(target);
            if (merged) {
                merged->means.set_stream(target);
                merged->colors.set_stream(target);
            }
            return PointCloudUpdateManager::Prepared{std::move(cloud), centroid, std::move(merged)};
        };
    }
} // namespace lfs::vis
