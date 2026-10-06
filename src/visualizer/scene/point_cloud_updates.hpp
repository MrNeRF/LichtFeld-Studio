/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

namespace lfs::vis {
    enum class PointCloudUpdateState { Queued,
                                       Uploading,
                                       Publishing,
                                       Published,
                                       Superseded,
                                       Cancelled,
                                       Failed };

    // Tickets never own payloads, GPU events, or Python objects. Their destruction cannot wait.
    class LFS_VIS_API PointCloudUpdateTicket {
    public:
        [[nodiscard]] std::string state() const;
        [[nodiscard]] std::string error() const;
        [[nodiscard]] bool inputsReleased() const { return inputs_released_.load(std::memory_order_acquire); }
        bool cancel();

    private:
        friend class PointCloudUpdateManager;
        std::atomic<PointCloudUpdateState> state_{PointCloudUpdateState::Queued};
        std::atomic<bool> inputs_released_{false};
        std::string error_;
        void fail(std::string error);
        void supersede();
    };

    struct PointCloudUpdateTarget {
        core::Scene* scene = nullptr; // compared only; never dereferenced on the worker
        uint64_t scene_generation = 0;
        core::Uuid uuid;
        std::shared_ptr<std::atomic<uint64_t>> revision;
        uint64_t expected_revision = 0;
        uint64_t render_generation = 0;
        std::shared_ptr<std::atomic<uint64_t>> scene_epoch;
    };

    struct PointCloudUpdateCompanion {
        std::shared_ptr<core::PointCloud> cloud;
        glm::mat4 transform{1.0f};
    };
    struct PointCloudUpdateInput {
        core::Tensor points;
        core::Tensor colors;
        std::optional<glm::vec3> centroid;
        std::shared_ptr<void> source_owners;
        std::vector<PointCloudUpdateCompanion> companions;
        glm::mat4 transform{1.0f};
        bool target_visible = true;
    };

    class LFS_VIS_API PointCloudUpdateManager {
    public:
        struct Prepared {
            std::shared_ptr<core::PointCloud> cloud;
            glm::vec3 centroid{0.0f};
            std::shared_ptr<core::PointCloud> merged;
        };
        using Prepare = std::function<Prepared(PointCloudUpdateInput&, const std::function<void()>&)>;
        using Publish = std::function<void(const PointCloudUpdateTarget&, const Prepared&, core::Scene::PointCloudRetirement&)>;
        // Limits include obsolete requests until their sources and GPU work are safely retired.
        static constexpr size_t kMaxRequests = 32;
        static constexpr size_t kMaxBytes = 1024ull * 1024 * 1024;
        PointCloudUpdateManager(Prepare prepare, std::function<void()> wake, bool resolve_on_scene = false);
        using Resolve = std::function<void(PointCloudUpdateTarget&, PointCloudUpdateInput&)>;
        void resolveQueued(const Resolve& resolve);
        ~PointCloudUpdateManager();
        PointCloudUpdateManager(const PointCloudUpdateManager&) = delete;
        PointCloudUpdateManager& operator=(const PointCloudUpdateManager&) = delete;
        std::shared_ptr<PointCloudUpdateTicket> submit(PointCloudUpdateTarget target, PointCloudUpdateInput input);
        // Called only on the scene thread. Never waits for preparation or GPU completion.
        void publishReady(const Publish& publish);
        void cancelAll();
        // Join before Python finalization or graphics teardown, with no GIL held.
        void shutdown();
        [[nodiscard]] bool hasReady() const;
        [[nodiscard]] size_t retainedBytes() const;
        [[nodiscard]] size_t retainedRequests() const;

    private:
        struct Request;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::vector<std::shared_ptr<Request>> requests_;
        Prepare prepare_;
        std::function<void()> wake_;
        bool stopping_ = false;
        bool resolve_on_scene_ = false;
        size_t retained_bytes_ = 0;
        std::thread worker_;
        void run();
    };

    // All allocation, staging, conversion, validation of values, GPU work, and
    // renderer-resource import happens in this callback, on the native worker.
    LFS_VIS_API PointCloudUpdateManager::Prepare preparePointCloudUpdate(
        core::SplatTensorAllocator allocator,
        std::function<void(core::PointCloud&, core::TensorCompletion)> prepare_renderer,
        void* vulkan_device = nullptr);
} // namespace lfs::vis
