/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_completion.hpp"
#include "core/tensor_upload.hpp"
#include "visualizer/nodes/modifier_manager.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace lfs::vis {

    struct ModifierObjectSnapshot {
        core::Uuid uuid;
        std::string name;
        core::NodeType type = core::NodeType::SPLAT;
        glm::mat4 world{1.0f};
        std::shared_ptr<const core::SplatData> splats;
        std::shared_ptr<const core::PointCloud> points;
        std::shared_ptr<const core::MeshData> mesh;
        ModifierStack stack;
    };

    struct ModifierEvaluationRequest {
        std::uint64_t generation = 0;
        std::uint64_t source_generation = 0;
        float seconds = 0.0f;
        float frames_per_second = 24.0f;
        std::vector<ModifierObjectSnapshot> objects;
        std::vector<lfs::nodes::EvaluationCamera> cameras;
        std::unordered_map<std::string, nlohmann::json> trees;
        std::vector<core::Uuid> targets;
        std::shared_ptr<core::TensorCompletion> inputs_ready;
        std::chrono::steady_clock::time_point requested_at;
        bool bake = false;
    };

    struct ModifierHostResult {
        core::Uuid uuid;
        ModifierEvaluation evaluation;
        std::shared_ptr<core::SplatData> splats;
        std::shared_ptr<core::PointCloud> points;
        std::shared_ptr<core::MeshData> mesh;
        std::unordered_map<std::string, core::Tensor> previews;
        bool enabled = false;
        std::uint64_t output_key = 0;
    };

    struct ModifierWorkerResult {
        std::uint64_t generation = 0;
        std::unordered_map<core::Uuid, ModifierHostResult> hosts;
        std::shared_ptr<core::TensorFence> ready;
        bool cancelled = false;
        std::chrono::steady_clock::time_point requested_at;
    };

    // A single queue owner. Only immutable captures cross into the worker; only
    // fence-complete results cross back. Live Scene and DOM are never accessed.
    class ModifierEvaluationWorker {
    public:
        explicit ModifierEvaluationWorker(const lfs::nodes::NodeTypeRegistry& registry);
        ~ModifierEvaluationWorker();
        void invalidate(std::uint64_t generation);
        void submit(ModifierEvaluationRequest request);
        void retire(ModifierWorkerResult result);
        [[nodiscard]] std::optional<ModifierWorkerResult> takeReady();
        void wait(std::uint64_t generation);
        [[nodiscard]] ModifierWorkerProgress progress() const;
        [[nodiscard]] nlohmann::json performance(bool reset);

    private:
        void run(std::stop_token stop);
        ModifierWorkerResult evaluate(const ModifierEvaluationRequest& request);

        const lfs::nodes::NodeTypeRegistry& registry_;
        mutable std::mutex mutex_;
        std::condition_variable_any changed_;
        std::optional<ModifierEvaluationRequest> pending_;
        std::optional<ModifierWorkerResult> ready_;
        std::vector<ModifierWorkerResult> retired_;
        ModifierWorkerProgress progress_;
        std::atomic<std::uint64_t> generation_{0};
        std::uint64_t finished_generation_ = 0;
        std::uint64_t requests_ = 0;
        std::uint64_t evaluations_ = 0;
        std::uint64_t discarded_ = 0;
        std::unordered_map<std::string, std::uint64_t> node_runs_;
        std::unordered_map<std::string, lfs::nodes::EvalCache> caches_;
        std::unordered_map<core::Uuid, lfs::nodes::Geometry> sources_;
        std::unordered_map<core::Uuid, std::shared_ptr<const core::MeshData>> source_meshes_;
        std::unordered_map<core::Uuid, ModifierHostResult> previous_hosts_;
        std::uint64_t source_generation_ = 0;
        lfs::nodes::GeometryDeviceCache source_devices_;
        std::unique_ptr<core::TensorWorkQueue> queue_;
        std::jthread thread_;
    };

} // namespace lfs::vis
