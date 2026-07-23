/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "control_boundary.hpp"
#include "core/tensor.hpp"

#include <atomic>
#include <cstddef>
#include <expected>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace lfs::core {
    class SplatData;
} // namespace lfs::core

namespace lfs::training {

    enum class CommandTarget { Model,
                               Optimizer,
                               Session };

    enum class SelectionKind { All,
                               Range,
                               Indices };

    struct Selection {
        SelectionKind kind = SelectionKind::All;
        int64_t start = 0;
        int64_t end = 0; // exclusive
        std::vector<int64_t> indices;
    };

    enum class TrainingPhase {
        Idle,
        IterationStart,
        Forward,
        Backward,
        OptimizerStep,
        SafeControl
    };

    enum class ArgType { Int,
                         Float,
                         Bool,
                         String,
                         IntList,
                         FloatList };

    using ArgValue = std::variant<int64_t, double, bool, std::string, std::vector<int64_t>, std::vector<double>>;

    struct ArgSpec {
        std::string name;
        ArgType type;
        bool required = true;
        std::optional<std::string> description;
    };

    struct OperationInfo {
        std::string name;
        CommandTarget target;
        std::vector<SelectionKind> selectors;
        std::vector<ArgSpec> args;
        std::string description;
    };

    struct MutableFieldInfo {
        std::string name;
        CommandTarget target;
        std::string shape;
        std::string description;
        bool writable = true;
    };

    struct Command {
        CommandTarget target;
        std::string op;
        Selection selection;
        std::unordered_map<std::string, ArgValue> args;
    };

    struct TrainingSnapshot {
        int iteration = 0;
        int max_iterations = 0;
        float loss = 0.0f;
        std::size_t num_gaussians = 0;
        bool is_refining = false;
        bool is_paused = false;
        bool is_running = false;
        bool stop_requested = false;
        TrainingPhase phase = TrainingPhase::Idle;
        Trainer* trainer = nullptr; // non-owning
    };

    struct LossHistoryPoint {
        int iteration;
        float loss;
    };

    class CommandCenter {
    public:
        static LFS_BRIDGE_API CommandCenter& instance();

        void set_phase(TrainingPhase phase);

        void update_snapshot(const HookContext& ctx, int max_iterations, bool is_paused, bool is_running, bool stop_requested, TrainingPhase phase);
        void clear_snapshot(const Trainer* trainer);

        [[nodiscard]] TrainingSnapshot snapshot() const;
        [[nodiscard]] std::vector<LossHistoryPoint> loss_history() const;
        void clear_loss_history();

        std::expected<void, std::string> execute(const Command& cmd);

        // Enqueue a command to be executed later on the training thread.
        void enqueue_command(const Command& cmd);
        void drain_enqueued(TrainingSnapshot& view);

        std::vector<OperationInfo> operations(std::optional<CommandTarget> target = std::nullopt) const;
        std::vector<MutableFieldInfo> mutables(std::optional<CommandTarget> target = std::nullopt) const;

    private:
        CommandCenter();

        std::expected<void, std::string> exec_model(const Command& cmd, TrainingSnapshot& view);
        std::expected<void, std::string> exec_optimizer(const Command& cmd, TrainingSnapshot& view);
        std::expected<void, std::string> exec_session(const Command& cmd, TrainingSnapshot& view);

        // Helpers
        static std::expected<core::Tensor, std::string> build_row_mask(const Selection& sel, std::size_t rows, core::Device device);
        static std::expected<void, std::string> apply_set(core::Tensor& tensor, const core::Tensor& mask_rows, const ArgValue& value);
        static std::expected<void, std::string> apply_scale(core::Tensor& tensor, const core::Tensor& mask_rows, double factor);
        static std::expected<void, std::string> apply_clamp(core::Tensor& tensor, const core::Tensor& mask_rows, const std::optional<double>& minv, const std::optional<double>& maxv);

        static std::expected<core::Tensor*, std::string> resolve_attribute(lfs::core::SplatData& model, const std::string& name, size_t& row_dim_out);

        // Registry
        std::vector<OperationInfo> ops_;
        std::vector<MutableFieldInfo> mutable_fields_;

        std::vector<Command> pending_commands_;

        mutable std::mutex mutex_;
        TrainingSnapshot snapshot_{};
        std::atomic<TrainingPhase> phase_{TrainingPhase::Idle};
        std::vector<LossHistoryPoint> loss_history_;
        int last_recorded_iteration_ = -1;
    };

} // namespace lfs::training
