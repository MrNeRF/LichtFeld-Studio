/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "command_api.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <cmath>

namespace lfs::training {

    namespace {
        core::Tensor expand_mask(const core::Tensor& row_mask, const core::TensorShape& target_shape) {
            if (row_mask.shape().rank() == 0 || target_shape.rank() == 0) {
                return row_mask;
            }
            if (row_mask.shape().rank() == 1 && target_shape.rank() > 1) {
                auto expanded = row_mask.unsqueeze(-1);
                std::vector<size_t> dims = target_shape.dims();
                expanded = expanded.expand(core::TensorShape{dims});
                return expanded;
            }
            return row_mask;
        }

        core::Tensor make_full_like_mask(const core::Tensor& mask, double value) {
            return core::Tensor::full(mask.shape(), static_cast<float>(value), mask.device(), core::DataType::Float32);
        }

        std::string target_name(CommandTarget t) {
            switch (t) {
            case CommandTarget::Model:
                return "model";
            case CommandTarget::Optimizer:
                return "optimizer";
            case CommandTarget::Session:
                return "session";
            }
            return "unknown";
        }

        const char* phase_name(TrainingPhase p) {
            switch (p) {
            case TrainingPhase::Idle:
                return "Idle";
            case TrainingPhase::IterationStart:
                return "IterationStart";
            case TrainingPhase::Forward:
                return "Forward";
            case TrainingPhase::Backward:
                return "Backward";
            case TrainingPhase::OptimizerStep:
                return "OptimizerStep";
            case TrainingPhase::SafeControl:
                return "SafeControl";
            }
            return "Unknown";
        }

        bool argument_matches_type(const ArgValue& value, ArgType type) {
            switch (type) {
            case ArgType::Int:
                return std::holds_alternative<int64_t>(value);
            case ArgType::Float:
                return std::holds_alternative<double>(value);
            case ArgType::Bool:
                return std::holds_alternative<bool>(value);
            case ArgType::String:
                return std::holds_alternative<std::string>(value);
            case ArgType::IntList:
                return std::holds_alternative<std::vector<int64_t>>(value);
            case ArgType::FloatList:
                return std::holds_alternative<std::vector<double>>(value);
            }
            return false;
        }

        bool argument_is_finite(const ArgValue& value) {
            if (const auto* scalar = std::get_if<double>(&value)) {
                return std::isfinite(*scalar);
            }
            if (const auto* values = std::get_if<std::vector<double>>(&value)) {
                return std::ranges::all_of(*values, [](double item) { return std::isfinite(item); });
            }
            return true;
        }

        bool is_known_model_attribute(const std::string& name) {
            return name == "means" || name == "position" || name == "scaling" || name == "scale" ||
                   name == "rotation" || name == "rot" || name == "opacity" || name == "alpha" ||
                   name == "sh0" || name == "shN" || name == "sh";
        }

        std::optional<ParamType> param_type_from_attribute(const std::string& name) {
            if (name == "means")
                return ParamType::Means;
            if (name == "scaling")
                return ParamType::Scaling;
            if (name == "rotation")
                return ParamType::Rotation;
            if (name == "opacity")
                return ParamType::Opacity;
            if (name == "sh0")
                return ParamType::Sh0;
            if (name == "shN")
                return ParamType::ShN;
            return std::nullopt;
        }
    } // namespace

    CommandCenter& CommandCenter::instance() {
        static CommandCenter inst;
        return inst;
    }

    CommandCenter::CommandCenter() {
        ops_.push_back({.name = "set_attribute",
                        .target = CommandTarget::Model,
                        .selectors = {SelectionKind::All, SelectionKind::Range, SelectionKind::Indices},
                        .args = {{"attribute", ArgType::String, true, "Attribute name (means, scaling, rotation, opacity, sh0, shN)"},
                                 {"value", ArgType::Float, false, "Scalar value"},
                                 {"values", ArgType::FloatList, false, "Vector value (broadcast)"}},
                        .description = "Set attribute values for selected splats (scalar or per-dim vector)."});

        ops_.push_back({.name = "scale_attribute",
                        .target = CommandTarget::Model,
                        .selectors = {SelectionKind::All, SelectionKind::Range, SelectionKind::Indices},
                        .args = {{"attribute", ArgType::String, true, "Attribute name"},
                                 {"factor", ArgType::Float, true, "Multiplicative scale"}},
                        .description = "Scale attribute by factor for selected splats."});

        ops_.push_back({.name = "clamp_attribute",
                        .target = CommandTarget::Model,
                        .selectors = {SelectionKind::All, SelectionKind::Range, SelectionKind::Indices},
                        .args = {{"attribute", ArgType::String, true, "Attribute name"},
                                 {"min", ArgType::Float, false, "Optional min"},
                                 {"max", ArgType::Float, false, "Optional max"}},
                        .description = "Clamp attribute values for selected splats."});

        ops_.push_back({.name = "set_lr",
                        .target = CommandTarget::Optimizer,
                        .selectors = {SelectionKind::All},
                        .args = {{"value", ArgType::Float, true, "Learning rate"}},
                        .description = "Set global learning rate."});

        ops_.push_back({.name = "scale_lr",
                        .target = CommandTarget::Optimizer,
                        .selectors = {SelectionKind::All},
                        .args = {{"factor", ArgType::Float, true, "Scale factor"}},
                        .description = "Scale global learning rate."});

        ops_.push_back({.name = "pause",
                        .target = CommandTarget::Session,
                        .selectors = {SelectionKind::All},
                        .args = {},
                        .description = "Request pause."});

        ops_.push_back({.name = "resume",
                        .target = CommandTarget::Session,
                        .selectors = {SelectionKind::All},
                        .args = {},
                        .description = "Request resume."});

        ops_.push_back({.name = "request_stop",
                        .target = CommandTarget::Session,
                        .selectors = {SelectionKind::All},
                        .args = {},
                        .description = "Request graceful stop."});

        // Mutable fields
        mutable_fields_.push_back({"means", CommandTarget::Model, "[N,3]", "Gaussian means", true});
        mutable_fields_.push_back({"scaling", CommandTarget::Model, "[N,3]", "Log scaling", true});
        mutable_fields_.push_back({"rotation", CommandTarget::Model, "[N,4]", "Quaternion rotation", true});
        mutable_fields_.push_back({"opacity", CommandTarget::Model, "[N]", "Opacity logits", true});
        mutable_fields_.push_back({"sh0", CommandTarget::Model, "[N,3]", "SH0 coefficients", true});
        mutable_fields_.push_back({"shN", CommandTarget::Model, "[N,?]", "Higher-order SH coefficients", true});
    }

    void CommandCenter::set_phase(TrainingPhase phase) {
        phase_.store(phase, std::memory_order_relaxed);
    }

    void CommandCenter::update_snapshot(const HookContext& ctx, int max_iterations, bool is_paused, bool is_running, bool stop_requested, TrainingPhase phase) {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.iteration = ctx.iteration;
        snapshot_.max_iterations = max_iterations;
        snapshot_.loss = ctx.loss;
        snapshot_.num_gaussians = ctx.num_gaussians;
        snapshot_.is_refining = ctx.is_refining;
        snapshot_.trainer = ctx.trainer;
        snapshot_.is_paused = is_paused;
        snapshot_.is_running = is_running;
        snapshot_.stop_requested = stop_requested;
        snapshot_.phase = phase;

        if (ctx.iteration > last_recorded_iteration_ && ctx.loss > 0.0f) {
            loss_history_.push_back({ctx.iteration, ctx.loss});
            last_recorded_iteration_ = ctx.iteration;
        }
    }

    void CommandCenter::clear_snapshot(const Trainer* trainer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (snapshot_.trainer != trainer) {
            return;
        }
        snapshot_ = {};
        pending_commands_.clear();
        phase_.store(TrainingPhase::Idle, std::memory_order_relaxed);
    }

    TrainingSnapshot CommandCenter::snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        TrainingSnapshot snap = snapshot_;
        snap.phase = phase_.load(std::memory_order_relaxed);
        return snap;
    }

    std::vector<LossHistoryPoint> CommandCenter::loss_history() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return loss_history_;
    }

    void CommandCenter::clear_loss_history() {
        std::lock_guard<std::mutex> lock(mutex_);
        loss_history_.clear();
        last_recorded_iteration_ = -1;
    }

    std::vector<OperationInfo> CommandCenter::operations(std::optional<CommandTarget> target) const {
        if (!target) {
            return ops_;
        }
        std::vector<OperationInfo> filtered;
        for (const auto& op : ops_) {
            if (op.target == *target) {
                filtered.push_back(op);
            }
        }
        return filtered;
    }

    std::vector<MutableFieldInfo> CommandCenter::mutables(std::optional<CommandTarget> target) const {
        if (!target) {
            return mutable_fields_;
        }
        std::vector<MutableFieldInfo> filtered;
        for (const auto& f : mutable_fields_) {
            if (f.target == *target) {
                filtered.push_back(f);
            }
        }
        return filtered;
    }

    void CommandCenter::enqueue_command(const Command& cmd) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_commands_.push_back(cmd);
    }

    void CommandCenter::drain_enqueued(TrainingSnapshot& view) {
        std::vector<Command> local;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            local.swap(pending_commands_);
        }
        for (const auto& cmd : local) {
            switch (cmd.target) {
            case CommandTarget::Model:
                if (auto result = exec_model(cmd, view); !result)
                    LOG_WARN("exec_model failed for op '{}': {}", cmd.op, result.error());
                break;
            case CommandTarget::Optimizer:
                if (auto result = exec_optimizer(cmd, view); !result)
                    LOG_WARN("exec_optimizer failed for op '{}': {}", cmd.op, result.error());
                break;
            case CommandTarget::Session:
                if (auto result = exec_session(cmd, view); !result)
                    LOG_WARN("exec_session failed for op '{}': {}", cmd.op, result.error());
                break;
            }
        }
    }

    std::expected<core::Tensor*, std::string> CommandCenter::resolve_attribute(lfs::core::SplatData& model, const std::string& name, size_t& row_dim_out) {
        if (name == "means" || name == "position") {
            row_dim_out = model.means_raw().shape()[0];
            return &model.means_raw();
        }
        if (name == "scaling" || name == "scale") {
            row_dim_out = model.scaling_raw().shape()[0];
            return &model.scaling_raw();
        }
        if (name == "rotation" || name == "rot") {
            row_dim_out = model.rotation_raw().shape()[0];
            return &model.rotation_raw();
        }
        if (name == "opacity" || name == "alpha") {
            row_dim_out = model.opacity_raw().shape()[0];
            return &model.opacity_raw();
        }
        if (name == "sh0") {
            row_dim_out = model.sh0_raw().shape()[0];
            return &model.sh0_raw();
        }
        if (name == "shN" || name == "sh") {
            // TODO(swizzle): shN is stored in vksplat swizzled layout (1D). Returning the
            // raw pointer exposes the swizzled buffer; callers that expect [N, K, 3] will
            // misinterpret it. Until set_attribute is rewritten with swizzle awareness,
            // report N=size() so row masks have correct length; the byte content is still
            // swizzled.
            row_dim_out = static_cast<std::size_t>(model.size());
            return &model.shN_raw();
        }
        return std::unexpected(std::string("Unknown attribute: ") + name);
    }

    std::expected<core::Tensor, std::string> CommandCenter::build_row_mask(const Selection& sel, std::size_t rows, core::Device device) {
        switch (sel.kind) {
        case SelectionKind::All:
            return core::Tensor::ones_bool({rows}, device);
        case SelectionKind::Range: {
            // Allow open-ended ranges with sel.end < 0 meaning "to end".
            Selection s = sel;
            if (s.start < 0) {
                s.start = 0; // clamp start to zero per request
            }
            const int64_t effective_end = (s.end < 0) ? static_cast<int64_t>(rows) : s.end;
            if (effective_end < s.start || static_cast<size_t>(effective_end) > rows) {
                return std::unexpected("Invalid range selection");
            }
            auto mask = core::Tensor::zeros_bool({rows}, device);
            const size_t len = static_cast<size_t>(effective_end - s.start);
            if (len == 0) {
                return mask;
            }
            auto slice = mask.slice(0, s.start, effective_end);
            slice = core::Tensor::ones_bool({len}, device);
            return mask;
        }
        case SelectionKind::Indices: {
            auto mask_vec = std::vector<bool>(rows, false);
            for (auto idx : sel.indices) {
                if (idx < 0 || static_cast<size_t>(idx) >= rows) {
                    return std::unexpected("Index out of bounds in selection");
                }
                mask_vec[static_cast<size_t>(idx)] = true;
            }
            return core::Tensor::from_vector(mask_vec, {rows}, device);
        }
        }
        return std::unexpected("Unsupported selection kind");
    }

    std::expected<void, std::string> CommandCenter::apply_set(core::Tensor& tensor, const core::Tensor& mask_rows, const ArgValue& value) {
        auto mask_full = expand_mask(mask_rows, tensor.shape());

        if (std::holds_alternative<double>(value)) {
            const double v = std::get<double>(value);
            const auto mask_float = mask_full.to(tensor.dtype());
            const auto keep = mask_full.logical_not().to(tensor.dtype());
            const auto fill = make_full_like_mask(mask_full, v).to(tensor.dtype());
            tensor = tensor * keep + fill * mask_float;
            return {};
        }
        if (std::holds_alternative<std::vector<double>>(value)) {
            const auto& vec = std::get<std::vector<double>>(value);
            if (tensor.shape().rank() < 2) {
                return std::unexpected("Vector value requires tensor with dim >=2");
            }
            const size_t dim = tensor.shape()[tensor.shape().rank() - 1];
            if (vec.size() != dim) {
                return std::unexpected("Vector length mismatch with attribute dimension");
            }
            std::vector<float> vec_f;
            vec_f.reserve(vec.size());
            for (double d : vec)
                vec_f.push_back(static_cast<float>(d));
            auto value_tensor = core::Tensor::from_vector(vec_f, {size_t{1}, dim}, tensor.device());
            value_tensor = value_tensor.broadcast_to(tensor.shape());
            const auto mask_float = mask_full.to(tensor.dtype());
            const auto keep = mask_full.logical_not().to(tensor.dtype());
            tensor = tensor * keep + value_tensor * mask_float;
            return {};
        }
        return std::unexpected("Unsupported value type for set_attribute");
    }

    std::expected<void, std::string> CommandCenter::apply_scale(core::Tensor& tensor, const core::Tensor& mask_rows, double factor) {
        const auto mask_full = expand_mask(mask_rows, tensor.shape());
        const auto mask_float = mask_full.to(tensor.dtype());
        const auto one = make_full_like_mask(mask_full, 1.0).to(tensor.dtype());
        const auto scale = make_full_like_mask(mask_full, factor).to(tensor.dtype());
        tensor = tensor * (one + (scale - one) * mask_float);
        return {};
    }

    std::expected<void, std::string> CommandCenter::apply_clamp(core::Tensor& tensor, const core::Tensor& mask_rows, const std::optional<double>& minv, const std::optional<double>& maxv) {
        if (!minv && !maxv) {
            return std::unexpected("clamp_attribute requires min or max");
        }
        const auto mask_full = expand_mask(mask_rows, tensor.shape());
        const auto mask_float = mask_full.to(tensor.dtype());
        const auto keep = mask_full.logical_not().to(tensor.dtype());
        auto clamped = tensor;
        if (minv && maxv) {
            clamped = tensor.clamp(static_cast<float>(*minv), static_cast<float>(*maxv));
        } else if (minv) {
            clamped = tensor.clamp_min(static_cast<float>(*minv));
        } else if (maxv) {
            clamped = tensor.clamp_max(static_cast<float>(*maxv));
        }
        tensor = tensor * keep + clamped * mask_float;
        return {};
    }

    std::expected<void, std::string> CommandCenter::exec_model(const Command& cmd, TrainingSnapshot& view) {
        if (!view.trainer) {
            return std::unexpected("No active trainer available for model command");
        }
        auto& strategy = view.trainer->get_strategy_mutable();
        auto& model = strategy.get_model();

        const auto attr_name = std::get<std::string>(cmd.args.at("attribute"));
        const bool is_shN = (attr_name == "shN" || attr_name == "sh");

        const size_t rows = model.size();

        // shN is stored swizzled — operate on a deswizzled [N, K, 3] working buffer,
        // then reswizzle. Other params can mutate in place via the resolved pointer.
        core::Tensor shN_canon;
        core::Tensor* tensor = nullptr;
        size_t prev_capacity = 0;
        if (is_shN) {
            if (!model.shN_raw().is_valid() || model.shN_raw().numel() == 0 ||
                model.max_sh_coeffs_rest() == 0) {
                return std::unexpected("shN storage is not allocated (max sh-degree 0)");
            }
            shN_canon = model.shN_canonical();
            prev_capacity = std::max<size_t>(model.means().capacity(), model.size());
            tensor = &shN_canon;
        } else {
            size_t row_dim = 0;
            const auto tensor_ref = resolve_attribute(model, attr_name, row_dim);
            (void)row_dim;
            if (!tensor_ref) {
                return std::unexpected(tensor_ref.error());
            }
            tensor = *tensor_ref;
            prev_capacity = tensor->capacity();
        }

        auto mask = build_row_mask(cmd.selection, rows, tensor->device());
        if (!mask) {
            return std::unexpected(mask.error());
        }
        std::expected<void, std::string> result;
        if (cmd.op == "set_attribute") {
            const auto it_scalar = cmd.args.find("value");
            const auto it_vec = cmd.args.find("values");
            if (it_scalar == cmd.args.end() && it_vec == cmd.args.end()) {
                return std::unexpected("set_attribute requires 'value' or 'values'");
            }
            if (it_vec != cmd.args.end()) {
                result = apply_set(*tensor, *mask, it_vec->second);
            } else {
                result = apply_set(*tensor, *mask, it_scalar->second);
            }
        } else if (cmd.op == "scale_attribute") {
            const auto it = cmd.args.find("factor");
            if (it == cmd.args.end()) {
                return std::unexpected("scale_attribute requires factor");
            }
            const double factor = std::get<double>(it->second);
            result = apply_scale(*tensor, *mask, factor);
        } else if (cmd.op == "clamp_attribute") {
            std::optional<double> minv;
            std::optional<double> maxv;
            if (auto it = cmd.args.find("min"); it != cmd.args.end()) {
                minv = std::get<double>(it->second);
            }
            if (auto it = cmd.args.find("max"); it != cmd.args.end()) {
                maxv = std::get<double>(it->second);
            }
            result = apply_clamp(*tensor, *mask, minv, maxv);
        } else {
            return std::unexpected("Unsupported model op: " + cmd.op);
        }
        if (!result) {
            return result;
        }

        // For shN, write the mutated canonical view back into swizzled storage.
        if (is_shN) {
            model.shN_set_from_canonical(shN_canon, prev_capacity);
        }

        if (auto p = param_type_from_attribute(attr_name)) {
            auto& opt = strategy.get_optimizer();
            opt.reset_state(*p);

            // shN capacity is in swizzled floats — managed by SplatData/allocator, skip
            // the row-count reserve here.
            if (!is_shN) {
                size_t desired_cap = prev_capacity;
                if (const auto* st = opt.get_state(*p)) {
                    desired_cap = std::max(desired_cap, st->capacity);
                }
                desired_cap = std::max(desired_cap, tensor->shape()[0]);
                if (desired_cap > 0 && tensor->capacity() < desired_cap) {
                    tensor->reserve(desired_cap);
                }
            }
        }

        return result;
    }

    std::expected<void, std::string> CommandCenter::exec_optimizer(const Command& cmd, TrainingSnapshot& view) {
        if (!view.trainer) {
            return std::unexpected("No active trainer available for optimizer command");
        }
        auto& opt = view.trainer->get_strategy_mutable().get_optimizer();
        if (cmd.op == "set_lr") {
            const auto it = cmd.args.find("value");
            if (it == cmd.args.end()) {
                return std::unexpected("set_lr requires value");
            }
            opt.set_lr(static_cast<float>(std::get<double>(it->second)));
            return {};
        }
        if (cmd.op == "scale_lr") {
            const auto it = cmd.args.find("factor");
            if (it == cmd.args.end()) {
                return std::unexpected("scale_lr requires factor");
            }
            const double f = std::get<double>(it->second);
            opt.set_lr(opt.get_lr() * static_cast<float>(f));
            return {};
        }
        return std::unexpected("Unsupported optimizer op: " + cmd.op);
    }

    std::expected<void, std::string> CommandCenter::exec_session(const Command& cmd, TrainingSnapshot& view) {
        if (!view.trainer) {
            return std::unexpected("No active trainer available for session command");
        }
        if (cmd.op == "pause") {
            view.trainer->request_pause();
            return {};
        }
        if (cmd.op == "resume") {
            view.trainer->request_resume();
            return {};
        }
        if (cmd.op == "request_stop") {
            view.trainer->request_stop();
            return {};
        }
        return std::unexpected("Unsupported session op: " + cmd.op);
    }

    std::expected<void, std::string> CommandCenter::execute(const Command& cmd) {
        const auto phase = phase_.load(std::memory_order_relaxed);
        const auto it_op = std::find_if(ops_.begin(), ops_.end(), [&](const OperationInfo& info) {
            return info.name == cmd.op && info.target == cmd.target;
        });
        if (it_op == ops_.end()) {
            return std::unexpected("Unknown operation: " + cmd.op + " for target " + target_name(cmd.target));
        }
        const auto& allowed_sel = it_op->selectors;
        if (std::find(allowed_sel.begin(), allowed_sel.end(), cmd.selection.kind) == allowed_sel.end()) {
            return std::unexpected("Selector kind not allowed for op " + cmd.op);
        }

        for (const auto& [name, value] : cmd.args) {
            const auto spec = std::ranges::find_if(it_op->args, [&](const ArgSpec& item) { return item.name == name; });
            if (spec == it_op->args.end()) {
                return std::unexpected("Unknown argument '" + name + "' for op " + cmd.op);
            }
            if (!argument_matches_type(value, spec->type)) {
                return std::unexpected("Wrong type for argument '" + name + "' in op " + cmd.op);
            }
            if (!argument_is_finite(value)) {
                return std::unexpected("Non-finite argument '" + name + "' in op " + cmd.op);
            }
        }
        for (const auto& spec : it_op->args) {
            if (spec.required && !cmd.args.contains(spec.name)) {
                return std::unexpected("Missing required argument '" + spec.name + "' for op " + cmd.op);
            }
        }
        if (cmd.op == "set_attribute" && cmd.args.contains("value") == cmd.args.contains("values")) {
            return std::unexpected("set_attribute requires exactly one of 'value' or 'values'");
        }
        if (cmd.op == "clamp_attribute" && !cmd.args.contains("min") && !cmd.args.contains("max")) {
            return std::unexpected("clamp_attribute requires 'min' or 'max'");
        }
        if (cmd.target == CommandTarget::Model) {
            const auto& attribute = std::get<std::string>(cmd.args.at("attribute"));
            if (!is_known_model_attribute(attribute)) {
                return std::unexpected("Unknown model attribute: " + attribute);
            }
        }

        // Snapshot clearing is the Trainer lifetime boundary. Keep this lock while
        // touching session atomics, and put GPU/model mutations on the worker queue.
        // The worker drains that queue only after waiting for outstanding model readers.
        std::lock_guard<std::mutex> lock(mutex_);
        if (!snapshot_.trainer) {
            return std::unexpected("No active trainer; cannot execute command");
        }
        switch (cmd.target) {
        case CommandTarget::Model:
        case CommandTarget::Optimizer:
            LOG_DEBUG("CommandCenter: queue {} op {} in phase {}", target_name(cmd.target), cmd.op, phase_name(phase));
            pending_commands_.push_back(cmd);
            return {};
        case CommandTarget::Session:
            LOG_DEBUG("CommandCenter: exec session op {} in phase {}", cmd.op, phase_name(phase));
            return exec_session(cmd, snapshot_);
        }
        return std::unexpected("Invalid command target");
    }

} // namespace lfs::training
