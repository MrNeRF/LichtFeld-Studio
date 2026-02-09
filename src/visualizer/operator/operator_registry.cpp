/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "operator_registry.hpp"
#include "control/command_api.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/logger.hpp"
#include "python/python_runtime.hpp"
#include "scene/scene_manager.hpp"
#include <cassert>

namespace lfs::vis::op {

    OperatorRegistry& OperatorRegistry::instance() {
        static OperatorRegistry registry;
        return registry;
    }

    void OperatorRegistry::registerOperator(BuiltinOp op, OperatorDescriptor desc, OperatorFactory factory) {
        std::lock_guard lock(mutex_);
        const auto idx = static_cast<size_t>(op);
        assert(idx < builtins_.size());

        desc.builtin_id = op;
        desc.source = OperatorSource::CPP;

        RegisteredOperator reg;
        reg.descriptor = std::move(desc);
        reg.factory = std::move(factory);
        reg.is_registered = true;
        builtins_[idx] = std::move(reg);
    }

    void OperatorRegistry::registerCallbackOperator(OperatorDescriptor desc, CallbackOperator callbacks) {
        std::lock_guard lock(mutex_);

        const std::string class_id = desc.python_class_id;
        desc.source = OperatorSource::PYTHON;

        RegisteredOperator reg;
        reg.descriptor = std::move(desc);
        reg.poll_fn = std::move(callbacks.poll);
        reg.invoke_fn = std::move(callbacks.invoke);
        reg.modal_fn = std::move(callbacks.modal);
        reg.cancel_fn = std::move(callbacks.cancel);
        reg.is_registered = true;
        python_operators_[class_id] = std::move(reg);
    }

    void OperatorRegistry::unregisterOperator(BuiltinOp op) {
        std::lock_guard lock(mutex_);
        const auto idx = static_cast<size_t>(op);
        assert(idx < builtins_.size());

        const std::string id = to_string(op);

        if (active_modal_builtin_.has_value() && *active_modal_builtin_ == op) {
            if (active_modal_) {
                if (auto ctx = makeContext()) {
                    active_modal_->cancel(*ctx);
                }
                active_modal_.reset();
            }
            active_modal_builtin_.reset();
        }

        builtins_[idx] = RegisteredOperator{};
        poll_cache_.erase(id);
    }

    void OperatorRegistry::unregisterOperator(const std::string& class_id) {
        std::lock_guard lock(mutex_);

        if (active_modal_id_ == class_id) {
            auto it = python_operators_.find(class_id);
            if (it != python_operators_.end() && it->second.cancel_fn) {
                it->second.cancel_fn();
            }
            active_modal_id_.clear();
        }

        python_operators_.erase(class_id);
        poll_cache_.erase(class_id);
    }

    void OperatorRegistry::unregisterAllPython() {
        std::lock_guard lock(mutex_);

        if (active_modal_ && !active_modal_builtin_.has_value()) {
            if (auto ctx = makeContext()) {
                active_modal_->cancel(*ctx);
            }
            active_modal_.reset();
        }

        if (!active_modal_id_.empty()) {
            auto it = python_operators_.find(active_modal_id_);
            if (it != python_operators_.end() && it->second.cancel_fn) {
                it->second.cancel_fn();
            }
            active_modal_id_.clear();
        }

        python_operators_.clear();

        std::erase_if(poll_cache_, [this](const auto& pair) {
            auto builtin = builtin_op_from_string(pair.first);
            if (builtin.has_value()) {
                return !builtins_[static_cast<size_t>(*builtin)].is_registered;
            }
            return true;
        });
    }

    std::vector<const OperatorDescriptor*> OperatorRegistry::getAllOperators() const {
        std::lock_guard lock(mutex_);
        std::vector<const OperatorDescriptor*> result;

        for (const auto& reg : builtins_) {
            if (reg.is_registered && hasFlag(reg.descriptor.flags, OperatorFlags::REGISTER)) {
                result.push_back(&reg.descriptor);
            }
        }

        for (const auto& [id, reg] : python_operators_) {
            if (reg.is_registered && hasFlag(reg.descriptor.flags, OperatorFlags::REGISTER)) {
                result.push_back(&reg.descriptor);
            }
        }

        return result;
    }

    const OperatorDescriptor* OperatorRegistry::getDescriptor(BuiltinOp op) const {
        std::lock_guard lock(mutex_);
        const auto idx = static_cast<size_t>(op);
        assert(idx < builtins_.size());
        return builtins_[idx].is_registered ? &builtins_[idx].descriptor : nullptr;
    }

    const OperatorDescriptor* OperatorRegistry::getDescriptor(const std::string& class_id) const {
        std::lock_guard lock(mutex_);

        auto builtin = builtin_op_from_string(class_id);
        if (builtin.has_value()) {
            const auto idx = static_cast<size_t>(*builtin);
            return builtins_[idx].is_registered ? &builtins_[idx].descriptor : nullptr;
        }

        const auto it = python_operators_.find(class_id);
        return it != python_operators_.end() ? &it->second.descriptor : nullptr;
    }

    bool OperatorRegistry::pollImpl(const RegisteredOperator& reg) const {
        const auto& ctx = python::context();
        const uint64_t gen = ctx.scene_generation;
        const bool has_sel = scene_manager_ && scene_manager_->hasSelectedNode();
        const auto* cc = lfs::event::command_center();
        const bool training = cc ? cc->snapshot().is_running : false;

        const PollDependency deps = reg.descriptor.poll_deps;
        const std::string id = reg.descriptor.id();

        auto cache_it = poll_cache_.find(id);
        if (cache_it != poll_cache_.end()) {
            const auto& e = cache_it->second;
            bool valid = true;
            if ((deps & PollDependency::SCENE) != PollDependency::NONE) {
                valid &= (e.scene_generation == gen);
            }
            if ((deps & PollDependency::SELECTION) != PollDependency::NONE) {
                valid &= (e.has_selection == has_sel);
            }
            if ((deps & PollDependency::TRAINING) != PollDependency::NONE) {
                valid &= (e.is_training == training);
            }
            if (valid) {
                return e.result;
            }
        }

        bool result = false;

        if (reg.poll_fn) {
            result = reg.poll_fn();
        } else if (reg.factory) {
            auto op = reg.factory();
            if (op) {
                auto local_ctx = makeContext();
                result = local_ctx && op->poll(*local_ctx);
            }
        }

        poll_cache_[id] = {result, gen, has_sel, training, deps};
        return result;
    }

    bool OperatorRegistry::poll(BuiltinOp op) const {
        std::lock_guard lock(mutex_);
        const auto idx = static_cast<size_t>(op);
        assert(idx < builtins_.size());

        if (!builtins_[idx].is_registered) {
            return false;
        }
        return pollImpl(builtins_[idx]);
    }

    bool OperatorRegistry::poll(const std::string& class_id) const {
        std::lock_guard lock(mutex_);

        auto builtin = builtin_op_from_string(class_id);
        if (builtin.has_value()) {
            const auto idx = static_cast<size_t>(*builtin);
            if (!builtins_[idx].is_registered) {
                return false;
            }
            return pollImpl(builtins_[idx]);
        }

        const auto it = python_operators_.find(class_id);
        if (it == python_operators_.end()) {
            return false;
        }
        return pollImpl(it->second);
    }

    void OperatorRegistry::invalidatePollCache(PollDependency changed) {
        std::lock_guard lock(mutex_);
        if (changed == PollDependency::ALL) {
            poll_cache_.clear();
            return;
        }

        std::erase_if(poll_cache_, [&](const auto& pair) {
            auto builtin = builtin_op_from_string(pair.first);
            PollDependency deps;
            if (builtin.has_value()) {
                const auto idx = static_cast<size_t>(*builtin);
                if (!builtins_[idx].is_registered)
                    return true;
                deps = builtins_[idx].descriptor.poll_deps;
            } else {
                auto it = python_operators_.find(pair.first);
                if (it == python_operators_.end())
                    return true;
                deps = it->second.descriptor.poll_deps;
            }
            return (deps & changed) != PollDependency::NONE;
        });
    }

    OperatorReturnValue OperatorRegistry::invokeImpl(RegisteredOperator& reg, const std::string& id,
                                                     OperatorProperties* props) {
        OperatorProperties local_props;
        OperatorProperties& props_ref = props ? *props : local_props;

        if (reg.invoke_fn) {
            if (reg.poll_fn && !reg.poll_fn()) {
                return OperatorReturnValue::cancelled();
            }

            mutex_.unlock();
            const OperatorResult result = reg.invoke_fn(props_ref);
            mutex_.lock();

            if (result == OperatorResult::RUNNING_MODAL && reg.modal_fn) {
                if (!active_modal_id_.empty() && active_modal_id_ != id) {
                    LOG_WARN("Modal operator '{}' replacing active modal '{}'", id, active_modal_id_);
                } else if (active_modal_) {
                    LOG_WARN("Modal operator '{}' replacing active modal", id);
                }
                active_modal_id_ = id;
                modal_props_ = props_ref;
            }

            return {result, {}};
        }

        if (!reg.factory) {
            LOG_ERROR("Operator has no factory or invoke callback: {}", id);
            return OperatorReturnValue::cancelled();
        }

        auto op = reg.factory();
        if (!op) {
            LOG_ERROR("Failed to create operator: {}", id);
            return OperatorReturnValue::cancelled();
        }

        auto ctx = makeContext();
        if (!ctx) {
            LOG_ERROR("No scene manager for operator context");
            return OperatorReturnValue::cancelled();
        }

        if (!op->poll(*ctx)) {
            return OperatorReturnValue::cancelled();
        }

        mutex_.unlock();
        const OperatorResult result = op->invoke(*ctx, props_ref);
        mutex_.lock();

        if (result == OperatorResult::RUNNING_MODAL) {
            if (active_modal_) {
                LOG_WARN("Modal operator '{}' replacing active modal", id);
            } else if (!active_modal_id_.empty()) {
                LOG_WARN("Modal operator '{}' replacing active modal '{}'", id, active_modal_id_);
            }
            active_modal_ = std::move(op);
            modal_props_ = props_ref;
        }

        return {result, {}};
    }

    OperatorReturnValue OperatorRegistry::invoke(BuiltinOp op, OperatorProperties* props) {
        std::unique_lock lock(mutex_);
        assert(!(active_modal_ && !active_modal_id_.empty() && !active_modal_builtin_.has_value()) &&
               "Both modal types active simultaneously");

        const auto idx = static_cast<size_t>(op);
        assert(idx < builtins_.size());

        if (!builtins_[idx].is_registered) {
            LOG_WARN("Builtin operator not registered: {}", to_string(op));
            return OperatorReturnValue::cancelled();
        }

        auto result = invokeImpl(builtins_[idx], to_string(op), props);

        if (result.status == OperatorResult::RUNNING_MODAL && active_modal_) {
            active_modal_builtin_ = op;
            active_modal_id_.clear();
        }

        return result;
    }

    OperatorReturnValue OperatorRegistry::invoke(const std::string& class_id, OperatorProperties* props) {
        std::unique_lock lock(mutex_);
        assert(!(active_modal_ && !active_modal_id_.empty() && !active_modal_builtin_.has_value()) &&
               "Both modal types active simultaneously");

        auto builtin = builtin_op_from_string(class_id);
        if (builtin.has_value()) {
            lock.unlock();
            return invoke(*builtin, props);
        }

        const auto it = python_operators_.find(class_id);
        if (it == python_operators_.end()) {
            LOG_WARN("Operator not found: {}", class_id);
            return OperatorReturnValue::cancelled();
        }

        return invokeImpl(it->second, class_id, props);
    }

    bool OperatorRegistry::hasModalOperator() const {
        std::lock_guard lock(mutex_);
        return active_modal_ != nullptr || !active_modal_id_.empty();
    }

    ModalState OperatorRegistry::modalState() const {
        std::lock_guard lock(mutex_);
        if (active_modal_)
            return ModalState::ACTIVE_CPP;
        if (!active_modal_id_.empty())
            return ModalState::ACTIVE_PYTHON;
        return ModalState::IDLE;
    }

    OperatorResult OperatorRegistry::dispatchModalEvent(const ModalEvent& event) {
        std::unique_lock lock(mutex_);

        if (!active_modal_id_.empty()) {
            auto it = python_operators_.find(active_modal_id_);
            if (it == python_operators_.end() || !it->second.modal_fn) {
                active_modal_id_.clear();
                return OperatorResult::CANCELLED;
            }

            auto& reg = it->second;
            auto props = modal_props_;
            lock.unlock();

            const OperatorResult result = reg.modal_fn(event, props);

            lock.lock();

            if (result == OperatorResult::FINISHED || result == OperatorResult::CANCELLED) {
                if (result == OperatorResult::CANCELLED && reg.cancel_fn) {
                    reg.cancel_fn();
                }
                active_modal_id_.clear();
            } else {
                modal_props_ = props;
            }

            return result;
        }

        if (!active_modal_) {
            return OperatorResult::PASS_THROUGH;
        }

        auto ctx = makeContext();
        if (!ctx) {
            return OperatorResult::CANCELLED;
        }

        ctx->setModalEvent(event);

        auto op = std::move(active_modal_);
        auto props = modal_props_;
        lock.unlock();

        const OperatorResult result = op->modal(*ctx, props);

        lock.lock();

        if (result == OperatorResult::RUNNING_MODAL || result == OperatorResult::PASS_THROUGH) {
            active_modal_ = std::move(op);
            modal_props_ = props;
        } else if (result == OperatorResult::CANCELLED) {
            op->cancel(*ctx);
            active_modal_builtin_.reset();
        } else {
            active_modal_builtin_.reset();
        }

        return result;
    }

    void OperatorRegistry::cancelModalOperator() {
        std::lock_guard lock(mutex_);

        if (!active_modal_id_.empty()) {
            auto it = python_operators_.find(active_modal_id_);
            if (it != python_operators_.end() && it->second.cancel_fn) {
                it->second.cancel_fn();
            }
            active_modal_id_.clear();
            return;
        }

        if (!active_modal_) {
            return;
        }

        if (auto ctx = makeContext()) {
            active_modal_->cancel(*ctx);
        }
        active_modal_.reset();
        active_modal_builtin_.reset();
    }

    void OperatorRegistry::clear() {
        std::lock_guard lock(mutex_);

        if (!active_modal_id_.empty()) {
            auto it = python_operators_.find(active_modal_id_);
            if (it != python_operators_.end() && it->second.cancel_fn) {
                it->second.cancel_fn();
            }
            active_modal_id_.clear();
        }

        if (active_modal_) {
            if (auto ctx = makeContext()) {
                active_modal_->cancel(*ctx);
            }
            active_modal_.reset();
        }

        active_modal_builtin_.reset();

        for (auto& reg : builtins_) {
            reg = RegisteredOperator{};
        }
        python_operators_.clear();
        poll_cache_.clear();
        scene_manager_ = nullptr;
    }

    std::optional<OperatorContext> OperatorRegistry::makeContext() const {
        if (!scene_manager_) {
            return std::nullopt;
        }
        return OperatorContext(*scene_manager_);
    }

} // namespace lfs::vis::op
