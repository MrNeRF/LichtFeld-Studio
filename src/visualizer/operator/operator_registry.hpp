/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include "operator.hpp"
#include "operator_id.hpp"
#include <array>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis::op {

    struct CallbackOperator {
        std::function<bool()> poll;
        std::function<OperatorResult(OperatorProperties&)> invoke;
        std::function<OperatorResult(const ModalEvent&, OperatorProperties&)> modal;
        std::function<void()> cancel;
    };

    struct PollCacheEntry {
        bool result;
        uint64_t scene_generation;
        bool has_selection;
        bool is_training;
        PollDependency deps;
    };

    enum class ModalState : uint8_t {
        IDLE,
        ACTIVE_CPP,
        ACTIVE_PYTHON
    };

    class LFS_VIS_API OperatorRegistry {
    public:
        static OperatorRegistry& instance();

        void registerOperator(BuiltinOp op, OperatorDescriptor desc, OperatorFactory factory);
        void registerCallbackOperator(OperatorDescriptor desc, CallbackOperator callbacks);
        void unregisterOperator(BuiltinOp op);
        void unregisterOperator(const std::string& class_id);
        void unregisterAllPython();

        [[nodiscard]] std::vector<const OperatorDescriptor*> getAllOperators() const;
        [[nodiscard]] const OperatorDescriptor* getDescriptor(BuiltinOp op) const;
        [[nodiscard]] const OperatorDescriptor* getDescriptor(const std::string& class_id) const;
        [[nodiscard]] bool poll(BuiltinOp op) const;
        [[nodiscard]] bool poll(const std::string& class_id) const;
        void invalidatePollCache(PollDependency changed = PollDependency::ALL);

        OperatorReturnValue invoke(BuiltinOp op, OperatorProperties* props = nullptr);
        OperatorReturnValue invoke(const std::string& class_id, OperatorProperties* props = nullptr);

        [[nodiscard]] bool hasModalOperator() const;
        [[nodiscard]] ModalState modalState() const;
        [[nodiscard]] std::string activeModalId() const;
        [[nodiscard]] bool canLockMutexForTest() const;
        OperatorResult dispatchModalEvent(const ModalEvent& event);
        void cancelModalOperator();

        void clear();
        void setSceneManager(SceneManager* scene) { scene_manager_ = scene; }

    private:
        OperatorRegistry() = default;
        ~OperatorRegistry() = default;
        OperatorRegistry(const OperatorRegistry&) = delete;
        OperatorRegistry& operator=(const OperatorRegistry&) = delete;

        struct RegisteredOperator {
            OperatorDescriptor descriptor;
            OperatorFactory factory;

            std::function<bool()> poll_fn;
            std::function<OperatorResult(OperatorProperties&)> invoke_fn;
            std::function<OperatorResult(const ModalEvent&, OperatorProperties&)> modal_fn;
            std::function<void()> cancel_fn;

            bool is_registered = false;
        };

        [[nodiscard]] std::optional<OperatorContext> makeContext() const;
        [[nodiscard]] bool pollImpl(const RegisteredOperator& reg,
                                    const OperatorProperties* props = nullptr) const;
        OperatorReturnValue invokeImpl(std::unique_lock<std::mutex>& lock,
                                       RegisteredOperator& reg, const std::string& id,
                                       OperatorProperties* props);

        mutable std::mutex mutex_;
        std::array<RegisteredOperator, static_cast<size_t>(BuiltinOp::_Count)> builtins_{};
        std::unordered_map<std::string, RegisteredOperator> python_operators_;
        OperatorPtr active_modal_;
        std::string active_modal_id_;
        std::optional<BuiltinOp> active_modal_builtin_;
        OperatorProperties modal_props_;
        SceneManager* scene_manager_ = nullptr;
        mutable std::unordered_map<std::string, PollCacheEntry> poll_cache_;
    };

    inline OperatorRegistry& operators() {
        return OperatorRegistry::instance();
    }

} // namespace lfs::vis::op
