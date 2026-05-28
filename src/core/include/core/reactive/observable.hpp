/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/reactive/store.hpp"

#include <functional>
#include <mutex>
#include <string_view>
#include <utility>

namespace lfs::core::reactive {

    template <typename T>
    class Observable {
    public:
        using value_type = T;

        Observable(Store& store, Store::FieldId field_id, std::string_view name, T initial = {})
            : store_(&store), field_id_(field_id), name_(name), value_(std::move(initial)) {}

        Observable(const Observable&) = delete;
        Observable& operator=(const Observable&) = delete;
        Observable(Observable&&) = delete;
        Observable& operator=(Observable&&) = delete;

        [[nodiscard]] Store::FieldId field_id() const noexcept { return field_id_; }
        [[nodiscard]] std::string_view name() const noexcept { return name_; }

        [[nodiscard]] T get() const {
            std::lock_guard lock(value_mutex_);
            return value_;
        }

        bool set(T value) {
            {
                std::lock_guard lock(value_mutex_);
                if (value_ == value)
                    return false;
                value_ = std::move(value);
            }
            store_->enqueue_dirty(field_id_);
            return true;
        }

        [[nodiscard]] SubscriptionToken subscribe(std::function<void(const T&)> callback) {
            return store_->subscribe(field_id_, [this, cb = std::move(callback)] {
                cb(get());
            });
        }

    private:
        Store* store_ = nullptr;
        Store::FieldId field_id_ = 0;
        std::string_view name_;
        mutable std::mutex value_mutex_;
        T value_;
    };

} // namespace lfs::core::reactive
