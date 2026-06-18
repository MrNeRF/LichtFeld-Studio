/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/reactive/store.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace lfs::core::reactive {

    namespace {
        struct ThreadBatchState {
            int depth = 0;
            std::vector<Store::FieldId> pending_fields;
        };

        thread_local std::unordered_map<Store*, ThreadBatchState> tls_batches;

        std::mutex g_wake_mutex;
        Store::WakeCallback g_wake_callback = nullptr;

        bool append_dirty_unique(std::vector<Store::FieldId>& fields,
                                 const Store::FieldId field_id) {
            if (std::find(fields.begin(), fields.end(), field_id) != fields.end())
                return false;
            fields.push_back(field_id);
            return true;
        }
    } // namespace

    SubscriptionToken::SubscriptionToken(Store* store,
                                         const std::uint32_t field_id,
                                         const std::uint64_t subscription_id) noexcept
        : store_(store), field_id_(field_id), subscription_id_(subscription_id) {}

    SubscriptionToken::~SubscriptionToken() {
        reset();
    }

    SubscriptionToken::SubscriptionToken(SubscriptionToken&& other) noexcept
        : store_(std::exchange(other.store_, nullptr)),
          field_id_(std::exchange(other.field_id_, 0)),
          subscription_id_(std::exchange(other.subscription_id_, 0)) {}

    SubscriptionToken& SubscriptionToken::operator=(SubscriptionToken&& other) noexcept {
        if (this != &other) {
            reset();
            store_ = std::exchange(other.store_, nullptr);
            field_id_ = std::exchange(other.field_id_, 0);
            subscription_id_ = std::exchange(other.subscription_id_, 0);
        }
        return *this;
    }

    void SubscriptionToken::reset() {
        if (!store_)
            return;
        store_->unsubscribe(field_id_, subscription_id_);
        store_ = nullptr;
        field_id_ = 0;
        subscription_id_ = 0;
    }

    SubscriptionToken Store::subscribe(const FieldId field_id, Subscriber callback) {
        if (!callback)
            return {};

        const auto id = next_subscription_id_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard lock(subscribers_mutex_);
            subscribers_[field_id].push_back(SubscriberEntry{
                .id = id,
                .callback = std::move(callback),
            });
        }
        return SubscriptionToken(this, field_id, id);
    }

    void Store::unsubscribe(const FieldId field_id, const std::uint64_t subscription_id) {
        std::lock_guard lock(subscribers_mutex_);
        auto it = subscribers_.find(field_id);
        if (it == subscribers_.end())
            return;
        auto& entries = it->second;
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [subscription_id](const SubscriberEntry& entry) {
                                         return entry.id == subscription_id;
                                     }),
                      entries.end());
        if (entries.empty())
            subscribers_.erase(it);
    }

    void Store::enqueue_dirty(const FieldId field_id) {
        auto batch_it = tls_batches.find(this);
        if (batch_it != tls_batches.end() && batch_it->second.depth > 0) {
            batch_it->second.pending_fields.push_back(field_id);
            return;
        }

        enqueue_dirty_now(field_id);
    }

    void Store::enqueue_dirty_now(const FieldId field_id) {
        bool inserted = false;
        {
            std::lock_guard lock(dirty_mutex_);
            inserted = append_dirty_unique(dirty_fields_, field_id);
        }
        if (inserted)
            wake_main_thread();
    }

    bool Store::has_dirty() const {
        std::lock_guard lock(dirty_mutex_);
        return !dirty_fields_.empty();
    }

    bool Store::drain_dirty_into_frame() {
        std::vector<FieldId> dirty;
        {
            std::lock_guard lock(dirty_mutex_);
            dirty.swap(dirty_fields_);
        }
        if (dirty.empty())
            return false;

        const bool was_draining = draining_.exchange(true, std::memory_order_acq_rel);
        assert(!was_draining && "Store::drain_dirty_into_frame is not reentrant");
        if (was_draining) {
            std::lock_guard lock(dirty_mutex_);
            dirty_fields_.insert(dirty_fields_.end(), dirty.begin(), dirty.end());
            return true;
        }

        std::sort(dirty.begin(), dirty.end());
        dirty.erase(std::unique(dirty.begin(), dirty.end()), dirty.end());

        try {
            for (const FieldId field_id : dirty) {
                std::vector<Subscriber> callbacks;
                {
                    std::lock_guard lock(subscribers_mutex_);
                    auto it = subscribers_.find(field_id);
                    if (it != subscribers_.end()) {
                        callbacks.reserve(it->second.size());
                        for (const auto& entry : it->second)
                            callbacks.push_back(entry.callback);
                    }
                }

                for (auto& callback : callbacks) {
                    try {
                        callback();
                    } catch (const std::exception& e) {
                        LOG_ERROR("Reactive store subscriber failed for field {}: {}", field_id, e.what());
                    } catch (...) {
                        LOG_ERROR("Reactive store subscriber failed for field {} with an unknown exception", field_id);
                    }
                }
            }
        } catch (...) {
            draining_.store(false, std::memory_order_release);
            throw;
        }

        draining_.store(false, std::memory_order_release);
        return true;
    }

    void Store::begin_batch() {
        auto& state = tls_batches[this];
        ++state.depth;
    }

    void Store::end_batch() {
        auto it = tls_batches.find(this);
        assert(it != tls_batches.end() && it->second.depth > 0);
        if (it == tls_batches.end() || it->second.depth <= 0)
            return;

        auto& state = it->second;
        --state.depth;
        if (state.depth > 0)
            return;

        std::vector<FieldId> pending;
        pending.swap(state.pending_fields);
        tls_batches.erase(it);

        if (pending.empty())
            return;

        bool inserted = false;
        {
            std::lock_guard lock(dirty_mutex_);
            for (const FieldId field_id : pending)
                inserted = append_dirty_unique(dirty_fields_, field_id) || inserted;
        }
        if (inserted)
            wake_main_thread();
    }

    void Store::set_wake_callback(const WakeCallback callback) {
        std::lock_guard lock(g_wake_mutex);
        g_wake_callback = callback;
    }

    void Store::wake_main_thread() {
        WakeCallback callback = nullptr;
        {
            std::lock_guard lock(g_wake_mutex);
            callback = g_wake_callback;
        }
        if (callback)
            callback();
    }

    BatchUpdate::BatchUpdate(Store& store) : store_(&store) {
        store_->begin_batch();
    }

    BatchUpdate::~BatchUpdate() {
        if (store_)
            store_->end_batch();
    }

} // namespace lfs::core::reactive
