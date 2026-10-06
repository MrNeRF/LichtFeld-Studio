/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// CPU-only frame-demand ledger shared by the viewer and its regression tests.

#pragma once

#include "dirty_flags.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis {

    enum class FrameReason : std::uint8_t {
        Startup,
        Input,
        CameraMotion,
        CameraSettle,
        SceneChange,
        SettingsChange,
        Selection,
        Overlay,
        ViewportResize,
        WindowResize,
        TrainingPreview,
        TrainingProgress,
        TrainingCompleted,
        AsyncCompletion,
        LodStreaming,
        ArenaRetry,
        GuiAnimation,
        GuiLayout,
        Tooltip,
        StatusMessage,
        HudSample,
        FpsIdle,
        Playback,
        Video,
        Export,
        PythonRedraw,
        PythonFrameCallback,
        Mcp,
        Count
    };

    [[nodiscard]] constexpr std::string_view frameReasonName(const FrameReason reason) {
        switch (reason) {
        case FrameReason::Startup: return "Startup";
        case FrameReason::Input: return "Input";
        case FrameReason::CameraMotion: return "CameraMotion";
        case FrameReason::CameraSettle: return "CameraSettle";
        case FrameReason::SceneChange: return "SceneChange";
        case FrameReason::SettingsChange: return "SettingsChange";
        case FrameReason::Selection: return "Selection";
        case FrameReason::Overlay: return "Overlay";
        case FrameReason::ViewportResize: return "ViewportResize";
        case FrameReason::WindowResize: return "WindowResize";
        case FrameReason::TrainingPreview: return "TrainingPreview";
        case FrameReason::TrainingProgress: return "TrainingProgress";
        case FrameReason::TrainingCompleted: return "TrainingCompleted";
        case FrameReason::AsyncCompletion: return "AsyncCompletion";
        case FrameReason::LodStreaming: return "LodStreaming";
        case FrameReason::ArenaRetry: return "ArenaRetry";
        case FrameReason::GuiAnimation: return "GuiAnimation";
        case FrameReason::GuiLayout: return "GuiLayout";
        case FrameReason::Tooltip: return "Tooltip";
        case FrameReason::StatusMessage: return "StatusMessage";
        case FrameReason::HudSample: return "HudSample";
        case FrameReason::FpsIdle: return "FpsIdle";
        case FrameReason::Playback: return "Playback";
        case FrameReason::Video: return "Video";
        case FrameReason::Export: return "Export";
        case FrameReason::PythonRedraw: return "PythonRedraw";
        case FrameReason::PythonFrameCallback: return "PythonFrameCallback";
        case FrameReason::Mcp: return "Mcp";
        case FrameReason::Count: break;
        }
        return "?";
    }

    enum class FrameScope : std::uint8_t { Gui,
                                           View,
                                           All };

    // One bit per 3D view. master has a single view (bit 0); dev editor areas
    // use one bit per visible view.
    using ViewMask = std::uint32_t;
    inline constexpr std::size_t kMaxViews = 32;
    inline constexpr ViewMask kAllViews = ViewMask{1};

    using FrameClock = std::chrono::steady_clock;
    using ReasonSet = std::bitset<static_cast<std::size_t>(FrameReason::Count)>;

    [[nodiscard]] inline double secondsUntilFrameDeadline(const FrameClock::time_point deadline,
                                                          const FrameClock::time_point now) {
        // Due work must wake immediately, including deadlines elapsed during present.
        return std::max(0.0, std::chrono::duration<double>(deadline - now).count());
    }

    struct FrameRequest {
        FrameReason reason;
        FrameScope scope;
        ViewMask views = 1;
        DirtyMask flags = 0;
        std::string detail{};
    };

    class FrameDemandLedger;

    // RAII handle for a continuous demand. Destroying it ends the holder.
    class [[nodiscard]] DemandToken {
    public:
        DemandToken() = default;
        DemandToken(const DemandToken&) = delete;
        DemandToken& operator=(const DemandToken&) = delete;
        DemandToken(DemandToken&& other) noexcept { swap(other); }
        DemandToken& operator=(DemandToken&& other) noexcept {
            if (this != &other) {
                release();
                swap(other);
            }
            return *this;
        }
        ~DemandToken() { release(); }

        [[nodiscard]] bool active() const { return ledger_ != nullptr; }
        void release();

    private:
        friend class FrameDemandLedger;
        DemandToken(FrameDemandLedger* ledger, const std::uint64_t id) : ledger_(ledger), id_(id) {}
        void swap(DemandToken& other) noexcept {
            std::swap(ledger_, other.ledger_);
            std::swap(id_, other.id_);
        }
        FrameDemandLedger* ledger_ = nullptr;
        std::uint64_t id_ = 0;
    };

    struct FramePlan {
        bool present = false;
        ViewMask render_views = 0;
        std::array<DirtyMask, kMaxViews> view_flags{};
        ReasonSet reasons;
        std::vector<std::string> details;

        [[nodiscard]] bool empty() const { return !present && render_views == 0; }
    };

    struct LiveHolderInfo {
        FrameReason reason;
        FrameScope scope;
        std::string detail;
        FrameClock::duration age{0};
    };

    struct FrameLedgerSnapshot {
        std::uint64_t frames_presented = 0;
        std::uint64_t wakes_without_frame = 0;
        std::uint64_t frames_without_reason = 0;
        std::uint64_t holders_expired = 0;
        std::uint64_t requests_dropped = 0;
        std::uint64_t preview_skipped_no_step = 0;
        std::uint64_t stale_detections = 0;
        std::array<std::uint64_t, kMaxViews> views_rendered{};
        std::array<std::uint64_t, static_cast<std::size_t>(FrameReason::Count)> presents_by_reason{};
        std::array<std::uint64_t, static_cast<std::size_t>(FrameReason::Count)> view_renders_by_reason{};
        ReasonSet last_frame_reasons;
        std::vector<std::string> last_frame_details;
        std::vector<LiveHolderInfo> live_holders;
    };

    class FrameDemandLedger {
    public:
        FrameDemandLedger() = default;
        FrameDemandLedger(const FrameDemandLedger&) = delete;
        FrameDemandLedger& operator=(const FrameDemandLedger&) = delete;

        // The main loop's wake (SDL push event). Called from any thread.
        void setWakeCallback(std::function<void()> wake) {
            std::lock_guard lock(mutex_);
            wake_ = std::move(wake);
        }

        void setDisplayInterval(const FrameClock::duration interval) {
            std::lock_guard lock(mutex_);
            display_interval_ = std::max(interval, FrameClock::duration{std::chrono::microseconds{1}});
        }

        // Thread-safe one-shot. A View request with an empty mask is dropped and
        // counted (and asserts in debug builds).
        void setVisibleViews(ViewMask views) {
            std::lock_guard lock(mutex_);
            visible_views_ = views;
        }

        void request(FrameRequest request) {
            if (request.scope != FrameScope::Gui && request.views == 0) {
#ifndef NDEBUG
                assert(request.views != 0 && "view-scoped frame request requires a non-empty view mask");
#endif
                requests_dropped_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            std::function<void()> wake;
            {
                std::lock_guard lock(mutex_);
                one_shots_.push_back(std::move(request));
                wake = wake_;
            }
            if (wake)
                wake();
        }

        [[nodiscard]] DemandToken hold(const FrameReason reason, const FrameScope scope, const ViewMask views,
                                       const DirtyMask flags, std::function<bool()> alive,
                                       std::string detail = {}) {
#ifndef NDEBUG
            assert(alive && "frame-demand holder requires a lifetime predicate");
            assert(scope == FrameScope::Gui || views != 0);
#endif
            std::uint64_t id = 0;
            {
                std::lock_guard lock(mutex_);
                id = ++next_holder_id_;
                holders_.push_back(std::make_shared<Holder>(Holder{
                    .id = id,
                    .reason = reason,
                    .scope = scope,
                    .views = scope == FrameScope::Gui ? ViewMask{0} : views,
                    .flags = flags,
                    .alive = std::move(alive),
                    .detail = std::move(detail),
                    .created = FrameClock::now(),
                    .last_fired = {},
                }));
            }
            return DemandToken{this, id};
        }

        [[nodiscard]] FramePlan plan(const FrameClock::time_point now) {
            auto sampled_holders = sampleHolders(now);
            // GUI predicates can publish thumbnail/overlay changes through request().
            // Keep their storage alive, but never call application code under mutex_.
            for (auto& sample : sampled_holders)
                sample.expired = !sample.holder->alive();
            std::lock_guard lock(mutex_);
            FramePlan plan;

            for (const auto& r : one_shots_)
                apply(plan, r.reason, r.scope, r.views, r.flags, r.detail);
            one_shots_.clear();

            for (const auto& sample : sampled_holders) {
                const auto it = std::find(holders_.begin(), holders_.end(), sample.holder);
                // A callback or another thread may have released this token.
                if (it == holders_.end())
                    continue;
                if (sample.expired) {
                    holders_.erase(it);
                    ++counters_.holders_expired;
                    continue;
                }
                auto& h = **it;
                if (sample.next_due > now)
                    continue;
                h.last_fired = now;
                apply(plan, h.reason, h.scope, h.views, h.flags, h.detail);
            }

            if (plan.empty())
                ++counters_.wakes_without_frame;
            else
                last_plan_started_ = now;
            return plan;
        }

        // Earliest time anything needs the loop awake; empty means block forever.
        [[nodiscard]] std::optional<FrameClock::time_point> nextDeadline(const FrameClock::time_point now) const {
            std::lock_guard lock(mutex_);
            if (!one_shots_.empty())
                return now;
            std::optional<FrameClock::time_point> earliest;
            for (const auto& holder : holders_) {
                const auto due = holderNextDue(*holder, now);
                if (!earliest || due < *earliest)
                    earliest = due;
            }
            return earliest;
        }

        void countPresented(const FramePlan& plan) {
            std::lock_guard lock(mutex_);
            ++counters_.frames_presented;
            if (plan.reasons.none())
                ++counters_.frames_without_reason;
            for (std::size_t i = 0; i < plan.reasons.size(); ++i)
                if (plan.reasons.test(i))
                    ++counters_.presents_by_reason[i];
            counters_.last_frame_reasons = plan.reasons;
            counters_.last_frame_details = plan.details;
        }

        void countViewRendered(const ViewMask views, const FramePlan& plan) {
            std::lock_guard lock(mutex_);
            for (std::size_t v = 0; v < kMaxViews; ++v)
                if (views & (ViewMask{1} << v))
                    ++counters_.views_rendered[v];
            for (std::size_t i = 0; i < plan.reasons.size(); ++i)
                if (plan.reasons.test(i))
                    ++counters_.view_renders_by_reason[i];
        }

        [[nodiscard]] FrameLedgerSnapshot snapshot(const FrameClock::time_point now = FrameClock::now()) const {
            std::lock_guard lock(mutex_);
            auto s = counters_;
            s.requests_dropped = requests_dropped_.load(std::memory_order_relaxed);
            for (const auto& h : holders_)
                s.live_holders.push_back(LiveHolderInfo{h->reason, h->scope, h->detail, now - h->created});
            return s;
        }

        // Test and MCP helper: counters only; holders and pending requests stay.
        void resetCounters() {
            std::lock_guard lock(mutex_);
            counters_ = {};
            requests_dropped_.store(0, std::memory_order_relaxed);
        }

        void countSkippedPreview() {
            std::lock_guard lock(mutex_);
            ++counters_.preview_skipped_no_step;
        }

        void countStaleView() {
            std::lock_guard lock(mutex_);
            ++counters_.stale_detections;
        }

    private:
        friend class DemandToken;

        struct Holder {
            std::uint64_t id;
            FrameReason reason;
            FrameScope scope;
            ViewMask views;
            DirtyMask flags;
            std::function<bool()> alive;
            std::string detail;
            FrameClock::time_point created;
            FrameClock::time_point last_fired;
        };
        struct HolderSample {
            std::shared_ptr<Holder> holder;
            FrameClock::time_point next_due;
            bool expired = false;
        };

        [[nodiscard]] std::vector<HolderSample> sampleHolders(const FrameClock::time_point now) const {
            std::lock_guard lock(mutex_);
            std::vector<HolderSample> samples;
            samples.reserve(holders_.size());
            for (const auto& holder : holders_)
                samples.push_back({holder, holderNextDue(*holder, now)});
            return samples;
        }

        void releaseHolder(const std::uint64_t id) {
            // Destruction of callback captures can also re-enter the ledger.
            std::shared_ptr<Holder> released;
            {
                std::lock_guard lock(mutex_);
                const auto it = std::find_if(holders_.begin(), holders_.end(),
                                             [id](const auto& h) { return h->id == id; });
                if (it != holders_.end()) {
                    released = std::move(*it);
                    holders_.erase(it);
                }
            }
        }

        void apply(FramePlan& plan, const FrameReason reason, const FrameScope scope, const ViewMask views,
                   const DirtyMask flags, const std::string& detail) {
            plan.reasons.set(static_cast<std::size_t>(reason));
            plan.present = true;
            if (scope != FrameScope::Gui) {
                const ViewMask mask = scope == FrameScope::All ? visible_views_ : views;
                plan.render_views |= mask;
                for (std::size_t v = 0; v < kMaxViews; ++v)
                    if (mask & (ViewMask{1} << v))
                        plan.view_flags[v] |= flags;
            }
            if (!detail.empty())
                plan.details.push_back(detail);
        }

        [[nodiscard]] FrameClock::time_point holderNextDue(const Holder& h, const FrameClock::time_point now) const {
            const auto base = std::max(h.last_fired, last_plan_started_);
            return base == FrameClock::time_point{} ? now : base + display_interval_;
        }

        mutable std::mutex mutex_;
        std::function<void()> wake_;
        FrameClock::duration display_interval_{std::chrono::microseconds{16667}};
        std::vector<FrameRequest> one_shots_;
        std::vector<std::shared_ptr<Holder>> holders_;
        ViewMask visible_views_ = kAllViews;
        std::uint64_t next_holder_id_ = 0;
        FrameClock::time_point last_plan_started_{};

        FrameLedgerSnapshot counters_;
        std::atomic<std::uint64_t> requests_dropped_{0};
    };

    inline void DemandToken::release() {
        if (ledger_) {
            ledger_->releaseHolder(id_);
            ledger_ = nullptr;
            id_ = 0;
        }
    }

} // namespace lfs::vis
