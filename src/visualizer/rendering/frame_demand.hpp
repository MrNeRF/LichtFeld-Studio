/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Frame-demand ledger: the single place that records why the viewer renders
// a 3D view or presents a GUI frame. The main loop asks plan(); nothing else
// may decide that a frame happens. Idle is the absence of reasons: with no
// one-shot requests, no live holders and no wake deadline, nextDeadline()
// is empty and the loop blocks in the OS event wait.
//
// Header-only and free of viewer dependencies so it is unit-tested on the
// CPU-only CI runner.

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

    [[nodiscard]] constexpr bool trainingPreviewStepAdvanced(const int previous_iteration,
                                                             const int current_iteration) {
        return current_iteration > previous_iteration;
    }

    // A one-shot demand. Consumed by the next plan() at or after not_before.
    struct FrameRequest {
        FrameReason reason;
        FrameScope scope;
        ViewMask views = 1;
        DirtyMask flags = 0;
        FrameClock::time_point not_before{};
        std::string detail{};
    };

    // How a continuous holder is paced. Display: never earlier than one display
    // interval after the previous frame start. Interval: a fixed period.
    // External: the holder's own scheduler says when it is next due (training
    // preview). PerEvent: every plan() while alive (input-driven only).
    struct Cadence {
        enum class Kind : std::uint8_t { PerEvent,
                                         Display,
                                         Interval,
                                         External };
        using NextDue = std::function<std::optional<FrameClock::time_point>(FrameClock::time_point now)>;

        Kind kind = Kind::Display;
        FrameClock::duration interval{0};
        NextDue next_due{};

        [[nodiscard]] static Cadence perEvent() { return Cadence{Kind::PerEvent, {}, {}}; }
        [[nodiscard]] static Cadence display() { return Cadence{Kind::Display, {}, {}}; }
        [[nodiscard]] static Cadence every(const FrameClock::duration period) {
            return Cadence{Kind::Interval, period, {}};
        }
        [[nodiscard]] static Cadence external(NextDue next_due) {
            return Cadence{Kind::External, {}, std::move(next_due)};
        }
    };

    // Every holder must end: either at a fixed time or when a predicate turns
    // false. There is deliberately no default constructor.
    class Finite {
    public:
        using Alive = std::function<bool()>;

        [[nodiscard]] static Finite until(const FrameClock::time_point deadline) {
            Finite f;
            f.deadline_ = deadline;
            return f;
        }
        [[nodiscard]] static Finite alive(Alive predicate) {
            Finite f;
            f.alive_ = std::move(predicate);
            return f;
        }
        [[nodiscard]] static Finite untilOrAlive(const FrameClock::time_point deadline, Alive predicate) {
            Finite f;
            f.deadline_ = deadline;
            f.alive_ = std::move(predicate);
            return f;
        }

        [[nodiscard]] bool expired(const FrameClock::time_point now) const {
            if (deadline_ && now >= *deadline_)
                return true;
            if (alive_ && !alive_())
                return true;
            return false;
        }
        [[nodiscard]] std::optional<FrameClock::time_point> deadline() const { return deadline_; }
        [[nodiscard]] bool valid() const { return deadline_.has_value() || static_cast<bool>(alive_); }

    private:
        Finite() = default;
        std::optional<FrameClock::time_point> deadline_;
        Alive alive_;
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

        // Display refresh period used by Cadence::display(); the loop refreshes
        // it from the window's display mode. Defaults to 60 Hz.
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

        // Continuous demand. Ends when `finite` says so or the token is released.
        [[nodiscard]] DemandToken hold(const FrameReason reason, const FrameScope scope, const ViewMask views,
                                       const DirtyMask flags, Cadence cadence, Finite finite,
                                       std::string detail = {}, const bool wake_loop = true) {
#ifndef NDEBUG
            assert(finite.valid() && "frame-demand holder must have a finite condition");
            assert(scope == FrameScope::Gui || views != 0);
#endif
            std::function<void()> wake;
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
                    .cadence = std::move(cadence),
                    .finite = std::move(finite),
                    .detail = std::move(detail),
                    .created = FrameClock::now(),
                    .last_fired = {},
                }));
                wake = wake_;
            }
            if (wake && wake_loop)
                wake();
            return DemandToken{this, id};
        }

        // A wake that runs loop maintenance without producing a frame.
        void wakeAt(const FrameClock::time_point when, std::string why) {
            std::function<void()> wake;
            {
                std::lock_guard lock(mutex_);
                wakes_.push_back(Wake{when, std::move(why)});
                wake = wake_;
            }
            if (wake)
                wake();
        }

        // Called once per loop iteration after the event wait. Consumes due
        // one-shots and wakes, advances holders, drops expired ones.
        [[nodiscard]] FramePlan plan(const FrameClock::time_point now) {
            auto sampled_holders = sampleHolders(now);
            // GUI predicates can publish thumbnail/overlay changes through request().
            // Keep their storage alive, but never call application code under mutex_.
            for (auto& sample : sampled_holders) {
                sample.expired = sample.holder->finite.expired(now);
                if (!sample.expired)
                    sampleExternalDeadline(sample, now);
            }
            std::lock_guard lock(mutex_);
            FramePlan plan;

            // Wakes: only pruning. They exist to bound the event wait.
            std::erase_if(wakes_, [&](const Wake& w) { return w.when <= now; });

            // One-shots.
            std::vector<FrameRequest> later;
            for (auto& r : one_shots_) {
                if (r.not_before > now) {
                    later.push_back(std::move(r));
                    continue;
                }
                apply(plan, r.reason, r.scope, r.views, r.flags, r.detail);
            }
            one_shots_ = std::move(later);

            // Holders.
            for (const auto& sample : sampled_holders) {
                const auto it = std::find(holders_.begin(), holders_.end(), sample.holder);
                // A callback or another thread may have released this token.
                if (it == holders_.end())
                    continue;
                if (sample.expired) {
                    holders_.erase(it);
                    ++holders_expired_;
                    continue;
                }
                auto& h = **it;
                if (h.cadence.kind != Cadence::Kind::PerEvent &&
                    (!sample.next_due || *sample.next_due > now))
                    continue;
                h.last_fired = now;
                apply(plan, h.reason, h.scope, h.views, h.flags, h.detail);
            }

            if (plan.empty())
                ++wakes_without_frame_;
            else
                last_plan_started_ = now;
            return plan;
        }

        // Earliest time anything needs the loop awake; empty means block forever.
        [[nodiscard]] std::optional<FrameClock::time_point> nextDeadline(const FrameClock::time_point now) const {
            auto sampled_holders = sampleHolders(now);
            for (auto& sample : sampled_holders)
                sampleExternalDeadline(sample, now);
            std::lock_guard lock(mutex_);
            std::optional<FrameClock::time_point> earliest;
            const auto consider = [&](const FrameClock::time_point t) {
                if (!earliest || t < *earliest)
                    earliest = t;
            };
            for (const auto& r : one_shots_)
                consider(r.not_before > now ? r.not_before : now);
            for (const auto& w : wakes_)
                consider(w.when);
            for (const auto& sample : sampled_holders) {
                if (std::find(holders_.begin(), holders_.end(), sample.holder) == holders_.end())
                    continue;
                if (const auto d = sample.holder->finite.deadline())
                    consider(*d);
                if (sample.next_due)
                    consider(*sample.next_due);
            }
            return earliest;
        }

        void notePresented(const FramePlan& plan) {
            std::lock_guard lock(mutex_);
            ++frames_presented_;
            if (plan.reasons.none())
                ++frames_without_reason_;
            for (std::size_t i = 0; i < plan.reasons.size(); ++i)
                if (plan.reasons.test(i))
                    ++presents_by_reason_[i];
            last_frame_reasons_ = plan.reasons;
            last_frame_details_ = plan.details;
        }

        void noteViewRendered(const ViewMask views, const FramePlan& plan) {
            std::lock_guard lock(mutex_);
            for (std::size_t v = 0; v < kMaxViews; ++v)
                if (views & (ViewMask{1} << v))
                    ++views_rendered_[v];
            for (std::size_t i = 0; i < plan.reasons.size(); ++i)
                if (plan.reasons.test(i))
                    ++view_renders_by_reason_[i];
        }

        [[nodiscard]] FrameLedgerSnapshot snapshot(const FrameClock::time_point now = FrameClock::now()) const {
            std::lock_guard lock(mutex_);
            FrameLedgerSnapshot s;
            s.frames_presented = frames_presented_;
            s.wakes_without_frame = wakes_without_frame_;
            s.frames_without_reason = frames_without_reason_;
            s.holders_expired = holders_expired_;
            s.requests_dropped = requests_dropped_.load(std::memory_order_relaxed);
            s.preview_skipped_no_step = preview_skipped_no_step_;
            s.stale_detections = stale_detections_;
            s.views_rendered = views_rendered_;
            s.presents_by_reason = presents_by_reason_;
            s.view_renders_by_reason = view_renders_by_reason_;
            s.last_frame_reasons = last_frame_reasons_;
            s.last_frame_details = last_frame_details_;
            for (const auto& h : holders_)
                s.live_holders.push_back(LiveHolderInfo{h->reason, h->scope, h->detail, now - h->created});
            return s;
        }

        // Test and MCP helper: counters only; holders and pending requests stay.
        void resetCounters() {
            std::lock_guard lock(mutex_);
            frames_presented_ = wakes_without_frame_ = frames_without_reason_ = holders_expired_ = 0;
            requests_dropped_.store(0, std::memory_order_relaxed);
            views_rendered_.fill(0);
            presents_by_reason_.fill(0);
            view_renders_by_reason_.fill(0);
            last_frame_reasons_.reset();
            last_frame_details_.clear();
            preview_skipped_no_step_ = 0;
            stale_detections_ = 0;
        }

        void notePreviewSkippedNoStep() {
            std::lock_guard lock(mutex_);
            ++preview_skipped_no_step_;
        }

        void noteStaleDetection() {
            std::lock_guard lock(mutex_);
            ++stale_detections_;
        }

        [[nodiscard]] std::size_t liveHolderCount() const {
            std::lock_guard lock(mutex_);
            return holders_.size();
        }

    private:
        friend class DemandToken;

        struct Holder {
            std::uint64_t id;
            FrameReason reason;
            FrameScope scope;
            ViewMask views;
            DirtyMask flags;
            Cadence cadence;
            Finite finite;
            std::string detail;
            FrameClock::time_point created;
            FrameClock::time_point last_fired;
        };
        struct Wake {
            FrameClock::time_point when;
            std::string why;
        };

        struct HolderSample {
            std::shared_ptr<Holder> holder;
            std::optional<FrameClock::time_point> next_due;
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

        static void sampleExternalDeadline(HolderSample& sample, const FrameClock::time_point now) {
            const auto& cadence = sample.holder->cadence;
            if (cadence.kind == Cadence::Kind::External && cadence.next_due)
                sample.next_due = cadence.next_due(now);
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

        [[nodiscard]] std::optional<FrameClock::time_point> holderNextDue(const Holder& h,
                                                                          const FrameClock::time_point now) const {
            switch (h.cadence.kind) {
            case Cadence::Kind::PerEvent:
                return std::nullopt; // input-driven: the event itself wakes the loop
            case Cadence::Kind::Display: {
                const auto base = std::max(h.last_fired, last_plan_started_);
                return base == FrameClock::time_point{} ? now : base + display_interval_;
            }
            case Cadence::Kind::Interval:
                return h.last_fired == FrameClock::time_point{} ? now : h.last_fired + h.cadence.interval;
            case Cadence::Kind::External:
                return std::nullopt; // evaluated outside the lock by sampleExternalDeadline()
            }
            return std::nullopt;
        }

        mutable std::mutex mutex_;
        std::function<void()> wake_;
        FrameClock::duration display_interval_{std::chrono::microseconds{16667}};
        std::vector<FrameRequest> one_shots_;
        std::vector<std::shared_ptr<Holder>> holders_;
        std::vector<Wake> wakes_;
        ViewMask visible_views_ = kAllViews;
        std::uint64_t next_holder_id_ = 0;
        FrameClock::time_point last_plan_started_{};

        std::uint64_t frames_presented_ = 0;
        std::uint64_t wakes_without_frame_ = 0;
        std::uint64_t frames_without_reason_ = 0;
        std::uint64_t holders_expired_ = 0;
        std::uint64_t preview_skipped_no_step_ = 0;
        std::uint64_t stale_detections_ = 0;
        std::atomic<std::uint64_t> requests_dropped_{0};
        std::array<std::uint64_t, kMaxViews> views_rendered_{};
        std::array<std::uint64_t, static_cast<std::size_t>(FrameReason::Count)> presents_by_reason_{};
        std::array<std::uint64_t, static_cast<std::size_t>(FrameReason::Count)> view_renders_by_reason_{};
        ReasonSet last_frame_reasons_;
        std::vector<std::string> last_frame_details_;
    };

    inline void DemandToken::release() {
        if (ledger_) {
            ledger_->releaseHolder(id_);
            ledger_ = nullptr;
            id_ = 0;
        }
    }

} // namespace lfs::vis
