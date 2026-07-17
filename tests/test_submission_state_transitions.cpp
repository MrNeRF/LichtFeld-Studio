/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Phase 7A: closed SubmissionState transition table (T0,T1,T3–T7).
// Amendment 1: no T2 pre-submit-cancel event; replacement only on T3 for
// the reset/pre-wait row. Ordering (not atomicity) for T4→T5.

#include "rendering/vulkan_wait.hpp"

#include <gtest/gtest.h>

using lfs::ErrorCode;
using lfs::ErrorDomain;
using lfs::rendering::SubmissionFencePolicy;
using lfs::rendering::SubmissionState;
using lfs::rendering::SubmissionTransition;
using lfs::rendering::apply_submission_transition;

namespace {

    void expect_fresh(const SubmissionState& s) {
        EXPECT_FALSE(s.fence_reset);
        EXPECT_FALSE(s.submit_accepted);
        EXPECT_FALSE(s.timeline_published);
    }

} // namespace

TEST(SubmissionStateTransitions, T0BeginLifecycle) {
    SubmissionState s;
    s.submit_accepted = true; // dirty
    auto r = apply_submission_transition(s, SubmissionTransition::BeginLifecycle,
                                         SubmissionFencePolicy::NoResetNoReplacement,
                                         /*candidate*/ 42);
    ASSERT_TRUE(r.has_value());
    expect_fresh(s);
    EXPECT_EQ(s.candidate_timeline, 42u);
    EXPECT_FALSE(r->replace_fence_signaled);
    EXPECT_FALSE(r->terminal_device_lost);
}

TEST(SubmissionStateTransitions, T1FenceResetPreSubmit) {
    SubmissionState s;
    auto r = apply_submission_transition(s, SubmissionTransition::FenceReset);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(s.fence_reset);
    EXPECT_FALSE(s.submit_accepted);
    EXPECT_FALSE(s.timeline_published);
}

TEST(SubmissionStateTransitions, T1FenceResetIllegalAfterAccept) {
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::SubmitAccepted));
    auto r = apply_submission_transition(s, SubmissionTransition::FenceReset);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code(), ErrorCode::ContractViolation);
    EXPECT_EQ(r.error().domain(), ErrorDomain::Vulkan);
}

TEST(SubmissionStateTransitions, T3RejectedNoResetNoReplace) {
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::BeginLifecycle,
                                            SubmissionFencePolicy::NoResetNoReplacement,
                                            7));
    // gs_pipeline never fence_resets before submit.
    auto r = apply_submission_transition(s, SubmissionTransition::SubmitRejected,
                                         SubmissionFencePolicy::NoResetNoReplacement);
    ASSERT_TRUE(r.has_value());
    EXPECT_FALSE(r->replace_fence_signaled);
    EXPECT_FALSE(s.submit_accepted);
    EXPECT_FALSE(s.timeline_published);
}

TEST(SubmissionStateTransitions, T3RejectedResetPreWaitReplaces) {
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::FenceReset,
                                            SubmissionFencePolicy::ResetPreWaitReplacement));
    EXPECT_TRUE(s.fence_reset);
    auto r = apply_submission_transition(s, SubmissionTransition::SubmitRejected,
                                         SubmissionFencePolicy::ResetPreWaitReplacement);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->replace_fence_signaled);
    EXPECT_FALSE(s.submit_accepted);
    EXPECT_FALSE(s.timeline_published);
    EXPECT_FALSE(s.fence_reset); // replaced fence is signaled, not "reset"
}

TEST(SubmissionStateTransitions, T3RejectedNoResetEvenIfFenceResetBitSet) {
    // Policy NoResetNoReplacement never authorizes replace, even if the bit
    // is somehow set (defensive; gs_pipeline never sets it pre-submit).
    SubmissionState s;
    s.fence_reset = true;
    auto r = apply_submission_transition(s, SubmissionTransition::SubmitRejected,
                                         SubmissionFencePolicy::NoResetNoReplacement);
    ASSERT_TRUE(r.has_value());
    EXPECT_FALSE(r->replace_fence_signaled);
}

TEST(SubmissionStateTransitions, T3IllegalAfterPublish) {
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::SubmitAccepted));
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::PublishTimeline,
                                            SubmissionFencePolicy::NoResetNoReplacement,
                                            9));
    auto r = apply_submission_transition(s, SubmissionTransition::SubmitRejected);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code(), ErrorCode::ContractViolation);
}

TEST(SubmissionStateTransitions, T4ThenT5Ordering) {
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::BeginLifecycle,
                                            SubmissionFencePolicy::NoResetNoReplacement,
                                            100));
    auto t4 = apply_submission_transition(s, SubmissionTransition::SubmitAccepted,
                                          SubmissionFencePolicy::NoResetNoReplacement,
                                          100);
    ASSERT_TRUE(t4.has_value());
    EXPECT_TRUE(s.submit_accepted);
    EXPECT_FALSE(s.timeline_published);

    // Invariant: timeline_published implies submit_accepted — publish is
    // illegal without accept.
    SubmissionState orphan;
    auto bad = apply_submission_transition(orphan, SubmissionTransition::PublishTimeline);
    ASSERT_FALSE(bad.has_value());

    auto t5 = apply_submission_transition(s, SubmissionTransition::PublishTimeline,
                                          SubmissionFencePolicy::NoResetNoReplacement,
                                          100);
    ASSERT_TRUE(t5.has_value());
    EXPECT_TRUE(s.submit_accepted);
    EXPECT_TRUE(s.timeline_published);
    EXPECT_EQ(s.candidate_timeline, 100u);
}

TEST(SubmissionStateTransitions, T5SecondPublishIllegal) {
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::SubmitAccepted));
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::PublishTimeline,
                                            SubmissionFencePolicy::NoResetNoReplacement,
                                            1));
    auto r = apply_submission_transition(s, SubmissionTransition::PublishTimeline,
                                         SubmissionFencePolicy::NoResetNoReplacement,
                                         1);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code(), ErrorCode::ContractViolation);
}

TEST(SubmissionStateTransitions, T6GpuReadyClearsLifecycle) {
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::SubmitAccepted));
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::PublishTimeline,
                                            SubmissionFencePolicy::NoResetNoReplacement,
                                            5));
    auto r = apply_submission_transition(s, SubmissionTransition::GpuReady);
    ASSERT_TRUE(r.has_value());
    expect_fresh(s);
    EXPECT_EQ(s.candidate_timeline, 0u);
}

TEST(SubmissionStateTransitions, T7DeviceLostFromAnyState) {
    {
        SubmissionState s;
        auto r = apply_submission_transition(s, SubmissionTransition::DeviceLost);
        ASSERT_TRUE(r.has_value());
        EXPECT_TRUE(r->terminal_device_lost);
    }
    {
        SubmissionState s;
        ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::SubmitAccepted));
        ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::PublishTimeline,
                                                SubmissionFencePolicy::NoResetNoReplacement,
                                                3));
        auto r = apply_submission_transition(s, SubmissionTransition::DeviceLost);
        ASSERT_TRUE(r.has_value());
        EXPECT_TRUE(r->terminal_device_lost);
        // Bits retained (no host-signal / no free); terminal is external latch.
        EXPECT_TRUE(s.submit_accepted);
        EXPECT_TRUE(s.timeline_published);
    }
}

TEST(SubmissionStateTransitions, InvariantTimelineImpliesAccepted) {
    // Property check across the happy path.
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::BeginLifecycle,
                                            SubmissionFencePolicy::NoResetNoReplacement,
                                            11));
    ASSERT_FALSE(s.timeline_published);
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::SubmitAccepted));
    ASSERT_TRUE(s.submit_accepted);
    ASSERT_FALSE(s.timeline_published);
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::PublishTimeline,
                                            SubmissionFencePolicy::NoResetNoReplacement,
                                            11));
    EXPECT_TRUE(s.timeline_published);
    EXPECT_TRUE(s.submit_accepted);
}

TEST(SubmissionStateTransitions, AcceptedSubmitWithoutTimelineIsLegal) {
    // Pure-fence submits may leave timeline_published false after accept
    // (spec §2.4 invariant 2 for fence-only paths).
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::SubmitAccepted));
    EXPECT_TRUE(s.submit_accepted);
    EXPECT_FALSE(s.timeline_published);
}

// Phase 7C-P2: ResetPreWaitReplacement end-to-end (point-cloud pure-fence row).

TEST(SubmissionStateTransitions, ResetPreWaitHappyPathT0T1T4NoTimeline) {
    // Appendix A.1 happy path: T0 → T1 → T4; no T5 on pure-fence.
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::BeginLifecycle,
                                            SubmissionFencePolicy::ResetPreWaitReplacement));
    expect_fresh(s);

    auto t1 = apply_submission_transition(s, SubmissionTransition::FenceReset,
                                          SubmissionFencePolicy::ResetPreWaitReplacement);
    ASSERT_TRUE(t1.has_value());
    EXPECT_TRUE(s.fence_reset);
    EXPECT_FALSE(t1->replace_fence_signaled);

    auto t4 = apply_submission_transition(s, SubmissionTransition::SubmitAccepted,
                                          SubmissionFencePolicy::ResetPreWaitReplacement);
    ASSERT_TRUE(t4.has_value());
    EXPECT_TRUE(s.submit_accepted);
    EXPECT_FALSE(s.timeline_published);
    EXPECT_FALSE(t4->replace_fence_signaled);
}

TEST(SubmissionStateTransitions, ResetPreWaitRejectEndToEndThenRestartT0) {
    // T0 → T1 → T3 (replace) → T0 restarts clean for the next attempt.
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::BeginLifecycle,
                                            SubmissionFencePolicy::ResetPreWaitReplacement));
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::FenceReset,
                                            SubmissionFencePolicy::ResetPreWaitReplacement));
    EXPECT_TRUE(s.fence_reset);

    auto t3 = apply_submission_transition(s, SubmissionTransition::SubmitRejected,
                                          SubmissionFencePolicy::ResetPreWaitReplacement);
    ASSERT_TRUE(t3.has_value());
    EXPECT_TRUE(t3->replace_fence_signaled);
    EXPECT_FALSE(s.fence_reset); // replaced signaled fence is not "reset"
    EXPECT_FALSE(s.submit_accepted);
    EXPECT_FALSE(s.timeline_published);

    auto restart = apply_submission_transition(s, SubmissionTransition::BeginLifecycle,
                                               SubmissionFencePolicy::ResetPreWaitReplacement);
    ASSERT_TRUE(restart.has_value());
    expect_fresh(s);
    EXPECT_FALSE(restart->replace_fence_signaled);
}

TEST(SubmissionStateTransitions, T3RejectedWithoutFenceResetDoesNotReplace) {
    // §1.C.3: never replace when fence was not reset. Ruling-1 reset-failure
    // recovery is modeled outside the table (host-side replace only).
    SubmissionState s;
    ASSERT_TRUE(apply_submission_transition(s, SubmissionTransition::BeginLifecycle,
                                            SubmissionFencePolicy::ResetPreWaitReplacement));
    EXPECT_FALSE(s.fence_reset);

    auto t3 = apply_submission_transition(s, SubmissionTransition::SubmitRejected,
                                          SubmissionFencePolicy::ResetPreWaitReplacement);
    ASSERT_TRUE(t3.has_value());
    EXPECT_FALSE(t3->replace_fence_signaled);
    EXPECT_FALSE(s.fence_reset);
    EXPECT_FALSE(s.submit_accepted);
}
