/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "io/json_chapter_dom.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::io::project {

    // User-global GUI fields that must never persist in project GUIL chapters.
    // Shared by sanitize/validate (session_chapters) and runtime capture checks
    // (session_state). Keys are matched case-insensitively after lowercasing.
    inline constexpr std::array<std::string_view, 7>
        kUserGlobalGuiFieldKeys = {
            "theme",
            "language",
            "scale",
            "ui_scale",
            "hud",
            "vram_hud",
            "vram_hud_visible",
    };

    enum class SessionJsonChapterKind : std::uint8_t {
        GuiLayout,
        Editor,
        View,
        Sequencer,
    };

    [[nodiscard]] LFS_IO_API JsonChapterDom
    default_session_chapter_dom(SessionJsonChapterKind kind);

    [[nodiscard]] LFS_IO_API lfs::Result<void>
    validate_session_chapter_dom(
        SessionJsonChapterKind kind,
        const JsonChapterDom& dom);

    // Foreign GUIL writers may accidentally persist user-global fields.
    // Loading strips those fields transactionally; validation used by save
    // remains strict so LichtFeld never writes them itself.
    [[nodiscard]] LFS_IO_API lfs::Result<void>
    sanitize_session_chapter_for_load(
        SessionJsonChapterKind kind,
        JsonChapterDom& dom);

    // Recursively edits only fields supplied by known_state. Existing
    // object/array-element extension members remain in the retained DOM.
    [[nodiscard]] LFS_IO_API lfs::Result<void>
    merge_session_chapter_known_state(
        JsonChapterDom& destination,
        const JsonChapterDom::Json& known_state);

    template <SessionJsonChapterKind Kind>
    class BasicSessionJsonChapter {
    public:
        BasicSessionJsonChapter()
            : dom_(default_session_chapter_dom(Kind)) {}

        explicit BasicSessionJsonChapter(JsonChapterDom dom)
            : dom_(std::move(dom)) {}

        [[nodiscard]] static lfs::Result<
            BasicSessionJsonChapter>
        parse(const std::string_view bytes) {
            auto dom = JsonChapterDom::parse(bytes);
            if (!dom) {
                return std::move(dom).error();
            }
            if (auto sanitized =
                    sanitize_session_chapter_for_load(
                        Kind, *dom);
                !sanitized) {
                return std::move(sanitized).error();
            }
            if (auto valid =
                    validate_session_chapter_dom(
                        Kind, *dom);
                !valid) {
                return std::move(valid).error();
            }
            return BasicSessionJsonChapter(
                std::move(*dom));
        }

        [[nodiscard]] static lfs::Result<
            BasicSessionJsonChapter>
        from_bytes(
            const std::span<const std::byte> bytes) {
            auto dom = JsonChapterDom::from_bytes(bytes);
            if (!dom) {
                return std::move(dom).error();
            }
            if (auto sanitized =
                    sanitize_session_chapter_for_load(
                        Kind, *dom);
                !sanitized) {
                return std::move(sanitized).error();
            }
            if (auto valid =
                    validate_session_chapter_dom(
                        Kind, *dom);
                !valid) {
                return std::move(valid).error();
            }
            return BasicSessionJsonChapter(
                std::move(*dom));
        }

        [[nodiscard]] const JsonChapterDom& dom() const noexcept {
            return dom_;
        }

        [[nodiscard]] JsonChapterDom& dom() noexcept {
            return dom_;
        }

        [[nodiscard]] std::vector<std::byte>
        to_bytes() const {
            return dom_.to_bytes();
        }

        [[nodiscard]] lfs::Result<void>
        validate() const {
            return validate_session_chapter_dom(
                Kind, dom_);
        }

        [[nodiscard]] lfs::Result<void>
        merge_known_state(
            const JsonChapterDom::Json& known_state) {
            JsonChapterDom staged = dom_;
            if (auto merged =
                    merge_session_chapter_known_state(
                        staged, known_state);
                !merged) {
                return merged;
            }
            if (auto valid =
                    validate_session_chapter_dom(
                        Kind, staged);
                !valid) {
                return valid;
            }
            dom_ = std::move(staged);
            return {};
        }

    private:
        JsonChapterDom dom_;
    };

    using GuiLayoutChapter =
        BasicSessionJsonChapter<
            SessionJsonChapterKind::GuiLayout>;
    using EditorSessionChapter =
        BasicSessionJsonChapter<
            SessionJsonChapterKind::Editor>;
    using ViewSessionChapter =
        BasicSessionJsonChapter<
            SessionJsonChapterKind::View>;
    using SequencerSessionChapter =
        BasicSessionJsonChapter<
            SessionJsonChapterKind::Sequencer>;

    struct MetricHistorySample {
        std::int32_t iteration = 0;
        float value = 0.0f;

        friend bool operator==(
            const MetricHistorySample&,
            const MetricHistorySample&) = default;
    };

    struct LastEvaluationMetrics {
        std::int32_t iteration = 0;
        float psnr = 0.0f;
        float ssim = 0.0f;

        friend bool operator==(
            const LastEvaluationMetrics&,
            const LastEvaluationMetrics&) = default;
    };

    // Additive METR trailer. Absent/zero means a resumable pause. Written
    // only when the run actually finished so mid-training files stay
    // byte-compatible with VERSION 1 readers.
    enum class TrainingFinishReason : std::uint32_t {
        None = 0,
        Completed = 1,
        UserStopped = 2,
        Error = 3,
    };

    class LFS_IO_API MetricsChapter {
    public:
        static constexpr std::uint32_t VERSION = 1;
        static constexpr std::size_t MAX_HISTORY_SAMPLES =
            10'000'000;

        [[nodiscard]] static lfs::Result<MetricsChapter>
        from_bytes(std::span<const std::byte> bytes);

        [[nodiscard]] lfs::Result<void>
        validate() const;
        [[nodiscard]] lfs::Result<std::vector<std::byte>>
        to_bytes() const;

        std::vector<MetricHistorySample> loss_history;
        std::vector<MetricHistorySample> psnr_history;
        double accumulated_training_seconds = 0.0;
        std::optional<LastEvaluationMetrics> last_evaluation;
        TrainingFinishReason finish_reason =
            TrainingFinishReason::None;

        friend bool operator==(
            const MetricsChapter&,
            const MetricsChapter&) = default;
    };

    struct ProjectSessionChapters {
        GuiLayoutChapter gui_layout;
        EditorSessionChapter editor;
        ViewSessionChapter view;
        SequencerSessionChapter sequencer;
        MetricsChapter metrics;
    };

} // namespace lfs::io::project
