/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <array>
#include <set>
#include <string>
#include <string_view>

namespace lfs::test::licht {

    inline constexpr std::string_view P3_MATRIX_ROW_DATA =
        "PROJ-39 PROJ-40 PROJ-41 PROJ-42 PROJ-43 PROJ-44 PROJ-45 "
        "PRMS-53 PRMS-54 PRMS-55 PRMS-56 SCNG-64 SCNG-65 SCNG-66 SCNG-67 SCNG-68 "
        "SCNG-69 SCNG-70 SCNG-71 SCNG-72 SCNG-73 SCNG-74 SCNG-75 SCNG-76 SCNG-77 "
        "SELM-83 SELM-84 SELM-85 SELM-86 REFS-96 REFS-97 REFS-98 REFS-99 REFS-100 "
        "REFS-101 REFS-102 SPLT-110 SPLT-111 SPLT-112 SPLT-113 SPLT-114 SPLT-115 SPLT-116";
    inline constexpr std::string_view P4_CKPT_MATRIX_ROW_DATA =
        "CKPT-126 CKPT-127 CKPT-128 CKPT-129 CKPT-130 CKPT-131 CKPT-132 CKPT-133 "
        "CKPT-134 CKPT-135 CKPT-136 CKPT-137 CKPT-138 CKPT-139 CKPT-140 CKPT-141 "
        "CKPT-142 CKPT-143 CKPT-144 CKPT-145 CKPT-146 CKPT-147 CKPT-148 CKPT-149";
    inline constexpr std::string_view P4_PPIS_MATRIX_ROW_DATA =
        "PPIS-157 PPIS-158 PPIS-159 PPIS-160";
    inline constexpr std::string_view P5_MATRIX_ROW_DATA =
        "GUIL-166 GUIL-167 GUIL-168 GUIL-169 GUIL-170 GUIL-171 EDTR-179 EDTR-180 "
        "EDTR-181 EDTR-182 EDTR-183 EDTR-184 VIEW-192 VIEW-193 VIEW-194 VIEW-195 "
        "VIEW-196 VIEW-197 VIEW-198 VIEW-199 VIEW-200 VIEW-201 VIEW-202 VIEW-203 "
        "VIEW-204 VIEW-205 VIEW-206 VIEW-207 VIEW-208 VIEW-209 VIEW-210 SEQR-218 "
        "SEQR-219 SEQR-220 SEQR-221 SEQR-222 METR-230 METR-231 METR-232 METR-233";

    struct PendingParameterExclusion {
        std::string_view field;
        std::string_view phase;
    };

    inline constexpr std::array PENDING_PARAMETER_EXCLUSIONS{
        PendingParameterExclusion{"headless", "P6/process launch"},
        PendingParameterExclusion{"auto_train", "P6/process launch"},
        PendingParameterExclusion{"no_splash", "P6/process launch"},
        PendingParameterExclusion{"debug_python", "P6/process launch"},
        PendingParameterExclusion{"debug_python_port", "P6/process launch"},
        PendingParameterExclusion{"config_file", "P6/process launch"},
    };

    [[nodiscard]] inline std::set<std::string> word_set(const std::string_view words) {
        std::set<std::string> result;
        for (std::size_t begin = 0; begin < words.size();) {
            while (begin < words.size() && words[begin] == ' ') {
                ++begin;
            }
            const std::size_t end = words.find(' ', begin);
            if (begin < words.size()) {
                result.emplace(words.substr(begin, end - begin));
            }
            begin = end == std::string_view::npos ? words.size() : end + 1;
        }
        return result;
    }

} // namespace lfs::test::licht
