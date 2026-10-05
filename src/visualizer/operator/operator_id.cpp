/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "operator_id.hpp"
#include <array>
#include <cassert>

namespace lfs::vis::op {

    namespace {

        struct OpInfo {
            const char* id_string;
        };

        constexpr std::array<OpInfo, static_cast<size_t>(BuiltinOp::_Count)> OP_INFO = {{
            {"selection.stroke"},
            {"transform.set"},
            {"transform.translate"},
            {"transform.rotate"},
            {"transform.scale"},
            {"transform.apply_batch"},
            {"align.pick_point"},
            {"ed.undo"},
            {"ed.redo"},
            {"ed.delete"},
            {"selection.clear"},
            {"scene.select_node"},
            {"crop_box.add"},
            {"crop_box.set"},
            {"crop_box.fit"},
            {"crop_box.reset"},
            {"ellipsoid.add"},
            {"ellipsoid.set"},
            {"ellipsoid.fit"},
            {"ellipsoid.reset"},
            {"selection.depth_window_drag"},
        }};

    } // namespace

    const char* to_string(BuiltinOp op) {
        const auto idx = static_cast<size_t>(op);
        assert(idx < OP_INFO.size());
        return OP_INFO[idx].id_string;
    }

    std::optional<BuiltinOp> builtin_op_from_string(std::string_view s) {
        for (size_t i = 0; i < OP_INFO.size(); ++i) {
            if (s == OP_INFO[i].id_string) {
                return static_cast<BuiltinOp>(i);
            }
        }
        return std::nullopt;
    }

} // namespace lfs::vis::op
