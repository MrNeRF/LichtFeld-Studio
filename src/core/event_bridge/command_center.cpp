/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/control/command_api.hpp"

namespace lfs::training {

    CommandCenter& CommandCenter::instance() {
        // lfs_training is linked into multiple modules, so the singleton storage
        // must stay in this shared library.
        static CommandCenter inst;
        return inst;
    }

    CommandCenter::CommandCenter() {
        ops_.push_back({.name = "set_attribute",
                        .target = CommandTarget::Model,
                        .selectors = {SelectionKind::All, SelectionKind::Range, SelectionKind::Indices},
                        .args = {{"attribute", ArgType::String, true, "Attribute name (means, scaling, rotation, opacity, sh0, shN)"},
                                 {"value", ArgType::Float, false, "Scalar value"},
                                 {"values", ArgType::FloatList, false, "Vector value (broadcast)"}},
                        .description = "Set attribute values for selected splats (scalar or per-dim vector)."});

        ops_.push_back({.name = "scale_attribute",
                        .target = CommandTarget::Model,
                        .selectors = {SelectionKind::All, SelectionKind::Range, SelectionKind::Indices},
                        .args = {{"attribute", ArgType::String, true, "Attribute name"},
                                 {"factor", ArgType::Float, true, "Multiplicative scale"}},
                        .description = "Scale attribute by factor for selected splats."});

        ops_.push_back({.name = "clamp_attribute",
                        .target = CommandTarget::Model,
                        .selectors = {SelectionKind::All, SelectionKind::Range, SelectionKind::Indices},
                        .args = {{"attribute", ArgType::String, true, "Attribute name"},
                                 {"min", ArgType::Float, false, "Optional min"},
                                 {"max", ArgType::Float, false, "Optional max"}},
                        .description = "Clamp attribute values for selected splats."});

        ops_.push_back({.name = "set_lr",
                        .target = CommandTarget::Optimizer,
                        .selectors = {SelectionKind::All},
                        .args = {{"value", ArgType::Float, true, "Learning rate"}},
                        .description = "Set global learning rate."});

        ops_.push_back({.name = "scale_lr",
                        .target = CommandTarget::Optimizer,
                        .selectors = {SelectionKind::All},
                        .args = {{"factor", ArgType::Float, true, "Scale factor"}},
                        .description = "Scale global learning rate."});

        ops_.push_back({.name = "pause",
                        .target = CommandTarget::Session,
                        .selectors = {SelectionKind::All},
                        .args = {},
                        .description = "Request pause."});

        ops_.push_back({.name = "resume",
                        .target = CommandTarget::Session,
                        .selectors = {SelectionKind::All},
                        .args = {},
                        .description = "Request resume."});

        ops_.push_back({.name = "request_stop",
                        .target = CommandTarget::Session,
                        .selectors = {SelectionKind::All},
                        .args = {},
                        .description = "Request graceful stop."});

        // Mutable fields
        mutable_fields_.push_back({"means", CommandTarget::Model, "[N,3]", "Gaussian means", true});
        mutable_fields_.push_back({"scaling", CommandTarget::Model, "[N,3]", "Log scaling", true});
        mutable_fields_.push_back({"rotation", CommandTarget::Model, "[N,4]", "Quaternion rotation", true});
        mutable_fields_.push_back({"opacity", CommandTarget::Model, "[N]", "Opacity logits", true});
        mutable_fields_.push_back({"sh0", CommandTarget::Model, "[N,3]", "SH0 coefficients", true});
        mutable_fields_.push_back({"shN", CommandTarget::Model, "[N,?]", "Higher-order SH coefficients", true});
    }

} // namespace lfs::training
