/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/decimate/types.hpp"

namespace lfs::io::decimate {
    using core::decimate::allocate;
    using core::decimate::Candidates;
    using core::decimate::candidates_k;
    using core::decimate::Data;
    using core::decimate::invalid;
    using core::decimate::knn_k;
    using core::decimate::Selection;
    using core::decimate::View;
    Selection select(const Candidates&, size_t n, int k, size_t needed);
    Candidates cpu_candidates(const Data&);
    Candidates gpu_candidates(const Data&);
    Data cpu_merge(const Data&, const Selection&);
    Data gpu_merge(const Data&, const Selection&);
} // namespace lfs::io::decimate
