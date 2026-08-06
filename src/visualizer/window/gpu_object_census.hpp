/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis {

    enum class GpuObjectKind : std::uint8_t {
        ExternalImage = 0,
        ExternalBuffer = 1,
        ExternalSemaphore = 2,
    };

    struct LFS_VIS_API GpuObjectCensusRow {
        GpuObjectKind kind{};
        std::string scope;
        std::int64_t count = 0;
    };

    // Live-count census for External* objects (#1488).
    class LFS_VIS_API GpuObjectCensus {
    public:
        void onCreate(GpuObjectKind kind, std::string_view scope);
        void onDestroy(GpuObjectKind kind, std::string_view scope);

        // Non-zero live rows only, sorted by (kind, scope).
        [[nodiscard]] std::vector<GpuObjectCensusRow> report() const;

        // True if any destroy-without-create underflow was observed (clamped to 0).
        [[nodiscard]] bool underflowFlagged() const noexcept;

        void reset();

    private:
        using Key = std::pair<GpuObjectKind, std::string>;
        std::map<Key, std::int64_t> counts_;
        bool underflow_flagged_ = false;
    };

} // namespace lfs::vis
