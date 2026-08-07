/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gpu_object_census.hpp"

namespace lfs::vis {

    void GpuObjectCensus::onCreate(const GpuObjectKind kind, const std::string_view scope) {
        ++counts_[Key{kind, std::string(scope)}];
    }

    void GpuObjectCensus::onDestroy(const GpuObjectKind kind, const std::string_view scope) {
        const Key key{kind, std::string(scope)};
        const auto it = counts_.find(key);
        if (it == counts_.end() || it->second <= 0) {
            underflow_flagged_ = true;
            if (it != counts_.end()) {
                it->second = 0;
            }
            return;
        }
        --it->second;
        if (it->second == 0) {
            counts_.erase(it);
        }
    }

    std::vector<GpuObjectCensusRow> GpuObjectCensus::report() const {
        std::vector<GpuObjectCensusRow> rows;
        rows.reserve(counts_.size());
        for (const auto& [key, count] : counts_) {
            if (count <= 0) {
                continue;
            }
            rows.push_back(GpuObjectCensusRow{
                .kind = key.first,
                .scope = key.second,
                .count = count,
            });
        }
        // std::map iteration is already ordered by (kind, scope).
        return rows;
    }

    bool GpuObjectCensus::underflowFlagged() const noexcept {
        return underflow_flagged_;
    }

} // namespace lfs::vis
