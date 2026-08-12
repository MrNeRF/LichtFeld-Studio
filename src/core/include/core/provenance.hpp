/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <string>

namespace lfs::core {

    struct LFS_CORE_API ProvenanceStamp {
        std::string export_id;   // fresh UUIDv4 per export
        int iteration = -1;      // training iteration; -1 = unknown
        std::string strategy;    // canonical optimizer strategy name; empty = unknown
        std::string app_version; // GIT_TAGGED_VERSION
        std::string exported_at; // ISO-8601 UTC
        // Project identity (.licht project format, #1525 / #1507). Empty until the
        // project session wiring lands; serialized only when non-empty.
        std::string project_id;
        std::string commit_id;
        std::string node_id;
        std::string dataset_id;
    };

    // Builds a stamp with export_id/app_version/exported_at filled; iteration and
    // strategy are the caller's to set when known.
    [[nodiscard]] LFS_CORE_API ProvenanceStamp make_provenance_stamp();

    // Single-line JSON: {"lichtfeld_provenance":1,"export_id":...,...}
    // Omits iteration when < 0 and every string field that is empty. Field order:
    // lichtfeld_provenance, export_id, project, commit, node, dataset, iteration,
    // strategy, app_version, exported_at. No newlines.
    [[nodiscard]] LFS_CORE_API std::string provenance_to_json(const ProvenanceStamp& stamp);

} // namespace lfs::core
