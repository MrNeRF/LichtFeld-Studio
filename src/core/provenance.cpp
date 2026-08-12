/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/provenance.hpp"
#include "core/utc_time.hpp"

#include "git_version.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

namespace lfs::core {

    namespace {

        bool is_safe_identity_id(const std::string& value) {
            if (value.empty() || value.size() > 64)
                return false;
            for (const unsigned char c : value) {
                if (!std::isxdigit(c) && c != '-')
                    return false;
            }
            return true;
        }

        std::string serialize_provenance_json(const ProvenanceStamp& stamp, const bool with_identity) {
            nlohmann::ordered_json json;
            json["lichtfeld_provenance"] = 1;
            if (!stamp.export_id.empty())
                json["export_id"] = stamp.export_id;
            if (with_identity) {
                if (is_safe_identity_id(stamp.project_id))
                    json["project"] = stamp.project_id;
                if (is_safe_identity_id(stamp.commit_id))
                    json["commit"] = stamp.commit_id;
                if (is_safe_identity_id(stamp.node_id))
                    json["node"] = stamp.node_id;
                if (is_safe_identity_id(stamp.dataset_id))
                    json["dataset"] = stamp.dataset_id;
            }
            if (stamp.iteration >= 0)
                json["iteration"] = stamp.iteration;
            if (!stamp.strategy.empty())
                json["strategy"] = stamp.strategy;
            if (!stamp.app_version.empty())
                json["app_version"] = stamp.app_version;
            if (!stamp.build_commit.empty())
                json["build_commit"] = stamp.build_commit;
            if (!stamp.exported_at.empty())
                json["exported_at"] = stamp.exported_at;
            return json.dump();
        }

        std::string uuid_v4() {
            std::random_device rd;
            std::array<std::uint8_t, 16> bytes{};
            for (std::size_t i = 0; i < 4; ++i) {
                const auto word = static_cast<std::uint32_t>(rd());
                bytes[i * 4 + 0] = static_cast<std::uint8_t>(word);
                bytes[i * 4 + 1] = static_cast<std::uint8_t>(word >> 8);
                bytes[i * 4 + 2] = static_cast<std::uint8_t>(word >> 16);
                bytes[i * 4 + 3] = static_cast<std::uint8_t>(word >> 24);
            }

            bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0f) | 0x40);
            bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3f) | 0x80);

            static constexpr char kHex[] = "0123456789abcdef";
            std::string out(36, '-');
            auto write2 = [&](const std::size_t pos, const std::uint8_t value) {
                out[pos] = kHex[value >> 4];
                out[pos + 1] = kHex[value & 0x0f];
            };
            write2(0, bytes[0]);
            write2(2, bytes[1]);
            write2(4, bytes[2]);
            write2(6, bytes[3]);
            write2(9, bytes[4]);
            write2(11, bytes[5]);
            write2(14, bytes[6]);
            write2(16, bytes[7]);
            write2(19, bytes[8]);
            write2(21, bytes[9]);
            write2(24, bytes[10]);
            write2(26, bytes[11]);
            write2(28, bytes[12]);
            write2(30, bytes[13]);
            write2(32, bytes[14]);
            write2(34, bytes[15]);
            return out;
        }

    } // namespace

    ProvenanceStamp make_provenance_stamp() {
        ProvenanceStamp stamp;
        stamp.export_id = uuid_v4();
        stamp.app_version = GIT_TAGGED_VERSION;
        stamp.build_commit = GIT_COMMIT_HASH_SHORT;
        stamp.exported_at = utc_now();
        return stamp;
    }

    ProvenanceStamp make_minimal_provenance_stamp() {
        ProvenanceStamp stamp;
        stamp.app_version = GIT_TAGGED_VERSION;
        stamp.build_commit = GIT_COMMIT_HASH_SHORT;
        return stamp;
    }

    std::string provenance_to_json(const ProvenanceStamp& stamp) {
        std::string json = serialize_provenance_json(stamp, true);
        if (json.size() > 4096)
            json = serialize_provenance_json(stamp, false);
        return json;
    }

} // namespace lfs::core
