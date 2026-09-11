/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SPZ_PROVENANCE_LICHTFELD_H_
#define SPZ_PROVENANCE_LICHTFELD_H_

#include "splat-extensions.h"

#include <string>

namespace spz {

    constexpr uint32_t kMaxLichtFeldProvenanceBytes = 4096;

    // LichtFeld provenance JSON attached to an SPZ v4 container.
    // The payload is a single-line UTF-8 JSON string (no framing inside the payload;
    // type and byteLength are written by write() per the extension stream format).
    // This extension has no PLY representation.
    struct SpzExtensionProvenanceLichtFeld : public SpzExtensionBase {
        std::string json;

        SpzExtensionProvenanceLichtFeld();
        uint32_t payloadBytes() const override;
        void write(std::ostream& os) const override;
        SpzExtensionBase* copyAsRawData() const override;
        std::optional<std::shared_ptr<SpzExtensionBase>> tryReadFromPly(
            std::istream& in, const std::unordered_set<std::string>& elementNames) const override;
        void writePlyHeader(std::ostream& out) const override;
        void writePlyData(std::ostream& out) const override;
        static std::optional<SpzExtensionBasePtr> read(std::istream& is);
        static SpzExtensionType type();
    };

} // namespace spz

#endif // SPZ_PROVENANCE_LICHTFELD_H_
