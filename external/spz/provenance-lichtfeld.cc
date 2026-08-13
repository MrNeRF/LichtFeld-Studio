/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "provenance-lichtfeld.h"

#include <limits>
#include <utility>

namespace spz {

    SpzExtensionProvenanceLichtFeld::SpzExtensionProvenanceLichtFeld()
        : SpzExtensionBase(SpzExtensionType::SPZ_LICHTFELD_provenance) {}

    uint32_t SpzExtensionProvenanceLichtFeld::payloadBytes() const {
        return static_cast<uint32_t>(json.size());
    }

    void SpzExtensionProvenanceLichtFeld::write(std::ostream& os) const {
        const uint32_t t = static_cast<uint32_t>(extensionType);
        const uint32_t len = payloadBytes();
        os.write(reinterpret_cast<const char*>(&t), sizeof(t));
        os.write(reinterpret_cast<const char*>(&len), sizeof(len));
        if (len > 0) {
            os.write(json.data(), static_cast<std::streamsize>(len));
        }
    }

    std::optional<SpzExtensionBasePtr> SpzExtensionProvenanceLichtFeld::read(std::istream& is) {
        std::string json;
        json.resize(static_cast<std::size_t>(kMaxLichtFeldProvenanceBytes) + 1);
        is.read(json.data(), static_cast<std::streamsize>(json.size()));
        const auto n = static_cast<uint32_t>(is.gcount());
        if (n > kMaxLichtFeldProvenanceBytes) {
            if (is)
                is.ignore(std::numeric_limits<std::streamsize>::max());
            return std::nullopt;
        }
        json.resize(n);
        auto rec = std::make_shared<SpzExtensionProvenanceLichtFeld>();
        rec->json = std::move(json);
        return std::optional{std::move(rec)};
    }

    SpzExtensionType SpzExtensionProvenanceLichtFeld::type() {
        return SpzExtensionType::SPZ_LICHTFELD_provenance;
    }

    SpzExtensionBase* SpzExtensionProvenanceLichtFeld::copyAsRawData() const {
        return new SpzExtensionProvenanceLichtFeld(*this);
    }

    // No PLY representation for this extension.
    std::optional<SpzExtensionBasePtr> SpzExtensionProvenanceLichtFeld::tryReadFromPly(
        std::istream&, const std::unordered_set<std::string>&) const {
        return std::nullopt;
    }

    void SpzExtensionProvenanceLichtFeld::writePlyHeader(std::ostream&) const {}

    void SpzExtensionProvenanceLichtFeld::writePlyData(std::ostream&) const {}

} // namespace spz
