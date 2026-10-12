/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/loader.hpp"
#include "tinyply.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace lfs::io::detail {

    // Tinyply packs a requested group in header order, not request order.
    // Canonical headers need no data pass; reorder other scalar triples in place.
    inline void order_ply_vertex_properties(const tinyply::PlyFile& file,
                                            tinyply::PlyData& data,
                                            const std::array<std::string_view, 3>& names,
                                            const LoadOptions& options) {
        std::array<size_t, 3> offsets{};
        size_t found = 0;
        for (const auto& element : file.get_elements()) {
            if (element.name != "vertex")
                continue;
            for (const auto& property : element.properties) {
                for (size_t i = 0; i < names.size(); ++i) {
                    if (property.name == names[i])
                        offsets[i] = found++;
                }
            }
            break;
        }
        if (found != names.size() || offsets == std::array<size_t, 3>{0, 1, 2} || data.isList)
            return;

        // Fixed-width copies let the compiler reorder a row without per-scalar memcpy calls.
        const auto reorder = [&]<typename T>() {
            auto* buffer = data.buffer.get();
            for (size_t i = 0; i < data.count; ++i) {
                if ((i % 4096) == 0)
                    throw_if_load_cancel_requested(options, "PLY property ordering cancelled");
                auto* destination = buffer + i * 3 * sizeof(T);
                std::array<T, 3> row;
                std::memcpy(row.data(), destination, sizeof(row));
                const std::array<T, 3> ordered{row[offsets[0]], row[offsets[1]], row[offsets[2]]};
                std::memcpy(destination, ordered.data(), sizeof(ordered));
            }
        };
        switch (tinyply::PropertyTable.at(data.t).stride) {
        case 1: reorder.template operator()<uint8_t>(); break;
        case 2: reorder.template operator()<uint16_t>(); break;
        case 4: reorder.template operator()<uint32_t>(); break;
        case 8: reorder.template operator()<uint64_t>(); break;
        }
    }

} // namespace lfs::io::detail
