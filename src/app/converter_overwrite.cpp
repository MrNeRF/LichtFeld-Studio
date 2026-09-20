/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/converter_overwrite.hpp"

#include "core/path_utils.hpp"

#include <cctype>
#include <istream>
#include <ostream>
#include <string>

namespace lfs::app {

    OverwriteChoice ask_overwrite(
        const std::filesystem::path& path,
        std::istream& input,
        std::ostream& output) {
        output << "File exists: " << lfs::core::path_to_utf8(path.filename())
               << "\nOverwrite? [y]es / [n]o / [a]ll: ";

        std::string response;
        if (!std::getline(input, response) || response.empty()) {
            return OverwriteChoice::NO;
        }
        const char choice = static_cast<char>(
            std::tolower(static_cast<unsigned char>(response.front())));
        if (choice == 'y') {
            return OverwriteChoice::YES;
        }
        if (choice == 'a') {
            return OverwriteChoice::ALL;
        }
        return OverwriteChoice::NO;
    }

} // namespace lfs::app
