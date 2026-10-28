/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <archive.h>
#include <archive_entry.h>

namespace lfs::io {

    // Archive member names are UTF-8. Windows asks libarchive for UTF-8 names
    // explicitly; on POSIX the native spelling already is the UTF-8 byte string,
    // and libarchive may be built without the charset conversion.
    [[nodiscard]] inline bool use_utf8_zip_member_names([[maybe_unused]] archive* const writer) {
#ifdef _WIN32
        return archive_write_set_format_option(writer, "zip", "hdrcharset", "UTF-8") == ARCHIVE_OK;
#else
        return true;
#endif
    }

    inline void set_archive_member_name(archive_entry* const entry, const char* const name) {
#ifdef _WIN32
        archive_entry_set_pathname_utf8(entry, name);
#else
        archive_entry_set_pathname(entry, name);
#endif
    }

    [[nodiscard]] inline const char* archive_member_name(archive_entry* const entry) {
#ifdef _WIN32
        return archive_entry_pathname_utf8(entry);
#else
        return archive_entry_pathname(entry);
#endif
    }

} // namespace lfs::io
