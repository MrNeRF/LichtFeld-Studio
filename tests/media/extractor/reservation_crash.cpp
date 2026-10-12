// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Loaded only into a disposable test child on macOS/exFAT. Stop after the
// core's O_EXCL reservation, immediately before its replacing rename.
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

extern "C" int stopAtReservedPublication(const char* source, const char* target) {
    const auto* requested = std::getenv("LFS_MEDIA_TEST_RESERVED_FILENAME");
    const std::string_view path(target);
    const auto slash = path.find_last_of('/');
    const auto name = path.substr(slash == std::string_view::npos ? 0 : slash + 1);
    struct stat state{};
    if (requested && name == requested && ::lstat(target, &state) == 0 &&
        S_ISREG(state.st_mode) && state.st_size == 0)
        ::_exit(73);
    return ::rename(source, target);
}
// dyld's interpose section avoids a dependency on the non-public SDK header.
__attribute__((used, section("__DATA,__interpose"))) static const struct {
    const void* replacement;
    const void* original;
} reserved_publication_interpose = {reinterpret_cast<const void*>(&stopAtReservedPublication),
                                    reinterpret_cast<const void*>(&::rename)};
