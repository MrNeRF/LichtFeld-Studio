#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

if (( $# > 1 )); then
    echo "Usage: $0 [builddir]" >&2
    exit 2
fi

builddir="${1:-build}"
deps_file="$(mktemp)"
trap 'rm -f "$deps_file"' EXIT

if ! ninja -C "$builddir" -t deps >"$deps_file"; then
    echo "FAIL: ninja dependency database could not be read: $builddir" >&2
    exit 1
fi

read -r cpp_count cu_count zero_count < <(
    awk '
        /\.cpp\.o: #deps / { cpp++ }
        /\.cu\.o: #deps /  { cu++ }
        /\.(cpp|cu)\.o: #deps 0([, ]|$)/ { zero++ }
        END { printf "%d %d %d\n", cpp + 0, cu + 0, zero + 0 }
    ' "$deps_file"
)

if (( zero_count > 0 )); then
    echo "FAIL: $zero_count object(s) have #deps 0 in $builddir" >&2
    awk '/\.(cpp|cu)\.o: #deps 0([, ]|$)/ { sub(/: #deps.*/, ""); print }' "$deps_file" >&2
    exit 1
fi

printf 'PASS: build=%s objects=%d cpp=%d cu=%d zero=0\n' \
    "$builddir" "$((cpp_count + cu_count))" "$cpp_count" "$cu_count"
