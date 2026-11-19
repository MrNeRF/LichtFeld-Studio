/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <ostream>

namespace nvimgcodec {

struct LibraryVersion {
    int major_ver = -1;
    int minor_ver = -1;
    int patch_ver = -1;
    bool valid = false;

    constexpr LibraryVersion(int _major, int _minor, int _patch)
        : major_ver(_major), minor_ver(_minor), patch_ver(_patch), valid(true) {}
    constexpr LibraryVersion() : valid(false) {}

    int flat_version() const {
        if (!valid) {
            return -1;
        }
        return (major_ver * 1000000 + minor_ver * 1000 + patch_ver);
    }

    friend std::ostream& operator<<(std::ostream& os, const LibraryVersion& v) {
        if (!v.valid) {
            os << "invalid";
        } else {
            os << v.major_ver << "." << v.minor_ver << "." << v.patch_ver;
        }
        return os;
    }

    explicit operator bool() const {
        return valid;
    }

    bool operator==(const LibraryVersion& other) const {
        return flat_version() == other.flat_version();
    }
    bool operator!=(const LibraryVersion& other) const {
        return !(*this == other);
    }
    bool operator<(const LibraryVersion& other) const {
        return flat_version() < other.flat_version();
    }
    bool operator>(const LibraryVersion& other) const {
        return other < *this;
    }
    bool operator<=(const LibraryVersion& other) const {
        return !(other < *this);
    }
    bool operator>=(const LibraryVersion& other) const {
        return !(*this < other);
    }
};

} // namespace nvimgcodec
