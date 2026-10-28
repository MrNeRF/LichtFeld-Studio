/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/path_utils.hpp"
#include <RmlUi/Core/FileInterface.h>
#include <cstdio>

namespace lfs::vis::gui {
    // RmlUi's default file interface uses narrow fopen on Windows. Resource
    // names stay UTF-8/logical; conversion occurs only while opening the file.
    class RmlFileInterface final : public Rml::FileInterface {
    public:
        Rml::FileHandle Open(const Rml::String& name) override {
            return reinterpret_cast<Rml::FileHandle>(lfs::core::open_file(lfs::core::utf8_to_path(name), "rb"));
        }
        void Close(Rml::FileHandle handle) override { std::fclose(file(handle)); }
        size_t Read(void* buffer, size_t size, Rml::FileHandle handle) override {
            return std::fread(buffer, 1, size, file(handle));
        }
        bool Seek(Rml::FileHandle handle, long offset, int origin) override {
#ifdef _WIN32
            return _fseeki64(file(handle), offset, origin) == 0;
#else
            return fseeko(file(handle), offset, origin) == 0;
#endif
        }
        size_t Tell(Rml::FileHandle handle) override {
#ifdef _WIN32
            const auto offset = _ftelli64(file(handle));
#else
            const auto offset = ftello(file(handle));
#endif
            return offset < 0 ? 0 : static_cast<size_t>(offset);
        }

    private:
        static std::FILE* file(Rml::FileHandle handle) { return reinterpret_cast<std::FILE*>(handle); }
    };
} // namespace lfs::vis::gui
