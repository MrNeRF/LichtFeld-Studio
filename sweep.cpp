#include "core/native_fs.hpp"
#include <cstdio>
#include <fstream>
#include <string>

namespace fs = lfs::core::native_fs;
using std::filesystem::path;

int wmain() {
    wchar_t buffer[512]{};
    GetEnvironmentVariableW(L"PROBE_ROOT", buffer, 512);
    for (int pad = 0; pad <= 40; pad += 2) {
        auto dir = path(buffer) / (L"w" + std::wstring(static_cast<size_t>(pad), L'x'));
        for (int i = 0; i < 12; ++i)
            dir /= path(u8"unicode_測試_à_component_01234567890123456789");
        std::error_code e;
        fs::create_directories(dir, e);
        const auto file = dir / "network.bin", renamed = dir / "renamed.bin";
        {
            std::ofstream out(fs::native_path(file), std::ios::binary);
            out << "network";
        }
        std::error_code size_error;
        (void)fs::file_size(file, size_error);
        const auto source = fs::native_path(file, true), target = fs::native_path(renamed, true);
        const auto result = MoveFileExW(source.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING) ? 0 : GetLastError();
        std::printf("len=%zu bytes=%zu mkdir=%d size=%d rename=%lu\n", source.native().size(), source.native().size() * 2, e.value(),
                    size_error.value(), result);
    }
    return 0;
}
