#include "core/native_fs.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

namespace fs = lfs::core::native_fs;
using std::filesystem::path;

static unsigned long move(const path& a, const path& b) {
    return MoveFileExW(fs::native_path(a, true).c_str(), fs::native_path(b, true).c_str(), MOVEFILE_REPLACE_EXISTING) ? 0 : GetLastError();
}

int wmain() {
    wchar_t buffer[512]{};
    GetEnvironmentVariableW(L"PROBE_ROOT", buffer, 512);
    const path base(buffer);
    std::printf("== root %ls\n", buffer);
    const char* names[] = {"write+move", "write+file_size+move", "write+attrs+move", "write+deleteopen+move",
                           "write+readopen+move", "write+file_size+sleep3s+move", "write+file_size+retry10s"};
    int n = 0;
    for (int depth : {0, 6, 12}) {
        for (int s = 0; s < 7; ++s) {
            auto dir = base / ("s" + std::to_string(n++));
            for (int i = 0; i < depth; ++i)
                dir /= path(u8"unicode_測試_à_component_01234567890123456789");
            std::error_code e;
            fs::create_directories(dir, e);
            const auto file = dir / "network.bin", renamed = dir / "renamed.bin";
            {
                std::ofstream out(fs::native_path(file), std::ios::binary);
                out << "network";
            }
            std::error_code size_error;
            if (s == 1 || s == 5 || s == 6)
                (void)fs::file_size(file, size_error);
            if (s == 2) {
                WIN32_FILE_ATTRIBUTE_DATA data{};
                GetFileAttributesExW(fs::native_path(file, true).c_str(), GetFileExInfoStandard, &data);
            }
            if (s == 3 || s == 4) {
                HANDLE h = CreateFileW(fs::native_path(file, true).c_str(), s == 3 ? DELETE : FILE_READ_ATTRIBUTES, 7, nullptr,
                                       OPEN_EXISTING, 0, nullptr);
                if (h != INVALID_HANDLE_VALUE)
                    CloseHandle(h);
            }
            if (s == 5)
                Sleep(3000);
            const auto start = std::chrono::steady_clock::now();
            unsigned long result = move(file, renamed);
            int tries = 1;
            for (; s == 6 && result != 0 && tries < 100; ++tries) {
                Sleep(100);
                result = move(file, renamed);
            }
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            std::printf("depth=%2d %-30s rename=%lu tries=%d ms=%lld\n", depth, names[s], result, tries, static_cast<long long>(ms));
        }
    }
    return 0;
}
