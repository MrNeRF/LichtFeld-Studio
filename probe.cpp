#include "core/native_fs.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>

namespace fs = lfs::core::native_fs;
using std::filesystem::path;

static unsigned long code(bool ok) { return ok ? 0 : GetLastError(); }

static unsigned long rename_by_handle(const path& from, const path& to, bool posix) {
    HANDLE h = CreateFileW(from.c_str(), DELETE | SYNCHRONIZE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE)
        return 100000 + GetLastError();
    const auto& name = to.native();
    const auto bytes = sizeof(FILE_RENAME_INFO) + name.size() * sizeof(wchar_t);
    std::string storage(bytes, '\0');
    auto* info = reinterpret_cast<FILE_RENAME_INFO*>(storage.data());
    if (posix)
        info->Flags = 0x1 /*REPLACE_IF_EXISTS*/ | 0x2 /*POSIX_SEMANTICS*/;
    else
        info->ReplaceIfExists = TRUE;
    info->FileNameLength = static_cast<DWORD>(name.size() * sizeof(wchar_t));
    std::memcpy(info->FileName, name.data(), info->FileNameLength);
    const BOOL ok = SetFileInformationByHandle(h, posix ? static_cast<FILE_INFO_BY_HANDLE_CLASS>(22) /*FileRenameInfoEx*/ : FileRenameInfo,
                                               info, static_cast<DWORD>(bytes));
    const auto error = code(ok);
    CloseHandle(h);
    return error;
}

int wmain() {
    wchar_t buffer[512]{};
    GetEnvironmentVariableW(L"PROBE_ROOT", buffer, 512);
    const path base(buffer);
    std::printf("== root %ls\n", buffer);
    struct Method {
        const char* name;
        std::function<unsigned long(const path&, const path&)> run;
    };
    const Method methods[] = {
        {"MoveFileExW(replace)", [](const path& a, const path& b) { return code(MoveFileExW(fs::native_path(a).c_str(), fs::native_path(b).c_str(), MOVEFILE_REPLACE_EXISTING)); }},
        {"MoveFileExW(0)", [](const path& a, const path& b) { return code(MoveFileExW(fs::native_path(a).c_str(), fs::native_path(b).c_str(), 0)); }},
        {"std::rename(native)", [](const path& a, const path& b) { std::error_code e; std::filesystem::rename(fs::native_path(a), fs::native_path(b), e); return static_cast<unsigned long>(e.value()); }},
        {"native_fs::rename", [](const path& a, const path& b) { std::error_code e; fs::rename(a, b, e); return static_cast<unsigned long>(e.value()); }},
        {"handle rename", [](const path& a, const path& b) { return rename_by_handle(fs::native_path(a, true), fs::native_path(b, true), false); }},
        {"handle rename posix", [](const path& a, const path& b) { return rename_by_handle(fs::native_path(a, true), fs::native_path(b, true), true); }},
    };
    using Query = BOOLEAN(NTAPI*)();
    auto query = reinterpret_cast<Query>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlAreLongPathsEnabled"));
    std::printf("process long paths enabled: %d\n", query ? static_cast<int>(query()) : -1);
    int n = 0;
    for (int depth : {0, 3, 4, 5, 6, 7, 8, 10, 12}) {
        for (int m : {0, 4}) {
            const auto& method = methods[m];
            auto dir = base / ("p" + std::to_string(n++));
            for (int i = 0; i < depth; ++i)
                dir /= path(u8"unicode_\u6E2C\u8A66_\u00E0_component_01234567890123456789");
            std::error_code e;
            fs::create_directories(dir, e);
            const auto file = dir / "network.bin", renamed = dir / "renamed.bin";
            {
                std::ofstream out(fs::native_path(file), std::ios::binary);
                out << "network";
            }
            const auto result = method.run(file, renamed);
            std::printf("depth=%2d len=%3zu %-22s mkdir=%d rename=%lu exists=%d\n", depth, fs::native_path(file, true).native().size(),
                        method.name, e.value(), result, static_cast<int>(fs::exists(renamed)));
        }
    }
    return 0;
}
