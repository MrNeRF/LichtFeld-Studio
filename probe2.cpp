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
    int n = 0;
    for (int depth : {0, 12}) {
        for (bool ofstream_write : {true, false}) {
            for (const auto& method : methods) {
                if (GetEnvironmentVariableW(L"PROBE_NOPOSIX", nullptr, 0) && std::string(method.name) == "handle rename posix")
                    continue;
                auto dir = base / ("p" + std::to_string(n++));
                for (int i = 0; i < depth; ++i)
                    dir /= path(u8"unicode_\u6E2C\u8A66_\u00E0_component_01234567890123456789");
                std::error_code e;
                fs::create_directories(dir, e);
                const auto file = dir / "network.bin", renamed = dir / "renamed.bin";
                unsigned long write_error = 0;
                if (ofstream_write) {
                    std::ofstream out(fs::native_path(file), std::ios::binary);
                    out << "network";
                    write_error = out ? 0 : 1;
                } else {
                    HANDLE h = CreateFileW(fs::native_path(file).c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
                    write_error = h == INVALID_HANDLE_VALUE ? GetLastError() : 0;
                    DWORD w = 0;
                    if (h != INVALID_HANDLE_VALUE) {
                        WriteFile(h, "network", 7, &w, nullptr);
                        CloseHandle(h);
                    }
                }
                std::error_code size_error;
                const auto size = fs::file_size(file, size_error);
                HANDLE probe = CreateFileW(fs::native_path(file, true).c_str(), DELETE, 7, nullptr, OPEN_EXISTING, 0, nullptr);
                const auto delete_open = probe == INVALID_HANDLE_VALUE ? GetLastError() : 0;
                if (probe != INVALID_HANDLE_VALUE)
                    CloseHandle(probe);
                const auto result = method.run(file, renamed);
                std::printf("depth=%2d %-8s %-22s mkdir=%d write=%lu size=%llu/%d deleteOpen=%lu rename=%lu exists=%d\n", depth,
                            ofstream_write ? "ofstream" : "win32", method.name, e.value(), write_error,
                            static_cast<unsigned long long>(size), size_error.value(), delete_open, result,
                            static_cast<int>(fs::exists(renamed)));
            }
        }
    }
    return 0;
}
