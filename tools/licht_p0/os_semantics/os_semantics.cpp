#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr int kLockBusyExit = 73;
constexpr std::chrono::seconds kChildTimeout{5};
fs::path executable_path;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void check(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

#ifdef _WIN32
[[noreturn]] void fail_system(const std::string& operation, const DWORD error) {
    fail(operation + " failed with Windows error " + std::to_string(error));
}

[[noreturn]] void fail_last_error(const std::string& operation) {
    fail_system(operation, GetLastError());
}
#else
[[noreturn]] void fail_errno(const std::string& operation, const int error) {
    fail(operation + " failed: " + std::string(std::strerror(error)) +
         " (errno=" + std::to_string(error) + ")");
}

[[noreturn]] void fail_last_error(const std::string& operation) {
    fail_errno(operation, errno);
}
#endif

class NativeFile {
public:
#ifdef _WIN32
    using Handle = HANDLE;
    static Handle invalid_value() noexcept { return INVALID_HANDLE_VALUE; }
#else
    using Handle = int;
    static constexpr Handle invalid_value() noexcept { return -1; }
#endif

    NativeFile() = default;
    explicit NativeFile(const Handle handle) : handle_(handle) {}
    NativeFile(const NativeFile&) = delete;
    NativeFile& operator=(const NativeFile&) = delete;

    NativeFile(NativeFile&& other) noexcept
        : handle_(std::exchange(other.handle_, invalid_value())) {}

    NativeFile& operator=(NativeFile&& other) noexcept {
        if (this != &other) {
            close_noexcept();
            handle_ = std::exchange(other.handle_, invalid_value());
        }
        return *this;
    }

    ~NativeFile() { close_noexcept(); }

    [[nodiscard]] Handle get() const { return handle_; }
    [[nodiscard]] bool valid() const { return handle_ != invalid_value(); }

    void close_checked() {
        if (!valid()) {
            return;
        }
#ifdef _WIN32
        const Handle closing = std::exchange(handle_, invalid_value());
        if (!CloseHandle(closing)) {
            fail_last_error("CloseHandle(file)");
        }
#else
        const Handle closing = std::exchange(handle_, invalid_value());
        if (::close(closing) != 0) {
            fail_last_error("close");
        }
#endif
    }

private:
    void close_noexcept() noexcept {
        if (!valid()) {
            return;
        }
#ifdef _WIN32
        const Handle closing = std::exchange(handle_, invalid_value());
        if (!CloseHandle(closing)) {
            std::fprintf(stderr, "FATAL CloseHandle(file) failed with Windows error %lu\n",
                         static_cast<unsigned long>(GetLastError()));
            std::abort();
        }
#else
        const Handle closing = std::exchange(handle_, invalid_value());
        if (::close(closing) != 0) {
            const int error = errno;
            std::fprintf(stderr, "FATAL close failed: %s (errno=%d)\n",
                         std::strerror(error), error);
            std::abort();
        }
#endif
    }

    Handle handle_ = invalid_value();
};

#ifdef _WIN32
constexpr DWORD kCompatibleShare =
    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

NativeFile open_windows_file(const fs::path& path,
                             const DWORD access,
                             const DWORD share,
                             const DWORD creation,
                             const bool positional) {
    const DWORD flags = FILE_ATTRIBUTE_NORMAL |
                        (positional ? FILE_FLAG_OVERLAPPED : 0);
    const HANDLE handle = CreateFileW(path.c_str(), access, share, nullptr, creation,
                                      flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        fail_last_error("CreateFileW(" + path.string() + ")");
    }
    return NativeFile(handle);
}
#endif

NativeFile create_new_read_write(const fs::path& path,
                                 const bool positional = false) {
#ifdef _WIN32
    return open_windows_file(path, GENERIC_READ | GENERIC_WRITE, kCompatibleShare,
                             CREATE_NEW, positional);
#else
    (void)positional;
    const int descriptor =
        ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        fail_last_error("open(O_CREAT|O_EXCL, " + path.string() + ")");
    }
    return NativeFile(descriptor);
#endif
}

NativeFile create_truncated_read_write(const fs::path& path,
                                       const bool positional = false) {
#ifdef _WIN32
    return open_windows_file(path, GENERIC_READ | GENERIC_WRITE, kCompatibleShare,
                             CREATE_ALWAYS, positional);
#else
    (void)positional;
    const int descriptor =
        ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        fail_last_error("open(O_TRUNC, " + path.string() + ")");
    }
    return NativeFile(descriptor);
#endif
}

NativeFile open_read_write(const fs::path& path,
                           const bool positional = false) {
#ifdef _WIN32
    return open_windows_file(path, GENERIC_READ | GENERIC_WRITE, kCompatibleShare,
                             OPEN_EXISTING, positional);
#else
    (void)positional;
    const int descriptor = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (descriptor < 0) {
        fail_last_error("open(O_RDWR, " + path.string() + ")");
    }
    return NativeFile(descriptor);
#endif
}

NativeFile open_read_only(const fs::path& path,
                          const bool positional = false) {
#ifdef _WIN32
    return open_windows_file(path, GENERIC_READ, kCompatibleShare, OPEN_EXISTING,
                             positional);
#else
    (void)positional;
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        fail_last_error("open(O_RDONLY, " + path.string() + ")");
    }
    return NativeFile(descriptor);
#endif
}

NativeFile open_append_only(const fs::path& path) {
#ifdef _WIN32
    return open_windows_file(path, FILE_APPEND_DATA | FILE_READ_ATTRIBUTES,
                             kCompatibleShare, OPEN_EXISTING, false);
#else
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CLOEXEC);
    if (descriptor < 0) {
        fail_last_error("open(O_APPEND, " + path.string() + ")");
    }
    return NativeFile(descriptor);
#endif
}

void write_all(NativeFile& file, const void* data, const std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::size_t written_total = 0;
    while (written_total < size) {
#ifdef _WIN32
        const std::size_t remaining = size - written_total;
        const DWORD request = static_cast<DWORD>(std::min<std::size_t>(
            remaining, std::numeric_limits<DWORD>::max()));
        DWORD written = 0;
        if (!WriteFile(file.get(), bytes + written_total, request, &written, nullptr)) {
            fail_last_error("WriteFile");
        }
        check(written != 0, "WriteFile made no progress");
        written_total += written;
#else
        const ssize_t written =
            ::write(file.get(), bytes + written_total, size - written_total);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail_last_error("write");
        }
        check(written != 0, "write made no progress");
        written_total += static_cast<std::size_t>(written);
#endif
    }
}

void write_single(NativeFile& file, const void* data, const std::size_t size) {
#ifdef _WIN32
    check(size <= std::numeric_limits<DWORD>::max(), "WriteFile request is too large");
    DWORD written = 0;
    if (!WriteFile(file.get(), data, static_cast<DWORD>(size), &written, nullptr)) {
        fail_last_error("WriteFile(single append)");
    }
    check(written == size, "WriteFile(single append) returned a short write");
#else
    const ssize_t written = ::write(file.get(), data, size);
    if (written < 0) {
        fail_last_error("write(single append)");
    }
    check(static_cast<std::size_t>(written) == size,
          "write(single append) returned a short write");
#endif
}

void read_exact_at(NativeFile& file,
                   void* data,
                   const std::size_t size,
                   const std::uint64_t offset) {
#ifdef _WIN32
    check(size <= std::numeric_limits<DWORD>::max(), "ReadFile request is too large");
    OVERLAPPED operation{};
    operation.Offset = static_cast<DWORD>(offset & 0xffffffffULL);
    operation.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD read = 0;
    const BOOL started =
        ReadFile(file.get(), data, static_cast<DWORD>(size), &read, &operation);
    if (!started) {
        const DWORD error = GetLastError();
        if (error != ERROR_IO_PENDING) {
            fail_system("ReadFile(OVERLAPPED)", error);
        }
        if (!GetOverlappedResult(file.get(), &operation, &read, TRUE)) {
            fail_last_error("GetOverlappedResult(ReadFile)");
        }
    }
    check(read == size, "ReadFile(OVERLAPPED) returned a short read");
#else
    const ssize_t read = ::pread(file.get(), data, size, static_cast<off_t>(offset));
    if (read < 0) {
        fail_last_error("pread");
    }
    check(static_cast<std::size_t>(read) == size, "pread returned a short read");
#endif
}

void write_exact_at_single(NativeFile& file,
                           const void* data,
                           const std::size_t size,
                           const std::uint64_t offset) {
#ifdef _WIN32
    check(size <= std::numeric_limits<DWORD>::max(), "WriteFile request is too large");
    OVERLAPPED operation{};
    operation.Offset = static_cast<DWORD>(offset & 0xffffffffULL);
    operation.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD written = 0;
    const BOOL started =
        WriteFile(file.get(), data, static_cast<DWORD>(size), &written, &operation);
    if (!started) {
        const DWORD error = GetLastError();
        if (error != ERROR_IO_PENDING) {
            fail_system("WriteFile(OVERLAPPED)", error);
        }
        if (!GetOverlappedResult(file.get(), &operation, &written, TRUE)) {
            fail_last_error("GetOverlappedResult(WriteFile)");
        }
    }
    check(written == size, "WriteFile(OVERLAPPED) returned a short write");
#else
    const ssize_t written =
        ::pwrite(file.get(), data, size, static_cast<off_t>(offset));
    if (written < 0) {
        fail_last_error("pwrite");
    }
    check(static_cast<std::size_t>(written) == size,
          "pwrite returned a short write");
#endif
}

void sync_file(NativeFile& file) {
#ifdef _WIN32
    if (!FlushFileBuffers(file.get())) {
        fail_last_error("FlushFileBuffers");
    }
#else
    while (::fdatasync(file.get()) != 0) {
        if (errno == EINTR) {
            continue;
        }
        fail_last_error("fdatasync");
    }
#endif
}

std::uint64_t file_size(NativeFile& file) {
#ifdef _WIN32
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file.get(), &size)) {
        fail_last_error("GetFileSizeEx");
    }
    check(size.QuadPart >= 0, "GetFileSizeEx returned a negative size");
    return static_cast<std::uint64_t>(size.QuadPart);
#else
    struct stat status {};
    if (::fstat(file.get(), &status) != 0) {
        fail_last_error("fstat");
    }
    check(status.st_size >= 0, "fstat returned a negative size");
    return static_cast<std::uint64_t>(status.st_size);
#endif
}

void truncate_file(NativeFile& file, const std::uint64_t size) {
#ifdef _WIN32
    check(size <= static_cast<std::uint64_t>(std::numeric_limits<LONGLONG>::max()),
          "truncate size exceeds LONGLONG");
    LARGE_INTEGER end{};
    end.QuadPart = static_cast<LONGLONG>(size);
    if (!SetFilePointerEx(file.get(), end, nullptr, FILE_BEGIN)) {
        fail_last_error("SetFilePointerEx");
    }
    if (!SetEndOfFile(file.get())) {
        fail_last_error("SetEndOfFile");
    }
#else
    if (::ftruncate(file.get(), static_cast<off_t>(size)) != 0) {
        fail_last_error("ftruncate");
    }
#endif
}

void write_file(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    NativeFile file = create_truncated_read_write(path);
    if (!bytes.empty()) {
        write_all(file, bytes.data(), bytes.size());
    }
    sync_file(file);
    file.close_checked();
}

void write_exclusive_text(const fs::path& path, const std::string& text) {
    NativeFile file = create_new_read_write(path);
    write_all(file, text.data(), text.size());
    sync_file(file);
    file.close_checked();
}

std::vector<std::uint8_t> read_file_bytes(const fs::path& path,
                                          const std::size_t maximum_size) {
    NativeFile file = open_read_only(path, true);
    const std::uint64_t raw_size = file_size(file);
    check(raw_size <= maximum_size, "file exceeds expected maximum size: " + path.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(raw_size));
    if (!bytes.empty()) {
        read_exact_at(file, bytes.data(), bytes.size(), 0);
    }
    file.close_checked();
    return bytes;
}

std::string read_text(const fs::path& path, const std::size_t maximum_size = 4096) {
    const std::vector<std::uint8_t> bytes = read_file_bytes(path, maximum_size);
    return std::string(bytes.begin(), bytes.end());
}

bool path_exists(const fs::path& path) {
    std::error_code error;
    const bool exists = fs::exists(path, error);
    if (error) {
        fail("filesystem::exists(" + path.string() + ") failed: " + error.message());
    }
    return exists;
}

void ensure_directory(const fs::path& path) {
    std::error_code error;
    fs::create_directories(path, error);
    if (error) {
        fail("filesystem::create_directories(" + path.string() + ") failed: " +
             error.message());
    }
    const bool is_directory = fs::is_directory(path, error);
    if (error) {
        fail("filesystem::is_directory(" + path.string() + ") failed: " +
             error.message());
    }
    check(is_directory, "path is not a directory: " + path.string());
}

std::uint64_t process_id() {
#ifdef _WIN32
    return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(::getpid());
#endif
}

fs::path make_run_directory(const fs::path& base) {
    ensure_directory(base);
    for (std::uint32_t suffix = 0; suffix < 1000; ++suffix) {
        std::string name = "run_" + std::to_string(process_id());
        if (suffix != 0) {
            name += "_" + std::to_string(suffix);
        }
        const fs::path candidate = base / name;
        std::error_code error;
        const bool created = fs::create_directory(candidate, error);
        if (error) {
            fail("filesystem::create_directory(" + candidate.string() + ") failed: " +
                 error.message());
        }
        if (created) {
            return candidate;
        }
    }
    fail("could not allocate a unique run directory under " + base.string());
}

fs::path make_scenario_directory(const fs::path& run_directory,
                                 const std::string& name) {
    const fs::path directory = run_directory / name;
    std::error_code error;
    const bool created = fs::create_directory(directory, error);
    if (error) {
        fail("filesystem::create_directory(" + directory.string() + ") failed: " +
             error.message());
    }
    check(created, "scenario directory already exists: " + directory.string());
    return directory;
}

std::vector<std::uint8_t> patterned_bytes(const std::size_t size,
                                          const std::uint8_t seed) {
    std::vector<std::uint8_t> bytes(size);
    for (std::size_t index = 0; index < size; ++index) {
        bytes[index] = static_cast<std::uint8_t>(
            (static_cast<std::uint32_t>(seed) + index * 131U + index / 17U) % 251U);
    }
    return bytes;
}

bool all_bytes_are(const std::vector<std::uint8_t>& bytes, const std::uint8_t value) {
    return std::all_of(bytes.begin(), bytes.end(),
                       [value](const std::uint8_t byte) { return byte == value; });
}

std::uint64_t parse_unsigned(const std::string& text, const std::string& label) {
    std::size_t parsed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &parsed, 10);
    } catch (const std::exception&) {
        fail("invalid " + label + ": " + text);
    }
    check(parsed == text.size(), "invalid " + label + ": " + text);
    return static_cast<std::uint64_t>(value);
}

class HeldFileLock {
public:
    enum class Api { none, ofd, flock, lock_file_ex };

    HeldFileLock() = default;
    HeldFileLock(const HeldFileLock&) = delete;
    HeldFileLock& operator=(const HeldFileLock&) = delete;
    ~HeldFileLock() { release_noexcept(); }

    [[nodiscard]] std::string api_name() const {
        switch (api_) {
        case Api::ofd:
            return "fcntl(F_OFD_SETLK)";
        case Api::flock:
            return "flock(LOCK_EX|LOCK_NB)";
        case Api::lock_file_ex:
            return "LockFileEx(exclusive|fail-immediately)";
        case Api::none:
            break;
        }
        return "none";
    }

private:
    friend bool try_acquire_lock(NativeFile&, HeldFileLock&);

    void release_noexcept() noexcept {
        if (file_ == nullptr) {
            return;
        }
#ifdef _WIN32
        OVERLAPPED operation{};
        if (!UnlockFileEx(file_->get(), 0, 1, 0, &operation)) {
            std::fprintf(stderr, "FATAL UnlockFileEx failed with Windows error %lu\n",
                         static_cast<unsigned long>(GetLastError()));
            std::abort();
        }
#else
        if (api_ == Api::ofd) {
            struct flock lock {};
            lock.l_type = F_UNLCK;
            lock.l_whence = SEEK_SET;
            lock.l_start = 0;
            lock.l_len = 1;
#ifdef F_OFD_SETLK
            if (::fcntl(file_->get(), F_OFD_SETLK, &lock) != 0) {
                const int error = errno;
                std::fprintf(stderr, "FATAL fcntl(F_OFD_SETLK unlock) failed: %s (errno=%d)\n",
                             std::strerror(error), error);
                std::abort();
            }
#else
            std::abort();
#endif
        } else if (api_ == Api::flock) {
            if (::flock(file_->get(), LOCK_UN) != 0) {
                const int error = errno;
                std::fprintf(stderr, "FATAL flock(LOCK_UN) failed: %s (errno=%d)\n",
                             std::strerror(error), error);
                std::abort();
            }
        }
#endif
        file_ = nullptr;
        api_ = Api::none;
    }

    NativeFile* file_ = nullptr;
    Api api_ = Api::none;
};

bool try_acquire_lock(NativeFile& file, HeldFileLock& held) {
    check(held.file_ == nullptr, "lock object is already holding a lock");
#ifdef _WIN32
    OVERLAPPED operation{};
    const DWORD flags = LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY;
    if (!LockFileEx(file.get(), flags, 0, 1, 0, &operation)) {
        const DWORD error = GetLastError();
        if (error == ERROR_LOCK_VIOLATION) {
            return false;
        }
        fail_system("LockFileEx", error);
    }
    held.file_ = &file;
    held.api_ = HeldFileLock::Api::lock_file_ex;
    return true;
#else
#ifdef F_OFD_SETLK
    struct flock lock {};
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 1;
    if (::fcntl(file.get(), F_OFD_SETLK, &lock) == 0) {
        held.file_ = &file;
        held.api_ = HeldFileLock::Api::ofd;
        return true;
    }
    const int ofd_error = errno;
    if (ofd_error == EACCES || ofd_error == EAGAIN) {
        return false;
    }
    if (ofd_error != EINVAL && ofd_error != ENOSYS) {
        fail_errno("fcntl(F_OFD_SETLK)", ofd_error);
    }
#endif
    if (::flock(file.get(), LOCK_EX | LOCK_NB) == 0) {
        held.file_ = &file;
        held.api_ = HeldFileLock::Api::flock;
        return true;
    }
    const int flock_error = errno;
    if (flock_error == EWOULDBLOCK || flock_error == EAGAIN) {
        return false;
    }
    fail_errno("flock(LOCK_EX|LOCK_NB)", flock_error);
#endif
}

class ChildProcess {
public:
#ifdef _WIN32
    explicit ChildProcess(const HANDLE process) : process_(process) {}
#else
    explicit ChildProcess(const pid_t process) : process_(process) {}
#endif
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;

    ChildProcess(ChildProcess&& other) noexcept
        : process_(std::exchange(other.process_, invalid_process())) {}

    ChildProcess& operator=(ChildProcess&& other) noexcept {
        if (this != &other) {
            cleanup_noexcept();
            process_ = std::exchange(other.process_, invalid_process());
        }
        return *this;
    }

    ~ChildProcess() { cleanup_noexcept(); }

    int wait(const std::chrono::milliseconds timeout =
                 std::chrono::duration_cast<std::chrono::milliseconds>(kChildTimeout)) {
#ifdef _WIN32
        check(process_ != invalid_process(), "child process was already consumed");
        const DWORD wait_ms = static_cast<DWORD>(timeout.count());
        const DWORD wait_result = WaitForSingleObject(process_, wait_ms);
        if (wait_result == WAIT_TIMEOUT) {
            if (!TerminateProcess(process_, 124)) {
                fail_last_error("TerminateProcess(timed-out child)");
            }
            if (WaitForSingleObject(process_, INFINITE) != WAIT_OBJECT_0) {
                fail_last_error("WaitForSingleObject(timed-out child)");
            }
            close_process_checked();
            fail("child process timed out");
        }
        if (wait_result != WAIT_OBJECT_0) {
            fail_last_error("WaitForSingleObject(child)");
        }
        DWORD exit_code = 0;
        if (!GetExitCodeProcess(process_, &exit_code)) {
            fail_last_error("GetExitCodeProcess");
        }
        close_process_checked();
        return static_cast<int>(exit_code);
#else
        check(process_ != invalid_process(), "child process was already consumed");
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true) {
            int status = 0;
            const pid_t waited = ::waitpid(process_, &status, WNOHANG);
            if (waited == process_) {
                process_ = invalid_process();
                if (WIFEXITED(status)) {
                    return WEXITSTATUS(status);
                }
                if (WIFSIGNALED(status)) {
                    return 128 + WTERMSIG(status);
                }
                fail("child exited with an unrecognized wait status");
            }
            if (waited < 0 && errno != EINTR) {
                fail_last_error("waitpid(WNOHANG)");
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                if (::kill(process_, SIGKILL) != 0) {
                    fail_last_error("kill(timed-out child, SIGKILL)");
                }
                wait_blocking_checked();
                fail("child process timed out");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
#endif
    }

    void kill_for_test() {
#ifdef _WIN32
        check(process_ != invalid_process(), "child process was already consumed");
        if (!TerminateProcess(process_, 137)) {
            fail_last_error("TerminateProcess");
        }
        if (WaitForSingleObject(process_, INFINITE) != WAIT_OBJECT_0) {
            fail_last_error("WaitForSingleObject(terminated child)");
        }
        DWORD exit_code = 0;
        if (!GetExitCodeProcess(process_, &exit_code)) {
            fail_last_error("GetExitCodeProcess(terminated child)");
        }
        check(exit_code == 137, "terminated child returned an unexpected exit code");
        close_process_checked();
#else
        check(process_ != invalid_process(), "child process was already consumed");
        if (::kill(process_, SIGKILL) != 0) {
            fail_last_error("kill(SIGKILL)");
        }
        int status = 0;
        while (::waitpid(process_, &status, 0) < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail_last_error("waitpid(killed child)");
        }
        process_ = invalid_process();
        check(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL,
              "child was not terminated by SIGKILL");
#endif
    }

private:
#ifdef _WIN32
    static HANDLE invalid_process() { return nullptr; }

    void close_process_checked() {
        const HANDLE closing = std::exchange(process_, invalid_process());
        if (!CloseHandle(closing)) {
            fail_last_error("CloseHandle(process)");
        }
    }
#else
    static pid_t invalid_process() { return -1; }

    void wait_blocking_checked() {
        int status = 0;
        while (::waitpid(process_, &status, 0) < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail_last_error("waitpid");
        }
        process_ = invalid_process();
    }
#endif

    void cleanup_noexcept() noexcept {
        if (process_ == invalid_process()) {
            return;
        }
#ifdef _WIN32
        const DWORD state = WaitForSingleObject(process_, 0);
        if (state == WAIT_TIMEOUT && !TerminateProcess(process_, 125)) {
            std::fprintf(stderr, "FATAL TerminateProcess(cleanup) failed with Windows error %lu\n",
                         static_cast<unsigned long>(GetLastError()));
            std::abort();
        }
        if (state == WAIT_FAILED || WaitForSingleObject(process_, INFINITE) != WAIT_OBJECT_0) {
            std::fprintf(stderr, "FATAL WaitForSingleObject(cleanup) failed with Windows error %lu\n",
                         static_cast<unsigned long>(GetLastError()));
            std::abort();
        }
        const HANDLE closing = std::exchange(process_, invalid_process());
        if (!CloseHandle(closing)) {
            std::fprintf(stderr, "FATAL CloseHandle(process cleanup) failed with Windows error %lu\n",
                         static_cast<unsigned long>(GetLastError()));
            std::abort();
        }
#else
        if (::kill(process_, SIGKILL) != 0 && errno != ESRCH) {
            const int error = errno;
            std::fprintf(stderr, "FATAL kill(cleanup) failed: %s (errno=%d)\n",
                         std::strerror(error), error);
            std::abort();
        }
        int status = 0;
        while (::waitpid(process_, &status, 0) < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == ECHILD) {
                break;
            }
            const int error = errno;
            std::fprintf(stderr, "FATAL waitpid(cleanup) failed: %s (errno=%d)\n",
                         std::strerror(error), error);
            std::abort();
        }
        process_ = invalid_process();
#endif
    }

#ifdef _WIN32
    HANDLE process_ = invalid_process();
#else
    pid_t process_ = invalid_process();
#endif
};

#ifdef _WIN32
std::wstring quote_windows_argument(const std::wstring& argument) {
    if (argument.find_first_of(L" \t\"") == std::wstring::npos) {
        return argument;
    }
    std::wstring quoted(1, L'\"');
    std::size_t backslashes = 0;
    for (const wchar_t character : argument) {
        if (character == L'\\') {
            ++backslashes;
            continue;
        }
        if (character == L'\"') {
            quoted.append(backslashes * 2 + 1, L'\\');
            quoted.push_back(L'\"');
            backslashes = 0;
            continue;
        }
        quoted.append(backslashes, L'\\');
        backslashes = 0;
        quoted.push_back(character);
    }
    quoted.append(backslashes * 2, L'\\');
    quoted.push_back(L'\"');
    return quoted;
}
#endif

ChildProcess spawn_self(const std::vector<fs::path>& arguments) {
#ifdef _WIN32
    std::wstring command_line = quote_windows_argument(executable_path.wstring());
    for (const fs::path& argument : arguments) {
        command_line.push_back(L' ');
        command_line += quote_windows_argument(argument.wstring());
    }
    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable_path.c_str(), mutable_command.data(), nullptr,
                        nullptr, FALSE, 0, nullptr, nullptr, &startup, &process)) {
        fail_last_error("CreateProcessW");
    }
    if (!CloseHandle(process.hThread)) {
        const DWORD error = GetLastError();
        TerminateProcess(process.hProcess, 126);
        WaitForSingleObject(process.hProcess, INFINITE);
        CloseHandle(process.hProcess);
        fail_system("CloseHandle(child thread)", error);
    }
    return ChildProcess(process.hProcess);
#else
    const pid_t child = ::fork();
    if (child < 0) {
        fail_last_error("fork");
    }
    if (child == 0) {
        std::vector<std::string> storage;
        storage.reserve(arguments.size() + 1);
        storage.push_back(executable_path.string());
        for (const fs::path& argument : arguments) {
            storage.push_back(argument.string());
        }
        std::vector<char*> argv;
        argv.reserve(storage.size() + 1);
        for (std::string& argument : storage) {
            argv.push_back(argument.data());
        }
        argv.push_back(nullptr);
        ::execv(executable_path.c_str(), argv.data());
        _exit(127);
    }
    return ChildProcess(child);
#endif
}

void wait_for_path(const fs::path& path) {
    const auto deadline = std::chrono::steady_clock::now() + kChildTimeout;
    while (!path_exists(path)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fail("timed out waiting for child signal: " + path.string());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

class ReadOnlyMapping {
public:
    ReadOnlyMapping(NativeFile& file, const std::size_t size) : size_(size) {
#ifdef _WIN32
        mapping_ = CreateFileMappingW(file.get(), nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_ == nullptr) {
            fail_last_error("CreateFileMappingW");
        }
        data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, size);
        if (data_ == nullptr) {
            const DWORD error = GetLastError();
            CloseHandle(mapping_);
            mapping_ = nullptr;
            fail_system("MapViewOfFile", error);
        }
#else
        data_ = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, file.get(), 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            fail_last_error("mmap");
        }
#endif
    }

    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;
    ~ReadOnlyMapping() { close_noexcept(); }

    [[nodiscard]] const std::uint8_t* data() const {
        return static_cast<const std::uint8_t*>(data_);
    }

private:
    void close_noexcept() noexcept {
        if (data_ == nullptr) {
            return;
        }
#ifdef _WIN32
        if (!UnmapViewOfFile(data_)) {
            std::fprintf(stderr, "FATAL UnmapViewOfFile failed with Windows error %lu\n",
                         static_cast<unsigned long>(GetLastError()));
            std::abort();
        }
        data_ = nullptr;
        if (!CloseHandle(mapping_)) {
            std::fprintf(stderr, "FATAL CloseHandle(mapping) failed with Windows error %lu\n",
                         static_cast<unsigned long>(GetLastError()));
            std::abort();
        }
        mapping_ = nullptr;
#else
        if (::munmap(data_, size_) != 0) {
            const int error = errno;
            std::fprintf(stderr, "FATAL munmap failed: %s (errno=%d)\n",
                         std::strerror(error), error);
            std::abort();
        }
        data_ = nullptr;
#endif
    }

    void* data_ = nullptr;
    std::size_t size_ = 0;
#ifdef _WIN32
    HANDLE mapping_ = nullptr;
#endif
};

int helper_try_lock(const std::vector<fs::path>& arguments) {
    check(arguments.size() == 1, "_try-lock requires LOCKFILE");
    NativeFile file = open_read_write(arguments[0]);
    HeldFileLock held;
    if (!try_acquire_lock(file, held)) {
        return kLockBusyExit;
    }
    return 0;
}

int helper_hold_lock(const std::vector<fs::path>& arguments) {
    check(arguments.size() == 2, "_hold-lock requires LOCKFILE READY");
    NativeFile file = open_read_write(arguments[0]);
    HeldFileLock held;
    check(try_acquire_lock(file, held), "holder could not acquire lock");
    write_exclusive_text(arguments[1], held.api_name());
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int helper_append_loop(const std::vector<fs::path>& arguments) {
    check(arguments.size() == 4,
          "_append-loop requires FILE COMMITTED_END READY STOP");
    const std::uint64_t committed_end =
        parse_unsigned(arguments[1].string(), "committed_file_end");
    NativeFile file = open_append_only(arguments[0]);
    check(file_size(file) == committed_end,
          "append fixture size does not equal committed_file_end");
    const std::vector<std::uint8_t> chunk(4096, 0xD3);
    write_single(file, chunk.data(), chunk.size());
    write_exclusive_text(arguments[2], "ready");
    for (std::uint32_t iteration = 1; iteration < 256; ++iteration) {
        if (path_exists(arguments[3])) {
            break;
        }
        write_single(file, chunk.data(), chunk.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    file.close_checked();
    return 0;
}

int helper_head_reader(const std::vector<fs::path>& arguments) {
    check(arguments.size() == 4, "_head-reader requires FILE READY STOP RESULT");
    NativeFile file = open_read_only(arguments[0], true);
    write_exclusive_text(arguments[1], "ready");
    std::vector<std::uint8_t> slot_a(4096);
    std::vector<std::uint8_t> slot_b(4096);
    std::uint64_t iterations = 0;
    while (iterations < 10'000'000) {
        read_exact_at(file, slot_a.data(), slot_a.size(), 4096);
        read_exact_at(file, slot_b.data(), slot_b.size(), 8192);
        const bool valid_a = all_bytes_are(slot_a, 0x11) || all_bytes_are(slot_a, 0x31);
        const bool valid_b = all_bytes_are(slot_b, 0x22) || all_bytes_are(slot_b, 0x42);
        if (!valid_a || !valid_b) {
            const auto describe = [](const std::vector<std::uint8_t>& slot,
                                     const std::string& name) {
                const auto different = std::find_if(
                    slot.begin() + 1, slot.end(),
                    [&slot](const std::uint8_t byte) { return byte != slot.front(); });
                if (different == slot.end()) {
                    return name + " uniform unexpected value=" +
                           std::to_string(slot.front());
                }
                const std::size_t index =
                    static_cast<std::size_t>(different - slot.begin());
                return name + " mixed: byte[0]=" + std::to_string(slot.front()) +
                       ", byte[" + std::to_string(index) + "]=" +
                       std::to_string(*different);
            };
            std::string detail;
            if (!valid_a) {
                detail = describe(slot_a, "slot A");
            }
            if (!valid_b) {
                if (!detail.empty()) {
                    detail += "; ";
                }
                detail += describe(slot_b, "slot B");
            }
            write_exclusive_text(arguments[3], detail);
            return 74;
        }
        ++iterations;
        if (iterations >= 2000 && iterations % 64 == 0 && path_exists(arguments[2])) {
            write_exclusive_text(arguments[3], std::to_string(iterations));
            return 0;
        }
    }
    fail("head reader reached its safety iteration limit");
}

int helper_prefix_reader(const std::vector<fs::path>& arguments) {
    check(arguments.size() == 5,
          "_prefix-reader requires FILE COMMITTED_END READY STOP RESULT");
    const std::uint64_t committed_end =
        parse_unsigned(arguments[1].string(), "committed_file_end");
    check(committed_end >= 8192, "committed_file_end is too small");
    NativeFile file = open_read_only(arguments[0], true);
    write_exclusive_text(arguments[2], "ready");
    const std::array<std::uint64_t, 3> offsets{
        0, committed_end / 2, committed_end - 4096};
    std::vector<std::uint8_t> block(4096);
    std::uint64_t iterations = 0;
    while (iterations < 10'000'000) {
        for (const std::uint64_t offset : offsets) {
            read_exact_at(file, block.data(), block.size(), offset);
            if (!all_bytes_are(block, 0x5A)) {
                write_exclusive_text(arguments[4], "prefix changed");
                return 75;
            }
        }
        ++iterations;
        if (iterations >= 2000 && iterations % 64 == 0 && path_exists(arguments[3])) {
            write_exclusive_text(arguments[4], std::to_string(iterations));
            return 0;
        }
    }
    fail("prefix reader reached its safety iteration limit");
}

int helper_crash_append(const std::vector<fs::path>& arguments) {
    check(arguments.size() == 3,
          "_crash-append requires FILE COMMITTED_END READY");
    const std::uint64_t committed_end =
        parse_unsigned(arguments[1].string(), "committed_file_end");
    NativeFile file = open_append_only(arguments[0]);
    check(file_size(file) == committed_end,
          "crash fixture size does not equal committed_file_end");
    const std::vector<std::uint8_t> chunk(4096, 0xC7);
    write_single(file, chunk.data(), chunk.size());
    write_exclusive_text(arguments[2], "wrote");
    for (std::uint32_t iteration = 1; iteration < 256; ++iteration) {
        write_single(file, chunk.data(), chunk.size());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int scenario_lockfile_exclusive(const fs::path& run_directory) {
    const fs::path directory =
        make_scenario_directory(run_directory, "lockfile-exclusive");
    const fs::path lock_path = directory / "container.licht.lock";
    std::string api;
    {
        NativeFile lock_file = create_new_read_write(lock_path);
        HeldFileLock held;
        check(try_acquire_lock(lock_file, held), "parent could not acquire lock");
        api = held.api_name();
        ChildProcess contender = spawn_self({"_try-lock", lock_path});
        const int contender_exit = contender.wait();
        check(contender_exit == kLockBusyExit,
              "child did not observe the held exclusive lock; exit=" +
                  std::to_string(contender_exit));
    }
#ifdef _WIN32
    const char* exclusive_create = "CREATE_NEW";
#else
    const char* exclusive_create = "O_CREAT|O_EXCL";
#endif
    std::cout << "PASS lockfile-exclusive: sibling lockfile created with "
              << exclusive_create << "; child blocked by " << api << '\n';
    return 0;
}

int scenario_lockfile_stale(const fs::path& run_directory) {
    const fs::path directory =
        make_scenario_directory(run_directory, "lockfile-stale");
    const fs::path lock_path = directory / "container.licht.lock";
    const fs::path ready_path = directory / "holder.ready";
    {
        NativeFile fixture = create_new_read_write(lock_path);
        fixture.close_checked();
    }
    ChildProcess holder = spawn_self({"_hold-lock", lock_path, ready_path});
    wait_for_path(ready_path);
    holder.kill_for_test();
    const auto acquisition_start = std::chrono::steady_clock::now();
    std::string api;
    {
        NativeFile lock_file = open_read_write(lock_path);
        HeldFileLock held;
        check(try_acquire_lock(lock_file, held),
              "lock remained held after holder process was killed");
        api = held.api_name();
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - acquisition_start);
#ifdef _WIN32
    const char* termination = "TerminateProcess";
#else
    const char* termination = "SIGKILL";
#endif
    std::cout << "PASS lockfile-stale: " << termination
              << " released the open-handle lock; reacquired via " << api << " in "
              << elapsed.count() << " us\n";
    return 0;
}

int scenario_positional_reads(const fs::path& run_directory) {
    const fs::path directory =
        make_scenario_directory(run_directory, "positional-reads");
    const fs::path fixture_path = directory / "container.licht";
    const fs::path ready_path = directory / "writer.ready";
    const fs::path stop_path = directory / "writer.stop";
    constexpr std::size_t committed_end = 16 * 1024;
    const std::vector<std::uint8_t> original = patterned_bytes(committed_end, 0x19);
    write_file(fixture_path, original);
    ChildProcess writer = spawn_self(
        {"_append-loop", fixture_path, std::to_string(committed_end), ready_path,
         stop_path});
    wait_for_path(ready_path);
    NativeFile reader = open_read_only(fixture_path, true);
    std::vector<std::uint8_t> observed(committed_end);
    constexpr std::uint32_t read_iterations = 1000;
    for (std::uint32_t iteration = 0; iteration < read_iterations; ++iteration) {
        read_exact_at(reader, observed.data(), observed.size(), 0);
        check(observed == original,
              "pinned committed region changed during concurrent append");
        if (iteration % 25 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    write_exclusive_text(stop_path, "stop");
    const int writer_exit = writer.wait();
    check(writer_exit == 0,
          "append child failed with exit=" + std::to_string(writer_exit));
    const std::uint64_t physical_size = file_size(reader);
    check(physical_size > committed_end, "concurrent writer did not append a tail");
    reader.close_checked();
#ifdef _WIN32
    const char* read_api = "ReadFile(OVERLAPPED)";
#else
    const char* read_api = "pread";
#endif
    std::cout << "PASS positional-reads: " << read_iterations << ' ' << read_api
              << " full-prefix reads stayed stable while append grew file from "
              << committed_end << " to " << physical_size << " bytes\n";
    return 0;
}

int scenario_head_slot_publish(const fs::path& run_directory) {
    const fs::path directory =
        make_scenario_directory(run_directory, "head-slot-publish");
    const fs::path fixture_path = directory / "container.licht";
    const fs::path ready_path = directory / "reader.ready";
    const fs::path stop_path = directory / "reader.stop";
    const fs::path result_path = directory / "reader.result";
    std::vector<std::uint8_t> fixture(3 * 4096, 0);
    std::fill(fixture.begin() + 4096, fixture.begin() + 8192, 0x11);
    std::fill(fixture.begin() + 8192, fixture.end(), 0x22);
    write_file(fixture_path, fixture);
    ChildProcess reader =
        spawn_self({"_head-reader", fixture_path, ready_path, stop_path, result_path});
    wait_for_path(ready_path);
    NativeFile writer = open_read_write(fixture_path, true);
    const std::vector<std::uint8_t> slot_a_old(4096, 0x11);
    const std::vector<std::uint8_t> slot_a_new(4096, 0x31);
    const std::vector<std::uint8_t> slot_b_old(4096, 0x22);
    const std::vector<std::uint8_t> slot_b_new(4096, 0x42);
    constexpr std::uint32_t publish_count = 2000;
    for (std::uint32_t publication = 0; publication < publish_count; ++publication) {
        const bool publish_b = publication % 2 == 0;
        const bool new_pattern = (publication / 2) % 2 == 0;
        const std::vector<std::uint8_t>& bytes = publish_b
                                                     ? (new_pattern ? slot_b_new : slot_b_old)
                                                     : (new_pattern ? slot_a_new : slot_a_old);
        const std::uint64_t offset = publish_b ? 8192 : 4096;
        sync_file(writer);
        write_exact_at_single(writer, bytes.data(), bytes.size(), offset);
        sync_file(writer);
    }
    writer.close_checked();
    write_exclusive_text(stop_path, "stop");
    const int reader_exit = reader.wait();
#ifdef _WIN32
    const char* write_api = "WriteFile(OVERLAPPED)";
    const char* sync_api = "FlushFileBuffers";
#else
    const char* write_api = "pwrite";
    const char* sync_api = "fdatasync";
#endif
    if (reader_exit == 74) {
        const std::string detail = path_exists(result_path)
                                       ? read_text(result_path)
                                       : "no reader diagnostic";
        std::cout
            << "TENSION head-slot-publish: concurrent reader observed a torn 4096 B "
               "slot during one aligned "
            << write_api << " bracketed by " << sync_api
            << " (" << detail
            << "). Plan §2.2 boundary 8 already requires CRC-invalid new heads to "
               "fall back; do not treat single-op publish as an OS atomic snapshot.\n";
        return 0;
    }
    if (reader_exit != 0) {
        const std::string detail = path_exists(result_path)
                                       ? read_text(result_path)
                                       : "no reader diagnostic";
        fail("head-slot reader failed with exit=" + std::to_string(reader_exit) +
             "; " + detail);
    }
    const std::uint64_t reader_iterations =
        parse_unsigned(read_text(result_path), "head reader iteration count");
    std::cout << "PASS head-slot-publish: " << publish_count
              << " alternating inactive-slot publishes used one 4096-byte " << write_api
              << " between " << sync_api << " calls; reader completed "
              << reader_iterations << " paired reads with no mixed slot\n";
    return 0;
}

int scenario_mapped_reader_vs_replace(const fs::path& run_directory) {
    const fs::path directory =
        make_scenario_directory(run_directory, "mapped-reader-vs-replace");
    const std::vector<std::uint8_t> old_bytes(4096, 0x4F);
    const std::vector<std::uint8_t> new_bytes(4096, 0x4E);
#ifdef _WIN32
    const fs::path destination = directory / "shared.licht";
    const fs::path replacement = directory / "shared.tmp";
    write_file(destination, old_bytes);
    write_file(replacement, new_bytes);
    {
        NativeFile mapped_file = open_windows_file(
            destination, GENERIC_READ, kCompatibleShare, OPEN_EXISTING, false);
        ReadOnlyMapping mapping(mapped_file, old_bytes.size());
        if (!ReplaceFileW(destination.c_str(), replacement.c_str(), nullptr,
                          REPLACEFILE_WRITE_THROUGH, nullptr, nullptr)) {
            fail_last_error("ReplaceFileW(compatible shares)");
        }
        check(std::equal(old_bytes.begin(), old_bytes.end(), mapping.data()),
              "old mapping changed after ReplaceFileW");
        check(read_file_bytes(destination, 4096) == new_bytes,
              "destination did not expose replacement bytes");
    }
    const fs::path blocked_destination = directory / "blocked.licht";
    const fs::path blocked_replacement = directory / "blocked.tmp";
    write_file(blocked_destination, old_bytes);
    write_file(blocked_replacement, new_bytes);
    DWORD blocked_error = ERROR_SUCCESS;
    {
        NativeFile mapped_file = open_windows_file(
            blocked_destination, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
            OPEN_EXISTING, false);
        ReadOnlyMapping mapping(mapped_file, old_bytes.size());
        if (ReplaceFileW(blocked_destination.c_str(), blocked_replacement.c_str(),
                         nullptr, REPLACEFILE_WRITE_THROUGH, nullptr, nullptr)) {
            fail("ReplaceFileW unexpectedly succeeded without FILE_SHARE_DELETE");
        }
        blocked_error = GetLastError();
        check(blocked_error == ERROR_SHARING_VIOLATION,
              "wrong-share ReplaceFileW error was " + std::to_string(blocked_error) +
                  ", expected 32");
        check(std::equal(old_bytes.begin(), old_bytes.end(), mapping.data()),
              "blocked replacement changed the mapping");
    }
    std::cout
        << "PASS mapped-reader-vs-replace: ReplaceFileW succeeded with "
           "FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE and the old mapping "
           "stayed valid; omitting FILE_SHARE_DELETE failed with error="
        << blocked_error << '\n';
#else
    const fs::path destination = directory / "shared.licht";
    const fs::path replacement = directory / "shared.tmp";
    write_file(destination, old_bytes);
    write_file(replacement, new_bytes);
    {
        NativeFile mapped_file = open_read_only(destination);
        ReadOnlyMapping mapping(mapped_file, old_bytes.size());
        if (::rename(replacement.c_str(), destination.c_str()) != 0) {
            fail_last_error("rename(temp, destination)");
        }
        check(std::equal(old_bytes.begin(), old_bytes.end(), mapping.data()),
              "old mmap changed after rename");
        check(read_file_bytes(destination, 4096) == new_bytes,
              "renamed destination did not expose replacement bytes");
    }
    std::cout << "PASS mapped-reader-vs-replace: rename(temp,dest) succeeded with a "
                 "held mmap; old mapping retained old inode bytes and dest exposed "
                 "replacement bytes\n";
#endif
    return 0;
}

int scenario_replace_error_matrix(const fs::path& run_directory) {
    const fs::path directory =
        make_scenario_directory(run_directory, "replace-error-matrix");
#ifdef _WIN32
    const fs::path missing_destination = directory / "first.licht";
    const fs::path replacement = directory / "first.tmp";
    write_file(replacement, std::vector<std::uint8_t>(128, 0x67));
    if (ReplaceFileW(missing_destination.c_str(), replacement.c_str(), nullptr,
                     REPLACEFILE_WRITE_THROUGH, nullptr, nullptr)) {
        fail("ReplaceFileW unexpectedly published to a missing destination");
    }
    const DWORD missing_error = GetLastError();
    check(path_exists(replacement),
          "failed ReplaceFileW removed the first-publication replacement");
    if (!MoveFileExW(replacement.c_str(), missing_destination.c_str(),
                     MOVEFILE_WRITE_THROUGH)) {
        fail_last_error("MoveFileExW(first publication)");
    }
    std::cout << "SKIP replace-error-matrix 1175: requires a nondeterministic "
                 "filesystem failure while moving the replacement\n";
    std::cout << "SKIP replace-error-matrix 1176: requires rollback failure after a "
                 "replacement move failure\n";
    std::cout << "SKIP replace-error-matrix 1177: requires a nondeterministic failure "
                 "removing the replaced file\n";
    std::cout << "PASS replace-error-matrix: missing-destination ReplaceFileW error="
              << missing_error
              << "; MoveFileExW completed first publication; 1175-1177 skipped with "
                 "explicit reasons\n";
#else
    const fs::path first_source = directory / "first.tmp";
    const fs::path first_destination = directory / "first.licht";
    write_file(first_source, std::vector<std::uint8_t>(128, 0x67));
    if (::rename(first_source.c_str(), first_destination.c_str()) != 0) {
        fail_last_error("rename(missing destination)");
    }
    const fs::path missing_source = directory / "missing.tmp";
    const fs::path unused_destination = directory / "unused.licht";
    errno = 0;
    const int missing_result = ::rename(missing_source.c_str(), unused_destination.c_str());
    const int missing_error = errno;
    check(missing_result == -1, "rename unexpectedly succeeded with a missing source");
    check(missing_error == ENOENT,
          "rename missing-source error was not ENOENT: " +
              std::to_string(missing_error));
    std::cout << "SKIP replace-error-matrix cross-device: all fixtures are constrained "
                 "to one --dir, so EXDEV needs a second filesystem\n";
    std::cout << "PASS replace-error-matrix: rename created a missing destination; "
                 "missing source failed with ENOENT; cross-device case skipped\n";
#endif
    return 0;
}

int scenario_tail_truncate(const fs::path& run_directory) {
    const fs::path directory =
        make_scenario_directory(run_directory, "tail-truncate");
    const fs::path fixture_path = directory / "container.licht";
    const fs::path lock_path = directory / "container.licht.lock";
    const fs::path ready_path = directory / "reader.ready";
    const fs::path stop_path = directory / "reader.stop";
    const fs::path result_path = directory / "reader.result";
    constexpr std::size_t committed_end = 64 * 1024;
    std::vector<std::uint8_t> fixture(128 * 1024, 0xE3);
    std::fill(fixture.begin(), fixture.begin() + committed_end, 0x5A);
    write_file(fixture_path, fixture);
    std::string lock_api;
    std::uint64_t reader_iterations = 0;
    {
        NativeFile lock_file = create_new_read_write(lock_path);
        HeldFileLock held;
        check(try_acquire_lock(lock_file, held),
              "simulated writer could not acquire lock");
        lock_api = held.api_name();
        ChildProcess reader = spawn_self(
            {"_prefix-reader", fixture_path, std::to_string(committed_end), ready_path,
             stop_path, result_path});
        wait_for_path(ready_path);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        NativeFile writer = open_read_write(fixture_path);
        truncate_file(writer, committed_end);
        sync_file(writer);
        check(file_size(writer) == committed_end,
              "physical size does not equal committed_file_end after truncate");
        writer.close_checked();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        write_exclusive_text(stop_path, "stop");
        const int reader_exit = reader.wait();
        check(reader_exit == 0,
              "tail-truncate reader failed with exit=" +
                  std::to_string(reader_exit));
        reader_iterations =
            parse_unsigned(read_text(result_path), "prefix reader iteration count");
        NativeFile verifier = open_read_only(fixture_path, true);
        check(file_size(verifier) == committed_end,
              "reopened size does not equal committed_file_end");
        std::vector<std::uint8_t> prefix(committed_end);
        read_exact_at(verifier, prefix.data(), prefix.size(), 0);
        check(all_bytes_are(prefix, 0x5A), "prefix changed after truncate");
    }
#ifdef _WIN32
    const char* truncate_api = "SetFilePointerEx+SetEndOfFile";
    const char* read_api = "ReadFile(OVERLAPPED)";
#else
    const char* truncate_api = "ftruncate";
    const char* read_api = "pread";
#endif
    std::cout << "PASS tail-truncate: under " << lock_api << ", " << truncate_api
              << " set size to committed_file_end=" << committed_end << "; "
              << reader_iterations << " concurrent " << read_api
              << " iterations kept the prefix stable\n";
    return 0;
}

int scenario_append_crash_orphan(const fs::path& run_directory) {
    const fs::path directory =
        make_scenario_directory(run_directory, "append-crash-orphan");
    const fs::path fixture_path = directory / "container.licht";
    const fs::path ready_path = directory / "writer.ready";
    constexpr std::size_t committed_end = 64 * 1024;
    std::vector<std::uint8_t> committed = patterned_bytes(committed_end, 0x2B);
    for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte) {
        committed[byte] = static_cast<std::uint8_t>(
            (static_cast<std::uint64_t>(committed_end) >> (byte * 8U)) & 0xffU);
    }
    write_file(fixture_path, committed);
    ChildProcess writer = spawn_self(
        {"_crash-append", fixture_path, std::to_string(committed_end), ready_path});
    wait_for_path(ready_path);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    writer.kill_for_test();
    NativeFile verifier = open_read_only(fixture_path, true);
    const std::uint64_t physical_size = file_size(verifier);
    check(physical_size > committed_end,
          "killed writer left no physical orphan tail");
    std::vector<std::uint8_t> observed(committed_end);
    read_exact_at(verifier, observed.data(), observed.size(), 0);
    check(observed == committed,
          "bytes below committed_file_end changed after append crash");
    std::uint64_t marker = 0;
    for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte) {
        marker |= static_cast<std::uint64_t>(observed[byte]) << (byte * 8U);
    }
    check(marker == committed_end,
          "committed_file_end marker changed after append crash");
    verifier.close_checked();
#ifdef _WIN32
    const char* termination = "TerminateProcess";
#else
    const char* termination = "SIGKILL";
#endif
    std::cout << "PASS append-crash-orphan: " << termination
              << " left physical size=" << physical_size
              << " > committed_file_end=" << committed_end
              << "; committed prefix and marker remained intact\n";
    return 0;
}

void run_scenario(const std::string& command, const fs::path& run_directory) {
    if (command == "lockfile-exclusive") {
        scenario_lockfile_exclusive(run_directory);
    } else if (command == "lockfile-stale") {
        scenario_lockfile_stale(run_directory);
    } else if (command == "positional-reads") {
        scenario_positional_reads(run_directory);
    } else if (command == "head-slot-publish") {
        scenario_head_slot_publish(run_directory);
    } else if (command == "mapped-reader-vs-replace") {
        scenario_mapped_reader_vs_replace(run_directory);
    } else if (command == "replace-error-matrix") {
        scenario_replace_error_matrix(run_directory);
    } else if (command == "tail-truncate") {
        scenario_tail_truncate(run_directory);
    } else if (command == "append-crash-orphan") {
        scenario_append_crash_orphan(run_directory);
    } else {
        fail("unknown subcommand: " + command);
    }
}

fs::path default_base_directory() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    const DWORD length = GetTempPathW(static_cast<DWORD>(buffer.size()), buffer.data());
    if (length == 0) {
        fail_last_error("GetTempPathW");
    }
    check(length < buffer.size(), "GetTempPathW result exceeded buffer");
    return fs::path(std::wstring(buffer.data(), length)) /
           (L"licht_os_semantics_" + std::to_wstring(process_id()));
#else
    return fs::path("/tmp") / ("licht_os_semantics_" + std::to_string(process_id()));
#endif
}

int dispatch_helper(const std::string& command,
                    const std::vector<fs::path>& arguments) {
    if (command == "_try-lock") {
        return helper_try_lock(arguments);
    }
    if (command == "_hold-lock") {
        return helper_hold_lock(arguments);
    }
    if (command == "_append-loop") {
        return helper_append_loop(arguments);
    }
    if (command == "_head-reader") {
        return helper_head_reader(arguments);
    }
    if (command == "_prefix-reader") {
        return helper_prefix_reader(arguments);
    }
    if (command == "_crash-append") {
        return helper_crash_append(arguments);
    }
    fail("unknown helper subcommand: " + command);
}

int program_main(const std::vector<fs::path>& arguments) {
    check(!arguments.empty(), "missing process argv");
    fs::path base_directory = default_base_directory();
    std::size_t next = 1;
    if (next < arguments.size() && arguments[next] == fs::path("--dir")) {
        check(next + 1 < arguments.size(), "--dir requires a directory");
        base_directory = arguments[next + 1];
        next += 2;
    }
    check(next < arguments.size(),
          "usage: os_semantics [--dir DIR] <subcommand>");
    const std::string command = arguments[next].string();
    ++next;
    std::vector<fs::path> trailing(arguments.begin() + static_cast<std::ptrdiff_t>(next),
                                   arguments.end());
    if (!command.empty() && command.front() == '_') {
        return dispatch_helper(command, trailing);
    }
    check(trailing.empty(), "unexpected arguments after subcommand");
    const fs::path run_directory = make_run_directory(base_directory);
    if (command == "all") {
        const std::array<const char*, 8> scenarios{
            "lockfile-exclusive",       "lockfile-stale",
            "positional-reads",         "head-slot-publish",
            "mapped-reader-vs-replace", "replace-error-matrix",
            "tail-truncate",            "append-crash-orphan"};
        bool any_failure = false;
        for (const char* scenario : scenarios) {
            try {
                run_scenario(scenario, run_directory);
            } catch (const std::exception& error) {
                std::cout << "FAIL " << scenario << ": " << error.what() << '\n';
                any_failure = true;
            }
        }
        if (any_failure) {
            std::cout << "FAIL all: one or more scenarios failed\n";
            return 1;
        }
        std::cout << "PASS all: 8 scenarios passed\n";
        return 0;
    }
    run_scenario(command, run_directory);
    return 0;
}

std::string diagnostic_command(const std::vector<fs::path>& arguments) {
    std::size_t next = 1;
    if (next < arguments.size() && arguments[next] == fs::path("--dir")) {
        next += 2;
    }
    return next < arguments.size() ? arguments[next].string() : "os_semantics";
}

#ifdef _WIN32
fs::path current_executable_path() {
    std::vector<wchar_t> buffer(1024);
    while (true) {
        SetLastError(ERROR_SUCCESS);
        const DWORD length =
            GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) {
            fail_last_error("GetModuleFileNameW");
        }
        if (length < buffer.size() - 1) {
            return fs::path(std::wstring(buffer.data(), length));
        }
        buffer.resize(buffer.size() * 2);
    }
}
#else
fs::path current_executable_path(const fs::path& argv_zero) {
    std::error_code error;
    const fs::path absolute = fs::absolute(argv_zero, error);
    if (error) {
        fail("filesystem::absolute(executable) failed: " + error.message());
    }
    return absolute;
}
#endif

} // namespace

#ifdef _WIN32
int wmain(const int argc, wchar_t** argv) {
    std::string command = "os_semantics";
    try {
        std::vector<fs::path> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
            arguments.emplace_back(argv[index]);
        }
        std::cout.setf(std::ios::unitbuf);
        executable_path = current_executable_path();
        command = diagnostic_command(arguments);
        return program_main(arguments);
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << command << ": " << error.what() << '\n';
        return 1;
    }
}
#else
int main(const int argc, char** argv) {
    std::string command = "os_semantics";
    try {
        std::vector<fs::path> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
            arguments.emplace_back(argv[index]);
        }
        std::cout.setf(std::ios::unitbuf);
        executable_path = current_executable_path(arguments.front());
        command = diagnostic_command(arguments);
        return program_main(arguments);
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << command << ": " << error.what() << '\n';
        return 1;
    }
}
#endif
