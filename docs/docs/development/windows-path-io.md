# Windows filesystem paths

LichtFeld keeps application paths in their normal spelling for the UI, project
references, JSON, URLs, and command arguments. Filesystem operations use the shared
`lfs::core::native_fs` adapter (`core/native_fs.hpp`). On Windows it supplies absolute
extended drive or UNC paths at the I/O boundary when needed. It does not change
system policy or the registry. On other platforms it uses the standard filesystem.

Directory walks retain standard `std::filesystem` iterators. Start them through
`native_fs::directory_iterator` / `recursive_directory_iterator`, or construct a
standard iterator with `native_fs::traversal_path(root)` when using error codes.
Entries carry native filenames for I/O. Convert them with
`native_fs::entry_path(root, entry.path())` before storing paths, keys, or messages;
this also preserves the caller's relative root spelling. Do not persist an
iterator's native `entry.path()` directly.

Use the adapter for filesystem operations and file streams. Filename APIs in
external libraries must receive `native_path` (wide Windows APIs) or
`path_to_native_utf8` (documented UTF-8 APIs). Do not apply these conversions to
URLs or command arguments. Decode UTF-8 API/JSON/resource names with
`utf8_to_path` before path construction; an implicit narrow-string path constructor
on Windows uses the system code page. Process launch code adapts the executable filename
separately from its arguments. ZIP writers explicitly select UTF-8 header names.
stb image readers use `core/stb_file.hpp`, which opens through `open_file` and
passes `FILE*` to stb. A native prefix alone does not bypass stb's fixed Windows
filename buffer.

Python runtime code uses `_lfs_path_io.Path`: `str(path)` is an application path,
while `os.fspath(path)` supplies its I/O spelling. Use the module's `open`, `walk`,
and `TemporaryDirectory` for their respective I/O boundaries. Native bindings
expose `lichtfeld.io.native_path` for filename APIs that require a string.

Project locks, autosaves, recovery copies, and log rotation preserve existing
filenames when they fit. Longer derived names use bounded hashes. Windows name
limits are measured in UTF-16 units, and hashed names respect directory case
sensitivity. Project serialization never requires an extended-path prefix.

Startup diagnostics report effective Windows process long-path opt-in separately
from the enabled extended-path I/O adapter. Either opt-in state uses the adapter.
An unknown diagnostic is not a reason to disable I/O adaptation.

## Validation and limits

Dedicated Windows test executables have a `longPathAware=false` manifest and assert
that effective process opt-in is disabled. The Windows workflow also patches a
private copy of the Python executable to that manifest; it never changes the
runner registry. Tests cover Unicode paths over 500 characters, file streams,
recursive traversal and copying, atomic outputs, project recovery and sidecars,
logging, image codecs, UI resource reads, mesh materials, archives, and FFmpeg.
UNC tests use the explicitly provisioned CI share and fail if a required share is
unavailable. Local runs can set `LFS_TEST_UNC_ROOT`, `LFS_TEST_UNC_LOCAL`, and
`LFS_TEST_REQUIRE_UNC=1` to exercise the same checks.

This does not remove filesystem component limits, permissions, unavailable shares,
sharing violations, or limits imposed by external programs. Windows can reject a
working directory longer than MAX_PATH when process opt-in is disabled; use
absolute adapted filenames rather than changing the working directory there.
The corresponding optional working-directory test reports a skip for that OS
restriction; ordinary long-path file operations are tested independently.
