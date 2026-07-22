# P0(d) OS File-Semantics Matrix

Prototype: `tools/licht_p0/os_semantics/os_semantics.cpp`  
Plan: `PROJECT_FORMAT_PLAN.md` §1 decision 13, §2.2, §3, §7 P0(d)  
Spec: `docs/licht_format_spec.md` (head slots + `committed_file_end`)

| scenario | plan assumption | POSIX expected | POSIX observed | Windows expected | Windows observed |
|---|---|---|---|---|---|
| lockfile-exclusive | §1.13: sibling lockfile = exclusive create + exclusive OS lock held on the open fd/handle (not existence-checking). | `O_CREAT\|O_EXCL` create-once; second process cannot acquire OFD lock / `flock` while holder lives. | **PASS:** `O_CREAT\|O_EXCL`; child blocked by `fcntl(F_OFD_SETLK)`. | `CREATE_NEW` create-once; `LockFileEx(EXCLUSIVE\|FAIL_IMMEDIATELY)` denies second process while holder handle is open. | pending |
| lockfile-stale | §1.13: killed holder cannot deny writes — kernel releases the lock with the fd/handle. | `SIGKILL` of holder → next process acquires immediately. | **PASS:** reacquired via `fcntl(F_OFD_SETLK)` in ~2 µs after `SIGKILL`. | `TerminateProcess` of holder → next process acquires immediately. | pending |
| positional-reads | §1.13: readers use positional IO (`pread` / overlapped); pinned prefix below `committed_file_end` is stable while a writer appends an orphan tail. | Repeated `pread` of the committed prefix matches the fixture during concurrent append. | **PASS:** 1000 full-prefix `pread` checks stable; physical size grew 16384 → ~188 KiB. | Repeated `ReadFile(OVERLAPPED)` of the committed prefix matches the fixture during concurrent append. | pending |
| head-slot-publish | §2.2 steps 8–9: publish inactive head with one aligned 4096 B write, flush before and after; readers should not rely on a torn slot. | Concurrent `pread` of a slot sees only all-old or all-new patterns while writer does one 4096 B `pwrite` between `fdatasync` calls. | **TENSION:** mixed old/new bytes inside one 4096 B slot under concurrent `pread` (repro: `byte[0]=17` vs `byte[256]=49`; also 49/3584, slot B 34/66). Buffered single `pwrite` is **not** concurrent-reader-atomic on Linux+ext4. Scenario prints `TENSION` and still completes. | Concurrent overlapped reads see only all-old or all-new patterns while writer does one 4096 B overlapped `WriteFile` between `FlushFileBuffers` calls. | pending |
| mapped-reader-vs-replace | §1.13 + §3: compaction replace works against a held map when share flags are correct (Windows checklist). | `rename(temp,dest)` succeeds with held `mmap`; old map retains old inode bytes; path exposes replacement. | **PASS:** rename succeeded with held `mmap`; old map and dest path each retained expected bytes. | `ReplaceFileW` succeeds with `FILE_SHARE_READ\|WRITE\|DELETE`; omitting `FILE_SHARE_DELETE` fails with sharing violation (32). | pending |
| replace-error-matrix | §3: `ReplaceFileW` errors 1175–1177 retained/validated; first publication uses `MoveFileExW` when dest missing; POSIX rename equivalents. | Missing dest: `rename` creates; missing source: `ENOENT`; cross-device `EXDEV` optional if no second FS. | **PASS:** missing dest created; missing source `ENOENT`; cross-device **SKIP** (single `--dir` FS). | Missing dest: record `ReplaceFileW` error, then `MoveFileExW` first publication; 1175–1177 exercised or explicit SKIP. | pending |
| tail-truncate | §2.2: under writer lock, `ftruncate`/`SetEndOfFile` to `committed_file_end` reclaims orphan tail; concurrent positional readers below that offset are unaffected. | `ftruncate` → size == `committed_file_end`; concurrent `pread` prefix stable. | **PASS:** size 65536; ~13k–19k concurrent prefix `pread` iterations stable under OFD lock. | `SetFilePointerEx`+`SetEndOfFile` → size == `committed_file_end`; concurrent overlapped prefix reads stable. | pending |
| append-crash-orphan | §2.1 / spec: kill mid-append → physical size > `committed_file_end`; bytes below remain intact (orphan tail ignored by parsers). | `SIGKILL` after confirmed append leaves orphan tail and intact committed prefix/marker. | **PASS:** physical size ~77824 > 65536 after `SIGKILL`; committed prefix + marker intact. | `TerminateProcess` after confirmed append leaves orphan tail and intact committed prefix/marker. | pending |

## Tensions

### **TENSION — head-slot-publish** (loud)

**What was assumed (naive reading of “one aligned op”):** a single page-aligned 4096 B write of the inactive head, with `fdatasync`/`FlushFileBuffers` before and after, is observed by concurrent readers as an atomic all-old or all-new slot.

**What POSIX observed (Linux 6.12 / ext4, buffered `pwrite`+`pread`):** mixed old and new fill-bytes inside the same 4096 B slot while a second process hammered positional reads. Reproduced on every full `all` run here (examples: slot A `byte[0]=17` vs `byte[256]=49`; `byte[0]=49` vs `byte[3584]=17`; slot B `byte[0]=34` vs `byte[3584]=66`; first-diff also at 688/3200).

**Implication for P2 writer freeze:**

1. Do **not** claim OS-level concurrent-reader atomicity for a buffered 4096 B head write.
2. Plan §2.2 crash boundary 8 already states a torn/failed new slot is **invalid** and open falls back with warning — that CRC + dual-slot model is **load-bearing**, not optional.
3. Production readers must validate full head CRC (and the rest of the open state machine) before trusting a slot; never treat a mid-slot byte mix as a valid publish.
4. Windows day-2 run must still measure the same hammer; outcome is `pending` here.

### Non-tensions

- Cross-device POSIX `EXDEV` was **SKIP** because fixtures are constrained to one `--dir`. Permitted; not a plan contradiction.
- Windows columns remain **pending** until CI or a Windows box runs `build.bat` + `os_semantics.exe --dir <dir> all`.

## How to re-run

```bash
# POSIX
cd tools/licht_p0/os_semantics
bash build.sh
nice -n19 ./os_semantics --dir /tmp/licht_os_sem_run all

# Windows (day-2, no source edits expected)
cd tools\licht_p0\os_semantics
build.bat
os_semantics.exe --dir %TEMP%\licht_os_sem_run all
```

Exit code 0 from `all` means every scenario completed its measurement. A `TENSION` line is a documented OS finding, not a harness crash.
