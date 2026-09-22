---
sidebar_position: 2
---

# Installation

Windows binaries are available from the project's [GitHub Releases](https://github.com/MrNeRF/LichtFeld-Studio/releases).

For source builds and platform-specific instructions, see the [Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki/) and the repo-local docs in [docs/README.md](../../README.md).

## Windows Defender detections

If Defender quarantines a binary such as `lfs_core.dll`, record the detection name,
the build's source commit or release version, and the SHA256 of the affected file.
For a file that is still available, PowerShell can print the hash:

```powershell
Get-FileHash .\build\lfs_core.dll -Algorithm SHA256
```

Update Defender's security intelligence and rescan the trusted build. If the
detection persists, submit the affected binary to
[Microsoft Security Intelligence](https://www.microsoft.com/en-us/wdsi/filesubmission)
for review. Maintainers should select **Software developer** and include the
detection name, version or commit, and build configuration. A screenshot alone
cannot confirm a false positive.

Windows builds embed product and version information in the executable, DLLs,
and Python module, following the metadata mitigation used in
[Microsoft APM](https://github.com/microsoft/apm/issues/487). This metadata does
not provide an Authenticode signature or guarantee that Defender will accept a
particular binary.
