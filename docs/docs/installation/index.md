---
sidebar_position: 2
---

# Installation

Windows binaries are available from the project's [GitHub Releases](https://github.com/MrNeRF/LichtFeld-Studio/releases).

For source builds and platform-specific instructions, see the [Wiki](https://github.com/MrNeRF/LichtFeld-Studio/wiki/) and the repo-local docs in [docs/README.md](../../README.md).

## NVIDIA video export

Builds using the repository's FFmpeg 9.0.2 and NVENC API 13.0 headers require
NVIDIA driver **570.0 or newer on Windows and Linux** for NVENC H.264/HEVC
video export. The GPU must support the selected codec and resolution; CUDA or
the GPU model may require a newer driver. WSL uses the Windows host driver.
This is the codec API's existing driver minimum, not an additional increase
introduced by FFmpeg 9. macOS exports use VideoToolbox instead.

H.264 retains a software fallback when available. HEVC export fails explicitly
when its provider is unavailable; it does not silently switch to H.264.
Custom builds using newer NVIDIA codec headers can require newer drivers; see
the [pinned NVIDIA codec header requirements](https://github.com/FFmpeg/nv-codec-headers/blob/n13.0.19.0/README).

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
