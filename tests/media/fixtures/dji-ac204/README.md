# DJI AC204 packet fixture

`ac204_djmd_first300.bin` contains 300 djmd packet payloads extracted by
DaniilDem from an Osmo Action 5 Pro clip, `DJI_20260916095953_0098_D.MP4`.
Each payload is preceded by its unsigned 32-bit big-endian byte count.

Source: https://github.com/DaniilDem/dji-telemetry-export/tree/50daf74bf1b2ab5687c4375f39349e49cd78a6e2
Original path: `tests/fixtures/ac204_djmd_first300.bin`
SHA-256: `dbb61977c99b05832cbe753d3ca0895aa0fc136ce7a3cc189d5514a1dea3a62f`
License: MIT, reproduced in LICENSE.

The corpus identifies `dvtm_ac204.proto` version `2.0.1`. All recorded GPS
statuses are 1. It does not contain the original MP4, packet PTS/DTS, a valid
GPS-fix sample or independent calibration of units, axes or clocks. Synthetic
carrier timings in tests are test inputs, not timings recovered from this clip.
