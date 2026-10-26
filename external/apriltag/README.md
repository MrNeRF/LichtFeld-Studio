# Vendored AprilTag

Upstream: https://github.com/AprilRobotics/apriltag

Verified upstream baseline: `b7c0ebe9aa20f82ec7a828579004f9e706bfecd9`.
Upstream version: **3.4.5**, from that revision's
[CMakeLists.txt](https://github.com/AprilRobotics/apriltag/blob/b7c0ebe9aa20f82ec7a828579004f9e706bfecd9/CMakeLists.txt).

The original import did not retain its commit or upstream CMakeLists.txt.
This baseline was established by comparing Git blob hashes for every retained
upstream file: 67 files match exactly; the remaining header differs only as
listed below. This identifies reproducible source contents, not the historical
checkout used by the original importer.

Local modifications and integration:

- `common/pthreads_cross.h`: define `WIN32_LEAN_AND_MEAN` and `NOMINMAX`, if
  absent, before including `windows.h`.
- `../CMakeLists.txt`: build the selected sources as the private static,
  position-independent C99 target `lfs_apriltag`, with platform compile
  definitions and thread/math linkage. Upstream build scripts, examples,
  Python bindings and upstream tests are not included.
- Retain all six supported families: tag16h5, tag25h9, tag36h10, tag36h11,
  tagCircle21h7 and tagStandard41h12. Other upstream family tables are omitted.
- `common/LICENSE.pthreads_cross`: reproduce the MIT notice from the threading
  shim's source/header in a file discovered by the release notice generator.
- This README records the verified baseline and local changes. No detection,
  pose-estimation, or family-table implementation changes have been made.

The main BSD notice is in `LICENSE.md`. The threading shim is separately
licensed under MIT; ship `common/LICENSE.pthreads_cross` as well. The release
notice generator (`tools/generate_third_party_licenses.py`) discovers both.
