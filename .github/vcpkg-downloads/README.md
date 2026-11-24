# vcpkg downloads kept in the repository

Some vcpkg ports download a patch from GitHub's commit endpoint
(`https://github.com/<owner>/<repo>/commit/<sha>.patch?full_index=1`) while they build. That
endpoint is rate limited and can answer 503 for hours, which fails every build that has no warm
vcpkg cache. The workflows copy the files in this folder into vcpkg's downloads directory before
configuring. vcpkg finds them there, checks them against the hash in the port and does not ask
GitHub.

| File | Port | Version |
|---|---|---|
| `assimp_pr_6627.patch` | assimp | 6.0.5 |
| `draco-b283ba36eea52dc525ab4c9324ae6de2055d1191.patch` | draco | 1.5.7 |
| `fmt-backport-4813.patch` | fmt | 12.2.0 |
| `spdlog-backport-3541.patch`, `spdlog-backport-3543.patch` | spdlog | 1.17.0 |

After moving `builtin-baseline` in `vcpkg.json`, look for ports in the dependency closure whose
`portfile.cmake` calls `vcpkg_download_distfile` with such a URL, and add the file under the
`FILENAME` the port gives it. A file no port asks for any more is harmless and can be deleted.
