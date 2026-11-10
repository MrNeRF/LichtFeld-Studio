# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, off-thread metadata reads. Results are published by the panel's UI tick."""

import os
from concurrent.futures import ThreadPoolExecutor
from _lfs_path_io import native_path


class SourceChangedError(RuntimeError):
    """The sampled file revision no longer matches the displayed image."""


def _read(path, image_info, previous):
    try:
        st = os.stat(native_path(path))
        stat = (True, st.st_mtime_ns, st.st_size, st.st_dev, st.st_ino, st.st_ctime_ns)
    except OSError:
        return (False, 0, 0, 0, 0, 0), (0, 0, 0)
    if (previous and previous[0] == stat and previous[1] is not None
            and previous[1][0] > 0 and previous[1][1] > 0):
        return stat, previous[1] if image_info else None
    try:
        info = image_info(str(path)) if image_info else None
    except Exception:
        info = (0, 0, 0)
    return stat, info


def _sample(path, expected, x, y, sampler):
    if expected and _read(path, None, None)[0] != expected:
        raise SourceChangedError
    color = sampler(str(path), x, y, 10)
    if expected and _read(path, None, None)[0] != expected:
        raise SourceChangedError
    return tuple(color)


class PreviewMetadata:
    """At most one IO request, with no executor backlog across fast navigation."""

    def __init__(self):
        self._executor = None
        self._pending = None
        self._epoch = 0
        self._seen = set()

    def reset(self):
        self._epoch += 1
        self._seen.clear()

    def tick(self, wanted, image_info, cached=None):
        result = None
        if self._pending:
            epoch, path, full, future = self._pending
            if not future.done():
                return None, True
            self._pending = None
            stat, info = future.result()
            if epoch == self._epoch:
                result = (path, stat, info)
        for path, full in wanted:
            key = (str(path), full)
            if key in self._seen:
                continue
            self._seen.add(key)
            if full:
                self._seen.add((str(path), False))
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview-metadata")
            future = self._executor.submit(_read, path, image_info if full else None,
                                           cached.get(str(path)) if cached else None)
            self._pending = (self._epoch, path, full, future)
            break
        return result, self._pending is not None

    def close(self):
        self.reset()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
        self._pending = None

    def sample(self, path, expected, x, y, sampler):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview-metadata")
        return self._executor.submit(_sample, path, expected, x, y, sampler)
