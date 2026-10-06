# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run with runpy.run_path(..., run_name='__main__') in a live viewer editor.

Requires the running viewer; headless Python imports cannot publish scene updates.
Only the uniquely named node created by this script is removed at the end.
"""

import gc
import time
import uuid
import weakref

import lichtfeld as lf
import numpy as np


def published(ticket):
    deadline = time.monotonic() + 20
    while ticket.state in ("queued", "uploading") and time.monotonic() < deadline:
        time.sleep(0.001)
    assert ticket.state == "published", (ticket.state, ticket.error)
    assert ticket.inputs_released


def main():
    scene = lf.get_scene()
    name = f"async-api-test-{uuid.uuid4()}"
    scene.add_point_cloud(
        name, lf.Tensor.zeros([2, 3], device="cpu"), lf.Tensor.zeros([2, 3], device="cpu")
    )
    try:
        node = scene.get_node(name)
        original = node.point_cloud()
        points = np.full((1_000_000, 3), 0.25, dtype=np.float32)
        colors = np.full((1_000_000, 3), 128, dtype=np.uint8)
        owner = weakref.ref(points)
        started = time.perf_counter()
        ticket = original.set_data_async(points, colors)
        print("Python submit_ms:", (time.perf_counter() - started) * 1000)
        del points, colors
        published(ticket)
        gc.collect()
        assert owner() is None
        current = scene.get_node_by_uuid(node.uuid).point_cloud()
        assert current.size == 1_000_000
        assert original.size == 2
        assert node.centroid == (0.25, 0.25, 0.25)
        assert np.allclose(current.colors.numpy(), 128 / 255)

        gpu_ticket = original.set_data_async(
            lf.Tensor.ones([4096, 3]), lf.Tensor.zeros([4096, 3]), centroid=(1, 1, 1)
        )
        published(gpu_ticket)
        assert node.point_cloud().size == 4096
        assert node.centroid == (1, 1, 1)

        for points in (
            np.zeros((2, 4), np.float32),
            np.zeros((2, 3), np.float64),
            np.zeros((3, 3), np.float32),
            np.zeros((2, 6), np.float32)[:, ::2],
        ):
            try:
                original.set_data_async(points, np.zeros((2, 3), np.float32))
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("Invalid metadata accepted")

        bad = original.set_data_async(
            np.zeros((2, 3), np.float32), np.full((2, 3), 2, np.float32)
        )
        deadline = time.monotonic() + 20
        while bad.state in ("queued", "uploading") and time.monotonic() < deadline:
            time.sleep(0.001)
        assert bad.state == "failed", (bad.state, bad.error)
        assert bad.error
        assert node.point_cloud().size == 4096

        empty = original.set_data_async(
            np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
        )
        published(empty)
        assert node.point_cloud().size == 0
        assert node.centroid == (0, 0, 0)
        print("LIVE_POINT_CLOUD_UPDATES_PASS")
    finally:
        scene.remove_node(name)


if __name__ == "__main__":
    main()
