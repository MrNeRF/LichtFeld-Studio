# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import lichtfeld as lf

scene = lf.get_scene()
def tensor(array):
    return lf.Tensor.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).cuda()
tensors = (tensor([[0, 0, 0], [1, 0, 0]]), tensor(np.zeros((2, 1, 3))),
           tensor(np.empty((2, 0, 3))), tensor(np.full((2, 3), -2)),
           tensor([[1, 0, 0, 0], [1, 0, 0, 0]]), tensor(np.zeros((2, 1))))
name = "Renderer guard fixture"
scene.add_splat(name, *tensors, sh_degree=0)
before = scene.get_node(name).uuid
try:
    try:
        scene.add_splat(name, *tensors, sh_degree=0, transient=True, replace_existing=True)
    except RuntimeError as error:
        assert "transient" in str(error), error
    else:
        raise AssertionError("Saved scene node was overwritten")
    assert scene.get_node(name).uuid == before
    print("SAVED_NODE_REPLACEMENT_REJECTED_AND_NATIVE_SH0_OK")
finally:
    scene.remove_node(name)
