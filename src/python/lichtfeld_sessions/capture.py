# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Owned CPU snapshots captured only at a trainer's model read boundary."""
def spatial_indices(positions, limit):
    """Stratify occupied cells instead of following Gaussian storage order."""
    import numpy as np
    count = len(positions)
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    low = positions.min(axis=0)
    extent = np.maximum(positions.max(axis=0) - low, 1e-8)
    resolution = max(2, int(round(limit ** (1 / 3))))
    for _ in range(5):
        cells = np.minimum(((positions - low) / extent * resolution).astype(np.int64), resolution - 1)
        keys = cells[:, 0] + resolution * (cells[:, 1] + resolution * cells[:, 2])
        _, chosen = np.unique(keys, return_index=True)
        if len(chosen) >= limit:
            return chosen[np.linspace(0, len(chosen) - 1, limit, dtype=np.int64)]
        resolution *= 2
    remaining = np.setdiff1d(np.arange(count), chosen, assume_unique=True)
    fill = remaining[np.linspace(0, len(remaining) - 1, limit - len(chosen), dtype=np.int64)]
    return np.concatenate((chosen, fill))


class PreviewCapacityError(ValueError):
    """Requested preview exceeds its transport budget before attribute copies."""


def capture(lf, *, limit=None, degree=None, max_bytes=None):
    import numpy as np

    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Preview point limit must be a positive integer")
    if degree is not None and (type(degree) is not int or not 0 <= degree <= 3):
        raise ValueError("Preview SH degree must be between 0 and 3")
    model = lf.get_scene().training_model()
    if model is None:
        raise RuntimeError("No model is available at the training safe boundary")
    count = model.num_points
    indices = None
    if model.has_deleted_mask():
        mask = model.deleted.cpu()
        mask.sync()
        indices = np.flatnonzero(~mask.numpy().reshape(-1).astype(bool)).astype(np.int64)
        if len(indices) == count:
            indices = None
    active_count = len(indices) if indices is not None else count
    active = min(model.active_sh_degree, degree) if degree is not None else model.active_sh_degree
    output_count = min(active_count, limit) if limit is not None else active_count
    if max_bytes is not None and output_count * (11 + 3 * (active + 1) ** 2) * 4 > max_bytes:
        raise PreviewCapacityError("Full preview exceeds the shared-memory limit.")
    if not output_count:
        raise ValueError("No active Gaussians are available")
    if limit is not None and active_count > limit:
        samples = np.linspace(0, active_count - 1, min(active_count, limit * 4), dtype=np.int64)
        candidates = indices[samples] if indices is not None else samples
        positions = model.means_raw.index_select(0, lf.Tensor.from_numpy(candidates).cuda()).cpu()
        positions.sync()
        indices = candidates[spatial_indices(positions.numpy(), limit)]
    cpu_selection = lf.Tensor.from_numpy(indices) if indices is not None else None
    selection = cpu_selection.cuda() if cpu_selection is not None else None

    def owned(tensor):
        result = (tensor.index_select(0, selection) if selection is not None else tensor).cpu()
        result.sync()
        return result.numpy()

    dc = owned(model.sh0_raw).reshape(output_count, 1, 3)
    if active:
        rows = cpu_selection if cpu_selection is not None else lf.Tensor.from_numpy(np.arange(count, dtype=np.int64))
        rest = model.shN_rows(rows, active).numpy()
        sh = np.concatenate((dc, rest), axis=1)
    else:
        sh = dc
    return {"means": owned(model.means_raw), "scale": owned(model.scaling_raw),
            "rotation": owned(model.rotation_raw), "opacity": owned(model.opacity_raw), "sh": sh}
