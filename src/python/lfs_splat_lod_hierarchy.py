# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers for building multi-level splat simplify hierarchies in Python."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import lichtfeld as lf


_HIERARCHY_FORMAT = "lichtfeld.splat_lod_hierarchy/v1"
_FLOAT32_EXACT_INT_LIMIT = 1 << 24


def _coerce_progress_result(result: Any) -> bool:
    if isinstance(result, bool):
        return result
    return True


def _call_progress(callback: Callable[[float, str], Any] | None, value: float, stage: str) -> bool:
    if callback is None:
        return True
    return _coerce_progress_result(callback(float(value), stage))


def _wrap_stage_progress(
    callback: Callable[[float, str], Any] | None,
    stage_prefix: str,
) -> Callable[[float, str], bool] | None:
    if callback is None:
        return None

    def _wrapped(value: float, stage: str) -> bool:
        return _call_progress(callback, value, f"{stage_prefix}: {stage}")

    return _wrapped


def _visible_count(splat_data: Any) -> int:
    if hasattr(splat_data, "visible_count"):
        try:
            return int(splat_data.visible_count())
        except TypeError:
            pass
    return int(splat_data.num_points)


def _as_int_list(values: Any) -> list[int]:
    return [int(value) for value in values]


def _resolve_scene_source(source: Any) -> tuple[Any, str | None, int | None]:
    import lichtfeld as lf

    if source is None:
        if not hasattr(lf, "get_selected_node_name"):
            raise ValueError("No source provided and lichtfeld.get_selected_node_name() is unavailable")
        selected_name = str(lf.get_selected_node_name() or "").strip()
        if not selected_name:
            raise ValueError("No source provided and no scene node is currently selected")
        return _resolve_scene_source(selected_name)

    if hasattr(source, "num_points") and not hasattr(source, "splat_data"):
        return source, None, None

    node = None
    if isinstance(source, str):
        if not hasattr(lf, "get_scene"):
            raise ValueError("Scene lookup requested but lichtfeld.get_scene() is unavailable")
        scene = lf.get_scene()
        if scene is None:
            raise ValueError("No active scene available")
        if hasattr(scene, "is_valid") and not scene.is_valid():
            raise ValueError("Scene reference is no longer valid")
        node = scene.get_node(source)
        if node is None:
            raise ValueError(f"No scene node named '{source}'")
    elif hasattr(source, "splat_data"):
        node = source
    else:
        raise TypeError(
            "source must be a SplatData value, a scene node, a node name, or None "
            "(to use the current scene selection)"
        )

    splat_data = node.splat_data()
    if splat_data is None:
        node_name = getattr(node, "name", "<unknown>")
        raise ValueError(f"Scene node '{node_name}' has no splat data")

    node_name = getattr(node, "name", None)
    node_id = getattr(node, "id", None)
    return splat_data, None if node_name is None else str(node_name), None if node_id is None else int(node_id)


def _target_count_for(current_count: int, ratio: float, min_points: int) -> int:
    return max(
        1,
        min(
            current_count,
            max(min_points, int(math.ceil(float(current_count) * float(ratio)))),
        ),
    )


def _node_ids_to_extra_attribute(node_ids: list[int], allow_lossy: bool) -> Any:
    import numpy

    max_node_id = max(node_ids) if node_ids else 0
    if max_node_id >= _FLOAT32_EXACT_INT_LIMIT and not allow_lossy:
        raise ValueError(
            "PLY extra attributes are written as float32, so node ids above "
            f"{_FLOAT32_EXACT_INT_LIMIT - 1} cannot round-trip exactly. "
            "Keep the JSON hierarchy as authoritative or set allow_lossy_node_ids=True."
        )
    return numpy.asarray(node_ids, dtype=numpy.float32)


@dataclass
class SplatLodLevel:
    """One materialized LOD level in the hierarchy."""

    lod_level: int
    splat_data: Any
    row_node_ids: list[int]
    source_count: int
    requested_ratio: float
    target_count: int
    post_prune_count: int
    pruned_input_node_ids: list[int] = field(default_factory=list)
    new_merge_node_ids: list[int] = field(default_factory=list)
    ply_path: str | None = None

    @property
    def count(self) -> int:
        return len(self.row_node_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lod_level": int(self.lod_level),
            "count": int(self.count),
            "source_count": int(self.source_count),
            "requested_ratio": float(self.requested_ratio),
            "target_count": int(self.target_count),
            "post_prune_count": int(self.post_prune_count),
            "row_node_ids": list(self.row_node_ids),
            "pruned_input_node_ids": list(self.pruned_input_node_ids),
            "new_merge_node_ids": list(self.new_merge_node_ids),
            "ply_path": self.ply_path,
        }


@dataclass
class SplatLodHierarchy:
    """Exact multi-level hierarchy built from repeated splat simplify calls."""

    source_num_points: int
    source_visible_count: int
    source_node_name: str | None
    source_node_id: int | None
    ratio: float
    lod_base: float
    opacity_prune_threshold: float
    max_levels: int | None
    min_points: int
    levels: list[SplatLodLevel] = field(default_factory=list)
    merge_node_ids: list[int] = field(default_factory=list)
    merge_children: list[list[int]] = field(default_factory=list)
    created_lod: list[int] = field(default_factory=list)
    created_pass: list[int] = field(default_factory=list)

    @property
    def leaf_count(self) -> int:
        return int(self.source_visible_count)

    @property
    def node_count(self) -> int:
        return self.leaf_count + len(self.merge_node_ids)

    @property
    def final_row_node_ids(self) -> list[int]:
        if not self.levels:
            return []
        return list(self.levels[-1].row_node_ids)

    def is_leaf(self, node_id: int) -> bool:
        return int(node_id) < self.leaf_count

    def children(self, node_id: int) -> list[int] | None:
        node_id = int(node_id)
        if node_id < self.leaf_count:
            return None
        merge_offset = node_id - self.leaf_count
        if merge_offset < 0 or merge_offset >= len(self.merge_children):
            raise KeyError(f"Unknown node id: {node_id}")
        return list(self.merge_children[merge_offset])

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": _HIERARCHY_FORMAT,
            "source_num_points": int(self.source_num_points),
            "source_visible_count": int(self.source_visible_count),
            "source_node_name": self.source_node_name,
            "source_node_id": self.source_node_id,
            "leaf_count": int(self.leaf_count),
            "node_count": int(self.node_count),
            "ratio": float(self.ratio),
            "lod_base": float(self.lod_base),
            "opacity_prune_threshold": float(self.opacity_prune_threshold),
            "max_levels": None if self.max_levels is None else int(self.max_levels),
            "min_points": int(self.min_points),
            "levels": [level.to_dict() for level in self.levels],
            "merge_nodes": {
                "node_ids": list(self.merge_node_ids),
                "children": [list(c) for c in self.merge_children],
                "created_lod": list(self.created_lod),
                "created_pass": list(self.created_pass),
            },
        }

    def save_json(self, path: str | Path) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return str(output_path)

    def save_ply_levels(
        self,
        output_dir: str | Path,
        *,
        base_name: str = "splat",
        include_level0: bool = True,
        binary: bool = True,
        node_id_attribute: str | None = None,
        allow_lossy_node_ids: bool = False,
        progress: Callable[[float, str], Any] | None = None,
    ) -> list[str]:
        import lichtfeld as lf

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        written_paths: list[str] = []
        start_index = 0 if include_level0 else 1
        levels_to_write = self.levels[start_index:]
        total_levels = len(levels_to_write)
        for save_index, level in enumerate(levels_to_write):
            output_path = root / f"{base_name}_lod{level.lod_level}.ply"
            extra_attributes = None
            if node_id_attribute:
                extra_attributes = {
                    str(node_id_attribute): _node_ids_to_extra_attribute(
                        level.row_node_ids,
                        allow_lossy=allow_lossy_node_ids,
                    )
                }

            save_progress = None
            if progress is not None:
                level_prefix = f"Save LOD {level.lod_level}"

                def _level_progress(
                    value: float,
                    stage: str,
                    *,
                    _save_index: int = save_index,
                    _level_prefix: str = level_prefix,
                ) -> bool:
                    if total_levels > 0:
                        overall = (_save_index + float(value)) / float(total_levels)
                    else:
                        overall = 1.0
                    return _call_progress(progress, overall, f"{_level_prefix}: {stage}")

                save_progress = _level_progress

            lf.io.save_ply(
                level.splat_data,
                str(output_path),
                binary=binary,
                progress=save_progress,
                extra_attributes=extra_attributes,
            )
            level.ply_path = str(output_path)
            written_paths.append(str(output_path))
        return written_paths

    def save(
        self,
        output_dir: str | Path,
        *,
        base_name: str = "splat",
        include_level0: bool = True,
        binary: bool = True,
        node_id_attribute: str | None = None,
        allow_lossy_node_ids: bool = False,
        progress: Callable[[float, str], Any] | None = None,
        hierarchy_filename: str | None = None,
    ) -> dict[str, Any]:
        root = Path(output_dir)
        ply_paths = self.save_ply_levels(
            root,
            base_name=base_name,
            include_level0=include_level0,
            binary=binary,
            node_id_attribute=node_id_attribute,
            allow_lossy_node_ids=allow_lossy_node_ids,
            progress=progress,
        )

        json_name = hierarchy_filename or f"{base_name}_hierarchy.json"
        json_path = self.save_json(root / json_name)
        return {
            "json_path": json_path,
            "ply_paths": ply_paths,
        }


def build_splat_lod_hierarchy(
    source: Any = None,
    ratio: float = 0.5,
    lod_base: float = 2.0,
    opacity_prune_threshold: float = 0.1,
    max_levels: int | None = None,
    min_points: int = 1,
    progress: Callable[[float, str], Any] | None = None,
) -> SplatLodHierarchy:
    """Build a multi-level simplify hierarchy from a source splat or scene node."""

    import lichtfeld as lf

    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("ratio must be in the range (0, 1]")
    if lod_base <= 1.0:
        raise ValueError("lod_base must be > 1")
    if min_points < 1:
        raise ValueError("min_points must be at least 1")
    if max_levels is not None and max_levels < 1:
        raise ValueError("max_levels must be at least 1 when provided")

    source, source_node_name, source_node_id = _resolve_scene_source(source)
    source_num_points = int(source.num_points)
    source_visible_count = _visible_count(source)
    if source_visible_count <= 0:
        raise ValueError("source splat has no visible gaussians")

    hierarchy = SplatLodHierarchy(
        source_num_points=source_num_points,
        source_visible_count=source_visible_count,
        source_node_name=source_node_name,
        source_node_id=source_node_id,
        ratio=float(ratio),
        lod_base=float(lod_base),
        opacity_prune_threshold=float(opacity_prune_threshold),
        max_levels=None if max_levels is None else int(max_levels),
        min_points=int(min_points),
        levels=[
            SplatLodLevel(
                lod_level=0,
                splat_data=source,
                row_node_ids=list(range(source_visible_count)),
                source_count=source_visible_count,
                requested_ratio=1.0,
                target_count=source_visible_count,
                post_prune_count=source_visible_count,
            )
        ],
    )

    next_node_id = source_visible_count
    target_lod_level = 1
    while True:
        if max_levels is not None and len(hierarchy.levels) >= max_levels:
            break

        previous_level = hierarchy.levels[-1]
        current_count = previous_level.count
        if current_count <= min_points:
            break

        target_count = _target_count_for(current_count, ratio, min_points)
        simplify_progress = _wrap_stage_progress(progress, f"LOD {target_lod_level}")
        result = lf.simplify_splat_data_with_history(
            previous_level.splat_data,
            ratio=float(target_count) / float(current_count),
            lod_base=lod_base,
            opacity_prune_threshold=opacity_prune_threshold,
            progress=simplify_progress,
        )

        tree = result.merge_tree
        leaf_count = int(tree.leaf_count())
        if leaf_count != current_count:
            raise RuntimeError(
                f"Unexpected simplify leaf count for LOD {target_lod_level}: "
                f"expected {current_count}, got {leaf_count}"
            )

        local_to_global = list(previous_level.row_node_ids)
        pruned_leaf_ids = _as_int_list(tree.pruned_leaf_ids)
        for leaf_id in pruned_leaf_ids:
            if leaf_id < 0 or leaf_id >= leaf_count:
                raise RuntimeError(f"Invalid pruned leaf id {leaf_id} in LOD {target_lod_level}")
        pruned_input_node_ids = [previous_level.row_node_ids[leaf_id] for leaf_id in pruned_leaf_ids]

        merge_left = _as_int_list(tree.merge_left)
        merge_right = _as_int_list(tree.merge_right)
        merge_pass = _as_int_list(tree.merge_pass)
        if not (len(merge_left) == len(merge_right) == len(merge_pass)):
            raise RuntimeError(f"Inconsistent merge arrays in LOD {target_lod_level}")

        new_merge_node_ids = list(range(next_node_id, next_node_id + len(merge_left)))
        next_node_id += len(new_merge_node_ids)
        local_to_global.extend(new_merge_node_ids)

        max_local_node_id = len(local_to_global) - 1
        for merge_index, node_id in enumerate(new_merge_node_ids):
            left_local = merge_left[merge_index]
            right_local = merge_right[merge_index]
            if left_local < 0 or left_local > max_local_node_id:
                raise RuntimeError(f"Invalid merge left child id {left_local} in LOD {target_lod_level}")
            if right_local < 0 or right_local > max_local_node_id:
                raise RuntimeError(f"Invalid merge right child id {right_local} in LOD {target_lod_level}")

            hierarchy.merge_node_ids.append(int(node_id))
            hierarchy.merge_children.append(
                [int(local_to_global[left_local]), int(local_to_global[right_local])]
            )
            hierarchy.created_lod.append(int(target_lod_level))
            hierarchy.created_pass.append(int(merge_pass[merge_index]))

        final_roots = _as_int_list(tree.final_roots)
        final_row_node_ids: list[int] = []
        for local_node_id in final_roots:
            if local_node_id < 0 or local_node_id > max_local_node_id:
                raise RuntimeError(f"Invalid final root id {local_node_id} in LOD {target_lod_level}")
            final_row_node_ids.append(int(local_to_global[local_node_id]))

        if len(final_row_node_ids) != _visible_count(result.splat_data):
            raise RuntimeError(
                f"Output row count mismatch in LOD {target_lod_level}: "
                f"tree has {len(final_row_node_ids)} rows but splat has {_visible_count(result.splat_data)}"
            )

        if final_row_node_ids == previous_level.row_node_ids:
            break

        hierarchy.levels.append(
            SplatLodLevel(
                lod_level=target_lod_level,
                splat_data=result.splat_data,
                row_node_ids=final_row_node_ids,
                source_count=current_count,
                requested_ratio=float(tree.requested_ratio),
                target_count=int(tree.target_count),
                post_prune_count=int(tree.post_prune_count),
                pruned_input_node_ids=pruned_input_node_ids,
                new_merge_node_ids=new_merge_node_ids,
            )
        )
        target_lod_level += 1

    _call_progress(progress, 1.0, "Hierarchy complete")
    return hierarchy


__all__ = [
    "SplatLodLevel",
    "SplatLodHierarchy",
    "build_splat_lod_hierarchy",
]
