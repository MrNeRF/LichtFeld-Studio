# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read only the private, numeric staging files produced by native scene capture."""
import json
from pathlib import Path
import re
import stat
import uuid

from . import gallery_validation


def staging_path(root, value):
    root, path = Path(root).absolute(), Path(value).absolute()
    if path.parent != root or path.suffix != ".scene":
        raise ValueError("Scene preparation is outside its temporary folder.")
    try:
        if str(uuid.UUID(path.stem)) != path.stem:
            raise ValueError()
    except ValueError:
        raise ValueError("Scene preparation has an unexpected name.") from None
    for entry in (root, path):
        if entry.is_symlink() or getattr(entry, "is_junction", lambda: False)():
            raise ValueError("Scene preparation was redirected. Keep it for recovery.")
    return path


def staging_files(root, value):
    """Validate the whole deletion plan before removing anything; never recurse."""
    path = staging_path(root, value)
    if not path.exists():
        return []
    files = []
    for entry in path.iterdir():
        if len(files) >= gallery_validation.MAX_NODES + 5:
            raise ValueError("Scene preparation contains too many files.")
        if (entry.name not in ("manifest.json", "manifest.json.tmp", "environment.lfsenv", "project.licht", "project.licht.lock")
                and not re.fullmatch(r"(?:0|[1-9][0-9]{0,3})\.(?:ply|sog|ssog|spz)", entry.name)):
            raise ValueError("Scene preparation contains an unexpected file. Keep it for recovery.")
        if not stat.S_ISREG(entry.lstat().st_mode) or getattr(entry, "is_junction", lambda: False)():
            raise ValueError("Scene preparation contains a redirected file. Keep it for recovery.")
        files.append(entry)
    return [*files, path]


def read_staging(root, value):
    path = staging_path(root, value)
    staging_files(root, path)
    with (path / "manifest.json").open("rb") as source:
        encoded = source.read(gallery_validation.MAX_MANIFEST_BYTES + 1)
    if len(encoded) > gallery_validation.MAX_MANIFEST_BYTES:
        raise ValueError("Scene preparation metadata is too large.")
    data = json.loads(encoded, object_pairs_hook=gallery_validation._object, parse_constant=gallery_validation._constant)
    if (not isinstance(data, dict) or data.keys() not in ({"version", "nodes"}, {"version", "nodes", "environment"})
            or type(data["version"]) is not int or data["version"] != 1
            or not isinstance(data["nodes"], list) or not 0 < len(data["nodes"]) <= gallery_validation.MAX_NODES):
        raise ValueError("Scene preparation metadata is invalid. Prepare the scene again.")
    nodes, total = [], 0
    for number, node in enumerate(data["nodes"]):
        if (not isinstance(node, dict) or node.keys() not in
                ({"path", "transform", "shDegree"}, {"path", "name", "transform", "shDegree"})
                or node["path"] not in (f"{number}.ply", f"{number}.sog", f"{number}.ssog", f"{number}.spz")):
            raise ValueError("Scene preparation has an invalid node path.")
        name = node.get("name", "")
        if not isinstance(name, str) or len(name.encode("utf-8")) > 4096 or "\0" in name:
            raise ValueError("Scene preparation has an invalid node name.")
        source = path / node["path"]
        total += source.stat().st_size
        nodes.append({"path": source, "transform": gallery_validation._matrix(node["transform"]),
                      "shDegree": gallery_validation._degree(node["shDegree"]), "name": name})
    background = path / "environment.lfsenv"
    if "environment" in data:
        if data["environment"] != background.name:
            raise ValueError("Scene preparation has an invalid HDR background path.")
        size = background.stat().st_size
        with background.open("rb") as source:
            gallery_validation.environment_header(source, size)
        total += size
    elif background.exists():
        raise ValueError("Scene preparation has an unreferenced HDR background.")
    if (path / "project.licht").exists():
        total += (path / "project.licht").stat().st_size
    return nodes, total




def unpack_project(root, source, destination, *, progress=None):
    """Prepare embedded splats for merging; keep the .licht source authoritative."""
    from .portable_project import ProjectFile
    destination = staging_path(root, destination)
    destination.mkdir(mode=0o700)
    outputs = []
    try:
        with Path(source).open('rb') as stream:
            project = ProjectFile(stream)
            total = sum(asset['size'] for asset in project.assets.values())
            completed, nodes = 0, []
            for index, node in enumerate(project.manifest['nodes']):
                path = destination / (str(index) + Path(node['file']).suffix)
                outputs.append(path)
                with path.open('xb') as output:
                    completed += project.copy_node(index, output, progress=(lambda value: progress(completed + value, total)) if progress else None)
                nodes.append({'path': path.name, 'transform': node['transform'], 'shDegree': node['shDegree'],
                              'name': project.chapters[b'SCNG']['nodes'][index]['name']})
            metadata = {'version': 1, 'nodes': nodes}
            if 'environment' in project.manifest:
                path = destination / 'environment.lfsenv'
                outputs.append(path)
                with path.open('xb') as output:
                    completed += project.copy_environment(output, progress=(lambda value: progress(completed + value, total)) if progress else None)
                metadata['environment'] = path.name
            marker = destination / 'manifest.json'
            outputs.append(marker)
            with marker.open('x') as output:
                json.dump(metadata, output, allow_nan=False)
            if progress: progress(completed, total)
    except Exception:
        for path in outputs: path.unlink(missing_ok=True)
        destination.rmdir()
        raise


def publication_view_metadata(root, value):
    """The native file owns VIEW/SEQR; only mirror HDR for the packaging guard."""
    from .portable_project import ProjectFile
    path = staging_path(root, value)
    staging_files(root, path)
    with (path / "project.licht").open("rb") as source:
        project = ProjectFile(source)
        if "environment" not in project.manifest:
            return {}
        render = project.chapters[b"VIEW"]["render_settings"]
        return {"environment": {"exposure": render["environment_exposure"],
                                "rotation": render["environment_rotation_degrees"]}}
