# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read only the private, numeric staging files produced by native scene capture."""
import json
from pathlib import Path
import re
import stat
import uuid

from . import gallery_validation
from .gallery_logging import failure as log_failure


def staging_path(root, value):
    root, path = Path(root).absolute(), Path(value).absolute()
    if root.is_symlink() or getattr(root, "is_junction", lambda: False)():
        raise ValueError("Scene preparation was redirected. Keep it for recovery.")
    root = root.resolve()
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError("Scene preparation was redirected. Keep it for recovery.")
    path = path.resolve()
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
        if (entry.name not in ("manifest.json", "manifest.json.tmp", "environment.lfsenv", "project.licht", "project.licht.lock", "project.licht.preview.tmp")
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
    except Exception as exc:
        log_failure("download_preparation", exc, source=source, destination=destination)
        for path in outputs: path.unlink(missing_ok=True)
        destination.rmdir()
        raise


def publication_view_metadata(root, value):
    """Read the reviewed viewing settings from the exact prepared commit."""
    import math
    from .portable_project import ProjectFile
    from .gallery_view import TONEMAPPING, viewer_vector
    path = staging_path(root, value)
    staging_files(root, path)
    with (path / "project.licht").open("rb") as source:
        project = ProjectFile(source)
        view = project.chapters[b"VIEW"]
        render = view["render_settings"]
        panel = next(item for item in view["panel_cameras"] if item["panel"] == "primary")
        up = viewer_vector(panel["R"][3:6])
        length = math.hypot(*up)
        if not math.isfinite(length) or length < 1e-8:
            raise ValueError("The saved camera orientation is invalid.")
        vertical = "long_axis_fov_degrees" not in view
        fov = (2 * math.degrees(math.atan(12 / render["focal_length_mm"])) if vertical
               else view["long_axis_fov_degrees"])
        camera = dict(position=viewer_vector(panel["t"]), target=viewer_vector(panel["pivot"]),
                      up=[component / length for component in up], fov=fov)
        if render.get("orthographic"):
            camera.update(projection="orthographic", orthoScale=panel["ortho_extent_world"])
        result = dict(camera=camera, verticalFov=vertical,
                      exposure=render.get("color_exposure", 1.0),
                      tonemapping=TONEMAPPING[render.get("color_tonemapping", 0)],
                      renderProfile="studio" if render.get("splat_render_profile", 0) == 0 else "standard",
                      shDegree=render["sh_degree"], antialiasing=render["mip_filter"],
                      renderMode=render["raster_backend"], background=render["background_color"])
        sequencer = project.chapters.get(b"SEQR", {})
        timeline = sequencer.get("timeline", {})
        if timeline.get("keyframes"):
            result["cameraPath"] = dict(version=1, duration=timeline["clip_duration"],
                                       loopMode=sequencer["loop_mode"], playbackSpeed=sequencer["playback_speed"],
                                       keyframes=timeline["keyframes"])
        if "environment" in project.manifest:
            result["environment"] = {"exposure": render["environment_exposure"],
                                     "rotation": render["environment_rotation_degrees"]}
        return result


def attach_preview(path, png, *, cancel=None):
    """Add THMB to the fresh publishing subset, preserving every scene payload.

    Both inputs and the completed container are validated. The temporary file
    replaces only the private prepared copy, after it has been flushed to disk.
    """
    import os
    import struct
    from .portable_project import ProjectFile, _crc
    from .portal_gallery import GalleryTransferCanceled, disk_preflight

    path = Path(path)
    temporary = path.with_name("project.licht.preview.tmp")
    if path.name != "project.licht" or path.is_symlink() or temporary.is_symlink():
        raise ValueError("The prepared project was redirected.")
    if (not isinstance(png, bytes) or not 33 <= len(png) <= 2 * 1024**2
            or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR"
            or not all(0 < n <= 2048 for n in struct.unpack_from(">II", png, 16))):
        raise ValueError("The project thumbnail must be a bounded PNG image.")
    def aligned(value):
        return (value + 63) // 64 * 64
    def checksum(value):
        struct.pack_into("<I", value, len(value) - 4, _crc(value[:-4]))
    def read_at(source, offset, size):
        source.seek(offset)
        return bytearray(source.read(size))
    try:
        with path.open("rb") as source:
            project = ProjectFile(source)
            if b"THMB" in project.chapters:
                if project.chapters[b"THMB"] != png:
                    raise ValueError("The prepared thumbnail changed. Prepare the project again.")
                return
            stamp = gallery_validation._stamp(source)
            disk_preflight([(temporary, stamp[2] + len(png) + 1024)])
            head_offset = next(offset for offset in (4096, 8192) if read_at(source, offset, 8) == b"LFSHEAD\0")
            head = read_at(source, head_offset, 4096)
            commit = read_at(source, struct.unpack_from("<Q", head, 80)[0], 256)
            old_index_offset, old_index_size = struct.unpack_from("<QQ", commit, 136)
            index = read_at(source, old_index_offset, old_index_size)
            header_offset = aligned(old_index_offset)
            payload_offset = header_offset + 64
            index_offset = aligned(payload_offset + len(png))
            header = bytearray(64)
            struct.pack_into("<4sHH16s", header, 0, b"THMB", 1, 64, head[32:48])
            struct.pack_into("<QQ", header, 32, len(png), len(png))
            struct.pack_into("<I", header, 56, _crc(png))
            checksum(header)
            row = bytearray(96)
            struct.pack_into("<4sH", row, 0, b"THMB", 1)
            row[16:32] = head[32:48]
            struct.pack_into("<QQQQQII", row, 32, header_offset, payload_offset, len(png), len(png), 1,
                             struct.unpack_from("<I", header, 56)[0], struct.unpack_from("<I", header, 60)[0])
            rows = [index[offset:offset + 96] for offset in range(64, len(index), 96)] + [row]
            index = index[:64] + b"".join(sorted(rows, key=lambda value: value[:4] + value[16:32]))
            struct.pack_into("<Q", index, 16, len(rows))
            commit_offset = aligned(index_offset + len(index))
            end = commit_offset + 256
            struct.pack_into("<QQQII", commit, 136, index_offset, len(index), len(index), _crc(index), _crc(index))
            struct.pack_into("<Q", commit, 176, end)
            checksum(commit)
            struct.pack_into("<Q", head, 80, commit_offset)
            struct.pack_into("<Q", head, 96, end)
            struct.pack_into("<I", head, 104, struct.unpack_from("<I", commit, 252)[0])
            struct.pack_into("<QII", head, 112, payload_offset, len(png), 1)
            checksum(head)
            temporary.unlink(missing_ok=True)
            with temporary.open("xb") as output:
                source.seek(0)
                remaining = old_index_offset
                while remaining:
                    if cancel and cancel.is_set():
                        raise GalleryTransferCanceled()
                    block = source.read(min(remaining, gallery_validation.CHUNK_BYTES))
                    if not block:
                        raise ValueError("The prepared project changed.")
                    output.write(block)
                    remaining -= len(block)
                output.write(bytes(header_offset - output.tell()))
                output.write(header)
                output.write(png)
                output.write(bytes(index_offset - output.tell()))
                output.write(index)
                output.write(bytes(commit_offset - output.tell()))
                output.write(commit)
                output.seek(head_offset)
                output.write(head)
                output.flush()
                os.fsync(output.fileno())
            if gallery_validation._stamp(source) != stamp:
                raise ValueError("The prepared project changed.")
        with temporary.open("rb") as result:
            ProjectFile(result)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
