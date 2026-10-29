#!/usr/bin/env python3
"""Generate the release-specific third-party notices from installed inputs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

NVIDIA_CUDA_RUNTIME_COMPONENTS = {
    "cudart": ("NVIDIA CUDA Runtime", "CUDA Runtime"),
    "curand": ("NVIDIA CUDA Random Number Generation", "CUDA Random Number Generation Library"),
    "nvjpeg": ("NVIDIA JPEG Library", "NVIDIA JPEG Library"),
    "nvrtc": ("NVIDIA Runtime Compilation Library", "NVIDIA Runtime Compilation Library"),
    "nvjitlink": ("NVIDIA JIT Linking Library", "NVIDIA JIT Linking Library"),
    "nvtoolsext": ("NVIDIA Tools Extension Library", "NVIDIA Tools Extension Library"),
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def copyright_lines(text: str) -> str:
    rows = []
    for line in text.splitlines():
        value = line.strip()
        lower = value.lower()
        if "[yyyy]" in lower or "<year>" in lower:
            continue
        if re.match(r"(?i)^SPDX-FileCopyrightText:", value):
            rows.append(value)
        elif re.match(r"(?i)^copyright\b|^©", value) and not any(part in lower for part in ("copyright notice", "copyright law", "copyright holder", "copyright owner", "copyright license")):
            rows.append(value)
    return "; ".join(dict.fromkeys(rows))


def license_id_from_text(text: str, fallback: str) -> str:
    first = "\n".join(text.splitlines()[:30])
    label = re.match(r"(?i)^\s*(?:license|licensing)\s*[:=-]\s*(.+)$", text.strip())
    if label:
        normalized = label.group(1).strip().lower()
        label_ids = {"mit": "MIT", "mit license": "MIT", "apache-2.0": "Apache-2.0", "bsd-2-clause": "BSD-2-Clause", "bsd-3-clause": "BSD-3-Clause", "ofl-1.1": "OFL-1.1", "mpl-2.0": "MPL-2.0", "gpl-2.0-or-later": "GPL-2.0-or-later", "gpl-3.0-or-later": "GPL-3.0-or-later", "lgpl-2.1-or-later": "LGPL-2.1-or-later"}
        if normalized in label_ids:
            return label_ids[normalized]
    patterns = [
        (r"(?i)Apache-2\.0 WITH LLVM-exception", "Apache-2.0 WITH LLVM-exception"),
        (r"(?i)Apache License(?:,? Version)?\s*2\.0|Apache-2\.0", "Apache-2.0"),
        (r"(?i)MIT License", "MIT"),
        (r"(?i)BSD 3-Clause|BSD-3-Clause", "BSD-3-Clause"),
        (r"(?i)BSD 2-Clause|BSD-2-Clause", "BSD-2-Clause"),
        (r"(?i)SIL OPEN FONT LICENSE", "OFL-1.1"),
        (r"(?i)Mozilla Public License(?:,? Version)?\s*2\.0", "MPL-2.0"),
        (r"(?i)GNU LESSER GENERAL PUBLIC LICENSE\s+Version 2\.1", "LGPL-2.1-or-later"),
        (r"(?i)GNU GENERAL PUBLIC LICENSE\s+Version 3", "GPL-3.0-or-later"),
        (r"(?i)GNU GENERAL PUBLIC LICENSE\s+Version 2", "GPL-2.0-or-later"),
        (r"(?i)Boost Software License", "BSL-1.0"),
        (r"(?i)zlib License", "Zlib"),
    ]
    return next((spdx for pattern, spdx in patterns if re.search(pattern, first)), fallback)


def acknowledgement_components(file: Path, source_root: Path, revision: str) -> list[dict[str, str]]:
    components = []
    sections = re.split(r"(?m)^={12,}\s*$", read_text(file))
    for section in sections:
        section = section.strip()
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if not lines or not re.search(r"https?://", lines[0]):
            continue
        name = lines[0].split("(", 1)[0].strip()
        if not name:
            continue
        if re.search(r"(?i)Boost Software License", section):
            license_id = "BSL-1.0"
        else:
            license_id = license_id_from_text(section, "LicenseRef-nvImageCodec-acknowledgement")
            if license_id.startswith("LicenseRef-") and "Redistribution and use" in section:
                license_id = "BSD-3-Clause" if "Neither the name" in section else "BSD-2-Clause"
        components.append({"name": f"nvImageCodec acknowledgement: {name}", "version": f"not versioned separately (nvImageCodec revision {revision})", "license": license_id, "copyright": copyright_lines(section) or f"Copyright notice is included in the acknowledgement for {name}.", "text": section})
    return components


def installed_ports(triplet_dir: Path, vcpkg_root: Path | None = None) -> list[dict[str, str]]:
    result = []
    for metadata in sorted((triplet_dir / "share").glob("*/vcpkg.spdx.json")):
        data = json.loads(metadata.read_text(encoding="utf-8"))
        package = next((p for p in data.get("packages", []) if p.get("SPDXID") == "SPDXRef-port"), {})
        name = package.get("name", metadata.parent.name)
        version = package.get("versionInfo", "unknown")
        license_id = package.get("licenseConcluded") or package.get("licenseDeclared") or "NOASSERTION"
        copyright = package.get("copyrightText", "NOASSERTION")
        text = read_text(metadata.parent / "copyright")
        license_texts = [text] if text else []
        if vcpkg_root:
            references = re.findall(r"(?i)([A-Za-z0-9_./+-]*(?:LICENSE|COPYING|NOTICE|PATENTS)[A-Za-z0-9_.+-]*)", text)
            source_root = vcpkg_root / "buildtrees" / name / "src"
            for reference in references:
                basename = Path(reference).name
                candidates = [metadata.parent / reference]
                if source_root.exists():
                    candidates.extend(source_root.glob(f"**/{basename}"))
                for candidate in candidates:
                    body = read_text(candidate)
                    if body and body not in license_texts:
                        license_texts.append(body)
        if license_id in ("NOASSERTION", "LicenseRef-vcpkg-null"):
            # vcpkg's SPDX metadata may leave the expression empty while its
            # installed copyright file still contains the source license.
            license_id = license_id_from_text(text, f"LicenseRef-{name}")
        is_license_preamble = re.match(r"(?i)^\s*(?:GNU (?:LESSER )?GENERAL PUBLIC LICENSE|MIT License|Apache License|BSD [23]-Clause License)", text)
        actual_copyrights = "" if is_license_preamble else copyright_lines(text)
        if actual_copyrights:
            copyright = actual_copyrights
        elif copyright in ("NOASSERTION", "NONE", "") or "copyright information" in copyright.lower():
            copyright = "Copyright statement is included in the bundled vcpkg copyright text."
        result.append({"name": name, "version": version, "license": license_id, "copyright": copyright, "text": "\n\n".join(license_texts)})
    return result


def vendored_components(root: Path, source_root: Path) -> list[dict[str, str]]:
    result = []
    revisions: dict[str, str] = {}
    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        is_readme = file.name.lower() in {"readme", "readme.md", "readme.txt"}
        is_acknowledgements = "acknowledg" in file.name.lower()
        is_license_doc = any(token in file.name.lower() for token in ("license", "copying", "notice", "patent"))
        if is_acknowledgements:
            relative = file.relative_to(source_root)
            revision = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(relative)], cwd=source_root, capture_output=True, text=True, check=False).stdout.strip() or "unavailable"
            result.extend(acknowledgement_components(file, source_root, revision))
            continue
        if not is_readme and not is_license_doc:
            continue
        body = read_text(file)
        if not body:
            continue
        head = "\n".join(body.splitlines()[:20])
        if is_readme:
            match = re.search(r"(?im)^\s*(?:license|licensing)\s*[:=-]\s*(.+)$", body)
            if not match:
                continue
            body = match.group(0)
            head = body
        license_id = license_id_from_text(head, f"LicenseRef-{file.parent.name}-{file.name}")
        copyright = copyright_lines(body[:12000]) or f"See {file.relative_to(root)}"
        relative = file.relative_to(root)
        component_root = root / relative.parts[0] if len(relative.parts) > 1 else file
        version_path = str(component_root.relative_to(source_root))
        if version_path not in revisions:
            revisions[version_path] = subprocess.run(["git", "log", "-1", "--format=%h", "--", version_path], cwd=source_root, capture_output=True, text=True, check=False).stdout.strip()
        revision = revisions[version_path]
        version = f"vendored revision {revision}" if revision else "vendored source version not recorded"
        result.append({"name": f"Vendored: {file.relative_to(root)}", "version": version, "license": license_id, "copyright": copyright, "text": body})
    return result


def python_components(roots: list[Path]) -> list[dict[str, str]]:
    result = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for metadata in sorted(root.rglob("*.dist-info")):
            if not metadata.is_dir() or metadata in seen:
                continue
            seen.add(metadata)
            name, _, version = metadata.name.removesuffix(".dist-info").rpartition("-")
            licenses = [p for p in metadata.rglob("*") if p.is_file() and ("license" in p.name.lower() or "copying" in p.name.lower() or "notice" in p.name.lower())]
            text = "\n\n".join(read_text(p) for p in licenses if read_text(p))
            license_id = f"LicenseRef-Python-{name}"
            for candidate in (metadata / "METADATA", metadata / "PKG-INFO"):
                content = read_text(candidate)
                match = re.search(r"(?m)^License:\s*(.+)$", content)
                if match:
                    license_id = match.group(1).strip()
                    break
            result.append({"name": f"Python package {name}", "version": version or "unknown", "license": license_id, "copyright": copyright_lines(text) or "See package metadata and license files.", "text": text or "No license file was present in this installed dist-info directory."})
    return result


def ffmpeg_details(args: argparse.Namespace) -> str:
    status = read_text(args.vcpkg_root / "installed" / "vcpkg" / "status") if args.vcpkg_root else ""
    # CMake's installed root can differ from the vcpkg checkout.
    status = status or read_text(args.triplet_dir.parent / "vcpkg" / "status")
    packages = [block for block in re.split(r"\n\s*\n", status) if re.search(r"(?m)^Package: ffmpeg$", block)]
    package = next((block for block in packages if re.search(r"(?m)^Feature:", block) is None), "")
    features = []
    if package:
        features.extend(re.findall(r"(?m)^Default-Features: (.+)$", package))
    features.extend(re.search(r"(?m)^Feature: ([^\n]+)$", block).group(1) for block in packages if re.search(r"(?m)^Feature: ([^\n]+)$", block) and re.search(r"(?m)^Status: install ok installed$", block))
    features = sorted(set(", ".join(features).replace(", ", " ").split()))
    logs = []
    portfile = None
    if args.vcpkg_root:
        buildtrees = args.vcpkg_root / "buildtrees" / "ffmpeg"
        logs = list(buildtrees.glob("*-config.log")) if buildtrees.exists() else []
        candidate = args.vcpkg_root / "ports" / "ffmpeg" / "portfile.cmake"
        portfile = candidate if candidate.exists() else None
    options = []
    for log in logs:
        content = read_text(log)
        command = next((line for line in content.splitlines() if "/configure " in line and line.startswith("# ")), "")
        candidate = re.findall(r"--(?:enable|disable|with|without)-[A-Za-z0-9_+-]+(?:=[^\s'\"]+)?", command)
        aligned = ("x264" in features) == ("--enable-libx264" in candidate)
        aligned &= ("openh264" in features) == ("--enable-libopenh264" in candidate)
        if command and aligned:
            options = candidate
            break
    option_source = "matching vcpkg FFmpeg build configure log" if options else "no configure log matching installed feature status"
    manifest = json.loads(read_text(args.source / "vcpkg.json") or "{}")
    baseline = manifest.get("builtin-baseline", "not recorded")
    spdx = args.triplet_dir / "share" / "ffmpeg" / "vcpkg.spdx.json"
    source_tag = "not recorded"
    if spdx.exists():
        data = json.loads(spdx.read_text(encoding="utf-8"))
        source = next((p.get("downloadLocation", "") for p in data.get("packages", []) if p.get("name", "").startswith("ffmpeg/")), "")
        source_tag = source or source_tag
    overlays = sorted(p for p in args.source.glob(".github/overlays/*/portfile.cmake") if p.parent.name in {"ffmpeg", "x264"})
    overlay_text = ", ".join(str(p.relative_to(args.source).parent) for p in overlays) or "none found"
    x264_package = next((p for p in installed_ports(args.triplet_dir, args.vcpkg_root) if p["name"] == "x264"), None)
    x264_source = "not installed in this triplet"
    x264_spdx = args.triplet_dir / "share" / "x264" / "vcpkg.spdx.json"
    if x264_spdx.exists():
        x264_data = json.loads(x264_spdx.read_text(encoding="utf-8"))
        x264_source = next((p.get("downloadLocation", "") for p in x264_data.get("packages", []) if p.get("SPDXID") == "SPDXRef-port"), "") or x264_source
    requested_features = []
    for dependency in manifest.get("dependencies", []):
        if isinstance(dependency, dict) and dependency.get("name") == "ffmpeg":
            requested_features = dependency.get("features", [])
    status_note = ""
    if requested_features and set(requested_features) != set(features):
        status_note = f"; manifest requests {', '.join(requested_features)}, installed status differs"
    gpl = "x264" in features or "gpl" in features
    return "\n".join([
        "### FFmpeg build configuration",
        "",
        f"- Generated from the vcpkg triplet installed at package time: `{args.triplet_dir.name}`.",
        f"- Version: {next((p['version'] for p in installed_ports(args.triplet_dir, args.vcpkg_root) if p['name'] == 'ffmpeg'), 'not installed')}",
        f"- Installed vcpkg features: {', '.join(features) or 'not available in vcpkg status'}",
        f"- Configure options ({option_source}): {', '.join(sorted(set(options))) or 'not found in available vcpkg logs/port recipe'}",
        f"- Resulting build license: {'GPL (x264 feature is installed)' if gpl else 'x264 is not installed in this triplet; do not label this build GPL because of x264'}{status_note}",
        f"- Corresponding FFmpeg source: {source_tag}",
        f"- x264 version/source: {x264_package['version'] if x264_package else 'not installed'}; {x264_source}",
        f"- vcpkg baseline: {baseline}; repository overlay ports: {overlay_text}",
    ])


def validate_coverage(document: str, triplet_dir: Path, install_prefix: Path) -> list[str]:
    errors = []
    for metadata in sorted((triplet_dir / "share").glob("*/vcpkg.spdx.json")):
        data = json.loads(metadata.read_text(encoding="utf-8"))
        package = next((p for p in data.get("packages", []) if p.get("SPDXID") == "SPDXRef-port"), {})
        name = package.get("name", metadata.parent.name)
        if not re.search(rf"(?m)^\|\s*{re.escape(name)}\s*\|", document):
            errors.append(f"installed vcpkg port missing from notices: {name}")
    shared = {".so", ".dylib", ".dll"}
    for path in install_prefix.rglob("*"):
        if path.is_file() and (path.suffix.lower() in shared or ".so." in path.name):
            if path.name not in document:
                errors.append(f"installed shared library missing from notices: {path.name}")
    nvidia_license = {
        "nvjpeg2k": ("LicenseRef-NVIDIA-nvJPEG2000", "runtime files .so and .h, nvjpeg2k_0.dll"),
        **{
            token: ("LicenseRef-NVIDIA-CUDA-Toolkit", marker)
            for token, (_, marker) in NVIDIA_CUDA_RUNTIME_COMPONENTS.items()
        },
    }
    for path in install_prefix.rglob("*"):
        if not path.is_file() or not re.search(r"(?i)(?:^|lib)(?:cuda|nv)[A-Za-z0-9_]*", path.name):
            continue
        name = path.name.lower()
        source_built = any(part in name for part in ("nvimgcodec", "nvjpeg_ext", "nvjpeg2k_ext"))
        runtime_kind = next((kind for kind in nvidia_license if kind in name), None)
        if source_built:
            if path.name not in document:
                errors.append(f"installed NVIDIA source-built library missing from notices: {path.name}")
        elif runtime_kind:
            license_id, license_marker = nvidia_license[runtime_kind]
            if path.name not in document or license_id not in document or license_marker not in document:
                errors.append(f"installed NVIDIA runtime missing its file and license notice: {path.name}")
        else:
            errors.append(f"NVIDIA runtime file is not listed in the reviewed CUDA Toolkit EULA Attachment A: {path.name}")
    return errors


def generate(args: argparse.Namespace) -> str:
    ports = installed_ports(args.triplet_dir, args.vcpkg_root)
    components = ports + vendored_components(args.source / "external", args.source)
    components += vendored_components(args.source / "resources/licenses", args.source)
    components += python_components([args.install_prefix, args.triplet_dir / "tools" / "python3", args.source])
    # Include bundled font and icon notices that are shipped as application assets.
    assets = [args.source / "src/visualizer/gui/assets/fonts", args.source / "src/visualizer/gui/assets/icon", args.source / "src/rendering/resources/assets"]
    for root in assets:
        if root.exists():
            components += vendored_components(root, args.source)
    sdk_requirements = read_text(args.source / "tools/license_texts/NVIDIA-SDK-Distribution-Requirements.txt")
    nvidia_texts = {
        "LicenseRef-NVIDIA-CUDA-Toolkit": "\n\n".join((sdk_requirements, read_text(args.source / "tools/license_texts/NVIDIA-CUDA-Toolkit-redistribution.txt"))),
        "LicenseRef-NVIDIA-nvJPEG2000": "\n\n".join((sdk_requirements, read_text(args.source / "tools/license_texts/NVIDIA-nvJPEG2000-Supplement.txt"))),
    }
    seen_nvidia_runtimes = set()
    for path in sorted(args.install_prefix.rglob("*")):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if "nvjpeg2k" in lower and "ext" not in lower:
            license_id = "LicenseRef-NVIDIA-nvJPEG2000"
            name = "NVIDIA nvJPEG2000 runtime"
            version_match = re.search(r"(?:so\.|_)(\d+(?:\.\d+)+)", path.resolve().name, re.IGNORECASE)
            version = version_match.group(1) if version_match else "version supplied by NVIDIA runtime filename"
        elif any(token in lower for token in NVIDIA_CUDA_RUNTIME_COMPONENTS) and not any(
            token in lower for token in ("nvimgcodec", "_ext")
        ):
            license_id = "LicenseRef-NVIDIA-CUDA-Toolkit"
            token = next(token for token in NVIDIA_CUDA_RUNTIME_COMPONENTS if token in lower)
            name = NVIDIA_CUDA_RUNTIME_COMPONENTS[token][0]
            version_match = re.search(r"(?:so\.|dll[._]?|_)(\d+(?:\.\d+)+)", path.resolve().name, re.IGNORECASE)
            version = version_match.group(1) if version_match else "version supplied by NVIDIA runtime filename"
        else:
            continue
        runtime_key = (license_id, name, path.resolve())
        if runtime_key in seen_nvidia_runtimes:
            continue
        seen_nvidia_runtimes.add(runtime_key)
        components.append({"name": name, "version": version, "license": license_id, "copyright": "Copyright NVIDIA Corporation. All rights reserved.", "text": nvidia_texts[license_id]})
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for component in components:
        groups[component["license"]].append(component)

    lines = ["# Third-party licenses", "", "Generated at install time from the selected vcpkg triplet and files shipped with this distribution.", "",
             "LichtFeld Studio is licensed under GPL-3.0-or-later (LICENSE) with an additional permission for NVIDIA GPU libraries (LICENSE-EXCEPTION.txt).", "",
             f"Library count: **{len(components)}**", ""]
    for license_id in sorted(groups, key=str.casefold):
        rows = sorted(groups[license_id], key=lambda x: (x["name"].casefold(), x["version"]))
        lines += [f"## {license_id}", "", "| Library | Version | Copyright |", "|---|---|---|"]
        for row in rows:
            safe = lambda value: value.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {safe(row['name'])} | {safe(row['version'])} | {safe(row['copyright'])} |")
        # One representative full text for this license group, keeping distinct
        # texts for license expressions that combine multiple SPDX identifiers.
        texts = sorted({row["text"] for row in rows if row["text"]})
        if texts:
            lines += ["", "### License text", "", "```text", max(texts, key=len), "```"]
        lines.append("")
    shared_names = sorted({p.name for p in args.install_prefix.rglob("*") if p.is_file() and (p.suffix.lower() in {".so", ".dylib", ".dll"} or ".so." in p.name)})
    nvidia_paths = [
        path for path in sorted(args.install_prefix.rglob("*"))
        if path.is_file() and re.search(r"(?i)(?:^|lib)(?:cuda|nv)[A-Za-z0-9_]*", path.name)
    ]
    cuda_runtime_present = any(
        any(token in path.name.lower() for token in NVIDIA_CUDA_RUNTIME_COMPONENTS)
        and not any(token in path.name.lower() for token in ("nvimgcodec", "_ext"))
        and "nvjpeg2k" not in path.name.lower()
        for path in nvidia_paths
    )
    nvjpeg2k_present = any(
        "nvjpeg2k" in path.name.lower() and "ext" not in path.name.lower()
        for path in nvidia_paths
    )
    notice = [
        "This distribution includes NVIDIA components. Copyright NVIDIA Corporation. All rights reserved.",
        "The NVIDIA files found in the installed payload are listed below. Runtime redistribution is subject to the applicable NVIDIA agreement and its stated requirements.",
    ]
    if cuda_runtime_present:
        notice.append("The CUDA Toolkit agreement requirements and relevant Attachment A entries are reproduced under `LicenseRef-NVIDIA-CUDA-Toolkit`.")
    if nvjpeg2k_present:
        notice.append("The applicable nvJPEG2000 Supplement and NVIDIA SDK distribution requirements are reproduced under `LicenseRef-NVIDIA-nvJPEG2000`.")
    notice.append("nvImageCodec and its extensions are built from vendored source and carry their own source notices.")
    lines += ["## Installed shared libraries", "", *[f"- `{name}`" for name in shared_names], "", ffmpeg_details(args), "", "### NVIDIA notices", "", *notice, "", "References: https://docs.nvidia.com/cuda/eula/index.html and https://docs.nvidia.com/cuda/nvjpeg2000/license.html", ""]
    for path in nvidia_paths:
        name = path.name
        lower = name.lower()
        if any(part in lower for part in ("nvimgcodec", "nvjpeg_ext", "nvjpeg2k_ext")):
            classification = "vendored nvImageCodec source build; see its bundled source license"
        elif "nvjpeg2k" in lower:
            classification = "nvJPEG2000 runtime; see LicenseRef-NVIDIA-nvJPEG2000 and its Supplement"
        elif any(token in lower for token in NVIDIA_CUDA_RUNTIME_COMPONENTS):
            classification = "NVIDIA runtime; see LicenseRef-NVIDIA-CUDA-Toolkit and Attachment A"
        else:
            classification = "not listed in CUDA Toolkit EULA Attachment A; package check fails"
        lines.append(f"- `{name}`: {classification}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--triplet-dir", type=Path, required=True)
    parser.add_argument("--install-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vcpkg-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = generate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    errors = validate_coverage(document, args.triplet_dir, args.install_prefix) if args.check else []
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Generated {args.output} ({len(installed_ports(args.triplet_dir, args.vcpkg_root))} vcpkg ports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
