# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Display-only GPU inventory and stable device choices for Preferences."""
import re


def identity(value):
    value = str(value).strip().lower()
    if value.startswith("gpu-"):
        value = value[4:]
    return value.replace("-", "").replace("{", "").replace("}", "")


def selected_value(devices, requested):
    if not requested:
        return ""
    for device in devices:
        if identity(requested) == identity(device["uuid"]) or str(requested) == str(device["index"]):
            return device["uuid"]
    # A missing saved identity follows the runtime Auto fallback. Keep it in
    # preferences for reconnects, but do not invent a device in the selector.
    return ""


def device_choices(devices, requested, role, tr):
    choices = [{"value": "", "label": tr("preferences.tensor_auto"), "disabled": False}]
    names = [device["name"] for device in devices]
    for device in devices:
        reason = device.get(f"{role}_reason", "")
        label = device["name"]
        if names.count(label) > 1:
            label += f" · #{device['index']} · " + device["uuid"][-8:]
        if reason:
            label += " · " + tr("preferences.gpu_unavailable")
        if device.get("software"):
            label += " · " + tr("preferences.gpu_software")
        choices.append({"value": device["uuid"], "label": label, "disabled": bool(reason)})
    return choices


def hardware_rows(vulkan, cuda, tr, hip=()):
    rows = {}
    indices = {}
    inventories = (("Vulkan", vulkan), ("CUDA", cuda), ("ROCm / HIP", hip))
    physical_counts = {}
    def physical_identity(device):
        value = str(device.get("physical_id", "")).strip().lower()
        return value if re.fullmatch(r"pci:[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value) else ""
    for backend, devices in inventories:
        for device in devices:
            physical = physical_identity(device)
            if physical:
                physical_counts[backend, physical] = physical_counts.get((backend, physical), 0) + 1
    physical_rows, uuid_rows, row_physical = {}, {}, {}
    for backend, devices in inventories:
        for device in devices:
            uuid = identity(device["uuid"])
            physical = physical_identity(device)
            if physical_counts.get((backend, physical), 0) != 1:
                physical = ""
            key = physical_rows.get(physical) if physical else None
            if key is None:
                key = uuid_rows.get(uuid)
                if key is not None and physical and row_physical.get(key, physical) != physical:
                    key = None
            if key is None:
                key = uuid or f"{backend}:{device['index']}"
                if key in rows and physical and row_physical.get(key, physical) != physical:
                    key = f"{backend}:{device['index']}:{key}"
            if uuid:
                uuid_rows[uuid] = key
            if physical:
                physical_rows[physical] = key
                row_physical[key] = physical
            indices.setdefault(key, []).append(f"{backend} #{device['index']}")
            row = rows.setdefault(key, {"name": device["name"], "backends": [], "memory": "", "details": []})
            if device.get("software"):
                row["details"].append(tr("preferences.gpu_software"))
            reason = device.get("tensor_reason", "")
            row["backends"].append(backend + (" · " + tr("preferences.gpu_unavailable") if reason else ""))
            if reason:
                row["details"].append(backend + ": " + reason)
            size = device.get("memory_bytes", 0)
            if size:
                label = ("preferences.gpu_memory_system" if device.get("software") else
                         "preferences.gpu_memory_shared" if device.get("shared_memory") else
                         "preferences.gpu_memory_dedicated")
                memory = tr(label).replace("{size}", f"{size / (1024 ** 3):.1f}")
                if backend == "ROCm / HIP" and row["memory"] and memory != row["memory"]:
                    row["details"].append(backend + ": " + memory)
                else:
                    row["memory"] = memory
    names = [row["name"] for row in rows.values()]
    result = []
    for key, row in rows.items():
        if names.count(row["name"]) > 1:
            row["name"] += " · " + " / ".join(indices[key]) + " · " + key[-8:]
        result.append({**row, "backends": " · ".join(row["backends"]), "details": "\n".join(row["details"])})
    return result
