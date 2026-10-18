# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Display-only GPU inventory and stable device choices for Preferences."""


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
    return str(requested)


def device_choices(devices, requested, role, tr):
    selected = selected_value(devices, requested)
    choices = [{"value": "", "label": tr("preferences.tensor_auto"), "disabled": False}]
    names = [device["name"] for device in devices]
    for device in devices:
        reason = device.get(f"{role}_reason", "")
        label = device["name"]
        if names.count(label) > 1:
            label += f" · #{device['index']} · " + device["uuid"][-8:]
        if reason:
            label += " · " + tr("preferences.gpu_unavailable")
        choices.append({"value": device["uuid"], "label": label, "disabled": bool(reason)})
    if selected and not any(item["value"] == selected for item in choices):
        choices.append({"value": selected, "label": tr("preferences.gpu_unavailable") + " · " + selected,
                        "disabled": True})
    return choices


def hardware_rows(vulkan, cuda, tr):
    rows = {}
    indices = {}
    for backend, devices in (("Vulkan", vulkan), ("CUDA", cuda)):
        for device in devices:
            key = identity(device["uuid"]) or f"{backend}:{device['index']}"
            indices.setdefault(key, []).append(f"{backend} #{device['index']}")
            row = rows.setdefault(key, {"name": device["name"], "backends": [], "memory": "", "details": []})
            reason = device.get("tensor_reason", "")
            row["backends"].append(backend + (" · " + tr("preferences.gpu_unavailable") if reason else ""))
            if reason:
                row["details"].append(backend + ": " + reason)
            size = device.get("memory_bytes", 0)
            if size:
                label = "preferences.gpu_memory_shared" if device.get("shared_memory") else "preferences.gpu_memory_dedicated"
                row["memory"] = tr(label).replace("{size}", f"{size / (1024 ** 3):.1f}")
    names = [row["name"] for row in rows.values()]
    result = []
    for key, row in rows.items():
        if names.count(row["name"]) > 1:
            row["name"] += " · " + " / ".join(indices[key]) + " · " + key[-8:]
        result.append({**row, "backends": " · ".join(row["backends"]), "details": "\n".join(row["details"])})
    return result
