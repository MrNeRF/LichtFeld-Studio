# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Portal admission arithmetic and durable, UI-thread-localized storage failures."""
import re

STORAGE_SENTENCE = "Your gallery storage limit has been reached. Delete scenes or cancel pending uploads first."
STORAGE_PREFIX = "gallery_storage:"


def byte_count(value):
    return value if type(value) is int and value >= 0 else None


def gallery_quota(facts):
    quota, used, reserved = (byte_count(facts.get(key)) for key in ("quotaBytes", "usedBytes", "reservedBytes"))
    if quota is None or used is None:
        return None, 0, None
    used += reserved or 0
    return quota, used, max(0, quota - used)


def storage_requirement(size, facts, source_format="licht", replaced=0):
    """Match create_upload: clamp the replacement credit before adding wrapping."""
    size = byte_count(size)
    if size is None:
        return None
    overhead = (byte_count(facts.get("wrapOverheadBytes")) or 0) if source_format in ("ply", "sog", "ssog", "spz", "zip") else 0
    return max(0, size - (byte_count(replaced) or 0)) + overhead


class GalleryStorageError(ValueError):
    def __init__(self, needed=None, free=None, quota=None):
        values = (needed, free, quota)
        super().__init__(STORAGE_PREFIX + ":".join(str(value) for value in values)
                         if all(byte_count(value) is not None for value in values) else STORAGE_SENTENCE)


def is_storage_error(message):
    text = str(message)
    return text.startswith(STORAGE_PREFIX) or STORAGE_SENTENCE.casefold() in text.casefold()


def is_localized_storage_message(message):
    """Snapshots carry already translated job messages; they must survive a second localization."""
    from .gallery_messages import tr
    text = str(message)
    if text == tr("error.gallery_storage"):
        return True
    parts = re.split(r"\{(?:needed|free|quota)\}", tr("error.gallery_storage_size"))
    return len(parts) > 1 and re.fullmatch(".+?".join(map(re.escape, parts)), text, re.S) is not None


def storage_message(message):
    from .asset_format import format_size
    from .gallery_messages import tr
    match = re.fullmatch(r"gallery_storage:(\d+):(\d+):(\d+)", str(message))
    if match:
        needed, free, quota = map(int, match.groups())
        return tr("error.gallery_storage_size", needed=format_size(needed), free=format_size(free), quota=format_size(quota))
    return tr("error.gallery_storage")


def quota_message(facts):
    from .asset_format import format_size
    from .gallery_messages import tr
    quota, used, _ = gallery_quota(facts)
    return tr("quota.used", used=format_size(used), quota=format_size(quota)) if quota is not None else ""
