# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sequential v3 IMU trailer records used by the CLI and Python contracts."""
import struct

MAGIC = b"8db42d694ccc418790edff439fe026bf"
COMPACT = b"\xf0\x03\x01"  # Protobuf field 62, is_raw_gyro.
FIRST = struct.pack("<Q6H", 534773966, 31743, 32794, 32802, 32775, 32786, 32775)
SECOND = struct.pack("<Q6H", 534774966, 0, 65535, 32768, 32769, 32767, 32768)


def record(tag, payload):
    return payload + struct.pack("<HI", tag, len(payload))


def footer(length, *, version=3):
    return struct.pack("<32xII32s", length + 72, version, MAGIC)


def imu_trailer():
    records = record(0x500, b"unrelated") + record(0x300, FIRST + SECOND)
    records += record(0x200, b"preview") + record(0x101, COMPACT)
    return records + footer(len(records))
