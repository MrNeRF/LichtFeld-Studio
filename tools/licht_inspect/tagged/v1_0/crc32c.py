"""Small, dependency-free CRC32c Castagnoli implementation."""

from __future__ import annotations

_POLY = 0x82F63B78


def _make_table() -> tuple[int, ...]:
    table: list[int] = []
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = (crc >> 1) ^ (_POLY if crc & 1 else 0)
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)


_TABLE = _make_table()


def crc32c(data: bytes | bytearray | memoryview, crc: int = 0) -> int:
    """Return CRC32c using initial/final xor 0xffffffff."""

    state = (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF
    for value in memoryview(data).cast("B"):
        state = _TABLE[(state ^ value) & 0xFF] ^ (state >> 8)
    return (state ^ 0xFFFFFFFF) & 0xFFFFFFFF


if __name__ == "__main__":
    assert crc32c(b"123456789") == 0xE3069283
