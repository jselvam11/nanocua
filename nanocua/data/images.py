"""Tiny PNG writer (stdlib only) so fixtures work without Pillow."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def write_dummy_png(
    path: str | Path,
    rgb: tuple[int, int, int] = (70, 130, 180),
    size: int = 64,
) -> Path:
    """Write a solid-color RGB PNG. Real CUA shots are 1080p desktops."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width = height = size
    red, green, blue = (int(c) & 255 for c in rgb)
    raw = b"".join(b"\x00" + bytes([red, green, blue]) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = (
        (70, 130, 180),
        (60, 160, 90),
        (190, 120, 50),
        (140, 90, 180),
        (80, 80, 90),
        (200, 80, 80),
    )
    return palette[index % len(palette)]
