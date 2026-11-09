# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic subtitle cues; coordinates are test values."""


def cues(rows):
    def stamp(ms):
        seconds, fraction = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02},{fraction:03}"
    return ('\r\n'.join(f'{index}\r\n{stamp(start)} --> {stamp(end)}\r\n{text}\r\n'
                        for index, (start, end, text) in enumerate(rows, 1))).encode('utf-8')


def packet(offset_ms=0, points=False):
    return cues([(start+offset_ms, (start if points else end)+offset_ms,
                  f'<font size="28">FrameCnt: {ordinal}, DiffTime: 33ms\r\n'
                  f'é 日 [latitude: {lat}] [longtitude: -73.25] [rel_alt: 12.5 abs_alt: -2] [unknown: kept]</font>')
                 for ordinal, start, end, lat in ((1, 0, 150, 0), (2, 400, 1000, 45.5))])


def embedded(directory, ffmpeg):
    """Synthetic MP4 with a video and two independently timed subtitle tracks."""
    import subprocess
    first = directory / "embedded-a.srt"
    second = directory / "embedded-b.srt"
    first.write_bytes(packet())
    second.write_bytes(cues([(100, 350, "ordinary subtitle"), (900, 1000, "FrameCnt: 2 [latitude: 37]")]))
    output = directory / "embedded é 日本語.mp4"
    subprocess.run([str(ffmpeg), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=size=16x16:rate=10:duration=1.2", "-i", str(first), "-i", str(second),
                    "-map", "0:v:0", "-map", "1:0", "-map", "2:0", "-c:v", "mpeg4",
                    "-c:s", "mov_text", str(output)], check=True, capture_output=True, timeout=30)
    return output
