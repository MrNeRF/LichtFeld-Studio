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
