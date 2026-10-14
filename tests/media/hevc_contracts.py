# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent MP4 hvc1 configuration/sample checks shared by native tests."""
import json
import subprocess


def assert_hvc1_samples(test, ffprobe, path):
    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_streams", "-show_packets", "-show_data",
        "-show_entries", "stream=extradata:packet=data", "-of", "json", str(path)]))

    def unpack(dump):
        return bytes.fromhex("".join(line.split(":", 1)[1].split("  ", 1)[0].replace(" ", "")
                                    for line in dump.splitlines() if ":" in line))

    config = unpack(probe["streams"][0]["extradata"])
    test.assertGreaterEqual(len(config), 23)
    test.assertEqual(config[0], 1)
    length_bytes = (config[21] & 3) + 1
    offset = 23
    parameter_sets = set()
    for _ in range(config[22]):
        test.assertLessEqual(offset + 3, len(config))
        header = config[offset]
        count = int.from_bytes(config[offset + 1:offset + 3], "big")
        offset += 3
        if (header & 63) in (32, 33, 34):
            test.assertTrue(header & 128, "hvc1 parameter array must be complete")
            test.assertGreater(count, 0)
            parameter_sets.add(header & 63)
        for _ in range(count):
            test.assertLessEqual(offset + 2, len(config))
            size = int.from_bytes(config[offset:offset + 2], "big")
            offset += 2
            test.assertGreaterEqual(size, 2)
            test.assertLessEqual(offset + size, len(config))
            test.assertEqual((config[offset] >> 1) & 63, header & 63)
            offset += size
    test.assertEqual(parameter_sets, {32, 33, 34})
    test.assertEqual(offset, len(config))
    test.assertTrue(probe["packets"])
    for packet in probe["packets"]:
        data = unpack(packet["data"])
        offset = 0
        while offset < len(data):
            test.assertLessEqual(offset + length_bytes, len(data))
            size = int.from_bytes(data[offset:offset + length_bytes], "big")
            offset += length_bytes
            test.assertGreaterEqual(size, 2)
            test.assertLessEqual(offset + size, len(data))
            test.assertNotIn((data[offset] >> 1) & 63, (32, 33, 34),
                             "hvc1 samples must not repeat parameter sets")
            offset += size
