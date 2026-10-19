# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Animation objects can be created through the public timeline factory chain."""

import pytest


def test_animation_timeline_clip_and_track_are_reachable(lf):
    timeline = lf.animation.Timeline()
    assert timeline.has_animation_clip is False

    clip = timeline.animation_clip()
    clip.name = "test clip"
    track = clip.add_track("float", "node:example.opacity")
    track.add_keyframe(0.0, 0.25, "linear")

    assert timeline.has_animation_clip is True
    assert clip.name == "test clip"
    assert clip.track_count == 1
    assert track.keyframe_count == 1
    assert track.evaluate(0.0) == 0.25
    assert clip.get_track_by_path("node:example.opacity").id == track.id
    tracks = clip.tracks()
    assert len(tracks) == 1
    assert tracks[0].id == track.id

    clip_ref = clip
    del clip, timeline
    assert track.evaluate(0.0) == 0.25
    assert tracks[0].evaluate(0.0) == 0.25
    assert clip_ref.tracks()[0].id == track.id


def test_removed_animation_track_wrappers_raise(lf):
    timeline = lf.animation.Timeline()
    clip = timeline.animation_clip()
    track = clip.add_track("float", "node:example.opacity")
    track_id = track.id
    track_from_lookup = clip.get_track(track_id)
    assert track_from_lookup.id == track_id

    clip.remove_track(track_id)

    with pytest.raises(RuntimeError, match=f"animation track {track_id} was removed"):
        track.evaluate(0.0)
    with pytest.raises(RuntimeError, match=f"animation track {track_id} was removed"):
        track.add_keyframe(0.0, 0.25)
    with pytest.raises(RuntimeError, match=f"animation track {track_id} was removed"):
        track_from_lookup.evaluate(0.0)
