# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generated settings records from older, newer, and damaged writers."""

import json

import pytest


@pytest.mark.parametrize('version,upload_format,expected', [
    (999, 'sog', 'sog'), (1000, 'ssog', 'ssog'), (1000, 'new-format', 'auto')])
def test_gallery_future_preferences_keep_version_and_fields(tmp_path, version, upload_format, expected):
    from lfs_plugins.gallery_preferences import read_preferences, set_preference
    path = tmp_path / 'preferences.json'
    original = dict(version=version, uploadFormat=upload_format,
                    posterCacheMiB=64, futureFeature={'enabled': True})
    path.write_text(json.dumps(original))
    assert read_preferences(tmp_path)['uploadFormat'] == expected
    set_preference('posterCacheMiB', 128, tmp_path)
    assert json.loads(path.read_text()) == dict(original, posterCacheMiB=128)

@pytest.mark.parametrize('contents', ['{"version":', '{"version":"future","posterCacheMiB":64}'])
def test_gallery_damaged_file_is_backed_up_before_write(tmp_path, monkeypatch, contents):
    from lfs_plugins.gallery_preferences import set_preference
    monkeypatch.setenv('LFS_RESOLVED_DATA_DIR', str(tmp_path / 'data'))
    path = tmp_path / 'preferences.json'
    path.write_text(contents)
    set_preference('posterCacheMiB', 128, tmp_path)
    backup = next((tmp_path / 'data' / 'backups').glob(
        'reset-corrupt-gallery-preferences-*/preferences.json'))
    assert backup.read_text() == contents
    assert json.loads(path.read_text())['posterCacheMiB'] == 128


@pytest.mark.parametrize('contents', ['null', '42', '{"enabled":'])
def test_settings_invalid_file_defaults_and_backs_up_before_write(tmp_path, contents):
    from lfs_plugins.settings import PluginSettings
    settings_root = tmp_path / 'plugins'
    path = settings_root / 'settings_plugin' / 'settings.json'
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    settings = PluginSettings('settings_plugin', settings_root)
    assert settings.get('enabled', False) is False
    settings.set('new_option', 9)
    assert json.loads(path.read_text()) == {'new_option': 9}
    backups = list((tmp_path / 'data' / 'backups').glob(
        'reset-corrupt-plugin-settings-*/settings.json'))
    assert len(backups) == 1
    assert backups[0].read_text() == contents


def test_plugin_settings_preserve_future_fields_from_both_writers(tmp_path):
    from lfs_plugins.settings import PluginSettings
    path = tmp_path / 'settings_plugin' / 'settings.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'version': 999, 'shortcut': 'S', 'enabled': 'yes', 'count': None,
                                'future_feature': {'enabled': True}}))
    settings = PluginSettings('settings_plugin', tmp_path)
    assert settings.get('shortcut', '') == 'S'
    assert settings.get('enabled', False) is False
    assert settings.get('count', 5) == 5
    path.write_text(json.dumps({'version': 1000, 'shortcut': 'S', 'new_field': 8,
                                'enabled': 'yes', 'count': None,
                                'future_feature': {'enabled': True}}))
    settings.set('shortcut', 'A')
    assert json.loads(path.read_text()) == {'version': 1000, 'shortcut': 'A', 'new_field': 8,
                                           'enabled': 'yes', 'count': None,
                                           'future_feature': {'enabled': True}}
