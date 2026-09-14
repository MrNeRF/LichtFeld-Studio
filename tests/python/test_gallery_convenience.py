# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gallery convenience actions must keep the button paths' account/write guards."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from test_asset_manager_panel import panel_module, _gallery_fixture, _Handle, _BindingContext, _BindingModel, _Element, _Event
from test_gallery_controller import gallery

@pytest.fixture
def convenience(panel_module, monkeypatch):
    panel, local, remote = _gallery_fixture(panel_module)
    strings = json.loads((Path(__file__).parents[2] / 'src/visualizer/gui/resources/locales/en.json').read_text())
    monkeypatch.setattr(panel_module.lf.ui, 'tr', lambda key: strings.get(key, key))
    panel._handle = _Handle()
    panel._select_asset_id(local['id'])
    yield panel, local, remote
    if panel._gallery_toast_timer:
        panel._gallery_toast_timer.cancel()

def test_gallery_preferences_preserve_each_other_and_old_format(tmp_path):
    from lfs_plugins.gallery_preferences import read_preferences, set_preference
    (tmp_path / 'preferences.json').write_text('{"uploadFormat":"spz"}')
    assert read_preferences(tmp_path) == dict(uploadFormat='spz', askBeforePublic=True, posterCacheMiB=64)
    set_preference('posterCacheMiB', 128, tmp_path)
    set_preference('askBeforePublic', False, tmp_path)
    set_preference('uploadFormat', 'ssog', tmp_path)
    assert read_preferences(tmp_path) == dict(uploadFormat='ssog', askBeforePublic=False, posterCacheMiB=128)

@pytest.mark.parametrize('key,value', [('posterCacheMiB', 0), ('posterCacheMiB', 4097),
     ('askBeforePublic', 'false'), ('uploadFormat', 'bad')])
def test_gallery_preferences_reject_invalid_values(tmp_path, key, value):
    from lfs_plugins.gallery_preferences import set_preference
    with pytest.raises(ValueError):
        set_preference(key, value, tmp_path)
    assert not (tmp_path / 'preferences.json').exists()

def test_gallery_quota_requires_server_usage(convenience):
    panel, local, _ = convenience
    panel._gallery_state.update(quotaBytes=50_000_000_000, usedBytes=12_300_000_000)
    assert panel._gallery_quota() == '12.3 GB of 50 GB used'
    panel._gallery_state.update(quotaBytes=100, usedBytes=None)
    assert panel._gallery_quota_values() == (None, 0)
    panel._gallery_state["usedBytes"] = 84
    local['file_size_bytes'] = 20
    assert 'may not fit' in panel._gallery_quota_warning()
    local['file_size_bytes'] = 16
    assert panel._gallery_quota_warning() == ''
    panel._gallery_state = {}
    assert panel._gallery_quota() == ''

def test_completion_actions_pin_scene_and_expire_without_focus(convenience, monkeypatch):
    panel, local, remote = convenience
    calls = []
    panel._gallery_controller = SimpleNamespace(open_portal=lambda scene, action: calls.append((scene, action)))
    completion = {'id': 'one', 'kind': 'publish', 'scene': copy.deepcopy(remote)}
    panel._gallery_completions({}, dict(panel._gallery_state, completion=completion))
    assert panel._gallery_toast['text'] == 'Published ‹Published project›'
    panel._select_asset_id('remote:remote-only')
    panel._gallery_command('toast_copy')
    assert calls == [(remote, 'copy')]
    assert panel._gallery_toast_timer.interval == 8
    panel._gallery_toast_timer.function()
    assert panel._gallery_toast is None

def test_completion_actions_refuse_changed_account(convenience):
    panel, _, remote = convenience
    panel._show_gallery_toast('Published', scene=remote)
    panel._gallery_state['identity'] = 'other'
    panel._gallery_controller = SimpleNamespace(open_portal=lambda *_: pytest.fail('Stale account action'))
    panel._gallery_command('toast_copy')

def test_remove_and_pull_completions(convenience):
    panel, local, _ = convenience
    panel._gallery_completions({}, dict(panel._gallery_state, completion={'id': 'remove', 'kind': 'remove', 'title': 'Example'}))
    assert panel._gallery_toast['text'] == 'Removed ‹Example› from your gallery'
    panel._gallery_completions({}, dict(panel._gallery_state, pulledProject={'id': local['id'], 'jobId': 'pull', 'path': '/tmp/Scenes/Example.licht'}))
    assert 'to ‹Scenes›' in panel._gallery_toast['text']
    assert panel._gallery_toast['path'] == '/tmp/Scenes/Example.licht'

def test_update_all_candidates_exclude_equal_remote_and_conflicts(convenience):
    panel, local, remote = convenience
    assert panel._gallery_update_candidates() == []
    local['commit_uuid'] = 'changed'
    assert panel._gallery_update_candidates() == [local]
    remote['title'] = 'Remote edit'
    remote['metadataRevision'] = 'metadata-edited'
    assert panel._gallery_update_candidates() == []

def test_local_drop_reuses_primary_validation_and_equal_hint(convenience, monkeypatch):
    panel, local, _ = convenience
    calls = []
    monkeypatch.setattr(panel, '_gallery_command', lambda action: calls.append(action))
    panel._gallery_drop_asset(local['id'], '__gallery__', 'account')
    assert panel._gallery_toast['text'] == 'Already up to date'
    assert calls == []
    local['commit_uuid'] = 'changed'
    panel._gallery_drop_asset(local['id'], '__gallery__', 'account')
    assert calls == ['update']
    panel._gallery_state['links'] = {}
    panel._gallery_drop_asset(local['id'], '__gallery__', 'account')
    assert calls == ['update', 'publish']
    panel._gallery_drop_asset(local['id'], '__gallery__', 'other')
    assert calls == ['update', 'publish']

def test_remote_drag_has_only_origin_owner_scene_id(convenience):
    panel, _, _ = convenience
    identifier = str(uuid.uuid4())
    scene = panel._gallery_state['scenes'][-1]
    scene.update(id=identifier, viewerUrl='https://untrusted.example/ignored')
    panel._gallery_state.update(identity=('https://portal.example', 'email', 'session', True), owner='owner')
    asset = panel._asset_dict('remote:' + identifier)
    assert panel._gallery_drag_payload(asset) == dict(origin='https://portal.example', owner='owner', sceneId=identifier)
    panel._gallery_state['connected'] = False
    assert panel._gallery_drag_payload(asset) is None

def test_remote_folder_drop_selects_requested_folder_for_shared_review(convenience, monkeypatch):
    panel, _, _ = convenience
    panel._asset_index.folders['destination'] = {'id': 'destination', 'path': '/tmp/destination'}
    calls = []
    monkeypatch.setattr(panel, '_gallery_command', lambda action: calls.append((action, panel._gallery_last_folder, panel.get_selected_asset_id())))
    panel._gallery_drop_asset('remote:remote-only', 'destination', 'account')
    assert calls == [('pull', 'destination', 'remote:remote-only')]

def test_gallery_empty_states_follow_account_and_library(convenience):
    panel, _, _ = convenience
    model = _BindingModel()
    panel.on_bind_model(_BindingContext(model))
    panel._selected_folder_id = '__gallery__'
    panel._gallery_state['scenes'] = []
    assert model.func_bindings['gallery_empty']() is True
    panel._gallery_state['connected'] = False
    assert model.func_bindings['gallery_empty']() is False

def _batch_assets(state):
    from lfs_plugins.gallery_sync import shared_fields
    assets = []
    for number in range(2):
        identifier = f'project{number}'
        scene = dict(id=f'scene{number}', title=f'Public {number}', visibility='public', description='', revision='r1', viewerSettings={}, contentRevision='r1', metadataRevision='r1')
        state['scenes'].append(scene)
        state['links'][identifier] = dict(sceneId=scene['id'], commitUuid='old', revision='r1', sharedFields=shared_fields(scene), contentRevision='r1', metadataRevision='r1')
        assets.append(dict(id=identifier, path=f'/tmp/{identifier}.licht', commit_uuid='new', exists=True))
    return assets

def test_update_all_one_public_confirmation_continues_after_item_failure(gallery, monkeypatch, tmp_path):
    controller, state, _ = gallery
    controller.service.root = tmp_path
    assets = _batch_assets(state)
    confirmations, started = [], []
    monkeypatch.setattr(controller, 'confirm_action', lambda key, titles, callback: confirmations.append((key, titles, callback)))
    def publish(asset, details, format, *, update):
        scene = next(s for s in state['scenes'] if s['title'] == details['title'])
        assert controller._batch_public_approved(scene, details)
        started.append(asset['id'])
        if len(started) == 1:
            raise ValueError('Missing source payload')
    monkeypatch.setattr(controller, 'publish_asset', publish)
    controller.update_all(assets)
    assert len(confirmations) == 1
    assert confirmations[0][:2] == ('confirm.update_all', 'Public 0\nPublic 1')
    assert started == []
    confirmations[0][2]()
    controller._advance_update_all()
    assert started == ['project0', 'project1']
    failures = [j for j in controller.snapshot()['jobs'] if j.get('batchFailure')]
    assert len(failures) == 1 and failures[0]['project'] == 'project0'
    assert failures[0]['message'] == 'Missing source payload'
    from lfs_plugins.gallery_transfer_panel import transfer_rows
    row = next(r for r in transfer_rows(controller.snapshot()) if r['id'] == failures[0]['id'])
    assert row['can_resume'] and row['can_cancel']

def test_update_all_revalidates_account_and_reviewed_revision(gallery, monkeypatch, tmp_path):
    controller, state, _ = gallery
    controller.service.root = tmp_path
    assets = _batch_assets(state)
    confirmations, started = [], []
    monkeypatch.setattr(controller, 'confirm_action', lambda key, title, callback: confirmations.append(callback))
    monkeypatch.setattr(controller, 'publish_asset', lambda *args, **kwargs: started.append(args[0]['id']))
    controller.update_all(assets)
    state['scenes'][0]['metadataRevision'] = 'changed'
    confirmations[0]()
    assert started == [] and controller._batch_rows[0]['project'] == 'project0'
    state['identity'] = ('other', 'account')
    controller._advance_update_all()
    assert started == [] and controller._update_queue == []

def test_update_all_keeps_async_preparation_failure_visible(gallery, monkeypatch, tmp_path):
    controller, state, _ = gallery
    controller.service.root = tmp_path
    assets = _batch_assets(state)
    monkeypatch.setattr(controller, 'confirm_action', lambda key, title, callback: callback())
    monkeypatch.setattr(controller, 'publish_asset', lambda *args, **kwargs: None)
    controller.update_all(assets[:1])
    controller._message = 'Preparation failed: missing payload'
    controller._advance_update_all()
    assert controller._batch_rows[0]['message'] == 'Preparation failed: missing payload'

def test_preferences_format_setter_preserves_other_values(gallery, tmp_path):
    from lfs_plugins.gallery_preferences import set_preference, read_preferences
    controller, _, _ = gallery
    controller.service.root = tmp_path
    set_preference('posterCacheMiB', 9, tmp_path)
    set_preference('posterCacheMiB', 3, tmp_path)
    controller.upload_format = 'spz'
    assert read_preferences(tmp_path)['posterCacheMiB'] == 3
    assert controller.upload_format == 'spz'

@pytest.mark.parametrize('key,ctrl,shift,command', [(72,True,False,'primary'), (14,True,True,'copy'), (111,False,False,'refresh_scope'), (72,False,False,None), (14,True,False,None)])
def test_gallery_shortcut_defaults(key, ctrl, shift, command):
    from lfs_plugins.gallery_shortcuts import shortcut_command
    assert shortcut_command(None, key, ctrl, shift) == command

def test_shortcuts_respect_native_conflicts_capture_and_rebinding():
    from lfs_plugins.gallery_shortcuts import shortcut_command
    actions = SimpleNamespace(NONE=0, ASSET_GALLERY_PRIMARY=1, ASSET_GALLERY_COPY_LINK=2, ASSET_REFRESH=3)
    keymap = SimpleNamespace(Action=actions, ToolMode=SimpleNamespace(GLOBAL=0), is_capturing=lambda: False,
        get_action_for_key=lambda mode, key, mods: 1 if key == 66 else 0)
    assert shortcut_command(keymap, 13) == 'primary'  # rebound to B
    assert shortcut_command(keymap, 72, True) is None  # old default is unbound
    keymap.is_capturing = lambda: True
    assert shortcut_command(keymap, 13) is None
    keymap = SimpleNamespace(Action=SimpleNamespace(NONE=0), ToolMode=SimpleNamespace(GLOBAL=0), is_capturing=lambda: False,
        get_action_for_key=lambda *args: 99)
    assert shortcut_command(keymap, 72, True) is None

def test_keyboard_primary_does_not_open_and_f5_refreshes_active_scope(convenience, panel_module, monkeypatch):
    panel, _, _ = convenience
    calls = []
    monkeypatch.setattr(panel, '_gallery_command', lambda command: calls.append(command))
    monkeypatch.setattr(panel, 'refresh_catalog', lambda: calls.append('local_refresh'))
    event = _Event(_Element(), params={'key_identifier': '72'}, bool_params={'ctrl_key': True})
    panel._on_asset_results_keydown(event)
    assert calls == ['primary'] and event.stopped
    assert panel_module.lf._test_state.opened == []
    panel._selected_folder_id = '__all__'
    panel._on_asset_results_keydown(_Event(_Element(), params={'key_identifier':'111'}))
    panel._selected_folder_id = '__gallery__'
    panel._on_asset_results_keydown(_Event(_Element(), params={'key_identifier':'111'}))
    assert calls == ['primary', 'local_refresh', 'refresh']

def test_keyboard_does_not_take_text_field_input(convenience, monkeypatch):
    panel, _, _ = convenience
    monkeypatch.setattr(panel, '_gallery_command', lambda *_: pytest.fail('Text field shortcut stolen'))
    text = _Element()
    text.tag_name = 'input'
    event = _Event(text, params={'key_identifier': '72'}, bool_params={'ctrl_key':True})
    assert not panel._on_gallery_shortcut(event)
    assert not event.stopped

def test_viewport_drop_handoff_validates_identity_before_shared_pull_open(convenience, monkeypatch):
    panel, _, _ = convenience
    calls = []
    identifier = str(uuid.uuid4())
    panel._gallery_state['scenes'][-1]['id'] = identifier
    panel._gallery_state.update(identity=('https://portal.example', 'email', 'session', True), owner='owner')
    monkeypatch.setattr(panel, '_gallery_command', lambda action: calls.append(action))
    payload = dict(origin='https://portal.example', owner='owner', sceneId=identifier)
    assert panel.gallery_viewport_drop(json.dumps(payload))
    assert calls == ['pull_open']
    for bad in (dict(payload, owner='other'), dict(payload, origin='https://other.example'),
                dict(payload, url='https://evil.example'), dict(payload, sceneId='invalid'), {}):
        assert not panel.gallery_viewport_drop(json.dumps(bad))
    assert calls == ['pull_open']

def test_sidebar_drag_hover_drop_consumes_payload_before_drag_end(convenience, panel_module, monkeypatch):
    panel, local, _ = convenience
    calls = []
    shell = _Element()
    target = _Element({'data-folder-id':'__gallery__'}, shell)
    child = _Element({}, target)
    event = _Event(shell, child)
    panel._gallery_drag = (local['id'], 'account')
    panel._drag_payload_token = 42
    monkeypatch.setattr(panel, '_gallery_drop_asset', lambda *args: calls.append(args))
    panel._on_gallery_drag_over(event)
    assert target.is_class_set('is-drag-over')
    panel._on_gallery_drop(event)
    panel._on_asset_drag_end(event)
    assert not target.is_class_set('is-drag-over')
    assert calls == [(local['id'], '__gallery__', 'account')]
    assert panel_module.lf._test_state.drag_cancels == [42]
    assert panel_module.lf._test_state.drag_ends == []
    rcss = (Path(__file__).parents[2] / 'src/visualizer/gui/rmlui/resources/asset_manager.rcss').read_text()
    assert 'drag: drag-drop;' in rcss

def test_preferences_group_binds_all_gallery_defaults(panel_module, monkeypatch, tmp_path):
    # Exercise the same preference helper used by the Preferences setters without
    # coupling this suite to a second native-module fixture.
    from lfs_plugins.gallery_preferences import set_preference, read_preferences
    for key, value in [('uploadFormat','studio'), ('askBeforePublic',False), ('posterCacheMiB',32)]:
        set_preference(key, value, tmp_path)
    assert read_preferences(tmp_path)['posterCacheMiB'] == 32
    import xml.etree.ElementTree as ET
    rml = ET.fromstring((Path(__file__).parents[2] / 'src/visualizer/gui/rmlui/resources/preferences.rml').read_text())
    group = rml.find('.//*[@data-if="gallery_expanded"]')
    assert group is not None
    bindings = {e.get('data-value') or e.get('data-checked') for e in group.iter()}
    assert {'gallery_uploadFormat','gallery_askBeforePublic','gallery_posterCacheMiB'} <= bindings

def test_batch_preparation_failure_retry_uses_normal_publish_path(gallery, monkeypatch, tmp_path):
    controller, state, _ = gallery
    controller.service.root = tmp_path
    assets = _batch_assets(state)
    entry = {'asset':assets[0], 'scene':state['scenes'][0], 'identity':state['identity'], 'format':'sog'}
    controller._record_batch_failure(entry, 'Preparation failed')
    row = controller._batch_rows[0]
    calls = []
    monkeypatch.setattr(controller, 'publish_asset', lambda *args, **kwargs: calls.append((args,kwargs)))
    controller.command('resume', row['id'])
    assert calls[0][0][0] == assets[0]
    assert calls[0][1] == {'update':True}
    assert controller._batch_rows == []

def test_publish_opens_dialog_with_current_format_and_selected_file(convenience, monkeypatch):
    from lfs_plugins import gallery_file_panel
    panel, asset, _ = convenience
    opened = []
    panel._gallery_controller = SimpleNamespace(upload_format='spz')
    monkeypatch.setattr(gallery_file_panel, 'open_gallery_file_panel', lambda **kw: opened.append(kw))
    panel._gallery_command('publish')
    assert len(opened) == 1 and opened[0]['asset'] == asset
    assert opened[0]['fields']['upload_format'] == 'spz'
    assert opened[0]['action'] == 'publish'

def test_remote_card_starts_typed_native_drag(convenience, panel_module):
    panel, _, _ = convenience
    identifier = str(uuid.uuid4())
    panel._gallery_state['scenes'][-1]['id'] = identifier
    panel._gallery_state.update(identity=('https://portal.example', 'email', 'session', True), owner='owner')
    shell = _Element()
    card = _Element({'data-asset-action':'select', 'data-asset-id':'remote:' + identifier}, shell)
    event = _Event(shell, card)
    panel._on_asset_drag_start(event)
    token, payload_type, data, label = panel_module.lf._test_state.drag_begins[-1]
    assert payload_type == 'application/x-lichtfeld-gallery-scene'
    assert json.loads(data) == dict(origin='https://portal.example', owner='owner', sceneId=identifier)
    assert label == 'Remote only' and event.stopped
    panel._on_asset_drag_end(event)
    assert panel_module.lf._test_state.drag_ends == [token]
