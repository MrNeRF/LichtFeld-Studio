"""Stress-harness product regressions (Lane P)."""
import copy
from importlib import import_module
from types import SimpleNamespace

import pytest

from test_gallery_controller import gallery, panel_module, scene


@pytest.mark.parametrize('choice,content,expected', [('mine', True, 'upload'), ('portal', True, 'pull'), ('portal', False, 'patch')])
def test_resolve_confirmation_chain_executes_final_action(gallery, monkeypatch, choice, content, expected):
    panel, state, actions = gallery
    module = import_module('lfs_plugins.gallery_controller')
    asset = dict(id='project', path='/project.licht', commit_uuid='local' if content else 'base')
    remote = scene(viewerSettings={'exposure': 2, 'cameraPath': {'keyframes': [{'t': 1}]}})
    state['scenes'] = [remote]
    state['links'] = {'project': dict(sceneId=remote['id'], commitUuid='base')}
    monkeypatch.setattr(panel, '_project_identity', lambda: ('project', asset['path']))
    poll = {'path': asset['path'], 'generation': 1, 'running': False}
    monkeypatch.setattr(module.lf, 'project_poll_write', lambda: dict(poll), raising=False)
    monkeypatch.setattr(module, 'capture_view', lambda _: {'exposure': 1, 'cameraPath': {'keyframes': [{'t': 0}]}})
    monkeypatch.setattr(panel, '_schedule_tick', lambda: None)
    monkeypatch.setattr(panel, '_schedule_phase_poll', lambda: None)
    restored = []
    monkeypatch.setattr(module, 'restore_view', lambda _lf, view, **kw: restored.append(copy.deepcopy(view)))
    def save(**kwargs):
        assert restored and not actions
        poll['generation'] += 1
        return True
    monkeypatch.setattr(module.lf, 'project_save', save, raising=False)
    monkeypatch.setattr(module.lf, 'project_is_dirty', lambda: False, raising=False)
    monkeypatch.setattr(panel, '_publish', lambda *a, **k: actions.append(('upload', a)))
    monkeypatch.setattr(panel, 'pull_asset', lambda *a, **k: actions.append(('pull', a)))
    panel.service.edit = lambda *a, **k: actions.append(('patch', a))
    panel.resolve_asset(asset, dict(title='Mine', description='', visibility='private'))
    dialogs = module.lf._test_state.confirm_dialogs
    pressed = 0
    while pressed < len(dialogs):
        _, _, buttons, callback = dialogs[pressed]
        assert panel._decision_pending
        callback(module.tr('conflict.' + choice))
        pressed += 1
    assert pressed == (4 if content else 3)
    assert not panel._decision_pending
    if not content:
        assert panel._save_pending and not actions
        panel._finish_current_project_save()
        assert restored[-1] == remote['viewerSettings']
    assert [a[0] for a in actions] == [expected]


@pytest.mark.parametrize('diagnostic,key', [
    ('Gallery download exceeds its declared size', 'error.download_size'),
    ('Gallery download was incomplete', 'error.download_damaged'),
    ('Invalid portable LichtFeld project.', 'error.download_damaged'),
    ('Project checksum failed.', 'error.download_damaged'),
])
def test_download_validation_reason_is_localized_before_generic_error(panel_module, monkeypatch, diagnostic, key):
    import json
    from pathlib import Path
    from lfs_plugins.gallery_messages import localize_message
    from lfs_plugins.gallery_sync import friendly_error
    translations = json.loads((Path(__file__).parents[2] / 'src/visualizer/gui/resources/locales/en.json').read_text())
    monkeypatch.setattr(panel_module.lf.ui, 'tr', lambda k: translations.get(k, k))
    assert localize_message(friendly_error(ValueError(diagnostic))) == translations['asset_manager.gallery.' + key]


def test_crash_journal_resumes_only_missing_upload_parts(tmp_path, monkeypatch):
    import hashlib
    import io
    import uuid
    from lfs_plugins import gallery_sync, portal_gallery
    from test_gallery_sync import connected, finish
    service = connected(tmp_path, monkeypatch)
    source = tmp_path / 'scene.ply'
    source.write_bytes(b'12345678')
    monkeypatch.setattr(service, 'resume', lambda _: None)
    identifier = service.queue_upload(source, {'title': 'Scene', '_commitUuid': 'saved'}, 'project')
    job = service._job(identifier)
    upload_id = str(uuid.uuid4())
    job.update(status='running', completed=3, checkpoint=dict(origin=service.account.base_url, owner='one',
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(), idempotencyKey='stable', uploadId=upload_id,
        request=dict(title='Scene', sourceFormat='ply', contentLength=8)))
    service._save()
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    recovered = restarted.snapshot()['jobs'][0]
    assert recovered['status'] == 'paused' and recovered['interrupted']
    puts, creates = [], []
    result = dict(id=str(uuid.uuid4()), revision='new', title='Scene', description='', visibility='private', viewerSettings={})
    def request(_client, method, path, body=None):
        if path == '/me':
            return dict(id='one', gallerySyncVersion=1)
        if path == '/splats/uploads':
            creates.append(body)
            return dict(id=upload_id, status='uploading', partSize=3,
                uploadedParts=[dict(partNumber=1, etag='retained', size=3)])
        if path.endswith('/part-upload-urls'):
            number = body['parts'][0]
            return dict(urls=[dict(partNumber=number, url=f'https://portal.example/part/{number}')])
        if path.endswith('/complete'):
            assert body['parts'][0] == dict(partNumber=1, etag='retained')
            return dict(id=upload_id, status='completed', scene=result)
        raise AssertionError((method, path, body))
    class Response(io.BytesIO):
        status = 200
        headers = {'ETag': 'sent'}
    def opened(request, **kwargs):
        puts.append((request.full_url, request.data))
        return Response()
    monkeypatch.setattr(portal_gallery.PortalGalleryClient, '_request', request)
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    monkeypatch.setattr(gallery_sync, 'PortalGalleryClient', portal_gallery.PortalGalleryClient)
    restarted.resume(identifier)
    finish(restarted)
    assert [(url.rsplit('/', 1)[-1], data) for url, data in puts] == [('2', b'456'), ('3', b'78')]
    assert creates[0]['idempotencyKey'] == 'stable'
    state = restarted.snapshot()
    assert state['jobs'][0]['status'] == 'completed'
    from lfs_plugins.gallery_controller import asset_sync_state
    assert asset_sync_state(dict(id='project', commit_uuid='saved'), state['links']['project'], result)['freshness'] == 'equal'


@pytest.mark.parametrize('exception', [ConnectionRefusedError(), TimeoutError()])
def test_outage_exhaustion_pauses_with_automatic_resume(tmp_path, monkeypatch, exception):
    from test_gallery_sync import connected, finish, Client
    service = connected(tmp_path, monkeypatch)
    source = tmp_path / 'scene.ply'
    source.write_bytes(b'ply-data')
    def upload(*args, **kwargs):
        kwargs['on_checkpoint']({'uploadId': 'retained'})
        kwargs['on_progress'](4, 8)
        raise exception
    monkeypatch.setattr(Client, 'upload', upload, raising=False)
    service.queue_upload(source, {'title': 'Scene'}, 'project')
    finish(service)
    job = service.snapshot()['jobs'][0]
    assert job['status'] == 'paused' and job['retryAt'] > 0
    assert job['completed'] == 4 and job['checkpoint']['uploadId'] == 'retained'
    assert 'automatically' in job['message']


def test_controller_automatically_resumes_due_waiting_job(gallery, monkeypatch):
    panel, state, actions = gallery
    state['jobs'] = [dict(id='waiting', status='paused', retryAt=0)]
    panel.service.resume = lambda identifier: actions.append(identifier)
    for method in ('_advance_phases', '_advance_update_all', '_publish_runtime_state', '_finish_pulls'):
        monkeypatch.setattr(panel, method, lambda *a: None)
    panel._next_refresh = float('inf')
    panel._tick_body()
    assert actions == ['waiting']


def test_resolve_mine_preserves_hdr_source_through_native_publish(gallery, monkeypatch, tmp_path):
    panel, state, actions = gallery
    module = import_module('lfs_plugins.gallery_controller')
    asset = dict(id='project', path=str(tmp_path/'project.licht'), commit_uuid='local')
    view = dict(environment={'exposure': -1.25, 'rotation': 123}, cameraPath=None)
    state['scenes'] = [scene(viewerSettings=view)]
    state['source_formats'] = ['licht']
    state['links'] = {'project': dict(sceneId='private-one', commitUuid='base')}
    panel.service.root = tmp_path
    monkeypatch.setattr(panel, '_project_identity', lambda: ('project', asset['path']))
    monkeypatch.setattr(module.lf, 'project_poll_write', lambda: {'path': asset['path']}, raising=False)
    monkeypatch.setattr(module, 'capture_view', lambda _: copy.deepcopy(view))
    settings = SimpleNamespace(environment_map_path='/saved.lfsenv', environment_mode='EQUIRECTANGULAR',
        environment_exposure=-1.25, environment_rotation_degrees=123)
    monkeypatch.setattr(module.lf, 'get_render_settings', lambda: settings, raising=False)
    monkeypatch.setattr(module, 'restore_view', lambda _lf, selected, **kw:
        pytest.fail('Lost HDR source') if kw.get('environment_path') != '/saved.lfsenv' else None)
    monkeypatch.setattr(panel, '_visible_splats', lambda: [SimpleNamespace(name='geometry')])
    monkeypatch.setattr(panel, '_save_current_project', lambda callback: callback())
    monkeypatch.setattr(panel, '_schedule_tick', lambda: None)
    monkeypatch.setattr(panel, '_schedule_phase_poll', lambda: None)
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: {'active': False}, raising=False)
    monkeypatch.setattr(module.lf, 'prepare_gallery_scene', lambda *a: actions.append(a), raising=False)
    panel.resolve_asset(asset, dict(title='Mine', description='', visibility='private'))
    dialogs = module.lf._test_state.confirm_dialogs
    for _, _, _, callback in dialogs:
        callback(module.tr('conflict.mine'))
    assert panel._export_pending, panel._message
    assert len(actions) == 1 and panel._export_pending[1]['viewerSettings'] == view


def test_resolve_portal_retires_conflict_before_backup_and_relinks(gallery, tmp_path, monkeypatch):
    from test_gallery_sync import connected, finish, Client, downloaded_job
    from lfs_plugins import gallery_sync
    from lfs_plugins.gallery_controller import asset_sync_state
    panel, _, _ = gallery
    service = connected(tmp_path, monkeypatch)
    panel.service = service
    panel._identity = service.identity()
    source = tmp_path / 'local.licht'
    source.write_bytes(b'local saved project')
    monkeypatch.setattr(service, 'resume', lambda _: None)
    identifier = service.queue_upload(source, dict(title='Local'), 'project')
    pending = service._job(identifier)
    pending.update(status='conflict', metadata=dict(title='Local', replaceSceneId='scene'))
    job = downloaded_job(service)
    remote = job['result']
    service._bucket()['links']['project'] = gallery_sync.exchange_link(remote, 'before')
    service._save()
    monkeypatch.setattr(Client, 'scene', lambda *a: remote)
    monkeypatch.setattr(panel, '_schedule_tick', lambda: None)
    started = []
    def pull():
        started.append('pull')
        service.prepare_local_update(job['id'], 'project', str(source), gallery_sync.file_stamp(source))
    panel._resolve_pending_uploads('project', 'scene', service.identity(), pull)
    finish(service)
    assert service._job(identifier)['status'] == 'canceled'
    assert not started
    panel._after_service()
    finish(service)
    assert job['localUpdate']['state'] == 'ready'
    assert started == ['pull'] and not panel._decision_pending
    assert source.read_bytes() == b'local saved project'
    service.link_download(job['id'], 'project', 'after')
    finish(service)
    facts = asset_sync_state(dict(id='project', commit_uuid='after'), service.snapshot()['links']['project'], remote,
        service.snapshot()['jobs'])
    assert facts['freshness'] == 'equal' and facts['state'] == 'equal'


def test_resolve_metadata_saves_camera_before_patch_and_freshness(gallery, monkeypatch, tmp_path):
    from test_gallery_sync import connected, finish, Client
    from lfs_plugins import gallery_sync
    from lfs_plugins.gallery_controller import asset_sync_state
    panel, _, _ = gallery
    module = import_module('lfs_plugins.gallery_controller')
    service = connected(tmp_path, monkeypatch)
    panel.service = service
    panel._identity = service.identity()
    remote = scene(viewerSettings=dict(exposure=2, cameraPath={'keyframes': [{'t': 1}]}))
    service.scenes = [remote]
    service._bucket()['links']['project'] = gallery_sync.exchange_link(remote, 'before')
    service._save()
    asset = dict(id='project', path='/linked.licht', commit_uuid='before')
    monkeypatch.setattr(panel, '_project_identity', lambda: ('project', asset['path']))
    poll = dict(path=asset['path'], generation=1, running=False)
    monkeypatch.setattr(module.lf, 'project_poll_write', lambda: dict(poll), raising=False)
    monkeypatch.setattr(module, 'capture_view', lambda _: dict(exposure=1, cameraPath=None))
    monkeypatch.setattr(panel, '_schedule_tick', lambda: None)
    monkeypatch.setattr(panel, '_schedule_phase_poll', lambda: None)
    events = []
    live = {}
    monkeypatch.setattr(module, 'restore_view', lambda _, view, **k: (live.update(copy.deepcopy(view)), events.append('restore')))
    def save(**kwargs):
        assert live == remote['viewerSettings']
        events.append('save')
        poll['generation'] += 1
        return True
    monkeypatch.setattr(module.lf, 'project_save', save, raising=False)
    monkeypatch.setattr(module.lf, 'project_is_dirty', lambda: False, raising=False)
    monkeypatch.setattr(module.lf.io, 'inspect_project', lambda _: SimpleNamespace(commit_uuid='after'))
    def patch(_client, scene_id, revision, **metadata):
        events.append('PATCH')
        return dict(remote, **metadata, revision='patched')
    monkeypatch.setattr(Client, 'update', patch, raising=False)
    panel.resolve_asset(asset, dict(title='Local title', description='', visibility='private'))
    for _, _, _, callback in module.lf._test_state.confirm_dialogs:
        callback(module.tr('conflict.portal'))
    assert events == ['restore', 'save']
    panel._finish_current_project_save()
    finish(service)
    assert events == ['restore', 'save', 'PATCH']
    state = service.snapshot()
    assert asset_sync_state(dict(asset, commit_uuid='after'), state['links']['project'], state['scenes'][0])['freshness'] == 'equal'


def test_portal_rejecting_corrupt_project_keeps_specific_damage_reason(panel_module, monkeypatch):
    from lfs_plugins.gallery_sync import friendly_error
    from lfs_plugins.gallery_messages import localize_message
    from lfs_plugins.portal_account import PortalHTTPError
    monkeypatch.setattr(panel_module.lf.ui, 'tr', lambda key: key)
    reason = friendly_error(PortalHTTPError(400, 'Invalid portable LichtFeld project.'))
    assert localize_message(reason) == 'The downloaded file is damaged or was changed on the portal.'


@pytest.mark.parametrize('is_open', [False, True])
def test_linked_pull_targets_selected_project_before_apply(gallery, monkeypatch, is_open):
    panel, state, actions = gallery
    module = import_module('lfs_plugins.gallery_controller')
    selected = '/linked.licht'
    current = [selected if is_open else '/unrelated.licht']
    asset = dict(id='project', path=selected)
    job = dict(id='download', status='completed')
    state['jobs'] = [job]
    panel._pull_requests[job['id']] = (asset, state['identity'])
    panel._refresh_model()
    monkeypatch.setattr(module.lf, 'project_poll_write', lambda: {'path': current[0]}, raising=False)
    def open_project(path, *args, **kwargs):
        actions.append(('open', path))
        current[0] = path
    monkeypatch.setattr(module.lf, 'project_open', open_project, raising=False)
    monkeypatch.setattr(import_module('lfs_plugins.training_confirm'), 'confirm_discard_work_then',
        lambda title, callback: callback(False))
    monkeypatch.setattr(panel, '_action_update_local', lambda identifier: actions.append(('apply', current[0])))
    panel._finish_pulls()
    if not is_open:
        assert actions == [('open', selected)]
        assert panel._open_continuation[0] == selected
        panel._open_continuation[2]()
    assert actions[-1] == ('apply', selected)


@pytest.mark.parametrize('kind', ['corrupt', 'over_length'])
def test_bad_download_transport_reports_specific_localized_reason(panel_module, tmp_path, monkeypatch, kind):
    import json
    from pathlib import Path
    import io
    from test_portable_project import FIXTURES
    from lfs_plugins import portal_gallery
    from lfs_plugins.gallery_sync import friendly_error
    from lfs_plugins.gallery_messages import localize_message
    data = bytearray((FIXTURES/'portable-ply.licht').read_bytes())
    if kind == 'corrupt':
        data[-1] ^= 1
    total = len(data) - (1 if kind == 'over_length' else 0)
    scene = dict(id='00000000-0000-0000-0000-000000000001', contentLength=total, revision='r1')
    def request(method, path, body=None):
        if path.endswith('/me'):
            return {'maxFileBytes': total}
        if path.endswith('/download'):
            return {'url': 'https://portal.example/content', 'scene': scene}
        return scene
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url='https://portal.example',
        request_json_authenticated=request, _client_version='test'))
    def opened(*args, **kwargs):
        response = io.BytesIO(data)
        response.status, response.headers = 200, {}
        return response
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    destination = tmp_path/'download.licht'
    translations = json.loads((Path(__file__).parents[2]/'src/visualizer/gui/resources/locales/en.json').read_text())
    monkeypatch.setattr(panel_module.lf.ui, 'tr', lambda key: translations.get(key, key))
    with pytest.raises(Exception) as raised:
        client.download('00000000-0000-0000-0000-000000000001', destination)
    key = 'download_size' if kind == 'over_length' else 'download_damaged'
    assert localize_message(friendly_error(raised.value)) == translations['asset_manager.gallery.error.' + key], repr(raised.value)
    assert not destination.exists() and not (tmp_path/'.download.licht.part').exists()


def test_retry_classification_keeps_storage_5xx_resumable_without_retrying_tls_failure():
    import io
    import ssl
    import urllib.error
    from lfs_plugins.portal_retry import is_transient
    assert is_transient(urllib.error.HTTPError('https://portal.example/part', 500, 'Busy', {}, io.BytesIO()))
    assert is_transient(urllib.error.HTTPError('https://portal.example/part', 429, 'Busy', {}, io.BytesIO()))
    assert not is_transient(urllib.error.URLError(ssl.SSLCertVerificationError('Certificate rejected')))


def test_account_switch_releases_pending_resolution(gallery):
    panel, state, actions = gallery
    panel._decision_pending = True
    panel._after_service = lambda: actions.append('old-account mutation')
    state['identity'] = ('https://portal.example', 'other@example.com', 'second', True)
    panel._check_identity()
    assert not panel._decision_pending and panel._after_service is None
    assert not actions


def test_resolve_mine_chain_queues_prepared_upload_and_finishes_equal(gallery, tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    from test_gallery_sync import connected, finish, Client
    from test_portable_project import FIXTURES
    from lfs_plugins import gallery_sync, gallery_preparation
    from lfs_plugins.gallery_controller import asset_sync_state
    panel, _, _ = gallery
    module = import_module('lfs_plugins.gallery_controller')
    service = connected(tmp_path, monkeypatch)
    panel.service = service
    panel._identity = service.identity()
    service._source_formats = ['licht']
    remote = scene(viewerSettings=dict(exposure=2, cameraPath=None))
    service.scenes = [remote]
    service._bucket()['links']['project'] = gallery_sync.exchange_link(remote, 'before')
    service._save()
    source = tmp_path/'selected.licht'
    shutil.copyfile(FIXTURES/'portable-ply.licht', source)
    environment = {'exposure': -1.25, 'rotation': 123}
    monkeypatch.setattr(module.lf, 'get_render_settings', lambda: SimpleNamespace(
        environment_map_path='/saved.lfsenv', environment_mode='EQUIRECTANGULAR',
        environment_exposure=-1.25, environment_rotation_degrees=123), raising=False)
    asset = dict(id='project', path=str(source), commit_uuid='after')
    monkeypatch.setattr(panel, '_project_identity', lambda: ('project', str(source)))
    monkeypatch.setattr(module.lf, 'project_poll_write', lambda: {'path': str(source)}, raising=False)
    monkeypatch.setattr(module.lf.io, 'inspect_project', lambda _: SimpleNamespace(commit_uuid='after'))
    monkeypatch.setattr(module, 'capture_view', lambda _: dict(exposure=1, cameraPath=None, environment=environment))
    monkeypatch.setattr(module, 'restore_view', lambda *a, **k: None)
    monkeypatch.setattr(panel, '_save_current_project', lambda callback: callback())
    monkeypatch.setattr(panel, '_visible_splats', lambda: [SimpleNamespace(name='geometry')])
    monkeypatch.setattr(panel, '_schedule_phase_poll', lambda: None)
    monkeypatch.setattr(panel, '_schedule_tick', lambda: None)
    export_state = dict(active=False)
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: export_state, raising=False)
    def prepare(path, format):
        gallery_preparation.unpack_project(tmp_path, source, Path(path))
        shutil.copyfile(source, Path(path)/'project.licht')
        export_state.update(path=path, outcome='completed')
    monkeypatch.setattr(module.lf, 'prepare_gallery_scene', prepare, raising=False)
    uploads = []
    def upload(_client, path, metadata, **kwargs):
        assert Path(path).suffix == '.licht' and Path(path).is_file()
        uploads.append(copy.deepcopy(metadata))
        return {'scene': dict(remote, title=metadata['title'], viewerSettings=metadata['viewerSettings'], revision='uploaded')}
    monkeypatch.setattr(Client, 'upload', upload, raising=False)
    panel.resolve_asset(asset, dict(title='Mine', description='', visibility='private'))
    for _, _, _, callback in module.lf._test_state.confirm_dialogs:
        callback(module.tr('conflict.mine'))
    assert panel._export_pending, panel._message
    panel._finish_export()
    finish(service)
    assert len(uploads) == 1 and uploads[0]['title'] == 'Mine', service.snapshot()['jobs'][-1]['message']
    state = service.snapshot()
    assert state['jobs'][-1]['status'] == 'completed', state['jobs'][-1]['message']
    assert asset_sync_state(asset, state['links']['project'], state['scenes'][0], state['jobs'])['freshness'] == 'equal'
