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
        if method == 'GET' and path == f'/splats/uploads/{upload_id}':
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
    assert job['status'] == 'waiting' and job['retryDelay'] == 5 and job['retryAt'] > 0
    assert job['completed'] == 4 and job['checkpoint']['uploadId'] == 'retained'
    assert job['message'] == 'Waiting for connection…'
    service.pause()
    finish(service)



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
    from test_gallery_sync import connected, finish, Client, gallery_sync
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


@pytest.mark.parametrize('both', [False, True])
def test_resolve_mine_chain_queues_prepared_upload_and_finishes_equal(gallery, tmp_path, monkeypatch, both):
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
    def track(x):
        return dict(version=1, duration=6, loopMode='loop', playbackSpeed=1.5, keyframes=[
            dict(time=.5, position=[x, 2, 3], rotation=[1, 0, 0, 0], focal_length_mm=35, easing=1),
            dict(time=4, position=[-2, 1, 3], rotation=[.5, .5, .5, .5], focal_length_mm=200, easing=3)])
    remote = scene(viewerSettings=dict(exposure=2, cameraPath=track(12) if both else None))
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
    monkeypatch.setattr(module, 'capture_view', lambda _: dict(exposure=1, cameraPath=track(22) if both else None, environment=environment))
    restored = []
    from lfs_plugins.gallery_view import restore_camera_path
    def native_restore(path):
        # The actual native timeline loader requires time, not t.
        if any('time' not in frame or 't' in frame for frame in path['keyframes']):
            return False
        restored.append(copy.deepcopy(path))
        return True
    monkeypatch.setattr(module.lf.ui, 'set_camera_path', native_restore, raising=False)
    monkeypatch.setattr(module.lf.ui, 'clear_keyframes', lambda: None, raising=False)
    monkeypatch.setattr(module, 'restore_view', lambda lf, view, **kw: restore_camera_path(lf, view['cameraPath']))
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
    pressed = []
    for _, _, buttons, callback in module.lf._test_state.confirm_dialogs:
        choice = 'both' if both and module.tr('conflict.both') in buttons else 'mine'
        pressed.append(choice)
        callback(module.tr('conflict.' + choice))
    if both:
        assert pressed == ['mine', 'mine', 'both', 'mine']
        assert restored[0]['duration'] == 12
        assert [f['time'] for f in restored[0]['keyframes']] == [.5, 4, 6.5, 10]
    assert panel._export_pending, panel._message
    panel._finish_export()
    finish(service)
    if both:
        assert uploads[0]['viewerSettings']['cameraPath'] == restored[0]
    assert len(uploads) == 1 and uploads[0]['title'] == 'Mine', service.snapshot()['jobs'][-1]['message']
    state = service.snapshot()
    assert state['jobs'][-1]['status'] == 'completed', state['jobs'][-1]['message']
    assert asset_sync_state(asset, state['links']['project'], state['scenes'][0], state['jobs'])['freshness'] == 'equal'


@pytest.fixture
def connection_clock(monkeypatch):
    from lfs_plugins import gallery_sync
    now, timers = [1000.0], []
    class Timer:
        def __init__(self, delay, callback):
            self.delay, self.callback, self.canceled = delay, callback, False
            timers.append(self)
        def start(self):
            pass
        def cancel(self):
            self.canceled = True
        def fire(self):
            assert not self.canceled
            now[0] += self.delay
            self.callback()
    monkeypatch.setattr(gallery_sync.threading, 'Timer', Timer)
    monkeypatch.setattr(gallery_sync.time, 'time', lambda: now[0])
    return now, timers


def test_waiting_probes_backoff_then_resumes_without_controller(tmp_path, monkeypatch, connection_clock):
    from test_gallery_sync import connected, finish, Client
    from lfs_plugins.gallery_transfer_panel import transfer_rows
    from lfs_plugins.gallery_controller import asset_sync_state
    service = connected(tmp_path, monkeypatch)
    now, timers = connection_clock
    source = tmp_path / 'scene.ply'
    source.write_bytes(b'12345678')
    attempts, probes = [], []
    reachable = [False]
    def upload(client, path, metadata, **kwargs):
        attempts.append(kwargs['checkpoint'])
        if len(attempts) == 1:
            kwargs['on_checkpoint']({'uploadId': 'retained'})
            kwargs['on_progress'](4, 8)
            raise ConnectionRefusedError()
        assert kwargs['checkpoint']['uploadId'] == 'retained'
        return {'scene': scene()}
    monkeypatch.setattr(Client, 'upload', upload, raising=False)
    identifier = service.queue_upload(source, {'title': 'Scene'}, 'project')
    finish(service)
    def probe(client, method, path):
        assert (method, path) == ('GET', '/me')
        probes.append(now[0])
        if not reachable[0]:
            raise ConnectionRefusedError()
        return {'id': 'one', 'gallerySyncVersion': 1}
    monkeypatch.setattr(Client, '_request', probe)
    for delay in [5, 10, 20, 40, 80, 160, 300, 300]:
        assert timers[-1].delay == delay
        state = service.snapshot()
        assert state['jobs'][0]['status'] == 'waiting'
        assert state['jobs'][0]['completed'] == 4
        row = transfer_rows(state)[0]
        assert row['can_pause'] and row['can_resume']
        assert row['phase'].endswith('waiting') or row['phase'] == 'Waiting for connection…'
        assert asset_sync_state({'id': 'project'}, None, None, state['jobs'])['state'] == 'waiting'
        timers[-1].fire()
        finish(service)
        assert len(attempts) == 1
    reachable[0] = True
    timers[-1].fire()
    finish(service)
    assert len(probes) == 9 and len(attempts) == 2
    assert service._job(identifier)['status'] == 'completed'
    assert service._connection_timer is None


@pytest.mark.parametrize('pinned', [False, True], ids=['restart', 'range'])
@pytest.mark.parametrize('failure', ['socket', 'eof', 'incomplete_read'])
@pytest.mark.parametrize('pause', [False, True], ids=['automatic', 'user_pause'])
def test_download_outage_waits_then_recovers(tmp_path, monkeypatch, connection_clock, pinned, failure, pause):
    import hashlib
    import io
    import json
    from http.client import IncompleteRead
    from pathlib import Path
    from test_gallery_sync import connected, finish, gallery_sync
    from lfs_plugins import portal_gallery, portal_retry
    from lfs_plugins.gallery_controller import asset_sync_state
    from lfs_plugins.gallery_transfer_panel import transfer_rows

    service = connected(tmp_path, monkeypatch)
    data, prefix = b'ply\ncomplete-download', b'ply\n'
    remote = scene(sourceFormat='ply', contentLength=len(data))
    remote['id'] = '28d8880e-96d2-46a0-9226-bb62976392b2'
    reachable, requests, probes, messages, progress = [True], [], [], [], []
    headers = {'ETag': '"pinned-v1"', 'Accept-Ranges': 'bytes'} if pinned else {}

    class Interrupted(io.BytesIO):
        status = 200

        def read(self, size=-1):
            if not self.tell():
                reachable[0] = False
                return super().read(len(prefix))
            if failure == 'socket':
                raise ConnectionResetError('Portal stopped mid-stream')
            if failure == 'incomplete_read':
                raise IncompleteRead(b'', len(data) - len(prefix))
            return b''

    class Recovered(io.BytesIO):
        def read(self, size=-1):
            job = service.snapshot()['jobs'][0]
            messages.append(job['message'])
            progress.append(job['completed'])
            return super().read(size)

    def opened(request, **kwargs):
        requests.append(request)
        assert kwargs['no_redirect'] and request.get_header('Authorization') is None
        if len(requests) == 1:
            response = Interrupted(data)
            response.headers = headers
            return response
        if not reachable[0]:
            raise ConnectionRefusedError('Portal offline')
        offset = len(prefix) if pinned else 0
        assert request.get_header('Range') == (f'bytes={offset}-' if pinned else None)
        assert request.get_header('If-range') == ('"pinned-v1"' if pinned else None)
        response = Recovered(data[offset:])
        response.status = 206 if pinned else 200
        response.headers = dict(headers)
        if pinned:
            response.headers['Content-Range'] = f'bytes {offset}-{len(data)-1}/{len(data)}'
        return response

    def request(client, method, path, *args):
        if path == '/me':
            probes.append(reachable[0])
            if not reachable[0]:
                raise ConnectionRefusedError('Portal offline')
            return {'id': 'one', 'gallerySyncVersion': 1}
        assert reachable[0]
        if path.endswith('/download'):
            return {'scene': remote, 'url': 'https://portal.example/storage'}
        return remote

    monkeypatch.setattr(portal_gallery.PortalGalleryClient, '_request', request)
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    # Exercise the real bounded retry loop without wall-clock backoff sleeps.
    monkeypatch.setattr(portal_gallery, 'retry_call', lambda operation, **kwargs:
                        portal_retry.retry_call(operation, sleep=lambda _: None, **kwargs))
    monkeypatch.setattr(gallery_sync, 'PortalGalleryClient', portal_gallery.PortalGalleryClient)
    service.download(remote)
    finish(service)
    job = service.snapshot()['jobs'][0]
    assert job['status'] == 'waiting', (job['message'], len(requests), probes)
    assert job['message'] == 'Waiting for connection…'
    assert job['completed'] == len(prefix) and len(requests) == 4
    destination = Path(job['path'])
    partial = destination.with_name('.' + destination.name + '.part')
    assert not destination.exists()
    assert partial.exists() == pinned
    if pinned:
        assert partial.read_bytes() == prefix
    row = transfer_rows(service.snapshot())[0]
    assert row['phase'].endswith('waiting') or row['phase'] == 'Waiting for connection…'
    assert row['can_pause'] and row['can_resume']
    assert asset_sync_state({}, None, remote, [job])['state'] == 'waiting'
    persisted = json.loads((tmp_path / 'sync.json').read_text())
    assert 'storage' not in json.dumps(persisted)  # No signed URL in the journal.
    timer = connection_clock[1][-1]
    assert timer.delay == 5
    timer.fire()
    finish(service)
    assert service._job(job['id'])['status'] == 'waiting'
    assert connection_clock[1][-1].delay == 10 and len(requests) == 4
    if pause:
        timer = connection_clock[1][-1]
        service.pause(job['id'])
        finish(service)
        assert timer.canceled
        assert service._job(job['id'])['status'] == 'paused'
        assert 'retryAt' not in service._job(job['id'])
    reachable[0] = True
    if pause:
        service._retry_connection()
        assert not service.busy and service._connection_timer is None
        assert len(requests) == 4
        service.resume(job['id'])
    else:
        connection_clock[1][-1].fire()
    finish(service)
    job = service._job(job['id'])
    assert job['status'] == 'completed', job['message']
    assert destination.read_bytes() == data and not partial.exists()
    assert job['sha256'] == hashlib.sha256(data).hexdigest()
    assert job['completed'] == len(data) and len(requests) == 5
    assert probes == ([False] if pause else [False, True])
    assert service._connection_timer is None
    if not pinned:
        assert 'Restarting from zero' in messages[0]
        assert progress[0] == 0


@pytest.mark.parametrize('restart', [False, True])
def test_explicit_pause_never_auto_resumes_after_outage(tmp_path, monkeypatch, connection_clock, restart):
    from test_gallery_sync import connected, finish, Client, gallery_sync
    service = connected(tmp_path, monkeypatch)
    source = tmp_path / 'scene.ply'
    source.write_bytes(b'12345678')
    monkeypatch.setattr(Client, 'upload', lambda *a, **kw: (_ for _ in ()).throw(TimeoutError()), raising=False)
    identifier = service.queue_upload(source, {'title': 'Scene'}, 'project')
    finish(service)
    timer = connection_clock[1][-1]
    service.pause(identifier)
    finish(service)
    assert timer.canceled
    if restart:
        service = gallery_sync.GallerySync(service.account, tmp_path)
        service.refresh()
        finish(service)
        assert service._owner, service.message
    job = service._job(identifier)
    assert job['status'] == 'paused' and 'retryAt' not in job
    assert service._connection_timer is None
    service._retry_connection()
    assert service._connection_timer is None and not service.busy


def test_waiting_recovers_after_restart_but_old_account_cannot_resume(tmp_path, monkeypatch, connection_clock):
    from test_gallery_sync import connected, finish, Client, gallery_sync
    service = connected(tmp_path, monkeypatch)
    source = tmp_path / 'scene.ply'
    source.write_bytes(b'12345678')
    monkeypatch.setattr(Client, 'upload', lambda *a, **kw: (_ for _ in ()).throw(TimeoutError()), raising=False)
    identifier = service.queue_upload(source, {'title': 'Scene'}, 'project')
    finish(service)
    service._connection_timer.cancel()
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    assert restarted._owner, restarted.message
    assert restarted._job(identifier)['status'] == 'waiting'
    assert restarted._connection_timer.delay == 5
    service.account.email = 'different@example.com'
    restarted._connection_timer.fire()
    assert not restarted.busy and restarted._connection_timer is None
    assert restarted.snapshot()['jobs'] == []


def test_pause_while_connection_probe_is_in_flight_stays_paused(tmp_path, monkeypatch, connection_clock):
    import threading
    from test_gallery_sync import connected, finish, Client
    service = connected(tmp_path, monkeypatch)
    source = tmp_path / 'scene.ply'
    source.write_bytes(b'12345678')
    monkeypatch.setattr(Client, 'upload', lambda *a, **kw: (_ for _ in ()).throw(TimeoutError()), raising=False)
    identifier = service.queue_upload(source, {'title': 'Scene'}, 'project')
    finish(service)
    probing, release = threading.Event(), threading.Event()
    def probe(*args):
        probing.set()
        assert release.wait(2)
        raise TimeoutError()
    monkeypatch.setattr(Client, '_request', probe)
    connection_clock[1][-1].fire()
    assert probing.wait(2)
    service.pause()
    release.set()
    finish(service)
    assert service._job(identifier)['status'] == 'paused'
    assert service._connection_timer is None


@pytest.mark.skipif(__import__('sys').platform == 'win32', reason='Exercises SIGKILL and the POSIX process lock')
def test_sigkill_recovery_revalidates_server_parts_and_completes_valid_licht(tmp_path, monkeypatch):
    import hashlib
    import io
    import subprocess
    import sys
    import uuid
    from test_gallery_sync import connected, finish, gallery_sync
    from test_portable_project import FIXTURES
    from lfs_plugins import portal_gallery
    from lfs_plugins.portable_project import ProjectFile
    service = connected(tmp_path, monkeypatch)
    source = tmp_path / 'scene.licht'
    data = (FIXTURES / 'portable-ply.licht').read_bytes()
    source.write_bytes(data)
    part_size = len(data) // 3
    upload_id = str(uuid.uuid4())
    monkeypatch.setattr(service, 'resume', lambda _: None)
    identifier = service.queue_upload(source, {'title': 'Scene', '_commitUuid': 'saved'}, 'project')
    service._job(identifier).update(status='running', completed=part_size * 2,
        checkpoint=dict(origin=service.account.base_url, owner='one', sha256=hashlib.sha256(data).hexdigest(),
            idempotencyKey='stable-key', uploadId=upload_id,
            request=dict(title='Scene', sourceFormat='licht', contentLength=len(data))))
    service._save()
    # A process holds the actual sidecar lock until it is killed. The lock file
    # survives; flock ownership does not. Recovery must distinguish the two.
    child = subprocess.Popen([sys.executable, '-c',
        'import fcntl,sys,time; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX); print("locked",flush=True); time.sleep(30)',
        str(tmp_path / 'sync.lock')], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'locked'
        child.kill()
        assert child.wait(timeout=3) == -9
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)
        child.stdout.close()
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    recovered = restarted._job(identifier)
    assert recovered['status'] == 'paused' and recovered['interrupted']
    assert recovered['completed'] == part_size * 2
    storage, puts, requests = {1: data[:part_size], 2: b'truncated'}, [], []
    remote_scene = dict(id=str(uuid.uuid4()), revision='published', title='Scene', viewerSettings={})
    def request(client, method, path, body=None):
        requests.append((method, path))
        if path == '/me':
            return dict(id='one', gallerySyncVersion=1, sourceFormats=['licht'])
        if path == '/splats/uploads':
            assert body['idempotencyKey'] == 'stable-key'
            # A replay response has no authoritative part inventory.
            return dict(id=upload_id, status='uploading', partSize=part_size, uploadedParts=[])
        if method == 'GET' and path == f'/splats/uploads/{upload_id}':
            return dict(id=upload_id, status='uploading', partSize=part_size,
                uploadedParts=[dict(partNumber=n, size=len(value), etag=f'tag-{n}') for n, value in storage.items()])
        if path.endswith('/part-upload-urls'):
            number = body['parts'][0]
            return dict(urls=[dict(partNumber=number, url=f'https://portal.example/part/{number}')])
        if path.endswith('/complete'):
            assert body['parts'][0] == dict(partNumber=1, etag='tag-1')
            assembled = b''.join(storage[n] for n in sorted(storage))
            assert assembled == data
            ProjectFile(io.BytesIO(assembled))  # The real publishing-subset reader.
            return dict(id=upload_id, status='completed', scene=remote_scene)
        raise AssertionError((method, path, body))
    class Response(io.BytesIO):
        status = 200
    def opened(request, **kwargs):
        number = int(request.full_url.rsplit('/', 1)[-1])
        puts.append(request.data)
        storage[number] = request.data
        response = Response()
        response.headers = {'ETag': f'tag-{number}'}
        return response
    monkeypatch.setattr(portal_gallery.PortalGalleryClient, '_request', request)
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    monkeypatch.setattr(gallery_sync, 'PortalGalleryClient', portal_gallery.PortalGalleryClient)
    restarted.resume(identifier)
    finish(restarted)
    assert restarted._job(identifier)['status'] == 'completed', restarted._job(identifier)['message']
    assert sum(map(len, puts)) == len(data) - part_size < len(data)
    assert ('GET', f'/splats/uploads/{upload_id}') in requests
    from lfs_plugins.gallery_controller import asset_sync_state
    assert asset_sync_state({'id': 'project', 'commit_uuid': 'saved'},
        restarted.snapshot()['links']['project'], remote_scene)['freshness'] == 'equal'


def test_waiting_wording_is_not_replaced_by_generic_connection_error(panel_module, monkeypatch):
    from lfs_plugins.gallery_messages import localize_message
    monkeypatch.setattr(panel_module.lf.ui, 'tr', lambda key: {
        'asset_manager.gallery.state.waiting': 'Waiting for connection…',
        'asset_manager.gallery.error.connection': 'Connection failed',
    }.get(key, key))
    assert localize_message('Waiting for connection…') == 'Waiting for connection…'
    assert localize_message('Waiting for the portal connection. The transfer will resume automatically.') == 'Waiting for connection…'


@pytest.mark.parametrize('probe_result', ['wrong_owner', 'unauthorized'])
def test_connection_probe_does_not_resume_under_another_account(tmp_path, monkeypatch, connection_clock, probe_result):
    from test_gallery_sync import connected, finish, Client
    from lfs_plugins.portal_account import PortalHTTPError
    service = connected(tmp_path, monkeypatch)
    source = tmp_path / 'scene.ply'
    source.write_bytes(b'12345678')
    attempts = []
    def upload(*args, **kwargs):
        attempts.append(1)
        raise TimeoutError()
    monkeypatch.setattr(Client, 'upload', upload, raising=False)
    identifier = service.queue_upload(source, {'title': 'Scene'}, 'project')
    finish(service)
    def probe(*args):
        if probe_result == 'unauthorized':
            raise PortalHTTPError(401, 'Expired')
        return {'id': 'two', 'gallerySyncVersion': 1}
    monkeypatch.setattr(Client, '_request', probe)
    connection_clock[1][-1].fire()
    finish(service)
    assert service._job(identifier)['status'] == 'error'
    assert len(attempts) == 1 and service._connection_timer is None


def test_legacy_outage_pause_migrates_to_waiting_on_restart(tmp_path, monkeypatch, connection_clock):
    from test_gallery_sync import connected, finish, gallery_sync
    service = connected(tmp_path, monkeypatch)
    source = tmp_path / 'scene.ply'
    source.write_bytes(b'12345678')
    monkeypatch.setattr(service, 'resume', lambda _: None)
    identifier = service.queue_upload(source, {'title': 'Scene'}, 'project')
    service._job(identifier).update(status='paused', retryAt=1005,
        message='Waiting for the portal connection. The transfer will resume automatically.')
    service._save()
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    assert restarted._job(identifier)['status'] == 'waiting'
    assert restarted._connection_timer.delay == 5
