"""Gallery stress-harness product regressions."""
import copy
from importlib import import_module
from types import SimpleNamespace

import pytest

from test_gallery_controller import gallery, panel_module, scene

@pytest.fixture
def removal_sequence(gallery, tmp_path, monkeypatch):
    """Real journal/controller/client; byte transfers and HTTP responses are controlled."""
    import io
    import json
    import threading
    from pathlib import Path
    from test_asset_manager_panel import _gallery_fixture
    from test_gallery_sync import finish
    controller, _, _ = gallery
    sync = import_module('lfs_plugins.gallery_sync')
    portal = import_module('lfs_plugins.portal_gallery')
    account_module = import_module('lfs_plugins.portal_account')
    module = import_module('lfs_plugins.gallery_controller')
    manager, local, remote = _gallery_fixture(import_module('lfs_plugins.asset_manager_panel'))
    remote.update(id='7e812ba8-6cfb-4307-a0bc-da8e395bb721', revision='reviewed', contentRevision='reviewed', metadataRevision='reviewed')
    other = dict(remote, id='52dde335-aab0-412d-a788-de4a5b414d4f', sourceFormat='licht', contentLength=8)
    remote_scenes = [other]
    requests = []
    reply = {'status': 204}
    deleting, release_delete = threading.Event(), threading.Event()
    release_delete.set()
    class Account:
        base_url = 'https://portal.lichtfeld.io'
        _client_version, _timeout = 'test', 1
        email = 'test@example.com'
        def snapshot(self):
            return SimpleNamespace(signed_in=True, email=self.email, connected_since='session')
        def request_json_authenticated(self, method, path, body=None, **kwargs):
            return account_module.PortalAccountService._request_json_once(self, method, path, body)
        def request_response_authenticated(self, method, path, *, body=None, **kwargs):
            return account_module.PortalAccountService._request_json_once(self, method, path, body,
                response_options={'max_bytes': 1024 * 1024})
    def opened(request, **kwargs):
        body = json.loads(request.data) if request.data else None
        requests.append((request.method, request.full_url, body))
        status = 200
        if request.method == 'DELETE':
            assert request.full_url.endswith('/splats/' + remote['id'])
            assert body == {'baseRevisions': {'content': 'reviewed', 'metadata': 'reviewed'}}
            deleting.set()
            assert release_delete.wait(3)
            status = reply['status']
            if status == 'timeout':
                raise TimeoutError('Timed out deleting scene')
            if status in (200, 204, 404):
                remote_scenes[:] = [s for s in remote_scenes if s['id'] != remote['id']]
            payload = {} if status in (200, 204) else {'error': 'delete refused'}
        elif request.full_url.endswith('/me'):
            payload = {"storageHosts": ["portal.example"], "sourceFormats": ["licht"], 'id': 'owner', 'gallerySyncVersion': 1, "revisionDomains": 1}
        else:
            assert request.full_url.endswith('/splats')
            payload = {'scenes': copy.deepcopy(remote_scenes), 'nextCursor': None}
        response = io.BytesIO(b'' if status == 204 else json.dumps(payload).encode())
        response.status, response.headers = status, {}
        return response
    monkeypatch.setattr(account_module, 'urlopen', opened)
    service = sync.GallerySync(Account(), tmp_path / 'sync')
    monkeypatch.setattr(service, '_cache_posters', lambda *a: None)
    service.refresh()
    finish(service)
    source = tmp_path / 'prepared.licht'
    source.write_bytes(b'payload!')
    attempts = []
    def upload(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise ValueError('Unsafe portal URL')
        remote_scenes.append(copy.deepcopy(remote))
        return {'scene': copy.deepcopy(remote)}
    def download(_client, scene_id, destination, **kwargs):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'payload!')
        return copy.deepcopy(other)
    monkeypatch.setattr(portal.PortalGalleryClient, 'upload', upload)
    monkeypatch.setattr(portal.PortalGalleryClient, 'download', download)
    upload_id = service.queue_upload(source, {'title': remote['title']}, local['id'])
    finish(service)
    assert service._job(upload_id)['status'] == 'error'
    assert not service.snapshot()['links']
    service.resume(upload_id)  # Retry the prepared snapshot of the closed project.
    finish(service)
    service.download(other)
    finish(service)
    assert [j['status'] for j in service.snapshot()['jobs']] == ['completed', 'completed']
    controller.service = service
    controller._identity = service.identity()
    controller._state = service.snapshot()
    controller._message = ''
    monkeypatch.setattr(controller, '_schedule_poll', lambda: None)
    monkeypatch.setattr(controller, '_schedule_poll', lambda: None)
    manager._gallery_controller = controller
    manager._gallery_state = service.snapshot()
    manager._select_asset_id(local['id'])
    # Mimic a native guard awaiting its next phase poll after the completed pull.
    controller._native_use = service.local_use(service.snapshot()['jobs'][-1]['id'])
    controller._native_use.__enter__()
    controller._phase_poll_scheduled = True
    yield SimpleNamespace(service=service, manager=manager, controller=controller, local=local,
        remote=remote, requests=requests, reply=reply, deleting=deleting, release_delete=release_delete,
        remote_scenes=remote_scenes, dialogs=module.lf._test_state.confirm_dialogs, finish=lambda: finish(service))
    release_delete.set()
    finish(service)
    controller._release_native_use()

@pytest.mark.parametrize('status', [200, 204, 404, 202, 403, 409, 500, 'timeout'])
def test_P5_publish_pull_other_confirm_remove_waits_for_delete_response(removal_sequence, status):
    import json
    run = removal_sequence
    service, manager = run.service, run.manager
    before = copy.deepcopy(service.snapshot()['links'])
    journal = service._journal.read_bytes()
    run.reply['status'] = status
    run.release_delete.clear()
    manager._gallery_notice = 'Wait for the current operation or pause it first.'
    manager._gallery_command('remove')
    assert len(run.dialogs) == 1 and run.controller._decision_pending
    assert run.controller._native_use is None
    manager._gallery_command('remove')
    assert len(run.dialogs) == 1 and not manager._gallery_notice
    assert not any(method == 'DELETE' for method, _, _ in run.requests)
    _, _, buttons, callback = run.dialogs[0]
    callback(buttons[-1])
    assert run.deleting.wait(3)
    assert service.snapshot()['links'] == before
    assert service._journal.read_bytes() == journal
    callback(buttons[-1])  # A duplicate native callback cannot issue another DELETE.
    run.release_delete.set()
    run.finish()
    deletes = [r for r in run.requests if r[0] == 'DELETE']
    assert deletes == [('DELETE', 'https://portal.lichtfeld.io/api/gallery/v1/splats/' + run.remote['id'],
                        {'baseRevisions': {'content': 'reviewed', 'metadata': 'reviewed'}})]
    state = service.snapshot()
    persisted = next(iter(json.loads(service._journal.read_bytes())['accounts'].values()))['links']
    if status in (200, 204, 404):
        assert state['links'][run.local['id']]['remoteDeleted'] is True
        assert persisted == state['links']
        assert all(s['id'] != run.remote['id'] for s in state['scenes'])
        assert all(s['id'] != run.remote['id'] for s in run.remote_scenes)
        assert state['completion']['kind'] == 'remove'
    else:
        assert state['links'] == before and persisted == before
        assert service._journal.read_bytes() == journal
        assert any(s['id'] == run.remote['id'] for s in state['scenes'])
        assert state['completion']['kind'] != 'remove'
        assert state['message'] and 'Removed' not in state['message']
    assert not run.controller._message and not manager._gallery_notice

@pytest.mark.parametrize('operation', ['remove', 'unlink', 'update'])
def test_P5_metadata_waits_for_listing_refresh_and_uses_reloaded_journal(removal_sequence, monkeypatch, operation):
    import threading
    run = removal_sequence
    service = run.service
    refreshing, release = threading.Event(), threading.Event()
    def posters(*args):
        refreshing.set()
        assert release.wait(3)
    monkeypatch.setattr(service, '_cache_posters', posters)
    # Force refresh to replace the bucket, as it does after a peer journal write.
    old_bucket = service._bucket()
    service._disk_digest = None
    service.refresh()
    try:
        assert refreshing.wait(3)
        assert service.busy and not service.metadata_busy
        assert service.message == 'Gallery is up to date.'
        assert service._bucket() is not old_bucket
        if operation == 'update':
            portal = import_module('lfs_plugins.portal_gallery')
            monkeypatch.setattr(portal.PortalGalleryClient, 'scene', lambda *a: copy.deepcopy(run.remote))
            monkeypatch.setattr(portal.PortalGalleryClient, 'update',
                lambda *a, **k: dict(run.remote, title='Edited'))
            service.edit(run.remote['id'], run.remote, {'title': 'Edited'})
        else:
            run.manager._gallery_command(operation)
            assert len(run.dialogs) == 1
            _, _, buttons, callback = run.dialogs[0]
            callback(buttons[-1])
        assert not run.controller._message and not run.manager._gallery_notice
        assert service.metadata_busy
        assert not run.deleting.is_set()
    finally:
        release.set()
        run.finish()
    links = service.snapshot()['links']
    if operation == 'remove':
        assert links[run.local['id']]['remoteDeleted'] is True
    elif operation == 'unlink':
        assert run.local['id'] not in links
    else:
        assert links[run.local['id']]['metadata']['title'] == 'Edited'
    assert not old_bucket['links'][run.local['id']].get('remoteDeleted')

@pytest.mark.parametrize('refusal', ['busy', 'launch', 'replacement', 'cancel'])
def test_P5_refused_remove_keeps_link_and_explains(removal_sequence, monkeypatch, refusal):
    import threading
    run = removal_sequence
    service = run.service
    before = copy.deepcopy(service.snapshot()['links'])
    journal = service._journal.read_bytes()
    run.manager._gallery_command('remove')
    assert len(run.dialogs) == 1
    _, _, buttons, callback = run.dialogs[0]
    release = threading.Event()
    if refusal == 'busy':
        service._launch(lambda: release.wait(3))
    elif refusal == 'launch':
        monkeypatch.setattr(service, '_launch', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Worker unavailable')))
    elif refusal == 'replacement':
        job = service._bucket()['jobs'][0]
        job.update(status='paused', metadata={'replaceSceneId': run.remote['id']})
    try:
        callback(buttons[0] if refusal == 'cancel' else buttons[-1])
    finally:
        release.set()
        run.finish()
    assert not any(r[0] == 'DELETE' for r in run.requests)
    assert service.snapshot()['links'] == before
    assert service._journal.read_bytes() == journal
    if refusal != 'cancel':
        assert run.controller._message or service.message == 'Discard the unfinished replacement before removing this gallery item.'

def test_P5_queued_remove_cannot_cross_account_boundary(removal_sequence, monkeypatch):
    import threading
    run = removal_sequence
    refreshing, release = threading.Event(), threading.Event()
    def posters(*args):
        refreshing.set()
        assert release.wait(3)
    monkeypatch.setattr(run.service, '_cache_posters', posters)
    before = run.service._journal.read_bytes()
    run.service.refresh()
    try:
        assert refreshing.wait(3)
        run.service.remove(run.remote['id'], run.remote['revision'])
        run.service.account.email = 'different@example.com'
    finally:
        release.set()
        run.finish()
    assert not any(r[0] == 'DELETE' for r in run.requests)
    assert run.service._journal.read_bytes() == before
    assert 'account changed' in run.service.message

def test_P5_metadata_thread_start_failure_retains_refresh_ownership(removal_sequence, monkeypatch):
    import threading
    run = removal_sequence
    refreshing, release = threading.Event(), threading.Event()
    def posters(*args):
        refreshing.set()
        assert release.wait(3)
    monkeypatch.setattr(run.service, '_cache_posters', posters)
    before = run.service._journal.read_bytes()
    run.service.refresh()
    try:
        assert refreshing.wait(3)
        original = run.service._thread
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, 'start', lambda self: (_ for _ in ()).throw(RuntimeError('Cannot start worker')))
            with pytest.raises(RuntimeError, match='Cannot start worker'):
                run.service.remove(run.remote['id'], run.remote['revision'])
        assert run.service._thread is original and run.service.busy and not run.service.metadata_busy
    finally:
        release.set()
        run.finish()
    assert run.service._journal.read_bytes() == before
    assert not any(r[0] == 'DELETE' for r in run.requests)

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
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
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
    source = tmp_path / 'scene.licht'
    source.write_bytes(b'12345678')
    monkeypatch.setattr(service, 'resume', lambda _: None)
    identifier = service.queue_upload(source, {'title': 'Scene', '_commitUuid': 'saved'}, 'project')
    job = service._job(identifier)
    upload_id = str(uuid.uuid4())
    job.update(status='running', completed=3, checkpoint=dict(origin=service.account.base_url, owner='one',
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(), idempotencyKey='stable', uploadId=upload_id,
        request=dict(title='Scene', sourceFormat='licht', contentLength=8)))
    service._save()
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    recovered = restarted.snapshot()['jobs'][0]
    assert recovered['status'] == 'paused' and recovered['interrupted']
    puts, creates = [], []
    result = dict(id=str(uuid.uuid4()), revision='new', title='Scene', description='', visibility='private', viewerSettings={}, contentRevision='new', metadataRevision='new')
    def request(_client, method, path, body=None):
        if path == '/me':
            return dict(id='one', gallerySyncVersion=1, sourceFormats=["licht"])
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
def test_outage_exhaustion_requires_manual_resume(tmp_path, monkeypatch, exception):
    from test_gallery_sync import connected, finish, Client
    service = connected(tmp_path, monkeypatch)
    source = tmp_path / 'scene.licht'
    source.write_bytes(b'ply-data')
    def upload(*args, **kwargs):
        kwargs['on_checkpoint']({'uploadId': 'retained'})
        kwargs['on_progress'](4, 8)
        raise exception
    monkeypatch.setattr(Client, 'upload', upload, raising=False)
    service.queue_upload(source, {'title': 'Scene'}, 'project')
    finish(service)
    job = service.snapshot()['jobs'][0]
    assert job['status'] == 'paused' and job['message'] == 'Paused (connection lost)'
    assert job['completed'] == 4 and job['checkpoint']['uploadId'] == 'retained'
    assert job['message'] == 'Paused (connection lost)'
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
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: {'active': False}, raising=False)
    monkeypatch.setattr(module.lf, 'prepare_gallery_project', lambda *a: actions.append(a), raising=False)
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
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
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
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
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
        return dict(remote, **metadata, revision='patched', contentRevision=remote['contentRevision'], metadataRevision='patched')
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
    scene = dict(id='00000000-0000-0000-0000-000000000001', contentLength=total, revision='r1', contentRevision='r1', metadataRevision='r1')
    def request(method, path, body=None):
        if path.endswith('/me'):
            return {"storageHosts": ["portal.example"], 'revisionDomains': 1, 'maxFileBytes': total}
        if path.endswith('/download'):
            return {'url': 'https://portal.example/content', 'scene': scene}
        return scene
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url='https://portal.example',
        request_json_authenticated=request, _client_version='test'))
    def opened(*args, **kwargs):
        response = io.BytesIO(data)
        response.status, response.headers = 200, {'Content-Length': str(len(data))}
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
    monkeypatch.setattr(panel, '_schedule_poll', lambda: None)
    export_state = dict(active=False)
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: export_state, raising=False)
    def prepare(project_path, path, format, commit):
        assert project_path == str(source) and commit == 'after'
        gallery_preparation.unpack_project(tmp_path, source, Path(path))
        shutil.copyfile(source, Path(path)/'project.licht')
        export_state.update(path=path, outcome='completed', commit_uuid=commit)
    monkeypatch.setattr(module.lf, 'prepare_gallery_project', prepare, raising=False)
    monkeypatch.setattr(gallery_preparation, 'publication_view_metadata', lambda *args: dict(
        exposure=1, cameraPath=restored[-1] if both else None, environment=environment))
    uploads = []
    def upload(_client, path, metadata, **kwargs):
        assert Path(path).suffix == '.licht' and Path(path).is_file()
        uploads.append(copy.deepcopy(metadata))
        return {'scene': dict(remote, title=metadata['title'], viewerSettings=metadata['viewerSettings'], revision='uploaded', contentRevision='uploaded', metadataRevision='uploaded')}
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

@pytest.mark.parametrize('pinned', [False, True], ids=['restart', 'range'])
@pytest.mark.parametrize('failure', ['socket', 'timeout', 'eof', 'incomplete_read', 'server_closed', 'server_closed_wrapped', 'oversized', 'header_short', 'header_long'])
def test_download_transport_failure_classification(tmp_path, monkeypatch, connection_clock, pinned, failure):
    import hashlib
    import io
    import json
    import urllib.error
    from http.client import IncompleteRead, RemoteDisconnected
    from pathlib import Path
    import test_gallery_sync
    from test_gallery_sync import connected, finish
    from lfs_plugins import gallery_sync
    monkeypatch.setattr(test_gallery_sync, 'gallery_sync', gallery_sync)
    from lfs_plugins import portal_gallery, portal_retry
    from lfs_plugins.gallery_controller import asset_sync_state
    from lfs_plugins.gallery_transfer_panel import transfer_rows

    service = connected(tmp_path, monkeypatch)
    data = (Path(__file__).parents[1] / 'data' / 'portable-sog.licht').read_bytes()
    prefix = data[:64]
    remote = scene(sourceFormat='licht', contentLength=len(data))
    remote['id'] = '28d8880e-96d2-46a0-9226-bb62976392b2'
    reachable, requests, probes, messages, progress = [True], [], [], [], []
    headers = {'Content-Length': str(len(data))}
    if pinned:
        headers.update({'ETag': '"pinned-v1"', 'Accept-Ranges': 'bytes'})

    class Interrupted(io.BytesIO):
        status = 200

        def read(self, size=-1):
            if failure.startswith('header_'):
                pytest.fail('Read body before rejecting storage size mismatch')
            if not self.tell():
                reachable[0] = False
                return super().read(len(prefix))
            if failure == 'socket':
                raise ConnectionResetError('Portal stopped mid-stream')
            if failure == 'timeout':
                raise TimeoutError('Portal stopped mid-stream')
            if failure == 'incomplete_read':
                raise IncompleteRead(b'', len(data) - len(prefix))
            if failure == 'server_closed':
                raise RemoteDisconnected('Portal closed the response')
            if failure == 'server_closed_wrapped':
                raise urllib.error.URLError(RemoteDisconnected('Portal closed the response'))
            if failure == 'oversized':
                return data
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
            response.headers = dict(headers)
            if failure.startswith('header_'):
                response.headers['Content-Length'] = str(len(data) + (-1 if failure == 'header_short' else 1))
            return response
        if not reachable[0]:
            raise ConnectionRefusedError('Portal offline')
        offset = len(prefix) if pinned else 0
        assert request.get_header('Range') == (f'bytes={offset}-' if pinned else None)
        assert request.get_header('If-range') == ('"pinned-v1"' if pinned else None)
        response = Recovered(data[offset:])
        response.status = 206 if pinned else 200
        response.headers = dict(headers, **{'Content-Length': str(len(data) - offset)})
        if pinned:
            response.headers['Content-Range'] = f'bytes {offset}-{len(data)-1}/{len(data)}'
        return response

    def request(client, method, path, *args):
        if path == '/me':
            probes.append(reachable[0])
            if not reachable[0]:
                raise ConnectionRefusedError('Portal offline')
            return {"storageHosts": ["portal.example"], "sourceFormats": ["licht"], 'id': 'one', 'gallerySyncVersion': 1, "revisionDomains": 1}
        assert reachable[0]
        if path.endswith('/download'):
            return {'scene': remote, 'url': 'https://portal.example/storage'}
        return remote

    monkeypatch.setattr(portal_gallery.PortalGalleryClient, '_request', request)
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    # An interruption must never enter backoff or retry, even when reachable.
    monkeypatch.setattr(portal_retry.time, 'sleep', lambda _: pytest.fail('Automatic download retry'))
    monkeypatch.setattr(gallery_sync, 'PortalGalleryClient', portal_gallery.PortalGalleryClient)
    service.download(remote)
    finish(service)
    job = service.snapshot()['jobs'][0]
    if failure in ('oversized', 'header_short', 'header_long'):
        assert job['status'] == 'error' and job['retryable'] is False
        assert job['message'] == ('Gallery download exceeds its declared size' if failure == 'oversized'
            else 'Gallery download is incomplete or larger than the portal declared')
        if failure.startswith('header_'):
            assert job['completed'] == 0 and not job.get('checkpoint')
        assert len(requests) == 1 and not connection_clock[1]
        assert not list((tmp_path / 'downloads').iterdir())
        assert not service.snapshot()['links'] and service.snapshot()['completion'] is None
        assert not transfer_rows(service.snapshot())[0]['can_resume']
        assert asset_sync_state({}, None, remote, [job])['action'] == ''
        with pytest.raises(ValueError, match=job['message']):
            service.resume(job['id'])
        # The refusal must survive journal reload, not just this worker.
        from test_gallery_sync import Client
        monkeypatch.setattr(gallery_sync, 'PortalGalleryClient', Client)
        restored = gallery_sync.GallerySync(service.account, tmp_path)
        restored.refresh()
        finish(restored)
        assert restored.snapshot()['jobs'][0]['retryable'] is False
        return
    assert job['status'] == 'paused', (job['message'], len(requests), probes)
    assert job['message'] == 'Paused (connection lost)'
    assert job['completed'] == len(prefix) and len(requests) == 1
    destination = Path(job['path'])
    partial = destination.with_name('.' + destination.name + '.part')
    assert not destination.exists()
    assert partial.exists() == pinned
    if pinned:
        assert partial.read_bytes() == prefix
    row = transfer_rows(service.snapshot())[0]
    assert row['phase'].endswith('paused') or row['phase'] == 'Paused'
    assert not row['can_pause'] and row['can_resume']
    assert asset_sync_state({}, None, remote, [job])['state'] == 'paused'
    persisted = json.loads((tmp_path / 'sync.json').read_text())
    assert 'storage' not in json.dumps(persisted)  # No signed URL in the journal.
    assert not connection_clock[1], "No outage retry timer may be scheduled"
    reachable[0] = True
    assert not service.busy and len(requests) == 1
    service.resume(job['id'])
    finish(service)
    job = service._job(job['id'])
    assert job['status'] == 'completed', job['message']
    assert destination.read_bytes() == data and not partial.exists()
    assert job['sha256'] == hashlib.sha256(data).hexdigest()
    assert job['completed'] == len(data) and len(requests) == 2
    assert probes == []
    if not pinned:
        assert 'Restarting from zero' in messages[0]
        assert progress[0] == 0


@pytest.mark.parametrize('failure', ['server_closed', 'server_closed_wrapped', 'incomplete_read', 'socket', 'timeout', 'unacknowledged', 'http500', 'http429'])
def test_upload_transport_failure_classification(tmp_path, monkeypatch, failure):
    import io
    import urllib.error
    from http.client import IncompleteRead, RemoteDisconnected
    from pathlib import Path
    import test_gallery_sync
    from lfs_plugins import gallery_sync, portal_gallery, portal_retry
    from lfs_plugins.gallery_transfer_panel import transfer_rows
    monkeypatch.setattr(test_gallery_sync, 'gallery_sync', gallery_sync)
    service = test_gallery_sync.connected(tmp_path, monkeypatch)
    path = tmp_path / 'upload.licht'
    path.write_bytes((Path(__file__).parents[1] / 'data' / 'portable-sog.licht').read_bytes())
    upload_id = '28d8880e-96d2-46a0-9226-bb62976392b2'
    calls, completed = [], []
    def request(client, method, route, body=None):
        if route == '/me':
            return dict(id='one', gallerySyncVersion=1, revisionDomains=1, sourceFormats=['licht'])
        if route.endswith('/part-upload-urls'):
            return {'urls': [dict(partNumber=body['parts'][0], url='https://portal.example/storage')]}
        if route.endswith('/complete'):
            completed.append(body)
            return dict(id=upload_id, status='completed', scene=scene())
        assert route.endswith('/uploads')
        return dict(id=upload_id, status='uploading', partSize=path.stat().st_size // 2)
    def opened(req, **kwargs):
        calls.append(req)
        assert req.method == 'PUT'
        if len(calls) > 1:
            if failure == 'server_closed':
                raise RemoteDisconnected('Portal closed without an upload acknowledgment')
            if failure == 'server_closed_wrapped':
                raise urllib.error.URLError(RemoteDisconnected('Portal closed without an upload acknowledgment'))
            if failure == 'incomplete_read':
                raise IncompleteRead(b'', 1)
            if failure == 'socket':
                raise ConnectionResetError('Socket reset during upload')
            if failure == 'timeout':
                raise TimeoutError('Socket timed out during upload')
            if failure.startswith('http'):
                raise urllib.error.HTTPError(req.full_url, int(failure[4:]), 'Busy', {}, io.BytesIO())
        response = io.BytesIO()
        response.status = 200
        response.headers = {} if len(calls) > 1 else {'ETag': '"part1"'}
        return response
    monkeypatch.setattr(portal_gallery.PortalGalleryClient, '_request', request)
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    monkeypatch.setattr(portal_gallery, 'retry_call', lambda operation, **kwargs:
                        portal_retry.retry_call(operation, sleep=lambda _: None, **kwargs))
    monkeypatch.setattr(gallery_sync, 'PortalGalleryClient', portal_gallery.PortalGalleryClient)
    identifier = service.queue_upload(path, {'title': 'Upload'}, 'project')
    test_gallery_sync.finish(service)
    job = service.snapshot()['jobs'][0]
    transient = failure in ('socket', 'timeout')
    resumable = transient or failure.startswith('http')
    assert job['status'] == ('paused' if transient else 'error'), job
    assert job['completed'] == path.stat().st_size // 2
    assert not completed and not service.snapshot()['links']
    assert transfer_rows(service.snapshot())[0]['can_resume'] == resumable
    assert len(calls) == (5 if resumable else 2)
    if transient:
        assert job['message'] == 'Paused (connection lost)'
    elif not resumable:
        assert job['retryable'] is False
        assert job['message'] == ('Storage did not acknowledge the upload part' if failure == 'unacknowledged'
                                  else 'The portal closed the upload without acknowledging it. Start a new upload.')
        with pytest.raises(ValueError, match=job['message']):
            service.resume(identifier)

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
    remote_scene = dict(id=str(uuid.uuid4()), revision='published', title='Scene', viewerSettings={}, contentRevision='published', metadataRevision='published')
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
