# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gallery storage, wire contract and durable recovery regressions."""
import hashlib
import io
import json
import shutil
import threading
import time
from types import SimpleNamespace
import urllib.error
import uuid

import pytest

from lfs_plugins import credential_storage, gallery_sync, portal_account, portal_gallery, portal_retry, portal_security
from test_portal_account import StubUrlopen, write_credentials
from test_gallery_sync import connected, finish, Client
from test_gallery_controller import gallery, scene
from test_asset_manager_panel import panel_module, _gallery_fixture

class FakeBackend:
    def __init__(self):
        self.value = None
        self.writes = 0
        self.deleted = 0

    def read(self):
        return self.value

    def write(self, value):
        self.value = value
        self.writes += 1

    def delete(self):
        self.value = None
        self.deleted += 1

@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(portal_retry.time, 'sleep', lambda _: None)

def test_credentials_migrate_once_and_refresh_uses_backend(tmp_path):
    path, backend = tmp_path / 'credentials.json', FakeBackend()
    write_credentials(path)
    account = portal_account.PortalAccountService(credentials_path=path, storage_backend=backend)
    assert account.snapshot().signed_in and not path.exists()
    assert json.loads(backend.value)['refresh_token'] == 'refresh-old'
    assert backend.writes == 1
    restarted = portal_account.PortalAccountService(credentials_path=path, storage_backend=backend)
    assert restarted.snapshot().signed_in and backend.writes == 1
    restarted._save_credentials(account._current_credentials())
    assert backend.writes == 2 and not path.exists()

def test_failed_secure_migration_preserves_plaintext(tmp_path):
    path, backend = tmp_path / 'credentials.json', FakeBackend()
    write_credentials(path)
    original = path.read_bytes()
    backend.write = lambda _: (_ for _ in ()).throw(OSError('locked'))
    account = portal_account.PortalAccountService(credentials_path=path, storage_backend=backend)
    assert path.read_bytes() == original and not account.snapshot().signed_in

@pytest.mark.parametrize('offline', [False, True])
def test_signout_revokes_and_wipes_backend_and_stale_files(tmp_path, monkeypatch, offline):
    path, backend = tmp_path / 'credentials.json', FakeBackend()
    write_credentials(path)
    account = portal_account.PortalAccountService(credentials_path=path, storage_backend=backend)
    path.write_bytes(backend.value)
    path.with_suffix('.dpapi').write_bytes(b'old encrypted copy')
    network = StubUrlopen(OSError('offline') if offline else {})
    monkeypatch.setattr(portal_account, 'urlopen', network)
    account.sign_out()
    assert network.requests[0].full_url.endswith(portal_account.REVOKE_PATH)
    assert network.requests[0].get_header('Authorization') == 'Bearer access-old'
    assert not account.snapshot().signed_in
    assert backend.value is None and backend.deleted == 1
    assert not path.exists() and not path.with_suffix('.dpapi').exists()

def test_signout_wipes_even_when_refresh_storage_fails(tmp_path, monkeypatch):
    path, backend = tmp_path / 'credentials.json', FakeBackend()
    write_credentials(path, access_expires_at=1)
    account = portal_account.PortalAccountService(credentials_path=path, storage_backend=backend)
    monkeypatch.setattr(account, '_refresh_tokens', lambda *_, **kw: (_ for _ in ()).throw(OSError('storage unavailable')))
    account.sign_out()
    assert backend.value is None and not account.snapshot().signed_in

@pytest.mark.parametrize('system,security,expected', [('Windows', None, credential_storage.DPAPIBackend),
    ('Darwin', '/usr/bin/security', credential_storage.KeychainBackend), ('Darwin', None, credential_storage.FileBackend),
    ('Linux', '/usr/bin/security', credential_storage.FileBackend)])
def test_default_credential_backend_selection(tmp_path, monkeypatch, system, security, expected):
    monkeypatch.setattr(credential_storage.platform, 'system', lambda: system)
    monkeypatch.setattr(credential_storage.shutil, 'which', lambda _: security)
    assert type(credential_storage.default_backend(tmp_path / 'credentials.json')) is expected

def test_dpapi_file_contains_only_protected_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(credential_storage, '_dpapi', lambda value, protect: b'cipher:' + value[::-1] if protect else value[7:][::-1])
    path = tmp_path / 'credentials.dpapi'
    backend = credential_storage.DPAPIBackend(path)
    backend.write(b'access_token plaintext')
    assert b'plaintext' not in path.read_bytes()
    assert backend.read() == b'access_token plaintext'
    backend.delete()
    assert not path.exists()

def test_keychain_secret_passes_via_stdin_not_argv(tmp_path, monkeypatch):
    import base64
    calls = []
    secret = b'{"refresh_token":"private"}'
    encoded = base64.b64encode(secret)
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert 'private' not in str(argv) and encoded.decode() not in str(argv)
        assert kwargs['timeout'] == 15
        return SimpleNamespace(returncode=0, stdout=encoded + b'\n', stderr=b'')
    monkeypatch.setattr(credential_storage.subprocess, 'run', run)
    backend = credential_storage.KeychainBackend(tmp_path / 'credentials.json', '/usr/bin/security')
    backend.write(secret)
    assert encoded in calls[0][1]['input']
    assert calls[0][0] == ['/usr/bin/security', '-i']
    backend.delete()
    assert 'delete-generic-password' in calls[-1][0]

@pytest.mark.parametrize('value', ['Authorization: Bearer secret', '{"access_token": "secret"}',
    "{'refresh_token': 'secret'}", 'user_code=secret', 'https://host/download?token=secret&ok=1',
    'x-amz-signature=secret'])
def test_redact_secret_fields(value):
    assert 'secret' not in portal_security.redact(value)

def test_redact_actual_tokens_and_user_code_in_unstructured_exception(tmp_path, monkeypatch, caplog):
    path = tmp_path / 'credentials.json'
    write_credentials(path, access='private-access', refresh='private-refresh')
    account = portal_account.PortalAccountService(credentials_path=path)
    account._set_linking(user_code='ABCD-1234', verification_uri='https://portal.lichtfeld.io/link/',
                         verification_uri_complete='', expires_at=time.time()+60, interval=1)
    text = portal_security.redact('failure private-access private-refresh ABCD-1234')
    assert text == 'failure [REDACTED] [REDACTED] [REDACTED]'
    assert 'ABCD-1234' not in caplog.text

@pytest.mark.parametrize('method,key,retried', [('GET', False, True), ('PATCH', False, False),
    ('DELETE', False, False), ('POST', False, False), ('POST', True, True)])
def test_account_retry_is_idempotency_gated(tmp_path, monkeypatch, method, key, retried):
    account = portal_account.PortalAccountService(credentials_path=tmp_path/'credentials.json', client_version='9.8.7')
    network = StubUrlopen((503, {}), {'ok': True})
    monkeypatch.setattr(portal_account, 'urlopen', network)
    body = {'idempotencyKey': 'stable'} if key else None
    if retried:
        assert account._request_json(method, '/api/gallery/v1/splats/uploads/x/complete', body) == {'ok': True}
    else:
        with pytest.raises(portal_account.PortalHTTPError):
            account._request_json(method, '/api/gallery/v1/splats/uploads/x/complete', body)
    assert len(network.requests) == (2 if retried else 1)
    assert all(r.get_header('User-agent') == 'LichtFeld-Studio/9.8.7' for r in network.requests)

@pytest.mark.parametrize('error', [TimeoutError(), urllib.error.URLError(TimeoutError()),
    portal_account.PortalHTTPError(429, 'busy', retry_after=500), portal_account.PortalHTTPError(502, 'gateway')])
def test_retry_delay_is_capped_and_attempt_count_is_bounded(error):
    calls, delays = [], []
    def fail():
        calls.append(1)
        raise error
    with pytest.raises(type(error)):
        portal_retry.retry_call(fail, idempotent=True, sleep=delays.append)
    assert len(calls) == 4 and len(delays) == 3 and all(0 < x <= 30 for x in delays)
    if getattr(error, 'status', None) == 429:
        assert delays == [30, 30, 30]

def _download_client(size, limit=None):
    identifier = str(uuid.uuid4())
    scene = {"contentRevision": 'r1', "metadataRevision": 'r1', 'id': identifier, 'contentLength': size, 'revision': 'r1'}
    def request(method, path, body=None):
        if path.endswith('/me'):
            return {"storageHosts": ["portal.example"], 'revisionDomains': 1, 'maxFileBytes': size if limit is None else limit}
        if path.endswith('/download'):
            return {'url': 'https://portal.example/presigned?token=opaque', 'scene': scene}
        return scene
    account = SimpleNamespace(base_url='https://portal.example', request_json_authenticated=request, _client_version='1.2.3')
    return portal_gallery.PortalGalleryClient(account), scene

def _response(data, status=200, headers=None):
    response = io.BytesIO(data)
    response.status, response.headers = status, headers or {}
    return response

def test_oversized_declared_download_rejected_before_writing(tmp_path, monkeypatch):
    client, scene = _download_client(100, limit=99)
    monkeypatch.setattr(portal_gallery, 'urlopen', lambda *a, **k: pytest.fail('Storage request before size check'))
    with pytest.raises(portal_account.PortalProtocolError, match='file-size limit'):
        client.download(scene['id'], tmp_path / 'absent' / 'scene.licht')
    assert list(tmp_path.iterdir()) == []

@pytest.mark.parametrize('response_kind', ['initial', 'range_total', 'range_length', 'restart'])
@pytest.mark.parametrize('storage_size', [126016, 1174593])
def test_storage_header_mismatch_rejected_before_streaming(tmp_path, monkeypatch, response_kind, storage_size):
    total, offset = 1174592, 64
    client, scene = _download_client(total)
    destination = tmp_path / 'scene.licht'
    destination.write_bytes(b'existing project')
    partial = tmp_path / '.scene.licht.part'
    checkpoint = None
    if response_kind != 'initial':
        partial.write_bytes(b'x' * offset)
        checkpoint = dict(representationId='"pinned"', scene={key: scene.get(key)
            for key in ('id', 'contentRevision', 'metadataRevision', 'contentLength')})
    headers = {'ETag': '"pinned"', 'Accept-Ranges': 'bytes'}
    if response_kind.startswith('range_'):
        range_total = storage_size if response_kind == 'range_total' else total
        headers['Content-Range'] = f'bytes {offset}-{range_total-1}/{range_total}'
        if response_kind == 'range_length':
            headers['Content-Length'] = str(storage_size - offset)
    else:
        headers['Content-Length'] = str(storage_size)

    class UnreadResponse(io.BytesIO):
        def read(self, *args):
            pytest.fail('A header mismatch must fail before any body read')

    calls, checkpoints, progress = [], [], []
    def opened(request, **kwargs):
        calls.append(request)
        assert request.get_header('Range') == (None if response_kind == 'initial' else f'bytes={offset}-')
        response = UnreadResponse()
        response.status = 206 if response_kind.startswith('range_') else 200
        response.headers = headers
        return response
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    with pytest.raises(portal_gallery.GalleryTransferInvalid, match='incomplete or larger than the portal declared'):
        client.download(scene['id'], destination, checkpoint=checkpoint,
            on_checkpoint=checkpoints.append, on_progress=lambda *args: progress.append(args))
    assert len(calls) == 1 and not progress and not checkpoints
    assert destination.read_bytes() == b'existing project'
    assert not partial.exists()

def test_download_disk_preflight_includes_staging_backup_and_destination(tmp_path, monkeypatch):
    client, scene = _download_client(100)
    monkeypatch.setattr(portal_gallery.shutil, 'disk_usage', lambda _: SimpleNamespace(free=299))
    monkeypatch.setattr(portal_gallery, 'urlopen', lambda *a, **k: pytest.fail('Storage request before preflight'))
    with pytest.raises(ValueError, match='disk space'):
        client.download(scene['id'], tmp_path / 'absent' / 'scene.licht')
    assert list(tmp_path.iterdir()) == []

def test_same_portal_host_presigned_download_has_no_bearer_and_records_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(portal_gallery, "validate_download", lambda *args: None)  # Byte-transport unit boundary.
    client, scene = _download_client(4)
    calls, checkpoints = [], []
    def opened(request, **kwargs):
        calls.append(request)
        assert request.get_header('Authorization') is None
        assert request.get_header('User-agent') == 'LichtFeld-Studio/1.2.3'
        assert kwargs == {'timeout': 120, 'no_redirect': True}
        return _response(b'data')
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    client.download(scene['id'], tmp_path/'file.licht', on_checkpoint=checkpoints.append)
    assert (tmp_path/'file.licht').read_bytes() == b'data'
    assert checkpoints[-1]['sha256'] == hashlib.sha256(b'data').hexdigest()
    assert 'token=opaque' not in json.dumps(checkpoints)

def test_pinned_download_resumes_range_if_range_and_hashes_all_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(portal_gallery, "validate_download", lambda *args: None)  # Byte-transport unit boundary.
    data = b'a' * (1024 * 1024) + b'ending'
    client, scene = _download_client(len(data))
    checkpoints, calls = [], []
    cancel = threading.Event()
    def opened(request, **kwargs):
        calls.append(request)
        offset = 1024*1024 if len(calls) == 2 else 0
        headers = {'ETag': '"representation1"', 'Accept-Ranges': 'bytes'}
        if offset:
            assert request.get_header('Range') == f'bytes={offset}-'
            assert request.get_header('If-range') == '"representation1"'
            headers['Content-Range'] = f'bytes {offset}-{len(data)-1}/{len(data)}'
        return _response(data[offset:], 206 if offset else 200, headers)
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    destination = tmp_path/'file.licht'
    with pytest.raises(portal_gallery.GalleryTransferCanceled):
        client.download(scene['id'], destination, cancel=cancel, on_checkpoint=checkpoints.append,
                        on_progress=lambda *a: cancel.set())
    assert not destination.exists()
    assert (tmp_path/'.file.licht.part').stat().st_size == 1024*1024
    client.download(scene['id'], destination, checkpoint=checkpoints[-1], on_checkpoint=checkpoints.append)
    assert destination.read_bytes() == data
    assert checkpoints[-1]['sha256'] == hashlib.sha256(data).hexdigest()
    assert not (tmp_path/'.file.licht.part').exists()

@pytest.mark.parametrize('etag', [None, 'W/"weak"'])
def test_unpinned_download_restarts_with_clear_message(tmp_path, monkeypatch, etag):
    monkeypatch.setattr(portal_gallery, "validate_download", lambda *args: None)  # Byte-transport unit boundary.
    client, scene = _download_client(4)
    messages, requests = [], []
    partial = tmp_path/'.file.licht.part'
    partial.write_bytes(b'old')
    def opened(request, **kwargs):
        requests.append(request)
        return _response(b'data', headers={'ETag': etag, 'Accept-Ranges': 'bytes'})
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    client.download(scene['id'], tmp_path/'file.licht', checkpoint={'representationId': etag}, on_message=messages.append)
    assert requests[0].get_header('Range') is None
    assert messages and 'Restarting from zero' in messages[0]
    assert (tmp_path/'file.licht').read_bytes() == b'data'

def test_native_licht_embedded_path_is_rejected_before_publish(tmp_path, monkeypatch):
    from test_portable_project import rewrite_chapter, FIXTURES
    data = rewrite_chapter((FIXTURES/'portable-sog.licht').read_bytes(), b'REFS',
        lambda chapter: chapter['references'][0]['locator'].update(preferred='../../escape'))
    client, scene = _download_client(len(data))
    monkeypatch.setattr(portal_gallery, 'urlopen', lambda *a, **k: _response(data))
    with pytest.raises(ValueError):
        client.download(scene['id'], tmp_path/'project.licht')
    assert list(tmp_path.iterdir()) == []

@pytest.mark.parametrize('title', ['..', '../../etc/passwd', 'CON', 'prn', 'AUX', 'NUL', 'COM1', 'LPT9', 'COM¹',
                                  'a' * 300, 'name.  ', 'a' * 113 + ' .'])
def test_destination_names_are_cross_platform_safe(title):
    name = portal_security.safe_filename(title)
    assert len(name) <= 120 and '..' not in name and not any(c in name for c in '/\\:')
    assert name.endswith('.licht') and not name[:-6].endswith((' ', '.'))
    assert name.split('.')[0].upper() not in {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM¹', 'LPT9'}

def test_stuck_processing_never_spins_or_links(tmp_path, monkeypatch):
    client = portal_gallery.PortalGalleryClient(SimpleNamespace())
    identifier = str(uuid.uuid4())
    client.processing_deadline = time.time() - 1
    upload = {'id': identifier, 'status': 'processing', 'processing': {
        'stage': 'validating', 'totalBytes': 8, 'bytesProcessed': 4}}
    with pytest.raises(portal_gallery.GalleryProcessingTimeout, match='taking longer'):
        client._await_processing(upload, identifier, 8, threading.Event(), lambda _: None)
    service = connected(tmp_path, monkeypatch)
    path = tmp_path/'scene.licht'
    path.write_bytes(b'ply-data')
    monkeypatch.setattr(Client, 'upload', lambda *a, **k: (_ for _ in ()).throw(
        portal_gallery.GalleryProcessingTimeout('Portal is taking longer than expected · Retry / Keep waiting')), raising=False)
    service.queue_upload(path, {'title': 'Scene'}, 'project')
    finish(service)
    snapshot = service.snapshot()
    assert snapshot['jobs'][0]['status'] == 'paused'
    assert snapshot['jobs'][0]['serverProcessing'] and snapshot['jobs'][0]['needsAttention']
    assert snapshot['links'] == {}

def test_account_switch_drops_in_memory_scenes_list_and_posters(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    service.scenes = [{'id': 'private'}]
    service._list_etag = 'private-etag'
    poster = tmp_path/'posters'/'private.png'
    poster.parent.mkdir()
    poster.write_bytes(b'private')
    service._poster_entries['private'] = {'path': str(poster)}
    first_bucket = service._bucket()
    first_bucket['links']['one-project'] = {"contentRevision": 'r1', "metadataRevision": 'r1', 'sceneId': 'private', 'revision': 'r1'}
    service.account.email = 'two@example.com'
    service.account.owner = 'two'
    assert service.snapshot()['scenes'] == []
    assert service.scenes == [] and not service._poster_entries and service._list_etag is None
    assert not poster.exists()
    service.refresh()
    finish(service)
    assert service._bucket() is not first_bucket
    assert first_bucket['links']['one-project']['sceneId'] == 'private'
    assert service.snapshot()['links'] == {}
    service.account.base_url = 'https://other.example'
    service.refresh()
    finish(service)
    assert len(service._data['accounts']) == 3

def test_journal_corruption_drill_keeps_links_after_restoring_backup(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    service._bucket()['links']['project'] = {"contentRevision": 'r1', "metadataRevision": 'r1', 'sceneId': 'remote', 'revision': 'r1'}
    service._save()
    service._save()  # Last known-good generation, including the link.
    backup = tmp_path/'sync.json.bak'
    assert json.loads(backup.read_bytes())['accounts']
    (tmp_path/'sync.json').write_bytes(b'{"version":2,"accounts":')
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    assert restarted.snapshot()['storage_issue']
    assert 'files have been kept' in restarted.snapshot()['message']
    shutil.copyfile(backup, tmp_path/'sync.json')
    restarted.refresh()
    finish(restarted)
    assert not restarted.snapshot()['storage_issue']
    assert restarted.snapshot()['links']['project'] == {"contentRevision": 'r1', "metadataRevision": 'r1', 'sceneId': 'remote', 'revision': 'r1',
        'checkedAt': restarted.snapshot()['links']['project']['checkedAt']}

def test_journal_history_bound_preserves_pending_and_recovery(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    now = time.time()
    jobs = [{'id': str(n), 'status': 'completed', 'finishedAt': now-n, 'createdAt': now-n} for n in range(230)]
    jobs += [{'id': 'old', 'status': 'completed', 'finishedAt': now-31*86400},
             {'id': 'backup', 'status': 'completed', 'finishedAt': now-31*86400, 'localUpdate': {'backupPath': '/kept.licht'}},
             {'id': 'pending', 'status': 'paused', 'createdAt': 1}]
    service._bucket()['jobs'] = jobs
    service._prune_jobs()
    kept = {job['id'] for job in service._bucket()['jobs']}
    assert len(kept) == 202 and {'backup', 'pending'} <= kept and 'old' not in kept

def test_transient_attempt_counts_are_durable(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path = tmp_path/'scene.licht'
    path.write_bytes(b'data')
    calls = []
    def upload(*a, **kwargs):
        def operation():
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError()
            return {'scene': {"contentRevision": 'r1', "metadataRevision": 'r1', 'id': 'remote', 'revision': 'r1'}}
        return portal_retry.retry_call(operation, idempotent=True)
    monkeypatch.setattr(Client, 'upload', upload, raising=False)
    service.queue_upload(path, {'title': 'Scene'}, 'project')
    finish(service)
    assert service.snapshot()['jobs'][0]['attempts'] == 2
    saved = json.loads((tmp_path/'sync.json').read_bytes())
    assert next(iter(saved['accounts'].values()))['jobs'][0]['attempts'] == 2

@pytest.mark.parametrize('was_public,target', [(False, 'public'), (True, 'public'), (True, 'private')])
def test_legacy_visibility_does_not_prompt_or_enter_updates(gallery, monkeypatch, was_public, target):
    controller, state, actions = gallery
    monkeypatch.setattr(controller, '_project_identity', lambda: ('project', '/project.licht'))
    prompts = []
    remote = scene()
    remote['visibility'] = 'public' if was_public else 'private'
    module = __import__('lfs_plugins.gallery_controller', fromlist=['lf'])
    monkeypatch.setattr(module, 'capture_view', lambda _: {})
    monkeypatch.setattr(controller, 'preferences', lambda: {'askBeforePublic': True})
    monkeypatch.setattr(module.lf.ui, 'confirm_dialog', lambda *a: prompts.append(a), raising=False)
    monkeypatch.setattr(controller, '_publish', lambda *a, **kw: actions.append((a, kw)))
    controller._review_publish(remote, {'title': 'Scene', 'description': '', 'visibility': target}, 'sog', False, update=True)
    assert not prompts and len(actions) == 1
    assert "visibility" not in actions[0][0][0]

@pytest.mark.parametrize('action', ['remove', 'publish_new'])
def test_remove_and_publish_as_new_keep_published_scene_safe(gallery, monkeypatch, panel_module, action):
    controller, state, actions = gallery
    manager, local, remote = _gallery_fixture(panel_module)
    remote['visibility'] = 'public'
    local['viewing_copy'] = True
    manager._gallery_controller = controller
    manager._select_asset_id(local['id'])
    prompts = []
    monkeypatch.setattr(panel_module.lf.ui, 'confirm_dialog', lambda *a: prompts.append(a), raising=False)
    controller.service.remove = lambda *a: actions.append(a)
    if action == 'remove':
        manager._gallery_command(action)
        assert prompts and not actions
        assert prompts[0][1].endswith('confirm.remove')
        prompts[0][-1](prompts[0][-2][0])
        assert not actions
        manager._gallery_command(action)
        prompts[-1][-1](prompts[-1][-2][-1])
        assert actions == [(remote['id'], remote)]
    else:
        from lfs_plugins import gallery_file_panel
        reviews = []
        monkeypatch.setattr(gallery_file_panel, 'open_gallery_file_panel', lambda **kwargs: reviews.append(kwargs))
        manager._gallery_command(action)
        assert not prompts and not actions
        assert len(reviews) == 1 and reviews[0]['publish_new']
        assert reviews[0]['scene'] == remote
        assert 'visibility' not in reviews[0]['fields']

def test_watchdog_resume_offers_retry_and_keep_waiting(gallery, monkeypatch):
    controller, state, actions = gallery
    module = __import__('lfs_plugins.gallery_controller', fromlist=['lf'])
    state['jobs'] = [{'id': 'stuck', 'status': 'error', 'needsAttention': True}]
    prompts = []
    monkeypatch.setattr(module.lf.ui, 'confirm_dialog', lambda *a: prompts.append(a), raising=False)
    controller.service.resume = lambda *a, **kw: actions.append((a, kw))
    controller._action_resume('stuck')
    assert not actions and prompts[0][2][-2:] == ['projects.gallery.action.retry', 'projects.gallery.action.keep_waiting']
    prompts[0][-1](prompts[0][2][2])
    assert actions == [(('stuck',), {'keep_waiting': True})]

def test_expired_part_url_is_renewed_without_bearer_or_changed_bytes(tmp_path, monkeypatch):
    identifier = str(uuid.uuid4())
    urls, puts = [], []
    completed = []
    def request(method, path, body=None):
        if path.endswith('/me'):
            return {"storageHosts": ["portal.example"], 'id': 'owner', 'gallerySyncVersion': 1, 'sourceFormats': ['licht'], "revisionDomains": 1}
        if path.endswith('/part-upload-urls'):
            urls.append(body)
            return {'urls': [{'partNumber': 1, 'url': 'https://portal.example/signed?token=' + str(len(urls))}]}
        if path.endswith('/complete'):
            completed.append(body)
            assert body['idempotencyKey']
            return {'id': identifier, 'status': 'completed', 'scene': {'id': str(uuid.uuid4())}}
        return {'id': identifier, 'status': 'uploading', 'partSize': 8, 'uploadedParts': []}
    def put(request, **kwargs):
        puts.append(request)
        assert request.get_header('Authorization') is None
        assert request.data == b'ply-data'
        assert kwargs['no_redirect'] and kwargs['timeout'] == 120
        if len(puts) == 1:
            raise urllib.error.HTTPError(request.full_url, 403, 'Expired', {}, io.BytesIO())
        return _response(b'', headers={'ETag': 'part-etag'})
    monkeypatch.setattr(portal_gallery, 'urlopen', put)
    account = SimpleNamespace(base_url='https://portal.example', request_json_authenticated=request)
    client = portal_gallery.PortalGalleryClient(account)
    source = tmp_path/'scene.licht'
    source.write_bytes(b'ply-data')
    client.upload(source, {'title': 'Scene'})
    assert len(puts) == len(urls) == 2 and len(completed) == 1
    assert puts[0].full_url != puts[1].full_url

def test_invalid_range_response_preserves_destination_and_removes_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(portal_gallery, "validate_download", lambda *args: None)  # Byte-transport unit boundary.
    data = b'a' * (1024 * 1024) + b'end'
    client, scene = _download_client(len(data))
    calls, checkpoints = [], []
    cancel = threading.Event()
    def opened(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return _response(data, headers={'ETag': '"one"', 'Accept-Ranges': 'bytes'})
        return _response(b'end', 206, {'ETag': '"two"', 'Accept-Ranges': 'bytes',
            'Content-Range': f'bytes {1024*1024}-{len(data)-1}/{len(data)}'})
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    destination = tmp_path/'file.licht'
    destination.write_bytes(b'original')
    with pytest.raises(portal_gallery.GalleryTransferCanceled):
        client.download(scene['id'], destination, cancel=cancel, on_checkpoint=checkpoints.append,
                        on_progress=lambda *args: cancel.set())
    with pytest.raises(portal_account.PortalProtocolError, match='representation'):
        client.download(scene['id'], destination, checkpoint=checkpoints[-1])
    assert destination.read_bytes() == b'original'
    assert not (tmp_path/'.file.licht.part').exists()

def test_timeout_mid_read_restarts_unpinned_without_appending(tmp_path, monkeypatch):
    monkeypatch.setattr(portal_gallery, "validate_download", lambda *args: None)  # Byte-transport unit boundary.
    client, scene = _download_client(4)
    requests = []
    class Interrupted(io.BytesIO):
        status, headers = 200, {}
        def read(self, size=-1):
            if self.tell():
                raise TimeoutError()
            return super().read(2)
    def opened(request, **kwargs):
        requests.append(request)
        return Interrupted(b'old!') if len(requests) == 1 else _response(b'data')
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    with pytest.raises(TimeoutError):
        client.download(scene['id'], tmp_path/'file.licht')
    assert len(requests) == 1
    assert not (tmp_path/'.file.licht.part').exists()
    # Manual Resume performs the second request, restarting without a pin.
    client.download(scene['id'], tmp_path/'file.licht')
    assert (tmp_path/'file.licht').read_bytes() == b'data'
    assert len(requests) == 2 and all(r.get_header('Range') is None for r in requests)

def test_valid_native_download_checks_embedded_crc_before_atomic_rename(tmp_path, monkeypatch):
    from test_portable_project import FIXTURES
    from lfs_plugins.portable_project import ProjectFile
    data = bytearray((FIXTURES/'portable-ply.licht').read_bytes())
    project = ProjectFile(io.BytesIO(data))
    offset = project._nodes[0]['offset'] + project._nodes[0]['size'] - 1
    data[offset] ^= 1
    client, scene = _download_client(len(data))
    monkeypatch.setattr(portal_gallery, 'urlopen', lambda *a, **k: _response(bytes(data)))
    destination = tmp_path/'scene.licht'
    destination.write_bytes(b'existing')
    with pytest.raises(ValueError, match='checksum'):
        client.download(scene['id'], destination)
    assert destination.read_bytes() == b'existing'
    assert not (tmp_path/'.scene.licht.part').exists()

def test_account_switch_during_backoff_stops_old_bearer(tmp_path, monkeypatch):
    path = tmp_path/'credentials.json'
    write_credentials(path)
    account = portal_account.PortalAccountService(credentials_path=path)
    network = StubUrlopen((503, {}))
    monkeypatch.setattr(portal_account, 'urlopen', network)
    monkeypatch.setattr(portal_retry.time, 'sleep', lambda _: account._clear_current_credentials())
    with pytest.raises(portal_account.PortalProtocolError, match='account changed'):
        account.request_json_authenticated('GET', '/api/gallery/v1/me')
    assert len(network.requests) == 1

def test_watchdog_deadline_survives_restart_and_keep_waiting_extends_it(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path = tmp_path/'scene.licht'
    path.write_bytes(b'data')
    def timed_out(self, *a, **kw):
        self.processing_deadline = time.time() - 1
        kw['on_processing']({'stage': 'validating', 'completed': 1, 'total': 4})
        raise portal_gallery.GalleryProcessingTimeout('Portal is taking longer than expected · Retry / Keep waiting')
    monkeypatch.setattr(Client, 'upload', timed_out, raising=False)
    job_id = service.queue_upload(path, {'title': 'Scene'}, 'project')
    finish(service)
    deadline = service.snapshot()['jobs'][0]['processingDeadline']
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    assert restarted.snapshot()['jobs'][0]['processingDeadline'] == deadline
    def done(self, *args, **kwargs):
        assert self.processing_deadline > time.time() + 890
        return {'scene': {"contentRevision": 'r1', "metadataRevision": 'r1', 'id': 'remote', 'revision': 'r1'}}
    monkeypatch.setattr(Client, 'upload', done)
    restarted.resume(job_id, keep_waiting=True)
    finish(restarted)
    assert restarted.snapshot()['jobs'][0]['status'] == 'completed'

def test_strict_partial_cleanup_owns_only_download_sidecar(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    identifier = str(uuid.uuid4())
    path = tmp_path/'downloads'/(identifier+'.licht')
    path.parent.mkdir()
    partial = path.with_name('.'+path.name+'.part')
    partial.write_bytes(b'partial')
    job = {'id': identifier, 'path': str(path), 'kind': 'download'}
    assert service._cleanup_paths(job, {}) == [path, partial]
    partial.unlink()
    external = tmp_path/'keep'
    external.write_bytes(b'keep')
    partial.symlink_to(external)
    with pytest.raises(ValueError, match='redirected'):
        service._cleanup_paths(job, {})
    assert external.read_bytes() == b'keep'

def test_processing_watchdog_uses_elapsed_time_if_wall_clock_stops(monkeypatch):
    client = portal_gallery.PortalGalleryClient(SimpleNamespace())
    client.processing_deadline = 1900.
    monotonic = iter([100., 1100.])
    monkeypatch.setattr(portal_gallery.time, 'time', lambda: 1000.)
    monkeypatch.setattr(portal_gallery.time, 'monotonic', lambda: next(monotonic))
    identifier = str(uuid.uuid4())
    state = {'id': identifier, 'status': 'processing', 'processing': {
        'stage': 'queued', 'totalBytes': 8, 'bytesProcessed': 0}}
    with pytest.raises(portal_gallery.GalleryProcessingTimeout):
        client._await_processing(state, identifier, 8, threading.Event(), lambda _: None)

@pytest.mark.parametrize('format', ['ply', 'sog', 'ssog'])
def test_native_download_validates_before_publication(tmp_path, monkeypatch, format):
    from test_portable_project import FIXTURES
    data = (FIXTURES/f'portable-{format}.licht').read_bytes()
    client, scene = _download_client(len(data))
    checkpoints = []
    monkeypatch.setattr(portal_gallery, 'urlopen', lambda *a, **k: _response(data))
    client.download(scene['id'], tmp_path/'project.licht', on_checkpoint=checkpoints.append)
    assert (tmp_path/'project.licht').read_bytes() == data
    assert checkpoints[-1]['sha256'] == hashlib.sha256(data).hexdigest()

def test_dpapi_ctypes_uses_user_scope_and_releases_native_buffer(monkeypatch):
    import ctypes
    calls, buffers = [], []
    class Function:
        def __init__(self, operation):
            self.operation = operation
        def __call__(self, *args):
            return self.operation(*args)
    def operation(incoming, description, entropy, reserved, prompt, flags, outgoing):
        calls.append(flags)
        assert description is entropy is reserved is prompt is None
        assert flags == 1  # UI forbidden, no machine-scope bit.
        data = ctypes.string_at(incoming._obj.pbData, incoming._obj.cbData)
        value = (ctypes.c_ubyte * len(data)).from_buffer_copy(data[::-1])
        buffers.append(value)
        outgoing._obj.cbData, outgoing._obj.pbData = len(data), ctypes.cast(value, ctypes.POINTER(ctypes.c_ubyte))
        return 1
    freed = []
    crypt = SimpleNamespace(CryptProtectData=Function(operation), CryptUnprotectData=Function(operation))
    kernel = SimpleNamespace(LocalFree=Function(lambda address: freed.append(address)))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda name, **kwargs: crypt if name == 'crypt32' else kernel, raising=False)
    cipher = credential_storage._dpapi(b'secret', protect=True)
    assert credential_storage._dpapi(cipher, protect=False) == b'secret'
    assert calls == [1, 1] and len(freed) == 2

def test_credential_initialization_handles_unwritable_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(portal_account, '_locked_sidecar', lambda *_: (_ for _ in ()).throw(OSError('read-only')))
    account = portal_account.PortalAccountService(credentials_path=tmp_path/'account'/'credentials.json')
    assert not account.snapshot().signed_in
    assert not (tmp_path/'account').exists()

def test_account_code_is_included_in_bug_report_redaction_material(tmp_path):
    account = portal_account.PortalAccountService(credentials_path=tmp_path/'credentials.json')
    account._set_linking(user_code='LINK-ONLY', verification_uri='', verification_uri_complete='',
                         expires_at=time.time()+60, interval=1)
    assert 'LINK-ONLY' in account._redaction_tokens()

def test_controller_drops_previous_account_scene_snapshots(gallery):
    controller, state, _ = gallery
    controller._state['scenes'] = [scene()]
    controller._last_snapshot = {'scenes': [scene()]}
    state['identity'] = ('https://other.example', 'two@example.com', 'new', True)
    controller._check_identity()
    assert controller._state['scenes'] == [] and controller._last_snapshot is None

def test_keep_waiting_only_polls_existing_upload_without_reading_export(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    identifier = str(uuid.uuid4())
    job = {'id': 'waiting', 'project': 'project', 'kind': 'upload', 'path': str(tmp_path/'missing.licht'),
           'message': '', 'status': 'error', 'needsAttention': True, 'total': 8, 'completed': 4,
           'metadata': {'title': 'Scene'}, 'checkpoint': {'uploadId': identifier}}
    service._bucket()['jobs'].append(job)
    service._save()
    requests = []
    def request(self, method, path):
        requests.append((method, path))
        return {'id': identifier, 'status': 'completed', 'scene': {"contentRevision": 'r1', "metadataRevision": 'r1', 'id': 'remote', 'revision': 'r1'}}
    monkeypatch.setattr(Client, '_request', request)
    monkeypatch.setattr(Client, '_await_processing', lambda self, result, *a: result, raising=False)
    monkeypatch.setattr(Client, 'upload', lambda *a, **kw: pytest.fail('Keep waiting must not start an upload'), raising=False)
    service.resume('waiting', keep_waiting=True)
    finish(service)
    assert requests == [('GET', '/splats/uploads/'+identifier)]
    assert service.snapshot()['jobs'][0]['status'] == 'completed'

def test_resume_explanation_survives_ui_message_translation(gallery, monkeypatch):
    from lfs_plugins import gallery_messages
    module = __import__('lfs_plugins.gallery_controller', fromlist=['lf'])
    monkeypatch.setattr(module.lf.ui, 'tr', lambda _: 'generic translated status')
    text = 'Paused. The download will restart from zero because the portal has no pinned representation.'
    assert gallery_messages.localize_message(text) == text

@pytest.mark.parametrize('origin,url,allowed', [
    ('http://127.0.0.1', 'http://127.0.0.1/data', True),
    ('http://127.0.0.1:8000', 'http://127.0.0.1:8001/data', False),
    ('https://portal.example', 'https://@portal.example/data', False),
])
def test_storage_loopback_exception_is_exact_origin(origin, url, allowed):
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url=origin))
    if allowed:
        assert client._storage_url(url) == url
    else:
        with pytest.raises(portal_account.PortalProtocolError):
            client._storage_url(url)
