# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Review FG2 regressions. IDs match reviewFG_report.md."""
import base64
import ctypes
import hashlib
import json
from pathlib import Path
import stat
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from lfs_plugins import credential_storage, portal_account, portal_gallery, portal_security, gallery_sync
from test_gallery_security import FakeBackend, _download_client, _response
from test_portal_account import write_credentials
from test_gallery_controller import gallery
from test_gallery_convenience import convenience
from test_asset_manager_panel import panel_module, _Element, _Event


BAD_URLS = ['https://foreign.example/share', 'http://portal.example/share',
            'file:///tmp/a', '//foreign.example/share', 'https://portal.example.evil/share',
            'https://portal.example:444/share', 'https://user@portal.example/share',
            'https://portal.example/\\evil', 'https://portal.example/\nshare']


@pytest.mark.parametrize('url', BAD_URLS)
def test_S1_copy_link_refuses_foreign_or_unsafe_url(gallery, panel_module, monkeypatch, url):
    controller, _, _ = gallery
    controller.service.account = SimpleNamespace(base_url='https://portal.example')
    monkeypatch.setattr(panel_module.lf.ui, 'set_clipboard_text', lambda _: pytest.fail('Copied unsafe URL'), raising=False)
    with pytest.raises(ValueError, match='unsafe_url'):
        controller.open_portal({'id': str(uuid.uuid4()), 'visibility': 'public', 'viewerUrl': url}, 'copy')


@pytest.mark.parametrize('field', ['verification_uri', 'verification_uri_complete'])
@pytest.mark.parametrize('url', BAD_URLS[:4])
def test_S2_device_flow_refuses_bad_origin_before_display(tmp_path, monkeypatch, field, url):
    account = portal_account.PortalAccountService(base_url='https://portal.example', credentials_path=tmp_path/'credentials.json')
    payload = dict(device_code='secret', user_code='ABCD', verification_uri='https://portal.example/device',
                   verification_uri_complete='https://portal.example/device?code=ABCD', expires_in=60, interval=1)
    payload[field] = url
    monkeypatch.setattr(account, '_request_json', lambda *a, **kw: payload)
    account._device_flow_worker()
    assert not account.snapshot().signed_in and not account.snapshot().verification_uri_complete
    assert account.snapshot().error == 'unsafe_portal_url'


@pytest.mark.parametrize('url', BAD_URLS[:4])
def test_X4_inline_browser_refuses_foreign_host_with_notice(convenience, panel_module, monkeypatch, url):
    panel, _, _ = convenience
    panel._gallery_controller = SimpleNamespace(service=SimpleNamespace(account=SimpleNamespace(
        base_url='https://portal.example', snapshot=lambda: SimpleNamespace(verification_uri_complete=url))))
    monkeypatch.setattr(panel_module.lf.ui, 'open_url', lambda _: pytest.fail('Opened unsafe URL'), raising=False)
    panel._gallery_command('connect_browser')
    assert panel._gallery_notice == 'This link does not belong to your portal and was blocked.'


@pytest.mark.parametrize('action', ['_open_verification_uri', '_maybe_open_verification_uri', '_copy_verification_uri'])
def test_X4_account_panel_browser_and_copy_validate_again(panel_module, monkeypatch, action):
    from lfs_plugins.account_panel import AccountPanel
    errors = []
    panel = AccountPanel.__new__(AccountPanel)
    panel._service = SimpleNamespace(base_url='https://portal.example',
        snapshot=lambda: SimpleNamespace(verification_uri_complete='https://foreign.example/device'),
        _set_signed_out=errors.append)
    panel._opened_verification_uri = ''
    monkeypatch.setattr(panel_module.lf.ui, 'open_url', lambda _: pytest.fail('Opened unsafe URL'), raising=False)
    monkeypatch.setattr(panel_module.lf.ui, 'set_clipboard_text', lambda _: pytest.fail('Copied unsafe URL'), raising=False)
    getattr(panel, action)(*(() if action == '_maybe_open_verification_uri' else (None,)))
    assert errors == ['unsafe_portal_url']


@pytest.mark.parametrize('url', BAD_URLS)
def test_X4_download_refuses_foreign_host_before_io(tmp_path, monkeypatch, url):
    client, scene = _download_client(4)
    original = client.account.request_json_authenticated
    client.account.request_json_authenticated = lambda method, path, body=None: (
        {'url': url, 'scene': scene} if path.endswith('/download') else original(method, path, body))
    monkeypatch.setattr(portal_gallery, 'urlopen', lambda *a, **kw: pytest.fail('Unsafe download'))
    with pytest.raises(portal_account.PortalProtocolError, match='Unsafe portal URL'):
        client.download(scene['id'], tmp_path/'file.ply')
    assert not list(tmp_path.iterdir())


def test_X4_me_explicit_owned_host_allows_download(tmp_path, monkeypatch):
    client, scene = _download_client(4)
    def request(method, path, body=None):
        if path.endswith('/me'):
            return {'maxFileBytes': 4, 'portalOwnedHosts': ['cdn.portal.example']}
        if path.endswith('/download'):
            return {'url': 'https://cdn.portal.example/file', 'scene': scene}
        return scene
    client.account.request_json_authenticated = request
    calls = []
    def opened(request, **kw):
        calls.append(request.full_url)
        assert request.get_header('Authorization') is None
        return _response(b'data')
    monkeypatch.setattr(portal_gallery, 'urlopen', opened)
    client.download(scene['id'], tmp_path/'file.ply')
    assert calls == ['https://cdn.portal.example/file']
    with pytest.raises(ValueError):
        portal_security.portal_url('https://portal.example', 'http://cdn.portal.example/file', client.portal_owned_hosts)


def test_X4_thumbnail_builds_endpoint_from_origin_not_json(gallery, monkeypatch):
    calls = []
    account = SimpleNamespace(base_url='https://portal.example', request_response_authenticated=lambda *a, **k:
        calls.append(a) or (200, {'ETag': '"poster"'}, b'\x89PNG\r\n\x1a\n'))
    client = portal_gallery.PortalGalleryClient(account)
    identifier = str(uuid.uuid4())
    client.thumbnail(identifier)
    assert calls == [('GET', '/api/gallery/v1/splats/' + identifier + '/thumbnail?size=256')]
    with pytest.raises(ValueError):
        client.thumbnail('https://foreign.example/poster')


@pytest.mark.parametrize('matches', [False, True])
def test_S3_existing_backend_is_rewritten_and_verified_before_plaintext_delete(tmp_path, matches):
    path, backend = tmp_path/'credentials.json', FakeBackend()
    path.write_bytes(b'new account')
    backend.value = b'new account' if matches else b'old account'
    storage = credential_storage.CredentialStorage(path, backend)
    assert storage.read() == b'new account'
    assert backend.writes == 1 and backend.value == b'new account' and not path.exists()
    assert storage.migrated.read() == b'verified\n'


def test_S3_failed_readback_preserves_only_unmigrated_plaintext(tmp_path):
    path, backend = tmp_path/'credentials.json', FakeBackend()
    path.write_bytes(b'good account')
    backend.write = lambda _: None
    backend.value = b'stale account'
    storage = credential_storage.CredentialStorage(path, backend)
    with pytest.raises(OSError, match='verified'):
        storage.read()
    assert path.read_bytes() == b'good account' and not storage.migrated.path.exists()
    with pytest.raises(OSError, match='verified'):
        storage.write(b'new account')
    assert path.read_bytes() == b'good account'


@pytest.mark.parametrize('previously_migrated', [False, True])
def test_S4_decrypt_failure_removes_plaintext_only_after_verified_migration(tmp_path, previously_migrated):
    path, backend = tmp_path/'credentials.json', FakeBackend()
    write_credentials(path)
    original = path.read_bytes()
    if previously_migrated:
        assert portal_account.PortalAccountService(credentials_path=path, storage_backend=backend).snapshot().signed_in
        path.write_bytes(original)  # interrupted cleanup left a plaintext copy
    backend.read = lambda: (_ for _ in ()).throw(OSError('decrypt/security failure'))
    restarted = portal_account.PortalAccountService(credentials_path=path, storage_backend=backend)
    assert not restarted.snapshot().signed_in
    assert path.exists() is not previously_migrated


def test_S5_signout_file_fallback_deletes_hidden_keychain(tmp_path, monkeypatch):
    path = tmp_path/'credentials.json'
    write_credentials(path)
    monkeypatch.setattr(credential_storage.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(credential_storage.shutil, 'which', lambda _: None)
    account = portal_account.PortalAccountService(credentials_path=path)
    path.with_suffix('.dpapi').write_bytes(b'old')
    calls = []
    def run(argv, **kw):
        calls.append((argv, kw))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(credential_storage.subprocess, 'run', run)
    monkeypatch.setattr(account, '_request_json', lambda *a, **kw: {})
    account.sign_out()
    item = credential_storage.KeychainBackend(path, '/usr/bin/security')
    assert calls == [(['/usr/bin/security', 'delete-generic-password', '-s', item.service, '-a', item.account],
                      dict(input=None, capture_output=True, timeout=15))]
    assert not path.exists() and not path.with_suffix('.dpapi').exists()
    assert not account.snapshot().signed_in


@pytest.mark.parametrize('kind', ['toast', 'undo'])
def test_T1_queued_expiry_never_dirties_unmounted_or_remounted_panel(convenience, panel_module, monkeypatch, kind):
    panel, _, _ = convenience
    queued = []
    monkeypatch.setattr(panel_module.lf.ui, 'schedule_on_ui_thread', queued.append)
    if kind == 'toast':
        panel._show_gallery_toast('Published')
    else:
        panel._set_gallery_undo(lambda: None)
    timer = getattr(panel, '_gallery_' + kind + '_timer')
    timer.function()  # The timer has already posted when unmount starts.
    panel.on_unmount(SimpleNamespace(remove_data_model=lambda _: None))
    assert timer.finished.is_set() and getattr(panel, '_gallery_' + kind + '_timer') is None
    monkeypatch.setattr(panel, '_request_model_update', lambda: pytest.fail('Dirty after unmount'))
    queued.pop()()
    panel._panel_mounted = True  # A new mount must not accept old callbacks either.
    timer.function()
    queued.pop()()


def test_T2_unsubscribe_cancels_phase_timer_and_invalidates_queued_callback(gallery, monkeypatch):
    controller, _, _ = gallery
    module = __import__('lfs_plugins.gallery_controller', fromlist=['lf'])
    queued, advanced = [], []
    monkeypatch.setattr(module.lf.ui, 'schedule_on_ui_thread', queued.append)
    monkeypatch.setattr(controller, '_schedule_tick', lambda: None)
    unsubscribe = controller.subscribe(lambda _: None)
    controller._import_pending = {'id': 'import'}
    controller._schedule_phase_poll()
    timer = controller._phase_poll_timer
    assert timer is not None
    timer.function()
    unsubscribe()
    assert timer.finished.is_set() and controller._phase_poll_timer is None
    monkeypatch.setattr(controller, '_advance_phases', lambda: advanced.append(True))
    queued.pop()()
    assert advanced == []
    # Background controller tick still advances owned work with no panel subscribers.
    controller._next_refresh = float('inf')
    controller._tick_body()
    assert advanced == [True]
    controller.close()
    controller._schedule_phase_poll()
    assert controller._phase_poll_timer is None


def test_C1_native_drop_handoff_and_python_pull_open(convenience, monkeypatch):
    panel, _, _ = convenience
    identifier = str(uuid.uuid4())
    panel._gallery_state['scenes'][-1]['id'] = identifier
    panel._gallery_state.update(identity=('https://portal.example', 'email', 'session', True), owner='owner')
    calls = []
    monkeypatch.setattr(panel, '_gallery_command', calls.append)
    payload = json.dumps(dict(origin='https://portal.example', owner='owner', sceneId=identifier))
    assert panel.gallery_viewport_drop(payload)
    assert calls == ['pull_open'] and panel.get_selected_asset_id() == 'remote:' + identifier
    assert not panel.gallery_viewport_drop(payload.replace('portal.example', 'foreign.example'))
    # Compile/runtime verification of the added native handoff belongs to the orchestrator.
    root = Path(__file__).parents[2]
    native = (root/'src/visualizer/gui/gui_manager.cpp').read_text()
    adapter = (root/'src/python/lfs/rml_python_panel_adapter.cpp').read_text()
    assert 'application/x-lichtfeld-gallery-scene' in native
    assert 'panel->onViewportDrop(released->type, released->data)' in native
    assert 'panel_instance_.attr("gallery_viewport_drop")(data)' in adapter


def test_C2_native_registry_exposes_defaults_and_preferences_rows():
    root = Path(__file__).parents[2]
    source = (root/'src/visualizer/input/input_bindings.cpp').read_text()
    bindings = (root/'src/python/lfs/py_keymap.cpp').read_text()
    from lfs_plugins.gallery_shortcuts import SHORTCUTS
    for name, *_ in SHORTCUTS:
        assert f'.value("{name}", Action::{name})' in bindings
        assert f'Action::{name}, "' in source
        assert f'case Action::{name}: return "{name.lower()}"' in source
    assert 'LAST_ACTION = Action::ASSET_REFRESH' in source


@pytest.mark.parametrize('kind,reason', [('diverged', 'Review changes first'), ('remote', 'Already in your gallery')])
def test_C3_published_drop_has_localized_reason(convenience, kind, reason):
    panel, local, remote = convenience
    if kind == 'diverged':
        local['commit_uuid'] = 'local change'
        remote['title'] = 'portal change'
        identifier = local['id']
    else:
        identifier = 'remote:remote-only'
    shell = _Element()
    target = _Element({'data-folder-id': '__gallery__'}, shell)
    event = _Event(shell, target)
    panel._gallery_drag = (identifier, 'account')
    panel._on_gallery_drag_over(event)
    assert target.is_class_set('is-drag-over')
    panel._on_gallery_drop(event)
    assert panel._gallery_drag is None and not target.is_class_set('is-drag-over')
    assert panel._gallery_toast['text'] == reason


@pytest.mark.parametrize('record', [{'localUpdate': {'backupPath': '/backup.licht'}},
    {'stagedImport': {'path': '/imports/a.licht'}}, {'recoveryPath': '/recovery/a.licht'},
    {'downloadPath': '/downloads/a.licht'}, {'kind': 'download', 'path': '/downloads/a.licht'}])
def test_C4_prune_preserves_all_owned_recovery_paths(record):
    owned = dict(id='owned', status='completed', finishedAt=1, **record)
    old = dict(id='old', status='completed', finishedAt=1)
    service = gallery_sync.GallerySync.__new__(gallery_sync.GallerySync)
    service._data = {'accounts': {'owner': {'jobs': [owned, old] + [dict(id=str(i), status='completed',
        finishedAt=time.time()) for i in range(205)]}}}
    service._prune_jobs()
    jobs = service._data['accounts']['owner']['jobs']
    assert owned in jobs and old not in jobs and len(jobs) == 201


def test_X3_dpapi_uses_user_scope_and_frees_os_buffers(tmp_path, monkeypatch):
    calls, buffers = [], []
    class Operation:
        def __init__(self, protect):
            self.protect = protect
        def __call__(self, incoming, description, entropy, reserved, prompt, flags, outgoing):
            blob = incoming._obj
            value = ctypes.string_at(blob.pbData, blob.cbData)
            calls.append(('protect' if self.protect else 'unprotect', value, description, entropy, reserved, prompt, flags))
            result = b'cipher:' + value[::-1] if self.protect else value[7:][::-1]
            buffer = (ctypes.c_ubyte * len(result)).from_buffer_copy(result)
            buffers.append(buffer)
            outgoing._obj.cbData, outgoing._obj.pbData = len(result), buffer
            return 1
    class Free:
        def __call__(self, pointer):
            calls.append(('free', ctypes.addressof(pointer.contents)))
    crypt = SimpleNamespace(CryptProtectData=Operation(True), CryptUnprotectData=Operation(False))
    kernel = SimpleNamespace(LocalFree=Free())
    loads = []
    def load(name, **kw):
        loads.append((name, kw))
        return {'crypt32': crypt, 'kernel32': kernel}[name]
    monkeypatch.setattr(ctypes, 'WinDLL', load, raising=False)
    path = tmp_path/'credentials.dpapi'
    backend = credential_storage.DPAPIBackend(path)
    secret = b'private refresh token'
    backend.write(secret)
    assert path.read_bytes() == b'cipher:' + secret[::-1]
    assert backend.read() == secret
    assert calls == [('protect', secret, None, None, None, None, 1), ('free', ctypes.addressof(buffers[0])),
                     ('unprotect', b'cipher:' + secret[::-1], None, None, None, None, 1),
                     ('free', ctypes.addressof(buffers[1]))]
    assert loads == [('crypt32', {'use_last_error': True}), ('kernel32', {'use_last_error': True})] * 2
    assert len(crypt.CryptProtectData.argtypes) == 7 and len(crypt.CryptUnprotectData.argtypes) == 7
    backend.delete()
    assert not path.exists()


def test_X3_keychain_exact_subprocess_contract_and_file_write_mode(tmp_path, monkeypatch):
    path, calls = tmp_path/'credentials.json', []
    backend = credential_storage.KeychainBackend(path, '/usr/bin/security')
    secret = b'private refresh token'
    encoded = base64.b64encode(secret)
    def run(argv, **kw):
        calls.append((argv, kw))
        return SimpleNamespace(returncode=0, stdout=encoded+b'\n')
    monkeypatch.setattr(credential_storage.subprocess, 'run', run)
    backend.write(secret)
    backend.delete()
    common = dict(capture_output=True, timeout=15)
    assert calls == [(['/usr/bin/security', '-i'], dict(common, input=(
        f'add-generic-password -U -s {backend.service} -a {backend.account} -w {encoded.decode()}\n').encode())),
        (['/usr/bin/security', 'find-generic-password', '-s', backend.service, '-a', backend.account, '-w'], dict(common, input=None)),
        (['/usr/bin/security', 'delete-generic-password', '-s', backend.service, '-a', backend.account], dict(common, input=None))]
    file = credential_storage.FileBackend(path)
    file.write(secret)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600  # before read can chmod it
    monkeypatch.setattr(credential_storage.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(OSError):
        backend.write(secret)


def test_X5_keep_waiting_real_http_only_polls_existing_upload(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    upload_id, scene_id = str(uuid.uuid4()), str(uuid.uuid4())
    requests, polls = [], []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            requests.append((self.command, self.path, self.headers.get('Authorization')))
            if self.path.endswith('/me'):
                value = {'id': 'owner', 'gallerySyncVersion': 1}
            elif self.path.endswith('/splats'):
                value = {'scenes': []}
            elif self.path.endswith('/splats/uploads/' + upload_id):
                polls.append(True)
                value = {'id': upload_id, 'status': 'processing', 'processing': {
                    'stage': 'validating', 'bytesProcessed': 4, 'totalBytes': 8}}
                if len(polls) == 2:
                    value = {'id': upload_id, 'status': 'completed', 'scene': {
                        'id': scene_id, 'revision': 'r1', 'title': 'Scene'}}
            else:
                self.send_error(404)
                return
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    except PermissionError:
        pytest.skip('Sandbox denies local sockets; run this real HTTP regression on the orchestrator')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = 'http://127.0.0.1:' + str(server.server_port)
        path = tmp_path/'credentials.json'
        write_credentials(path, origin=origin)
        account = portal_account.PortalAccountService(base_url=origin, credentials_path=path)
        service = gallery_sync.GallerySync(account, tmp_path/'gallery')
        def finish():
            service._thread.join(timeout=10)
            assert not service.busy
        service.refresh()
        finish()
        assert service.snapshot()['connected'], service.snapshot()['message']
        job = dict(id='waiting', project='project', kind='upload', path=str(tmp_path/'missing.ply'),
                   message='', status='error', needsAttention=True, total=8, completed=4,
                   metadata={'title': 'Scene'}, checkpoint={'uploadId': upload_id})
        service._bucket()['jobs'].append(job)
        service._save()
        requests.clear()
        service.resume('waiting', keep_waiting=True)
        finish()
        assert requests == [('GET', '/api/gallery/v1/splats/uploads/' + upload_id, 'Bearer access-old')] * 2
        assert service.snapshot()['jobs'][0]['status'] == 'completed', service.snapshot()['jobs'][0]
        assert service.snapshot()['links']['project']['sceneId'] == scene_id
        assert not (tmp_path/'missing.ply').exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_S4_missing_security_in_path_does_not_downgrade_verified_keychain(tmp_path, monkeypatch):
    path = tmp_path/'credentials.json'
    write_credentials(path)
    path.with_suffix('.migrated').write_bytes(b'verified\n')
    monkeypatch.setattr(credential_storage.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(credential_storage.shutil, 'which', lambda _: None)
    monkeypatch.setattr(credential_storage.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    account = portal_account.PortalAccountService(credentials_path=path)
    assert not account.snapshot().signed_in and not path.exists()
