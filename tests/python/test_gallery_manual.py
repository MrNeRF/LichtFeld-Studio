# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Manual entrypoints, finite status polling, and domain-protocol requirements."""
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from test_gallery_controller import gallery
from test_asset_manager_panel import panel_module, _gallery_fixture
from test_gallery_sync import Account, Client, finish
from lfs_plugins import gallery_sync, portal_gallery


def test_idle_subscription_and_elapsed_time_never_start_network_work(gallery, monkeypatch):
    controller, state, actions = gallery
    controller.service.refresh = lambda **_kwargs: actions.append('refresh')
    controller.service.resume = lambda _: actions.append('resume')
    monkeypatch.setattr(controller, '_advance_phases', lambda: None)
    seen = []
    stop = controller.subscribe(seen.append)
    for _ in range(3):
        controller._poll()
    stop()
    controller.subscribe(seen.append)()
    assert actions == []
    assert controller._timer is None
    assert seen


def test_transfer_poll_stops_and_delivers_terminal_state(gallery, monkeypatch):
    controller, state, _ = gallery
    scheduled, seen = [], []
    controller.subscribe(seen.append)
    monkeypatch.setattr(controller, '_schedule_poll', lambda: scheduled.append(True))
    monkeypatch.setattr(controller, '_advance_phases', lambda: None)
    controller.service.busy = True
    controller._poll()
    assert scheduled == [True]
    controller.service.busy = False
    state['message'] = 'Finished'
    controller._poll()
    assert scheduled == [True]
    assert seen[-1]['message'] == 'Finished'


def test_failed_explicit_refresh_is_not_retried(gallery, monkeypatch):
    controller, state, calls = gallery
    controller.service.refresh = lambda **_kwargs: calls.append('refresh')
    monkeypatch.setattr(controller, '_schedule_poll', lambda: None)
    controller.refresh()
    state['refresh_ok'] = False
    controller._poll()
    controller._poll()
    assert calls == ['refresh']
    assert controller.offline and not controller._work_pending()


def test_own_transfer_completion_refreshes_once(gallery, monkeypatch):
    controller, state, calls = gallery
    controller.service.refresh = lambda **_kwargs: calls.append('refresh')
    monkeypatch.setattr(controller, '_schedule_poll', lambda: None)
    state['completion'] = {'id': 'done', 'kind': 'download'}
    controller._poll()
    controller._poll()
    controller._poll()
    assert calls == ['refresh']


def test_scope_open_and_refresh_button_load_listing(panel_module):
    manager, _, _ = _gallery_fixture(panel_module)
    calls = []
    manager._gallery_controller = SimpleNamespace(refresh=lambda **_kwargs: calls.append('refresh'))
    from lfs_plugins.asset_gallery_ui import SCOPE_PUBLISHED
    manager._select_folder_id(SCOPE_PUBLISHED)
    manager._gallery_command('refresh')
    assert calls == ['refresh', 'refresh']


@pytest.mark.parametrize('version', [None, 0, '1', True])
def test_unsupported_portal_is_reported_once_and_stays_disabled(tmp_path, monkeypatch, version):
    requests = []
    def request(self, *args):
        requests.append(args)
        return {"storageHosts": ["portal.example"], 'id': 'one', 'gallerySyncVersion': 1, 'revisionDomains': version}
    monkeypatch.setattr(Client, '_request', request)
    monkeypatch.setattr(Client, 'list_scenes', lambda _: pytest.fail('Unsupported listing requested'))
    monkeypatch.setattr(gallery_sync, 'PortalGalleryClient', Client)
    service = gallery_sync.GallerySync(Account(), tmp_path)
    service.refresh()
    finish(service)
    service.refresh()
    assert len(requests) == 1
    assert service.snapshot()['unsupported'] and not service.snapshot()['connected']
    assert service.message == 'This portal version does not support gallery sync'
    with pytest.raises(portal_gallery.PortalProtocolError, match='does not support'):
        service.queue_upload(tmp_path / 'scene.licht', {'title': 'Scene'}, 'project')


def test_write_without_domain_baseline_never_reads_or_adopts_scene():
    calls = []
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(), revision_domains=1)
    client._request = lambda *args: calls.append(args)
    with pytest.raises(portal_gallery.PortalProtocolError, match='revision tokens'):
        client.guards({'revision': 'broad'}, ('content', 'metadata'))
    assert calls == []


def test_old_journal_is_rejected_without_migration(tmp_path):
    data = json.dumps({'version': 1, 'accounts': {}})
    (tmp_path / 'sync.json').write_text(data)
    service = gallery_sync.GallerySync(Account(), tmp_path)
    assert service.snapshot()['storage_issue']
    assert (tmp_path / 'sync.json').read_text() == data


@pytest.mark.parametrize('reason', ['gallery_project_not_supported: Unsupported file',
    'gallery_project_payload_unavailable: Missing payload', 'gallery_project_commit_mismatch: Changed commit'])
def test_native_refusal_keeps_exact_reason_and_never_opens(gallery, tmp_path, monkeypatch, reason):
    controller, state, _ = gallery
    import lfs_plugins.gallery_controller as module
    controller.service.root = tmp_path
    staging = tmp_path / 'export.scene'
    staging.mkdir()
    controller._publish_steps.pending = (staging, {}, 'project', 0)
    controller._publish_steps.identity = controller.service.identity()
    controller._publish_steps.prepared_commit = 'reviewed'
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: {
        'path': str(staging), 'active': False, 'outcome': 'failed', 'error': reason}, raising=False)
    monkeypatch.setattr(controller, '_remove_preparation', lambda _: None)
    monkeypatch.setattr(module.lf, 'project_open', lambda *a, **kw: pytest.fail('Refused publication opened a project'), raising=False)
    controller._finish_export()
    assert controller._publish_steps.pending is None
    expected_message = {
        'gallery_project_not_supported': 'projects.gallery.eligibility.format',
        'gallery_project_payload_unavailable': 'projects.gallery.eligibility.external_payloads',
        'gallery_project_commit_mismatch': 'projects.gallery.error.project_changed',
    }[reason.split(':', 1)[0]]
    assert controller._publish_steps.preparation_failure['message'] == reason
    assert controller.snapshot()['message'] == expected_message


def test_native_commit_mismatch_never_queues_upload(gallery, tmp_path, monkeypatch):
    controller, _, _ = gallery
    import lfs_plugins.gallery_controller as module
    staging = tmp_path / 'export.scene'
    staging.mkdir()
    controller._publish_steps.pending = (staging, {}, 'project', 0)
    controller._publish_steps.prepared_commit = 'reviewed'
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: {
        'path': str(staging), 'active': False, 'outcome': 'completed', 'commit_uuid': 'changed'}, raising=False)
    monkeypatch.setattr(controller, '_remove_preparation', lambda _: None)
    controller.service.queue_prepared_upload = lambda *a: pytest.fail('Queued a changed commit')
    controller._finish_export()
    assert 'commit does not match' in controller._message


def test_outage_stops_update_batch_until_another_user_action(gallery):
    controller, state, _ = gallery
    entry = {'identity': controller.service.identity(), 'asset': {'id': 'project'}, 'job_ids': set()}
    controller._batch_current = (entry, '')
    controller._update_queue = [{'asset': {'id': 'next'}}]
    state['jobs'] = [{'id': 'upload', 'project': 'project', 'status': 'paused', 'message': 'Paused (connection lost)'}]
    controller._advance_update_all()
    assert not controller._update_queue
    assert not hasattr(controller, "_batch_approval")


def test_domainless_link_is_rejected_as_a_whole():
    data = {'version': 2, 'accounts': {'owner': {'jobs': [], 'links': {
        'project': {'sceneId': 'scene', 'revision': 'broad', 'metadata': {}}}}}}
    with pytest.raises(ValueError, match='Invalid sync record'):
        gallery_sync._validate_journal(data)


def test_presentation_only_edit_does_not_invalidate_download(tmp_path, monkeypatch):
    import io
    from test_gallery_security import _download_client
    client, scene = _download_client(4)
    current = dict(scene, revision='new-cover')
    monkeypatch.setattr(client, 'scene', lambda _: current)
    monkeypatch.setattr(portal_gallery, 'urlopen', lambda *a, **kw: io.BytesIO(b'data'))
    monkeypatch.setattr(portal_gallery, 'validate_download', lambda *a: None)  # Isolate domain freshness.
    path = tmp_path / 'scene.licht'
    assert client.download(scene['id'], path) == scene
    assert path.read_bytes() == b'data'


@pytest.mark.parametrize('cancel', [False, True])
def test_pull_and_open_finishes_and_releases_idle_poll(gallery, monkeypatch, tmp_path, cancel):
    from contextlib import contextmanager
    from test_portable_project import FIXTURES
    from lfs_plugins import gallery_controller as module, asset_index
    from lfs_plugins.portable_project import ProjectFile
    controller, state, calls = gallery
    path = tmp_path / 'download.licht'
    path.write_bytes((FIXTURES / 'portable-sog.licht').read_bytes())
    with path.open('rb') as source:
        count = sum(node['count'] for node in ProjectFile(source).manifest['nodes'])
    native = SimpleNamespace(is_valid=lambda: True, get_nodes=lambda: [], total_gaussian_count=count)
    poll = {'path': None}
    scheduled, leases, opened = [], [], []
    monkeypatch.setattr(controller, '_schedule_poll', lambda: scheduled.append(True))
    monkeypatch.setattr(module.lf, 'get_scene', lambda: native, raising=False)
    monkeypatch.setattr(module.lf, 'project_is_dirty', lambda: False, raising=False)
    monkeypatch.setattr(module.lf, 'project_has_path', lambda: bool(poll['path']), raising=False)
    monkeypatch.setattr(module.lf, 'is_training_active', lambda: False, raising=False)
    monkeypatch.setattr(module.lf, 'project_poll_write', lambda: poll.copy(), raising=False)
    def open_project(value, **kwargs):
        assert kwargs == {'keep_asset_manager_open': True}
        opened.append(value)
        poll['path'] = value
    monkeypatch.setattr(module.lf, 'project_open', open_project, raising=False)
    monkeypatch.setattr(import_module('lfs_plugins.gallery_sync_steps'), 'restore_view', lambda *a, **kw: None)
    project = SimpleNamespace(id='project', project_uuid='project')
    monkeypatch.setattr(asset_index, 'AssetIndex', lambda: SimpleNamespace(load=lambda: True,
        update_asset=lambda *a, **kw: project,
        register_licht_asset=lambda *a, **kw: (project, None)))
    remote = {'id': 'scene', 'title': 'Downloaded', 'contentRevision': 'c', 'metadataRevision': 'm'}
    job = {'id': 'download', 'kind': 'download', 'status': 'running', 'project': 'project',
           'path': str(path), 'result': remote, 'metadata': {'title': remote['title']}}
    def download(*args, **kwargs):
        state['jobs'] = [job]
        controller.service.busy = True
    controller.service.download = download
    controller.service.environment_path = lambda _: None
    def stage(_):
        job['stagedImport'] = {'id': 'stage', 'state': 'ready', 'projectPath': str(path),
                               'projectId': 'project', 'projectStamp': module.file_stamp(path)}
        return 'stage'
    controller.service.stage_download = stage
    def link(*args, **kwargs):
        calls.append(('link', args, kwargs))
        job['linkOperation'] = {'id': 'link', 'state': 'ready'}
        return 'link'
    controller.service.link_download = link
    @contextmanager
    def local_use(_):
        leases.append('acquired')
        try:
            yield
        finally:
            leases.append('released')
    controller.service.local_use = local_use
    controller.service.refresh = lambda **_kwargs: calls.append('refresh')
    controller.pull_asset({'id': 'remote', 'remote_only': True}, remote, open_after=True)
    controller._poll()
    assert not opened
    controller.service.busy = False
    job['status'] = 'completed'
    state['completion'] = {'id': 'download-done'}
    controller._poll()
    assert controller._download_open_steps.pending['_opening']['phase'] == 'staging'
    assert leases == ['acquired']
    if cancel:
        controller.command('pause')
    for _ in range(5):
        controller._poll()
    assert opened == ([] if cancel else [str(path)])
    assert leases == ['acquired', 'released']
    assert len([call for call in calls if call != 'refresh']) == (0 if cancel else 1)
    assert calls.count('refresh') == 1
    assert controller._download_open_steps.pending is None and controller._native_use is None
    assert controller.phase() == 'idle' and not controller._panel_busy() and not controller._work_pending()
    before = len(scheduled)
    controller._poll()
    assert len(scheduled) == before and controller._timer is None


@pytest.mark.parametrize('action,expected', [('cancel', None), ('retry', {}), ('keep_waiting', {'keep_waiting': True}), ('Retry', None), (1, None)])
def test_watchdog_maps_localized_callback_labels_to_action_ids(gallery, monkeypatch, action, expected):
    from lfs_plugins import gallery_controller as module
    controller, state, calls = gallery
    state['jobs'] = [{'id': 'stuck', 'status': 'error', 'needsAttention': True}]
    prompts = []
    monkeypatch.setattr(module.lf.ui, 'tr', lambda key: 'FR:' + key)
    monkeypatch.setattr(module.lf.ui, 'confirm_dialog', lambda *args: prompts.append(args), raising=False)
    monkeypatch.setattr(controller, '_schedule_poll', lambda: None)
    controller.service.resume = lambda *args, **kwargs: calls.append(kwargs)
    controller.command('resume', 'stuck')
    title, message, buttons, selected = prompts[0]
    assert all(text.startswith('FR:') for text in [title, message, *buttons])
    selected('FR:projects.gallery.action.' + action if action in ('cancel', 'retry', 'keep_waiting') else action)
    assert calls == ([] if expected is None else [expected])


@pytest.mark.parametrize('saved,baseline,patch', [
    ('content:view', 'content:view', True), ('changed:view', 'content:view', False),
    ('content:changed-view', 'content:view', False), ('', 'content:view', False)])
def test_closed_update_uses_saved_file_proof_without_live_capture(gallery, monkeypatch, tmp_path, saved, baseline, patch):
    from lfs_plugins import gallery_controller as module, gallery_project_facts
    controller, state, calls = gallery
    controller.service.root = tmp_path
    state['source_formats'] = ['licht']
    state['links'] = {'project': {'sceneId': 'scene', 'contentStamp': baseline,
        'commitUuid': 'old', 'contentRevision': 'c', 'metadataRevision': 'm'}}
    state['scenes'] = [{'id': 'scene', 'contentRevision': 'c', 'metadataRevision': 'm', 'visibility': 'private'}]
    asset = {'id': 'project', 'path': str(tmp_path / 'saved.licht'), 'commit_uuid': 'new'}
    info = SimpleNamespace(project_uuid='project', commit_uuid='new')
    monkeypatch.setattr(module.lf.io, 'inspect_project', lambda _: info)
    monkeypatch.setattr(module.lf, 'project_poll_write', lambda: {'path': '/unrelated.licht'}, raising=False)
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: {'active': False}, raising=False)
    monkeypatch.setattr(module.lf, 'prepare_gallery_project', lambda *args: calls.append(('replace', args)), raising=False)
    monkeypatch.setattr(module.lf, 'project_open', lambda *a, **kw: pytest.fail('Opened closed project'), raising=False)
    monkeypatch.setattr(module, 'capture_view', lambda _: pytest.fail('Captured unrelated view'))
    monkeypatch.setattr(gallery_project_facts, 'saved_content_stamp', lambda path: saved)
    monkeypatch.setattr(controller, '_schedule_poll', lambda: None)
    controller.service.edit = lambda *args, **kwargs: calls.append(('patch', args, kwargs))
    details = {'title': 'Changed title', 'description': ''}
    controller.publish_asset(asset, details, 'sog', update=True)
    assert calls[0][0] == ('patch' if patch else 'replace') and len(calls) == 1
    if patch:
        assert calls[0][1] == ('scene', {'contentRevision': 'c', 'metadataRevision': 'm'}, details)
        assert calls[0][2] == {'commit_uuid': 'new', 'content_stamp': saved, 'project_id': 'project'}
        assert controller._publish_steps.pending is None
    else:
        assert controller._publish_steps.pending[1]['_contentStamp'] == saved
    if not saved:
        assert controller._publish_steps.reupload_reason['message'] == 'projects.gallery.info.reupload_encoding'


@pytest.mark.parametrize('field', [None, 'portalOwnedHosts'])
def test_me_without_storage_hosts_connects_without_adopting_alias(field):
    capabilities = {'revisionDomains': 1}
    if field:
        capabilities[field] = ['cdn.portal.example']
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(
        request_json_authenticated=lambda *a, **kw: capabilities))
    assert client._request('GET', '/me') == capabilities
    assert client.storage_hosts is None


@pytest.fixture
def refresh_account(gallery, tmp_path, monkeypatch):
    """Fresh journal and real controller/client with an in-memory portal transport."""
    from lfs_plugins import gallery_sync as sync
    controller, _, _ = gallery
    calls = []
    class RefreshAccount(Account):
        signed_in = False
        connected_since = 'first'
        capabilities = {'gallerySyncVersion': 1, 'revisionDomains': 1, 'sourceFormats': ['licht']}

        def snapshot(self):
            return SimpleNamespace(signed_in=self.signed_in, email=self.email,
                                   connected_since=self.connected_since)

        def request_json_authenticated(self, method, path, body=None, **kwargs):
            calls.append((self.email, self.connected_since, method, path))
            assert kwargs['expected_session'] == (self.email, self.connected_since)
            if path == '/api/gallery/v1/me':
                return dict(self.capabilities, id=self.owner)
            assert path == '/api/gallery/v1/splats'
            return {'scenes': [{'id': str(uuid.uuid5(uuid.NAMESPACE_URL, self.owner + '-scene')),
                                'status': 'ready', 'contentRevision': 'c', 'metadataRevision': 'm',
                                'viewerSettings': {}}]}

    account = RefreshAccount()
    account.owner = str(uuid.uuid5(uuid.NAMESPACE_URL, 'gallery-test-owner-one'))
    service = sync.GallerySync(account, tmp_path / 'fresh-journal')
    assert not service._journal.exists()
    controller.service = service
    controller._identity = service.identity()
    monkeypatch.setattr(controller, '_schedule_poll', lambda: None)

    def refresh():
        controller.refresh()
        if service._thread:
            finish(service)
        controller._poll()
        assert not controller._work_pending()

    return SimpleNamespace(account=account, service=service, controller=controller,
                           calls=calls, refresh=refresh)


@pytest.mark.parametrize('supported', [True, False])
def test_sign_in_refresh_checks_me_before_accepting_or_refusing(refresh_account, supported):
    run = refresh_account
    if not supported:
        run.account.capabilities = {'gallerySyncVersion': 1}
    run.account.signed_in = True
    run.refresh()
    state = run.service.snapshot()
    ui = run.controller.snapshot()
    assert state['connected'] is supported and state['refresh_ok'] is supported
    assert state['unsupported'] is not supported and ui['unsupported'] is not supported
    assert ui['offline'] is not supported
    paths = [call[-1] for call in run.calls]
    assert paths == ['/api/gallery/v1/me'] + (['/api/gallery/v1/splats'] if supported else [])
    if supported:
        assert [scene['id'] for scene in state['scenes']] == [str(uuid.uuid5(uuid.NAMESPACE_URL, run.account.owner + '-scene'))]
        assert ui['message'] == 'Gallery is up to date.'
    else:
        assert state['scenes'] == []
        assert state['message'] == portal_gallery.UNSUPPORTED_PORTAL
        assert ui['message'] == portal_gallery.UNSUPPORTED_PORTAL
        run.refresh()
        run.controller._poll()
        assert len(run.calls) == 1


@pytest.mark.parametrize('initially_supported', [True, False])
@pytest.mark.parametrize('change', ['account', 'session'])
def test_new_identity_refresh_fetches_me_and_listing(refresh_account, initially_supported, change):
    run = refresh_account
    supported = dict(run.account.capabilities, storageHosts=['portal.example'])
    run.account.capabilities = supported if initially_supported else {'gallerySyncVersion': 1}
    run.account.signed_in = True
    run.refresh()
    if change == 'account':
        run.account.email = 'two@example.com'
        run.account.owner = str(uuid.uuid5(uuid.NAMESPACE_URL, 'gallery-test-owner-two'))
    else:
        run.account.connected_since = 'second'
    run.account.capabilities = supported
    assert not run.service.snapshot()['unsupported']
    run.calls.clear()
    run.refresh()
    state = run.service.snapshot()
    assert state['connected'] and state['refresh_ok'] and not state['unsupported']
    assert [call[-1] for call in run.calls] == ['/api/gallery/v1/me', '/api/gallery/v1/splats']
    assert [scene['id'] for scene in state['scenes']] == [str(uuid.uuid5(uuid.NAMESPACE_URL, run.account.owner + '-scene'))]
    assert not run.controller.snapshot()['offline']
    assert run.controller.snapshot()['message'] == 'Gallery is up to date.'


def test_closed_update_rechecks_reviewed_commit_before_patch(gallery, monkeypatch, tmp_path):
    from lfs_plugins import gallery_controller as module, gallery_project_facts
    controller, state, _ = gallery
    controller.service.root = tmp_path
    state['source_formats'] = ['licht']
    state['links'] = {'project': {'sceneId': 'scene', 'contentStamp': 'content:view'}}
    state['scenes'] = [{'id': 'scene', 'contentRevision': 'c', 'metadataRevision': 'm', 'visibility': 'private'}]
    monkeypatch.setattr(module, 'asset_sync_state', lambda *a: {'freshness': 'local'})
    monkeypatch.setattr(module.lf, 'project_poll_write', lambda: {'path': '/unrelated.licht'}, raising=False)
    monkeypatch.setattr(module.lf.io, 'inspect_project', lambda _: SimpleNamespace(project_uuid='project', commit_uuid='changed'))
    monkeypatch.setattr(module.lf.ui, 'get_export_state', lambda: {'active': False}, raising=False)
    monkeypatch.setattr(gallery_project_facts, 'saved_content_stamp', lambda _: pytest.fail('Used an unreviewed commit'))
    controller.service.edit = lambda *a, **kw: pytest.fail('Patched from an unreviewed commit')
    with pytest.raises(ValueError, match='project_changed'):
        controller.publish_asset({'id': 'project', 'path': '/saved.licht', 'commit_uuid': 'reviewed'},
            {'title': 'Title', 'description': '', 'visibility': 'private'}, 'sog', update=True)
