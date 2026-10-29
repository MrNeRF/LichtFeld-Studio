# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise real editor keys through X11 and observe existing MCP responses."""

import json
import os
import shutil
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest

from runtime_test_app import isolated_app
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _fixture(path):
    fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
              'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    with path.open('wb') as output:
        output.write(('ply\nformat binary_little_endian 1.0\nelement vertex 16\n'
                      + ''.join(f'property float {field}\n' for field in fields)
                      + 'end_header\n').encode())
        for i in range(16):
            output.write(struct.pack('<17f', (i % 4 - 1.5) * .1, (i // 4 - 1.5) * .1, 0,
                                     0, 0, 0, 1, .2, .1, 3, -3, -3, -3, 1, 0, 0, 0))


def test_shader_editor_modes_commands_diagnostics_and_python_guard(tmp_path):
    if not os.environ.get('LFS_EXECUTABLE') or not shutil.which('xdotool'):
        pytest.skip('requires LFS_EXECUTABLE, Xvfb and xdotool')
    fixture = tmp_path / 'generated.ply'
    _fixture(fixture)
    input_env = dict(os.environ)
    with isolated_app(tmp_path, input_env=input_env) as endpoint:
        def xdo(*args):
            return subprocess.check_output(['xdotool', *args], env=input_env, text=True).strip()

        window = xdo('search', '--name', 'LichtFeld Studio').splitlines()[0]
        xdo('windowfocus', '--sync', window)
        # Insert and Pause: xdotool adds Alt to function keys on the default Xvfb keymap.
        setup = _tool(endpoint, 'editor_run', {'code': """
import lichtfeld as lf
lf.ui.set_language('en')
remaps = [('EDITOR_TOGGLE_KIND', 260), ('EDITOR_RUN', 284)]
for action, key in remaps:
    lf.keymap.set_binding(lf.keymap.ToolMode.EDITOR, getattr(lf.keymap.Action, action), key)
for action, key in remaps:
    actual = lf.keymap.get_trigger(getattr(lf.keymap.Action, action), lf.keymap.ToolMode.EDITOR)
    expected = {'type': 'key', 'key': key, 'modifiers': 0}
    assert actual == expected, f'Editor remap did not apply: {action}: expected {expected}, got {actual}'
print('STAGEE_REMAPS_VERIFIED')
stagee_run_count = 0
""", 'timeout_ms': 10000})
        assert setup['success'] and setup['completed'], f'Editor keymap setup failed: {setup}'
        assert 'STAGEE_REMAPS_VERIFIED' in setup['output']['text'], setup
        _call(endpoint, 'resources/read', {'uri': 'lichtfeld://ui/state'})

        def key(name):
            xdo('key', '--delay', '100', name)

        def output_contains(fragment):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                text = _tool(endpoint, 'editor_get_output')['text']
                if fragment in text:
                    return text
                time.sleep(.05)
            pytest.fail(f'Output did not contain {fragment!r}: {text}')

        def document(document_id=None):
            return _tool(endpoint, 'editor_get_code', {'document_id': document_id} if document_id else {})

        def wait_shader(document_id, **kwargs):
            return _tool(endpoint, 'editor_wait', {'document_id': document_id, 'timeout_ms': 30000, **kwargs})

        source = ('import lfs;\r\n// λ preserves source bytes  \r\n'
                  'struct Params { [LfsParam(0, 1, 0.5)] float strength; };\r\n'
                  'LfsElement effect(LfsElement e, LfsContext c, Params p) { e.color *= p.strength; return e; }\r\n')
        closed = wait_shader(str(uuid4()))
        assert closed['outcome'] == 'document_closed' and closed['completed'] and not closed['running']
        python_id = document()['document_id']
        assert document()['document_kind'] == 'python'
        shader = _tool(endpoint, 'editor_new_document', {'mode': 'shader', 'code': source})
        shader_id = shader['document_id']
        assert str(UUID(shader_id)) == shader_id and shader_id != python_id
        pending = _tool(endpoint, 'editor_run', {'wait_for_completion': False, 'wait_for_output': False})
        assert pending['document_id'] == shader_id and pending['owns_binding']
        assert pending['shader_status']['state'] == 'pending' and not pending['applied']
        # No scene means no draw demand: a bound draft is not a compiler job.
        assert not pending['running'] and not pending['completed']
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(wait_shader, shader_id, timeout_ms=200)
            _tool(endpoint, 'editor_activate_document', {'document_id': python_id})
            timed = waiting.result(timeout=10)
        assert timed['document_id'] == shader_id and timed['outcome'] == 'timeout'
        assert timed['timed_out'] and not timed['completed']
        stopped = _tool(endpoint, 'editor_interrupt', {'document_id': shader_id})
        assert stopped['released_bindings'] == [{'kind': 'global'}] and not stopped['owns_binding']
        assert wait_shader(shader_id)['outcome'] == 'unbound'
        _tool(endpoint, 'shader_remove', {'shader_id': pending['shader_id']})
        _tool(endpoint, 'scene_load_ply', {'path': str(fixture)})
        # Scene loading may restore the workspace; discover the active Python buffer again.
        _tool(endpoint, 'editor_set_mode', {'mode': 'python'})
        python_id = document()['document_id']
        code = "stagee_run_count += 1; print('STAGEE_RUN', stagee_run_count)"
        _tool(endpoint, 'editor_set_code', {'code': code})  # also focuses the editor
        key('Pause')
        output_contains('STAGEE_RUN 1')
        _tool(endpoint, 'editor_wait', {'timeout_ms': 10000})
        assert _tool(endpoint, 'editor_get_output')['text'].count('STAGEE_RUN') == 1
        key('Insert')
        assert document()['document_kind'] == 'shader'
        _tool(endpoint, 'editor_set_code', {'code': source})
        shader_id = document()['document_id']
        assert shader_id == python_id and document()['text'] == source
        key('Pause')
        applied = wait_shader(shader_id)
        assert applied['outcome'] == 'applied' and applied['applied'] and applied['completed'], applied
        assert applied['shader_status']['displayed_revision'] == applied['asset_revision']
        assert applied['owned_bindings'] == [{'kind': 'global'}]
        asset_id = applied['shader_id']
        assert _tool(endpoint, 'shader_get', {'shader_id': asset_id})['source'] == source
        assert 'Applied' not in _tool(endpoint, 'editor_get_output')['text']
        # The Stop key only interrupts work in flight, so release the applied binding through MCP.
        stopped = _tool(endpoint, 'editor_interrupt', {})
        assert stopped['released_bindings'] == [{'kind': 'global'}] and not stopped['owns_binding']
        assert wait_shader(shader_id)['outcome'] == 'unbound'

        other = _tool(endpoint, 'editor_new_document', {'code': "print('other document')"})['document_id']
        listed = _tool(endpoint, 'editor_list_documents')
        assert listed['active_document_id'] == other
        assert {shader_id, other} <= {d['document_id'] for d in listed['documents']}
        # Nonactive Run and edits do not switch tabs or dispatch shader text to Python.
        run = _tool(endpoint, 'editor_run', {'document_id': shader_id, 'timeout_ms': 30000})
        assert run['outcome'] == 'applied' and run['applied'], run
        assert _tool(endpoint, 'editor_list_documents')['active_document_id'] == other
        links = run['shader_status']['links']
        _tool(endpoint, 'shader_set_param', {'shader_id': asset_id, 'name': 'strength', 'value': .75})
        status = _tool(endpoint, 'shader_get', {'shader_id': asset_id})['status']
        assert status['values']['strength'] == .75 and status['links'] == links
        # Stop matches the Run asset UUID, including an external same-asset rebind.
        _tool(endpoint, 'shader_bind', {'shader_id': asset_id, 'target': {'kind': 'global'}})
        assert document(shader_id)['owned_bindings'] == [{'kind': 'global'}]
        stopped = _tool(endpoint, 'editor_interrupt', {'document_id': shader_id})
        assert stopped['released_bindings'] == [{'kind': 'global'}] and not stopped['owns_binding']
        assert _tool(endpoint, 'shader_get', {'shader_id': asset_id})['bindings'] == []
        _tool(endpoint, 'editor_run', {'document_id': shader_id, 'timeout_ms': 30000})
        replacement = _tool(endpoint, 'shader_set_source', {'source': source, 'name': 'Replacement'})['shader_id']
        _tool(endpoint, 'shader_bind', {'shader_id': replacement, 'target': {'kind': 'global'}})
        stopped = _tool(endpoint, 'editor_interrupt', {'document_id': shader_id})
        assert stopped['released_bindings'] == [] and not stopped['owns_binding']
        assert _tool(endpoint, 'shader_get', {'shader_id': replacement})['bindings'] == [{'kind': 'global'}]
        _tool(endpoint, 'shader_unbind', {'target': {'kind': 'global'}})

        nodes = json.loads(_call(endpoint, 'resources/read', {'uri': 'lichtfeld://scene/nodes'})['contents'][0]['text'])
        node = next(n for n in nodes['nodes'] if n['type'] == 'splat')
        target = {'kind': 'node', 'node_uuid': node['uuid']}
        targeted = _tool(endpoint, 'editor_set_target', {'document_id': shader_id, 'target': target})
        assert targeted['target'] == target and not targeted['owns_binding']
        broken = source.replace('return e;', 'missing_identifier; return e;')
        before = document(shader_id)
        edited = _tool(endpoint, 'editor_set_code', {'document_id': shader_id, 'code': broken})
        assert edited['source_revision'] > before['source_revision']
        failed = _tool(endpoint, 'editor_run', {'document_id': shader_id, 'timeout_ms': 30000})
        assert failed['outcome'] == 'failed' and failed['completed'] and not failed['applied'], failed
        assert failed['owned_bindings'] == [target]
        assert 'missing_identifier' in failed['output']['text']
        old = wait_shader(shader_id, asset_revision=applied['asset_revision'])
        assert old['outcome'] == 'superseded' and old['output']['text'] == '' and not old['applied']
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            marked = document(shader_id)
            if marked['marker_count']:
                break
            time.sleep(.05)
        assert marked['marker_count'] == len(marked['markers']) > 0, marked
        expected = broken.encode().index(b'missing_identifier')
        assert any(m['start_byte'] == expected and m['end_byte'] == expected + 1 for m in marked['markers'])
        assert all(m['source_revision'] == marked['source_revision'] for m in marked['markers'])
        assert marked['diagnostics'] and document(other)['markers'] == []
        assert document(other)['text'] == "print('other document')"
        bounded = _tool(endpoint, 'editor_get_output', {'document_id': shader_id, 'max_chars': 12, 'tail': False})
        assert bounded['truncated'] and len(bounded['text'].encode()) <= 12
        _tool(endpoint, 'editor_set_code', {'document_id': shader_id, 'code': source})
        clean = document(shader_id)
        assert clean['marker_count'] == 0 and clean['diagnostics'] == []
        assert clean['asset_revision'] == failed['asset_revision']
        for args in ({'document_id': shader_id, 'code': 'x' * (1024 * 1024 + 1)},
                     {'document_id': shader_id, 'code': 'untouched', 'timeout_ms': 30001}):
            rejected = _call(endpoint, 'tools/call', {'name': 'editor_run', 'arguments': args})
            assert rejected.get('isError'), rejected
            assert document(shader_id) == clean
        _tool(endpoint, 'editor_set_mode', {'document_id': other, 'mode': 'shader', 'shader_id': asset_id})
        assert document(other)['text'] == "print('other document')"
        assert document(other)['shader_id'] == asset_id and not document(other)['owns_binding']
        _tool(endpoint, 'editor_set_mode', {'document_id': other, 'mode': 'python'})
        _tool(endpoint, 'editor_activate_document', {'document_id': shader_id})
        key('Insert')
        assert document()['document_kind'] == 'python'
        # Python execution remains available when the active document is a shader.
        _tool(endpoint, 'editor_set_mode', {'mode': 'shader'})
        recovered = _tool(endpoint, 'editor_run', {'document_id': other, 'code': """
lf.ui.set_language('de')
lf.keymap.clear_binding(lf.keymap.ToolMode.EDITOR, lf.keymap.Action.EDITOR_TOGGLE_KIND)
print('python recovered')
""", 'timeout_ms': 10000})
        assert recovered['success'] and recovered['completed'], recovered
        assert 'python recovered' in recovered['output']['text']
        assert document()['document_id'] == shader_id and document()['document_kind'] == 'shader'
        _tool(endpoint, 'editor_activate_document', {'document_id': other})
        _tool(endpoint, 'editor_set_code', {'code': "print('toggle is unbound')"})
        key('Insert')  # now unbound: the document remains Python
        still_python = _tool(endpoint, 'editor_run', {'timeout_ms': 10000})
        assert still_python['document_kind'] == 'python' and still_python['completed'], still_python
        assert 'toggle is unbound' in still_python['output']['text']


def test_shader_api_observations_and_python_wait(tmp_path):
    if not os.environ.get('LFS_EXECUTABLE'):
        pytest.skip('requires LFS_EXECUTABLE and Xvfb')
    fixture = tmp_path / 'generated-api.ply'
    _fixture(fixture)
    with isolated_app(tmp_path) as endpoint:
        _tool(endpoint, 'scene_load_ply', {'path': str(fixture)})
        names = {t['name'] for t in _call(endpoint, 'tools/list')['tools']}
        assert {'shader_list', 'shader_get', 'shader_set_source', 'shader_bind',
                'shader_unbind', 'shader_set_param', 'shader_remove'} <= names
        assert not {'shader_compile', 'shader_cancel'} & names
        source = ('import lfs;\nstruct Params { [LfsParam(0, 1, 0.5)] float strength; };\n'
                  'LfsElement effect(LfsElement e, LfsContext c, Params p) { e.color *= p.strength; return e; }\n')
        created = _tool(endpoint, 'shader_set_source', {'source': source, 'name': 'API effect'})
        shader_id = created['shader_id']
        assert _tool(endpoint, 'shader_get', {'shader_id': shader_id})['status']['links'] == 0
        before = _tool(endpoint, 'shader_get', {'shader_id': shader_id})
        bad = _call(endpoint, 'tools/call', {'name': 'shader_set_source', 'arguments': {
            'shader_id': shader_id, 'source': 'invalid', 'expected_revision': 0}})
        assert bad.get('isError'), bad
        assert _tool(endpoint, 'shader_get', {'shader_id': shader_id}) == before
        result = _tool(endpoint, 'editor_run', {'code': f'''
import lichtfeld as lf
import threading
shader_id = {shader_id!r}
assert lf.shaders.get(shader_id)['source'] == {source!r}
assert any(a['shader_id'] == shader_id for a in lf.shaders.list())
draft = lf.shaders.set_source('draft', name='Temporary draft')['shader_id']
lf.shaders.remove(draft)
assert draft not in [a['shader_id'] for a in lf.shaders.list()]
lf.shaders.bind(shader_id, {{'kind': 'global'}})
waited = lf.shaders.wait(shader_id, timeout_ms=30000)
assert waited['outcome'] == 'applied', waited
status = lf.shaders.status(shader_id)
links = status['links']
lf.shaders.set_param(shader_id, 'strength', .75, expected_revision=status['source_revision'])
assert lf.shaders.status(shader_id)['links'] == links
assert lf.shaders.get(shader_id)['status']['values']['strength'] == .75
try:
    lf.shaders.set_param(shader_id, 'strength', True)
except lf.Error:
    pass
else:
    raise AssertionError('boolean parameter accepted')
assert lf.shaders.status(shader_id)['values']['strength'] == .75
lf.shaders.set_source('invalid source', shader_id=shader_id, expected_revision=status['source_revision'])
assert lf.shaders.wait(shader_id, source_revision=status['source_revision'])['outcome'] == 'superseded'
failed = lf.shaders.wait(shader_id, timeout_ms=30000)
assert failed['outcome'] == 'failed', failed
assert failed['status']['displayed_revision'] == status['source_revision']
assert failed['status']['schema_revision'] == status['source_revision']
lf.shaders.set_source({source!r}, shader_id=shader_id)
assert lf.shaders.wait(shader_id, timeout_ms=30000)['outcome'] == 'applied'
lf.shaders.unbind({{'kind': 'global'}}, shader_id=shader_id)
assert lf.shaders.status(shader_id)['state'] == 'unbound'
lf.shaders.bind(shader_id, {{'kind': 'global'}})
assert lf.shaders.wait(shader_id, timeout_ms=30000)['outcome'] == 'applied'
viewer_checked = threading.Event()
def check_viewer_wait(_):
    try:
        lf.shaders.wait(shader_id, timeout_ms=1)
    except ValueError:
        viewer_checked.set()
    finally:
        lf.stop_animation()
lf.on_frame(check_viewer_wait, duration_s=5)
assert viewer_checked.wait(10), 'viewer-thread wait did not reject'
print('F1_API_VERIFIED')
''', 'timeout_ms': 30000})
        if result.get('running'):
            result = _tool(endpoint, 'editor_wait', {'timeout_ms': 30000})
        assert result['completed'] and 'F1_API_VERIFIED' in result['output']['text'], result
        for uri in ('lichtfeld://shaders/catalog', 'lichtfeld://shaders/state',
                    f'lichtfeld://shaders/assets/{shader_id}', 'lichtfeld://runtime/jobs/shader.compile'):
            resource = _call(endpoint, 'resources/read', {'uri': uri})
            assert json.loads(resource['contents'][0]['text']), resource
        job = _tool(endpoint, 'runtime_job_wait', {'job_id': 'shader.compile', 'until': 'inactive'})
        assert not job['active'], job
        events = _tool(endpoint, 'runtime_events_tail', {'types': [
            'shader.compile.started', 'shader.compile.completed', 'shader.applied', 'shader.bindings.changed']})
        assert {'shader.compile.started', 'shader.compile.completed', 'shader.applied',
                'shader.bindings.changed'} <= {e['type'] for e in events['events']}, events

        # Scripted renders link only the hosts they draw: a tiled image its banded host,
        # a full image its full-image host, a depth capture its per-pixel host.
        def started():
            tail = _tool(endpoint, 'runtime_events_tail', {'types': ['shader.compile.started'], 'max_events': 1000})
            return [(e['sequence'], e['data']['host']) for e in tail['events'] if 'host' in e['data']]

        def scripted(render):
            mark = max((sequence for sequence, _ in started()), default=-1)
            result = _tool(endpoint, 'editor_run', {'code': (
                'import lichtfeld as lf\nview = lf.get_current_view()\n'
                f'assert {render} is not None\nprint("SCRIPTED_RENDER")\n'), 'timeout_ms': 60000})
            if result.get('running'):
                result = _tool(endpoint, 'editor_wait', {'timeout_ms': 60000})
            assert result['completed'] and 'SCRIPTED_RENDER' in result['output']['text'], result
            return [host for sequence, host in started() if sequence > mark]

        live = {host for _, host in started()}
        assert live and not any(host.endswith('_banded') for host in live), live
        tiled = scripted('lf.render_view_u8(view.rotation, view.translation, 4097, 4096)')
        assert len(tiled) == 1 and tiled[0].endswith('_banded'), tiled
        full = tiled[0].removesuffix('_banded')
        image = scripted('lf.render_view_u8(view.rotation, view.translation, 64, 64)')
        assert image == ([] if full in live else [full]), (image, live)
        per_pixel = full.replace('_survivors', '')
        depth = scripted('lf.render_view(view.rotation, view.translation, 64, 64, with_depth=True)')
        assert depth == ([] if per_pixel in live else [per_pixel]), (depth, live)
        _tool(endpoint, 'shader_unbind', {'target': {'kind': 'global'}, 'shader_id': shader_id})
        _tool(endpoint, 'shader_remove', {'shader_id': shader_id})
        _tool(endpoint, 'history_undo')
        assert _tool(endpoint, 'shader_get', {'shader_id': shader_id})['source'] == source
        _tool(endpoint, 'history_redo')
        assert shader_id not in {a['shader_id'] for a in _tool(endpoint, 'shader_list')['assets']}


def test_editor_wait_output_notification_precedes_completion(tmp_path):
    """A socket gate holds the script open after output; no sleep decides readiness."""
    import socket

    if not os.environ.get('LFS_EXECUTABLE'):
        pytest.skip('requires LFS_EXECUTABLE and Xvfb')
    with socket.create_server(('127.0.0.1', 0)) as server, isolated_app(tmp_path) as endpoint:
        server.settimeout(30)
        address = server.getsockname()
        run = _tool(endpoint, 'editor_run', {'code': f'''
import socket
with socket.create_connection({address!r}, timeout=30) as gate:
    gate.sendall(b'r')
    assert gate.recv(1) == b'o'
    print('WAIT_OUTPUT_GATE', flush=True)
    assert gate.recv(1) == b'q'
''', 'wait_for_completion': False, 'wait_for_output': False})
        assert run['success'], run
        connection, _ = server.accept()
        with connection, ThreadPoolExecutor(max_workers=1) as pool:
            connection.settimeout(30)
            assert connection.recv(1) == b'r'
            waiting = pool.submit(_tool, endpoint, 'editor_wait', {
                'document_id': run['document_id'], 'wait_for_completion': False,
                'wait_for_output': True, 'timeout_ms': 30000})
            try:
                connection.sendall(b'o')
                output = waiting.result(timeout=30)
                assert output['saw_output'] and not output['timed_out'], output
                assert 'WAIT_OUTPUT_GATE' in output['output']['text'], output
                assert output['running'] and not output['completed'], output
            finally:
                connection.sendall(b'q')
        done = _tool(endpoint, 'editor_wait', {'document_id': run['document_id'], 'timeout_ms': 30000})
        assert done['completed'] and not done['running'] and not done['timed_out'], done


def test_editor_run_observes_its_script_across_the_project_switch_it_makes(tmp_path):
    """new_project cancels shader work and project_open replaces the editor tabs; neither ends the
    wait of the Python run that asked for the switch. A socket gate holds each script after its switch."""
    import socket

    if not os.environ.get('LFS_EXECUTABLE'):
        pytest.skip('requires LFS_EXECUTABLE and Xvfb')
    project = tmp_path / 'switch.licht'
    with socket.create_server(('127.0.0.1', 0)) as server, isolated_app(tmp_path) as endpoint, \
            ThreadPoolExecutor(max_workers=1) as pool:
        server.settimeout(30)
        address = server.getsockname()

        def run_across(label, switch, switched=lambda: None):
            running = pool.submit(_tool, endpoint, 'editor_run', {'code': f'''
import lichtfeld as lf, socket
print('{label}_BEFORE', flush=True)
{switch}
with socket.create_connection({address!r}, timeout=30) as gate:
    gate.sendall(b'r')
    assert gate.recv(1) == b'q'
print('{label}_AFTER', flush=True)
''', 'show_console': False, 'timeout_ms': 30000})
            connection, _ = server.accept()
            with connection:
                connection.settimeout(30)
                assert connection.recv(1) == b'r'
                switched()
                connection.sendall(b'q')
            result = running.result(timeout=60)
            assert result['success'] and result['completed'], result
            assert not result['running'] and not result['timed_out'] and 'outcome' not in result, result
            text = result['output']['text']
            assert f'{label}_BEFORE' in text and text.index(f'{label}_BEFORE') < text.index(f'{label}_AFTER'), result
            # The console keeps the whole run's output too.
            console = _tool(endpoint, 'editor_get_output', {'max_chars': 20000})['text']
            assert f'{label}_BEFORE' in console and f'{label}_AFTER' in console, console
            return result

        run_across('NEW', f'lf.new_project(discard_changes=True)\nassert lf.project_save_as({str(project)!r}, wait=True)')

        tab = _tool(endpoint, 'editor_list_documents')['active_document_id']

        def tab_replaced():
            deadline = time.monotonic() + 30
            while tab in {d['document_id'] for d in _tool(endpoint, 'editor_list_documents')['documents']}:
                assert time.monotonic() < deadline, 'the opened project kept the running tab'
                time.sleep(.05)

        opened = run_across('OPEN', f'assert lf.project_open({str(project)!r}, discard_changes=True) == '
                                    'lf.ProjectOpenOutcome.OPENED', tab_replaced)
        assert opened['document_id'] == tab, opened
