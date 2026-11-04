# SPDX-License-Identifier: GPL-3.0-or-later
"""Own an isolated GUI process and run camera lifecycle contracts via discovered MCP."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
from mcp_test_client import MCP

parser = argparse.ArgumentParser()
for name in ('backend', 'area', 'runtime'):
    parser.add_argument(name, type=Path)
args = parser.parse_args()
area = args.area.resolve()
area.mkdir(parents=True, exist_ok=True)
client = MCP('http://127.0.0.1:45689/mcp')
try:
    client.rpc('ping')
except (OSError, urllib.error.URLError):
    pass
else:
    raise RuntimeError('Test port already in use; refusing to drive an existing app')
env = {**os.environ, 'LFS_HOME':str(args.runtime.resolve()), 'LFS_TEST_AREA':str(area)}
with (area/'gui.log').open('wb') as log:
    process = subprocess.Popen([str(args.backend.resolve()), '--no-splash', '--mcp-port', '45689'],
        env=env, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        deadline = time.monotonic()+60
        while time.monotonic()<deadline:
            assert process.poll() is None, 'GUI exited during startup'
            try:
                client.rpc('initialize', {'protocolVersion':'2024-11-05','capabilities':{},
                    'clientInfo':{'name':'training-review-tests','version':'1'}})
                break
            except (OSError, urllib.error.URLError):
                time.sleep(.2)
        else:
            raise AssertionError('MCP startup timed out')
        inventory = client.rpc('tools/list')
        tools = {item['name']:item for item in inventory['tools']}
        (area/'tools.json').write_text(json.dumps(inventory), encoding='utf-8')
        for uri in ('lichtfeld://runtime/catalog','lichtfeld://runtime/state','lichtfeld://ui/state',
                    'lichtfeld://scene/state','lichtfeld://selection/current'):
            state = client.resource(uri)
            (area/(uri.replace('lichtfeld://','').replace('/','-')+'.json')).write_text(json.dumps(state), encoding='utf-8')
        assert 'editor_run' in tools
        assert {'code','timeout_ms','show_console'} <= set(tools['editor_run']['inputSchema']['properties'])
        fixture = Path(__file__).with_name('renderer_camera_churn.py').read_text(encoding='utf-8')
        proof = area/'camera-churn-proof.json'
        proof.unlink(missing_ok=True)  # A previous run cannot satisfy this run's completion check.
        response = client.call('editor_run', {'code':fixture, 'timeout_ms':1000, 'show_console':False})
        (area/'editor-start.json').write_text(json.dumps(response), encoding='utf-8')
        output = response.get('structuredContent', {}).get('output', {}).get('text', '')
        assert 'Traceback (most recent call last)' not in output, output
        deadline = time.monotonic()+100
        while not proof.exists() and time.monotonic()<deadline:
            assert process.poll() is None, 'GUI exited during camera lifecycle test'
            time.sleep(.2)
        if not proof.exists():
            if 'editor_get_output' in tools:
                print(json.dumps(client.call('editor_get_output')), flush=True)
            raise AssertionError('Camera lifecycle test did not complete; see GUI/editor log')
        result = json.loads(proof.read_text(encoding='utf-8'))
        assert result.get('cameras') == 512 and result.get('cleanup') is True, result
        print('REAL_GUI_CAMERA_REVIEW_PASS', json.dumps(result), flush=True)
    finally:
        if process.poll() is None:
            try:
                client.call('editor_run', {'code':'import lichtfeld as lf\nlf.request_exit()',
                    'timeout_ms':1000,'show_console':False})
                process.wait(timeout=10)
            except Exception:
                process.terminate()  # Only this fixture's Popen child.
                process.wait(timeout=10)
