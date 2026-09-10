#!/usr/bin/env python3
"""Opt-in Linux native crash and disk probe. Launches/kills only its own app.

Run with --binary build/LichtFeld-Studio --display :92. Artifacts go to a
private directory on the selected disk; no existing app or user project is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--display', required=True)
    parser.add_argument('--directory', type=Path, default=Path.home()/'.cache')
    parser.add_argument('--count', type=int, default=1_048_576)
    args = parser.parse_args()
    if not 100_000 <= args.count <= 4_194_304:
        parser.error('--count must be between 100000 and 4194304')
    import numpy as np
    root = Path(tempfile.mkdtemp(prefix='lfs-gallery-resilience-', dir=args.directory))
    print('ARTIFACTS', root, flush=True)
    binary = args.binary.resolve()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    endpoint = f'http://127.0.0.1:{port}/mcp'
    results, process, handles = {}, None, []

    def call(method, params=None, timeout=30):
        req = urllib.request.Request(endpoint, data=json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params or {}}).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.load(response)
        if 'error' in value or value.get('result',{}).get('isError'):
            raise AssertionError(value)
        return value['result']

    def rpc(code):
        value = call('tools/call', {'name':'editor_run','arguments':{'code':code,'show_console':False,'timeout_ms':10000}})['structuredContent']
        output = value.get('output',{}).get('text','')
        if not value.get('success') or 'Traceback' in output or 'SyntaxError' in output:
            raise AssertionError(output)
        return output

    def wait(code, marker, timeout=60):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            text = rpc(code)
            if marker in text:
                return
            time.sleep(.05)
        raise AssertionError(text)

    def start():
        nonlocal process
        log = (root/f'app-{len(handles)}.log').open('w'); handles.append(log)
        process = subprocess.Popen([str(binary),'--mcp-port',str(port),'--no-splash'],
            env={**os.environ,'DISPLAY':args.display,'SDL_VIDEODRIVER':'x11','LFS_HOME':str(root/'home')},stdout=log,stderr=log)
        deadline = time.monotonic()+45
        while True:
            assert process.poll() is None, 'Owned app exited; inspect artifacts'
            try:
                call('initialize',{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'gallery-resilience','version':'1'}},timeout=1)
                break
            except (OSError, AssertionError):
                assert time.monotonic() < deadline, 'Owned app did not start'
                time.sleep(.1)
        # Discover metadata and state before invoking mutations.
        for method, params in [('tools/list',{}),('resources/list',{})]+[
            ('resources/read',{'uri':'lichtfeld://'+uri}) for uri in
            ['runtime/catalog','runtime/state','ui/state','scene/state','selection/current']]:
            call(method,params)
        rpc('import lichtfeld as lf\nfrom pathlib import Path\nimport json\nlf.new_project(discard_changes=True)')

    def kill():
        nonlocal process
        if process and process.poll() is None:
            process.kill(); process.wait(timeout=10)
        process = None

    def digest(path):
        with path.open('rb') as source:
            return hashlib.file_digest(source,'sha256').hexdigest()

    try:
        source = root/'source.ply'
        names = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2']+[f'f_rest_{i}' for i in range(45)]+['opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
        header = f'ply\nformat binary_little_endian 1.0\nelement vertex {args.count}\n'+''.join('property float '+n+'\n' for n in names)+'end_header\n'
        random = np.random.default_rng(42)
        with source.open('wb') as output:
            output.write(header.encode())
            for offset in range(0,args.count,16384):
                count = min(16384,args.count-offset)
                data = np.zeros((count,len(names)),dtype='<f4')
                index = np.arange(offset,offset+count)
                data[:,0] = index%1024-512; data[:,1] = index//1024-512
                data[:,6:54] = random.uniform(-.15,.15,(count,48))
                data[:,54] = random.uniform(.5,1,count); data[:,55:58] = -3; data[:,58] = 1
                output.write(data.tobytes())
            output.flush(); os.fsync(output.fileno())
        source_hash = digest(source)
        identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        batch = [{'path':str(source),'transform':identity,'shDegree':3}]
        project = root/'saved.licht'
        start()
        started = time.monotonic()
        rpc(f'lf.load_gallery_scene({batch!r},"Saved scene")')
        call('tools/call',{'name':'runtime_job_wait','arguments':{'job_id':'import.dataset','until':'inactive','timeout_ms':20000}})
        wait(f'print("LOADED" if sum(n.gaussian_count for n in lf.get_scene().get_nodes())=={args.count} else "WAIT")','LOADED')
        results['load_seconds'] = time.monotonic()-started
        rpc(f'assert lf.project_save_as({str(project)!r},wait=False)')
        wait('print("SAVED" if not lf.project_poll_write().get("running") else "WAIT")','SAVED')
        assert project.stat().st_size > args.count*64, "Fixture compressed too much to exercise large recovery files"
        project_hash = digest(project)
        output = root/'complete.scene'
        started = time.monotonic(); rpc(f'lf.prepare_gallery_scene({str(output)!r})')
        latencies = []
        while not (output/'manifest.json').exists():
            tick = time.monotonic(); rpc('print(lf.ui.get_export_state()["active"])'); latencies.append(time.monotonic()-tick)
            assert time.monotonic()-started < 90
        wait('print("EXPORTED" if not lf.ui.get_export_state()["active"] else "WAIT")','EXPORTED')
        results.update(export_seconds=time.monotonic()-started, splats=args.count,
            ply_bytes=source.stat().st_size, project_bytes=project.stat().st_size,
            ui_rpc_max_seconds=max(latencies,default=0), ui_rpc_samples=len(latencies),
            peak_rss_kib=int(next(line.split()[1] for line in Path(f'/proc/{process.pid}/status').read_text().splitlines() if line.startswith('VmHWM:'))))
        started=time.monotonic(); backup=root/'recovery.licht'; shutil.copyfile(project,backup)
        with backup.open('rb') as saved: os.fsync(saved.fileno())
        assert digest(backup)==project_hash
        results['disk_recovery_copy_seconds']=time.monotonic()-started
        crash_output=root/'interrupted.scene'
        # Eight copies widen the native worker interval without modifying the saved project.
        rpc(f'lf.load_gallery_scene({batch*8!r},"Interrupted import",hidden=True)\nstate=lf.ui.get_import_state()\nPath({str(root/"import-active.json")!r}).write_text(json.dumps(state))')
        state=json.loads((root/'import-active.json').read_text())
        assert state['active'] and state['progress'] < 1, state
        kill()
        assert digest(project)==project_hash and digest(backup)==project_hash and digest(source)==source_hash
        results['killed_during_import']=state
        start(); rpc(f'lf.project_open({str(project)!r},discard_changes=True)')
        wait(f'print("RESTORED" if not lf.ui.get_import_state()["active"] and sum(n.gaussian_count for n in lf.get_scene().get_nodes())=={args.count} else "WAIT")','RESTORED')
        # Observe disk writes while the HTTP call is still returning: fast SSDs
        # can complete an export before the caller receives its response.
        with ThreadPoolExecutor(max_workers=1) as executor:
            request=executor.submit(rpc,f'lf.prepare_gallery_scene({str(crash_output)!r})')
            deadline=time.monotonic()+20
            while True:
                partials=list(crash_output.glob('*.tmp'))
                partial=next((p for p in partials if p.exists() and 0 < p.stat().st_size < source.stat().st_size), None)
                if partial is not None:
                    results['partial_export_bytes']=partial.stat().st_size
                    kill(); break
                assert not (crash_output/'manifest.json').exists(), 'Export finished before the crash window; rerun with a larger count'
                assert time.monotonic()<deadline
                time.sleep(.001)
            try:
                request.result()
            except (OSError, AssertionError):
                pass  # Killing the owned process can interrupt its HTTP response.
        assert not (crash_output/'manifest.json').exists()
        assert digest(project)==project_hash and digest(source)==source_hash
        start(); rpc(f'lf.project_open({str(backup)!r},discard_changes=True)')
        wait(f'print("RECOVERED" if not lf.ui.get_import_state()["active"] and sum(n.gaussian_count for n in lf.get_scene().get_nodes())=={args.count} else "WAIT")','RECOVERED')
        results['crashes_preserved_saved_project_and_recovery']=True
        results['disk_root']=str(root)
        (root/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2),flush=True)
    finally:
        kill()
        for log in handles: log.close()


if __name__ == '__main__':
    main()
