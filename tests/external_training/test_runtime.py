# SPDX-License-Identifier: GPL-3.0-or-later
"""External viewer failure isolation and real worker lifecycle contracts."""
import builtins
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/python'))
from lfs_plugins.training_runtime import producer, reader, transport as t
from test_shared_sessions import arrays


def until(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = predicate()
        if value: return value
        time.sleep(.01)
    raise AssertionError('Worker timed out')


class RuntimeContracts(unittest.TestCase):
    def test_external_adapter_owns_publication_without_duplicate_hooks(self):
        for environment in ({"LFS_TRAINING_PUBLISHER": "external"}, {"LFS_BLENDER_JOB": "job.json"}):
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(producer, "_stream", None), mock.patch.object(producer, "AutoStream") as automatic:
                producer.install()
                automatic.assert_not_called()
                self.assertIsNone(producer._stream)
        with mock.patch.dict(os.environ, {"LFS_TRAINING_PUBLISHER": "automatic", "LFS_BLENDER_JOB": "legacy"}, clear=True):
            self.assertTrue(producer.automatic_publication_enabled())
        with mock.patch.dict(os.environ, {"LFS_TRAINING_PUBLISHER": "typo"}, clear=True):
            with self.assertRaisesRegex(ValueError, "automatic or external"):
                producer.automatic_publication_enabled()

    def test_invalid_metadata_does_not_hide_other_running_trainers(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({"label": "valid"})
            try:
                for key, value in (("state", []), ("iteration", "bad"), ("iterations", -1),
                                   ("count", True), ("heartbeat", float("nan")), ("heartbeat", 10**1000), ("pid", 10**1000), ("label", {})):
                    bad = {**publisher.record, key: value}
                    with self.assertRaises(ValueError): t.validate_session(bad)
                t.atomic_json(Path(root)/'invalid/session.json', {**publisher.record, 'id':'invalid', 'state':[]})
                self.assertEqual([record['id'] for record in t.discover()], [publisher.id])
                bad_json=Path(root)/'oversize/session.json';bad_json.parent.mkdir()
                bad_json.write_text(' '*(t.MAX_METADATA_CHARS+1))
                self.assertIsNone(t.read_json(bad_json))
                bad_json.write_bytes(b'\xff\xfe')
                self.assertIsNone(t.read_json(bad_json))
            finally:
                publisher.finish(); publisher.thread.join(3); publisher.pool.close()

    def test_invalid_snapshot_iteration_is_rejected_before_model_installation(self):
        with tempfile.TemporaryDirectory() as root:
            pool=t.SnapshotPool(root)
            try:
                record=pool.write(arrays(), 1)
                for key,value in (("iteration","bad"),("count",True),("sh_degree",4)):
                    with self.assertRaises(ValueError):t.read_snapshot({**record,key:value})
            finally:pool.close()

    def test_metadata_uses_exclusive_private_file_creation(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(t.os,'open',wraps=os.open) as opened:
            target=Path(root)/'metadata.json'
            t.atomic_json(target, {'value': 1})
            self.assertEqual(t.read_json(target), {'value': 1})
            private=[call for call in opened.call_args_list if len(call.args)>=3]
            self.assertTrue(private)
            self.assertEqual(private[0].args[2], 0o600)
            self.assertTrue(private[0].args[1] & os.O_EXCL)

    def test_terminal_reader_retires_after_last_acknowledged_frame(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ,LFS_TRAINING_SESSIONS_DIR=root):
            publisher=t.SessionPublisher({})
            publisher.publish(arrays(),17)
            observer=reader.Observer(publisher.root)
            try:
                frame=None
                def received():
                    nonlocal frame
                    frame=frame or observer.take()
                    return frame is not None
                until(received)
                observer.acknowledge(frame['record']['id'])
                publisher.finish('completed')
                observer.thread.join(5)
                self.assertFalse(observer.thread.is_alive())
                self.assertEqual(observer.status['state'],'completed')
                self.assertFalse(observer.error)
                self.assertFalse(list((publisher.root/'clients').glob('*.json')))
                self.assertEqual(len(frame['data']['means']),3)
            finally:
                observer.close();observer.thread.join(5)
                publisher.finish();publisher.pool.close()

    def test_terminal_reader_retires_without_snapshot_or_after_mapping_expiry(self):
        for expired in (False,True):
            with self.subTest(expired=expired), tempfile.TemporaryDirectory() as root, mock.patch.dict(
                    os.environ,LFS_TRAINING_SESSIONS_DIR=root):
                publisher=t.SessionPublisher({})
                if expired:publisher.publish(arrays(),1)
                publisher.finish('completed');publisher.pool.close()
                observer=reader.Observer(publisher.root)
                try:
                    observer.thread.join(5)
                    self.assertFalse(observer.thread.is_alive())
                    self.assertEqual(observer.status['state'],'completed')
                    self.assertFalse(observer.error)
                    self.assertFalse(list((publisher.root/'clients').glob('*.json')))
                finally:observer.close();observer.thread.join(5)

    def test_failed_stream_wrapper_closes_descriptor_and_removes_metadata_temp(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(t.os, 'fdopen', side_effect=OSError('stream unavailable')), mock.patch.object(
                    t.os, 'close', wraps=os.close) as closed:
                with self.assertRaises(OSError):t.atomic_json(Path(root)/'state.json', {})
                descriptor = closed.call_args.args[0]
            with self.assertRaises(OSError):os.fstat(descriptor)
            self.assertFalse(list(Path(root).glob('*.tmp')))

    @unittest.skipUnless(os.name=='nt', 'Windows native Unicode and extended paths')
    def test_session_metadata_roundtrips_through_long_unicode_registry(self):
        from _lfs_path_io import TemporaryDirectory
        with TemporaryDirectory() as base:
            directory=t.Path(base)
            while len(str(directory))<290:
                directory=directory/'training-カメラ-é'
            with mock.patch.dict(os.environ,LFS_TRAINING_SESSIONS_DIR=str(directory)):
                publisher=t.SessionPublisher({'label':'Camera é · カメラ'})
                client=None
                try:
                    client=t.SessionClient(publisher.root,'Long-path reader')
                    publisher.update(iteration=12)
                    status=client.status()
                    self.assertEqual(status['iteration'],12)
                    self.assertEqual(status['label'],'Camera é · カメラ')
                    self.assertNotIn('\\\\?\\',json_text:=str(publisher.root))
                    self.assertGreater(len(json_text),260)
                finally:
                    if client:client.close()
                    publisher.finish();publisher.thread.join(3);publisher.pool.close()

    @unittest.skipUnless(os.name=='nt', 'Windows native handle transfer')
    def test_windows_failed_handle_transfer_does_not_leak(self):
        import ctypes
        import msvcrt
        kernel=ctypes.WinDLL('kernel32', use_last_error=True)
        close_handle=kernel.CloseHandle
        close_handle.argtypes=[ctypes.c_void_p]
        def close(handle):
            self.assertTrue(close_handle(handle), 'The transferred native handle was not closed')
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'state.json';t.atomic_json(path,{})
            # Assert ownership of each actual resource, independently of handles
            # which unrelated process cleanup may release at the same time.
            with mock.patch.object(t.ctypes,'WinDLL',return_value=kernel), mock.patch.object(
                    kernel,'CloseHandle',side_effect=close) as closed, mock.patch.object(
                    msvcrt,'open_osfhandle',side_effect=OSError('transfer unavailable')) as transfer:
                for _ in range(8):
                    closed.reset_mock();transfer.reset_mock()
                    with self.assertRaises(OSError):t.read_text_shared(path)
                    transfer.assert_called_once()
                    closed.assert_called_once_with(transfer.call_args.args[0])

    @unittest.skipUnless(os.name=='nt', 'Windows mapped view retirement')
    def test_windows_failed_view_wrapper_closes_native_handle(self):
        import ctypes
        kernel=ctypes.WinDLL('kernel32', use_last_error=True)
        close_handle,unmap_view=kernel.CloseHandle,kernel.UnmapViewOfFile
        close_handle.argtypes=unmap_view.argtypes=[ctypes.c_void_p]
        retired=[]
        def close(handle):
            self.assertTrue(close_handle(handle), 'The snapshot handle was not closed')
            retired.append('close')
        def unmap(pointer):
            self.assertTrue(unmap_view(pointer), 'The snapshot view was not unmapped')
            retired.append('unmap')
        with tempfile.TemporaryDirectory() as root:
            pool=t.SnapshotPool(root)
            try:
                record=pool.write(arrays(),1)
                with mock.patch.object(t.ctypes,'WinDLL',return_value=kernel), mock.patch.object(
                        kernel,'CloseHandle',side_effect=close) as closed, mock.patch.object(
                        kernel,'UnmapViewOfFile',side_effect=unmap) as unmapped, mock.patch.object(
                        t,'memoryview',side_effect=MemoryError('view unavailable'),create=True):
                    for _ in range(8):
                        retired.clear();closed.reset_mock();unmapped.reset_mock()
                        with self.assertRaises(MemoryError):t.read_snapshot(record)
                        unmapped.assert_called_once()
                        closed.assert_called_once()
                        self.assertEqual(retired,['unmap','close'])
            finally:pool.close()

    @unittest.skipUnless(os.name=='posix', 'POSIX permission semantics')
    def test_posix_permissions_are_private_even_with_permissive_umask(self):
        import stat
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            previous=os.umask(0)
            publisher=None
            try:
                publisher=t.SessionPublisher({})
                record=publisher.publish(arrays(), 1)
                payload=Path(record['path'])
                for directory in (publisher.root, payload.parent):
                    self.assertEqual(stat.S_IMODE(directory.stat().st_mode),0o700)
                for path in (publisher.root/'session.json', payload):
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o600)
            finally:
                os.umask(previous)
                if publisher:
                    publisher.finish();publisher.thread.join(3);publisher.pool.close()

    @unittest.skipUnless(os.name=='posix' and os.geteuid()==0, 'Requires a disposable root runner to drop child UID')
    def test_other_uid_cannot_read_private_preview(self):
        import subprocess
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=str(Path(root)/'registry')):
            root=Path(root);root.chmod(0o755)
            control=root/'public-control';control.write_text('benign fixture');control.chmod(0o644)
            code="import sys;open(sys.argv[1],'rb').read()"
            options={'user':65534,'group':65534,'extra_groups':[], 'capture_output':True}
            self.assertEqual(subprocess.run([sys.executable,'-c',code,str(control)],**options).returncode,0)
            publisher=t.SessionPublisher({})
            try:
                record=publisher.publish(arrays(),1)
                for path in (publisher.root/'session.json',Path(record['path'])):
                    denied=subprocess.run([sys.executable,'-c',code,str(path)],**options)
                    self.assertNotEqual(denied.returncode,0)
                    self.assertIn(b'PermissionError',denied.stderr)
            finally:
                publisher.finish();publisher.thread.join(3);publisher.pool.close()

    @unittest.skipUnless(os.name=='posix', 'POSIX mapped-file transport')
    def test_failed_mapping_does_not_leave_temporary_payload(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(t.mmap,'mmap',side_effect=OSError('mapping unavailable')):
            pool=t.SnapshotPool(root)
            with self.assertRaises(OSError):pool.write(arrays(),1)
            self.assertFalse(list(Path(root).glob('*.bin')))
            self.assertFalse(pool.mappings)

    def test_current_process_identity_and_pid_reuse(self):
        token = t.process_identity(os.getpid())
        if os.name == 'nt' or sys.platform.startswith('linux'):
            self.assertIsNotNone(token)
        record = {'pid': os.getpid(), 'heartbeat': time.time(), 'process_identity': token}
        self.assertEqual(t.session_liveness(record), 'responsive')
        with mock.patch.object(t, 'process_identity', return_value='different'):
            if token is not None: self.assertEqual(t.session_liveness(record), 'exited')
        record['heartbeat'] = time.time() - 30
        self.assertEqual(t.session_liveness(record), 'unresponsive')

    def test_malformed_subscriber_does_not_disable_valid_observers(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({})
            client = t.SessionClient(publisher.root, "valid")
            try:
                t.atomic_json(publisher.root / 'clients' / 'broken.json', ['invalid'])
                t.atomic_json(publisher.root / 'clients' / 'bad-time.json', {'pid': os.getpid(), 'updated': 'invalid'})
                t.atomic_json(publisher.root / "clients" / "huge-time.json", {"pid": os.getpid(), "updated": 10**1000})
                t.atomic_json(publisher.root / "clients" / "huge-pid.json", {"pid": 10**1000, "updated": time.time()})
                self.assertEqual(len(publisher.clients()), 1)
            finally:
                client.close(); publisher.finish(); publisher.thread.join(3); publisher.pool.close()

    def test_incompatible_and_replaced_metadata_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            t.validate_session({'protocol': 999})
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            t.validate_session({'protocol': t.PROTOCOL, 'id': 'other', 'pid': 1}, 'expected')

    def test_no_implicit_package_install(self):
        original = builtins.__import__
        packages = types.SimpleNamespace(site_packages_dir=lambda: '', install=mock.Mock(), init=mock.Mock())
        def importing(name, *args, **kwargs):
            if name == 'numpy': raise ImportError('not installed')
            if name == 'lichtfeld': return types.SimpleNamespace(packages=packages)
            return original(name, *args, **kwargs)
        with mock.patch('builtins.__import__', side_effect=importing):
            with self.assertRaisesRegex(RuntimeError, 'not ready'):
                producer.numerical_runtime()
        packages.install.assert_not_called()
        packages.init.assert_not_called()

    def test_viewer_bootstrap_prepares_once_off_ui_thread(self):
        runtime = object()
        calls = []
        packages = types.SimpleNamespace(init=mock.Mock(), install=mock.Mock())
        def load():
            calls.append(threading.current_thread())
            if len(calls) == 1:
                raise producer.NumericalRuntimeUnavailable('missing')
            return runtime
        result = []
        with mock.patch.dict(sys.modules, lichtfeld=types.SimpleNamespace(packages=packages)), mock.patch.object(producer, '_load_numerical_runtime', side_effect=load):
            with self.assertRaisesRegex(RuntimeError,'worker'):
                producer.prepare_viewer_runtime()
            thread = threading.Thread(target=lambda:result.append(producer.prepare_viewer_runtime()))
            thread.start();thread.join(3)
        self.assertEqual(result,[runtime])
        self.assertTrue(all(t is not threading.main_thread() for t in calls))
        packages.init.assert_called_once()
        packages.install.assert_called_once_with('numpy>=2,<3')

    def test_ready_viewer_bootstrap_never_installs_and_failure_is_retryable(self):
        packages = types.SimpleNamespace(init=mock.Mock(),install=mock.Mock(side_effect=OSError('offline')))
        errors=[]
        def attempt():
            try:producer.prepare_viewer_runtime()
            except Exception as error:errors.append(str(error))
        with mock.patch.dict(sys.modules,lichtfeld=types.SimpleNamespace(packages=packages)), mock.patch.object(producer,'_load_numerical_runtime',return_value=object()):
            thread=threading.Thread(target=attempt);thread.start();thread.join(3)
        packages.install.assert_not_called()
        with mock.patch.dict(sys.modules,lichtfeld=types.SimpleNamespace(packages=packages)), mock.patch.object(producer,'_load_numerical_runtime',side_effect=producer.NumericalRuntimeUnavailable('missing')):
            thread=threading.Thread(target=attempt);thread.start();thread.join(3)
        self.assertIn('retry automatically',errors[0])

    def test_no_capture_without_subscribers_and_no_export_on_end(self):
        stream = producer.AutoStream.__new__(producer.AutoStream)
        stream.publisher = types.SimpleNamespace(update=mock.Mock(), clients=lambda: [],
            finish=mock.Mock(), pool=types.SimpleNamespace(close=mock.Mock()))
        stream.last_status = stream.last = 0
        stream.interval = 2
        with mock.patch.object(producer, 'capture') as capture:
            stream.step({'iteration': 5, 'num_gaussians': 3})
            stream.end({'iteration': 5, 'terminal_state': 'stopped'})
            capture.assert_not_called()
        stream.publisher.finish.assert_called_once_with('stopped', iteration=5)
        stream.publisher.pool.close.assert_called_once()

    def test_large_model_capture_uses_bounded_candidate_indices(self):
        import numpy as np
        selected = []
        class Tensor:
            def __init__(self, tail): self.tail = tail
            def index_select(self, axis, indices):
                selected.append(len(indices))
                return types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(
                    sync=lambda: None, numpy=lambda: np.zeros((len(indices), *self.tail), np.float32)))
        model = types.SimpleNamespace(num_points=10**9, has_deleted_mask=lambda: False,
            means_raw=Tensor((3,)), sh0_raw=Tensor((1,3)), scaling_raw=Tensor((3,)),
            rotation_raw=Tensor((4,)), opacity_raw=Tensor((1,)), active_sh_degree=0)
        lf = types.SimpleNamespace(Tensor=types.SimpleNamespace(
            from_numpy=lambda value: types.SimpleNamespace(cuda=lambda: value)))
        with mock.patch.dict(sys.modules, lichtfeld=lf), mock.patch.object(
                producer, 'spatial_indices', side_effect=lambda positions, limit: np.arange(limit)):
            result = producer.capture(model, limit=100, degree=0)
        self.assertEqual(len(result['means']), 100)
        self.assertLessEqual(max(selected), 400)

    def test_finish_stops_heartbeat_even_if_disk_write_fails(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({})
            try:
                with mock.patch.object(publisher, 'update', side_effect=OSError('disk full')):
                    with self.assertRaises(OSError): publisher.finish()
                self.assertTrue(publisher.stop.is_set())
            finally: publisher.stop.set(); publisher.thread.join(3)

    def test_worker_bounded_frames_reconnect_and_terminal_state(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({})
            observer = reader.Observer(publisher.root)
            try:
                first = publisher.publish(arrays(), 1)
                frame = until(observer.take)
                self.assertEqual(frame['record']['id'], first['id'])
                for step in range(2, 6): publisher.publish(arrays(), step)
                self.assertIsNone(observer.take())  # No new frame before acknowledgement.
                observer.acknowledge(first['id'])
                frame = until(observer.take)
                self.assertEqual(frame['record']['iteration'], 5)
                observer.acknowledge(frame['record']['id'])
                publisher.finish('completed', iteration=5)
                until(lambda: observer.status.get('state') == 'completed')
                self.assertNotIn('result', t.read_json(publisher.root / 'session.json'))
            finally:
                observer.close(); observer.thread.join(3)
                publisher.stop.set(); publisher.thread.join(3); publisher.pool.close()

    def test_bootstrap_failure_recovers_without_replacing_observer(self):
        attempts = []
        def bootstrap():
            attempts.append(threading.current_thread())
            if len(attempts) == 1:
                raise RuntimeError('Temporary setup failure')
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({})
            publisher.publish(arrays(), 1)
            observer = reader.Observer(publisher.root, bootstrap=bootstrap)
            try:
                until(lambda: observer.error)
                self.assertTrue(observer.thread.is_alive())
                frame = until(observer.take)
                self.assertEqual(frame['record']['iteration'], 1)
                self.assertEqual(observer.error, '')
                self.assertEqual(len(attempts), 2)
                self.assertTrue(all(thread is observer.thread for thread in attempts))
            finally:
                observer.close(); observer.thread.join(3)
                publisher.finish(); publisher.thread.join(3); publisher.pool.close()

    def test_slow_reader_never_blocks_take_or_close(self):
        gate, entered = threading.Event(), threading.Event()
        def load(*args):
            entered.set(); gate.wait(3)
            return arrays(), 'test'
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, LFS_TRAINING_SESSIONS_DIR=root):
            publisher = t.SessionPublisher({})
            publisher.publish(arrays(), 1)
            with mock.patch.object(reader, 'load_record', side_effect=load):
                observer = reader.Observer(publisher.root)
                try:
                    self.assertTrue(entered.wait(2))
                    started = time.perf_counter()
                    observer.take(); observer.close()
                    self.assertLess(time.perf_counter() - started, .02)
                finally:
                    gate.set(); observer.thread.join(3)
                    publisher.finish(); publisher.thread.join(3); publisher.pool.close()

    def test_pruning_never_removes_user_files_or_live_session(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'session'; directory.mkdir()
            (directory / 'session.json').write_text('{}')
            (directory / 'view_geometry.json').write_text('{}')
            (directory / 'user.licht').write_text('preserve')
            record = {'pid': os.getpid(), 'heartbeat': time.time()-90000}
            t.prune_session(directory, record)
            self.assertTrue((directory/'session.json').exists())
            self.assertTrue((directory/'view_geometry.json').exists())
            with mock.patch.object(t, 'process_alive', return_value=False): t.prune_session(directory, record)
            self.assertFalse((directory/'session.json').exists())
            self.assertFalse((directory/'view_geometry.json').exists())
            self.assertTrue((directory/'user.licht').exists())

if __name__ == '__main__': unittest.main()
