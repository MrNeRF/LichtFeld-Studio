# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure validator helpers; no Studio, Django, display, network or native module."""
import argparse
import base64
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'validate_gallery_sync.py'
spec = importlib.util.spec_from_file_location('validate_gallery_sync', SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ArgumentTests(unittest.TestCase):
    def args(self, *args):
        return validator.parse_args(['--build-dir', '/existing/build', *args])

    def test_local_default_and_format(self):
        args = self.args()
        self.assertEqual(args.portal_source, Path.home() / 'projects/lichtfeld-portal')
        self.assertIsNone(args.portal_origin)
        self.assertEqual(args.format, 'sog')

    def test_local_paths_and_keep(self):
        args = self.args('--portal-source', '/portal', '--portal-python', '/venv/bin/python',
                         '--display', ':104', '--keep', '--report', '/tmp/result.md', '--junit', '/tmp/result.xml')
        self.assertEqual(args.portal_source, Path('/portal'))
        self.assertEqual(args.portal_python, Path('/venv/bin/python'))
        self.assertEqual(args.report, Path('/tmp/result.md'))
        self.assertEqual(args.junit, Path('/tmp/result.xml'))
        self.assertTrue(args.keep)
        self.assertEqual(args.display, ':104')

    def test_remote_has_no_local_source(self):
        args = self.args('--portal-origin', 'https://portal.example:8443/')
        self.assertEqual(args.portal_origin, 'https://portal.example:8443')
        self.assertIsNone(args.portal_source)

    def test_local_and_remote_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('--portal-origin', 'https://portal.example', '--portal-source', '/portal')

    def test_build_required(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validator.parse_args([])

    def test_reject_bad_formats(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('--format', 'ply')

    def test_reject_bad_displays(self):
        for display in ('localhost:0', ':1.0', ':', '1', ':1;touch /tmp/no'):
            with self.subTest(display=display), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args('--display', display)

    def test_reject_nonpositive_and_nonfinite_deadlines(self):
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args('--timeout', value)

    def test_reject_invalid_origins(self):
        for origin in ('http://127.0.0.1:8000', 'http://portal.example', 'https://u:p@portal.example',
                       'https://portal.example/api', 'https://portal.example?key=x',
                       'https://portal.example/#token', 'https://', 'https://portal.example:invalid', 'https://['):
            with self.subTest(origin=origin), self.assertRaises(argparse.ArgumentTypeError):
                validator.portal_origin(origin)


class FormatTests(unittest.TestCase):
    def test_extensions(self):
        for name, suffix in [('studio', '.ply'), ('sog', '.sog'), ('ssog', '.ssog'), ('spz', '.spz')]:
            with self.subTest(name=name):
                self.assertEqual(validator.format_extension(name), suffix)

    def test_unknown_format_fails(self):
        with self.assertRaises(KeyError):
            validator.format_extension('unknown')

    def test_manifest_checks_every_node(self):
        for name in validator.FORMATS:
            files = ['nodes/0' + validator.format_extension(name), 'nodes/1' + validator.format_extension(name)]
            self.assertEqual(validator.manifest_files({'manifest': {'nodes': [{'file': f} for f in files]}}, name), files)

    def test_manifest_rejects_mixed_nodes(self):
        with self.assertRaises(AssertionError):
            validator.manifest_files({'manifest': {'nodes': [{'file': '0.sog'}, {'file': '1.ply'}]}}, 'sog')

    def test_manifest_rejects_empty_missing_or_nonstring_files(self):
        for index in ({}, {'manifest': {'nodes': []}}, {'manifest': {'nodes': [{}]}},
                      {'manifest': {'nodes': [{'file': None}]}}):
            with self.subTest(index=index), self.assertRaises(AssertionError):
                validator.manifest_files(index, 'sog')


class LogTests(unittest.TestCase):
    def test_each_forbidden_marker(self):
        lines = ['Traceback (most recent call last):', '[error] broken',
                 'Syntax error parsing property width', 'Missing localization key foo']
        self.assertEqual(validator.scan_logs('\n'.join(lines)), lines)

    def test_clean_logs(self):
        self.assertEqual(validator.scan_logs('[info] Gallery uploaded\n[debug] ready'), [])

    def test_case_insensitive_markers(self):
        self.assertEqual(validator.scan_logs('[ERROR] broken'), ['[ERROR] broken'])

    def test_allowlist_is_narrow(self):
        self.assertEqual(validator.scan_logs('[error] xdg-open: no method available for opening URL'), [])
        line = '[error] gallery upload failed while xdg-open was running'
        self.assertEqual(validator.scan_logs(line), [line])

    def test_explicit_allowlist_and_no_allowlist(self):
        self.assertEqual(validator.scan_logs('[error] known\n[error] new', (re.compile('known'),)), ['[error] new'])
        line = '[error] xdg-open: no method available for opening URL'
        self.assertEqual(validator.scan_logs(line, ()), [line])


class MCPTests(unittest.TestCase):
    def test_valid_result(self):
        self.assertEqual(validator.mcp_result({'jsonrpc': '2.0', 'id': 1, 'result': {'tools': []}}), {'tools': []})

    def test_rpc_error_and_tool_error(self):
        for payload in ({'error': {'code': -32601, 'message': 'not found'}}, {'result': {'isError': True}}):
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                validator.mcp_result(payload)

    def test_malformed_mcp(self):
        for payload in (None, [], {}, {'result': None}, {'result': []}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validator.mcp_result(payload)

    def test_editor_output(self):
        self.assertEqual(validator.editor_output({'structuredContent': {'success': True, 'output': {'text': 'ok\n'}}}), 'ok\n')

    def test_editor_unsuccessful_and_python_errors(self):
        for structured in ({}, {'success': False}, {'success': True, 'output': {'text': 'Traceback failed'}},
                           {'success': True, 'output': {'text': 'SyntaxError: bad code'}}):
            with self.subTest(structured=structured), self.assertRaises(RuntimeError):
                validator.editor_output({'structuredContent': structured})

    def test_marked_json_with_noise(self):
        self.assertEqual(validator.marked_json('log noise\n' + validator.MARKER + '{"ok": true}'
                                             + validator.END_MARKER + '\nmore noise'), {'ok': True})

    def test_missing_duplicate_and_invalid_json_marker(self):
        frame = validator.MARKER + '{}' + validator.END_MARKER
        for output in ('{}', frame + frame, validator.MARKER + '{broken',
                       validator.MARKER + '{broken' + validator.END_MARKER,
                       validator.END_MARKER + validator.MARKER):
            with self.subTest(output=output), self.assertRaises(ValueError):
                validator.marked_json(output)

    def test_generated_results_survive_terminal_wrapping_and_noise(self):
        value = {'title': 'with spaces ' * 30, 'unicode': 'Grüezi 🌍',
                 'markers': validator.MARKER + validator.END_MARKER, 'escaped': '\n\\"'}
        emitted = io.StringIO()
        with redirect_stdout(emitted):
            exec(validator.result_code('value'), {'value': value})
        for width in (20, 80, 137):
            with self.subTest(width=width):
                wrapped = '\n'.join(emitted.getvalue().strip()[i:i + width].rstrip()
                                    for i in range(0, len(emitted.getvalue().strip()), width))
                output = 'Using non-default LichtFeld portal host: 127.0.0.1:12345\n' + wrapped + '\nafterwards'
                self.assertEqual(validator.marked_json(output), value)
                self.assertIn('afterwards', validator.split_marked_output(output)[1])

    def test_payload_limit_is_enforced_before_printing(self):
        for value in ('x' * validator.JSON_PAYLOAD_LIMIT, ' ' * validator.JSON_PAYLOAD_LIMIT):
            emitted = io.StringIO()
            with redirect_stdout(emitted), self.assertRaisesRegex(ValueError, 'payload exceeds'):
                exec(validator.result_code('value'), {'value': value})
            self.assertEqual(emitted.getvalue(), '')
        self.assertLess(validator.JSON_PAYLOAD_LIMIT + len(validator.MARKER + validator.END_MARKER),
                        validator.EDITOR_OUTPUT_LIMIT)

    def test_rpc_and_value_frame_all_results_and_capture_diagnostics(self):
        diagnostics = []
        mcp = validator.MCP(12345, diagnostics.append)
        namespace = {}

        def editor(name, **args):
            self.assertEqual(name, 'editor_run')
            self.assertEqual(args['output_max_chars'], validator.EDITOR_OUTPUT_LIMIT)
            emitted = io.StringIO()
            with redirect_stdout(emitted):
                exec(args['code'], namespace)
            return {'structuredContent': {'success': True, 'output': {'text': emitted.getvalue()}}}

        with mock.patch.object(mcp, 'tool', side_effect=editor):
            self.assertIsNone(mcp.rpc("print('Using non-default LichtFeld portal host: 127.0.0.1:12345')\nx = 42"))
            self.assertEqual(mcp.value("{'answer': x}"), {'answer': 42})
        self.assertEqual(len(diagnostics), 1)
        self.assertIn('Using non-default', diagnostics[0])

    def test_truncated_or_unfinished_editor_fails_and_preserves_diagnostics(self):
        for extra in ({'running': True}, {'timed_out': True}, {'output': {'text': 'partial', 'truncated': True}}):
            diagnostics = []
            mcp = validator.MCP(12345, diagnostics.append)
            result = {'structuredContent': {'success': True, 'output': {'text': 'partial'}, **extra}}
            with self.subTest(extra=extra), mock.patch.object(mcp, 'tool', return_value=result):
                with self.assertRaises(RuntimeError):
                    mcp.value('True')
            self.assertEqual(diagnostics, ['partial'])

    def test_png_capture(self):
        data = b'\x89PNG\r\n\x1a\nexample'
        self.assertEqual(validator.png_data({'content': [{'type': 'text', 'text': 'capture'},
            {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(data).decode()}]}), data)

    def test_missing_or_invalid_png(self):
        for result in ({}, {'content': [{'type': 'image', 'mimeType': 'image/png', 'data': 'bm90LXBuZw=='}]},
                       {'content': [{'type': 'image', 'mimeType': 'image/jpeg', 'data': 'AAAA'}]}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                validator.png_data(result)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.steps = [dict(name='Publish', status='PASS', seconds=1.25, detail='ready', screenshot='/tmp/a b.png'),
                      dict(name='Pull', status='FAIL', seconds=2., detail='bad | result\nsecond line')]

    def test_markdown_step_table(self):
        result = validator.render_report(self.steps, 'python script.py', '/tmp/artifacts')
        self.assertIn('| Publish | PASS | 1.25 | ready | [PNG](</tmp/a b.png>) |', result)
        self.assertIn('bad &#124; result<br>second line', result)
        self.assertIn('python script.py', result)

    def test_last_thirty_logs_only_and_html_escaped(self):
        logs = [f'line {i} <error>' for i in range(40)]
        result = validator.render_report([], 'command', '/tmp', ['note'], logs)
        self.assertNotIn('line 9 ', result)
        self.assertIn('line 10 &lt;error&gt;', result)
        self.assertIn('line 39 &lt;error&gt;', result)
        self.assertIn('- note', result)

    def test_junit_counts_and_failure(self):
        root = ET.fromstring(validator.render_junit(self.steps))
        self.assertEqual(root.attrib['tests'], '2')
        self.assertEqual(root.attrib['failures'], '1')
        self.assertEqual(root.attrib['time'], '3.250')
        self.assertEqual(root.findall('testcase')[1].find('failure').text, self.steps[1]['detail'])
        self.assertEqual(root.findall('testcase')[0].find('system-out').text, '/tmp/a b.png')

    def test_empty_junit(self):
        root = ET.fromstring(validator.render_junit([]))
        self.assertEqual(root.attrib['tests'], '0')
        self.assertEqual(root.attrib['failures'], '0')


class CleanupTests(unittest.TestCase):
    def test_cleanup_scopes_to_this_run_and_excludes_tombstones(self):
        prefix = 'E2E validate ' + 'a' * 32 + ' '
        live = dict(id='owned', title=prefix + 'portable-multi', status='ready')
        scenes = [live, dict(id='deleted', title=prefix + 'old', status='deleted'),
                  dict(id='other', title='E2E validate ' + 'b' * 32 + ' portable-multi', status='ready'),
                  dict(id='normal', title='My scene', status='ready')]
        self.assertEqual(validator.owned_scenes(scenes, prefix), [live])

    def test_cleanup_rejects_broad_prefix(self):
        for prefix in ('', 'E2E validate ', 'E2E', 'E2E validate ' + 'a' * 32):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                validator.owned_scenes([], prefix)


class DisplayTests(unittest.TestCase):
    def test_occupied_displays_are_skipped_unless_explicit_and_strict(self):
        for flags, strict in (([], False), (['--strict-display'], False),
                              (['--display', ':93'], False),
                              (['--display=:93', '--strict-display'], True)):
            with self.subTest(flags=flags):
                run = object.__new__(validator.Run)
                run.args = validator.parse_args(['--build-dir', '/build', *flags])
                run.notes = []
                proc = mock.Mock()
                proc.poll.return_value = None
                run.spawn = mock.Mock(return_value=proc)

                def exists(path):
                    # Socket-only occupancy at :93, lock-only occupancy at :94.
                    return str(path) in ('/tmp/.X11-unix/X93', '/tmp/.X94-lock') or (
                        run.spawn.called and str(path) == '/tmp/.X11-unix/X95')

                with mock.patch.object(Path, 'exists', exists):
                    if strict:
                        with self.assertRaisesRegex(RuntimeError, 'occupied.*strict-display'):
                            run.start_display()
                        run.spawn.assert_not_called()
                    else:
                        run.start_display()
                        self.assertEqual(run.args.display, ':95')
                        self.assertEqual(run.spawn.call_args.args[0][1], ':95')
                        self.assertIn(':95', run.notes[-1])

    def test_free_explicit_strict_display_is_used(self):
        run = object.__new__(validator.Run)
        run.args = validator.parse_args(['--build-dir', '/build', '--display', ':104', '--strict-display'])
        run.notes = []
        proc = mock.Mock()
        proc.poll.return_value = None
        run.spawn = mock.Mock(return_value=proc)
        with mock.patch.object(Path, 'exists', lambda path: run.spawn.called):
            run.start_display()
        self.assertEqual(run.args.display, ':104')


class MainTests(unittest.TestCase):
    def run_main(self, workflow=None, cleanup=None, log_scan=None, close=None):
        def passed(run):
            with run.step('Successful workflow'):
                run.record_diagnostics('unrelated editor diagnostic')

        def clean_logs(run):
            with run.step('Studio and portal log scan'):
                pass

        roots = []
        close_run = close or validator.Run.close

        def close_and_track(run):
            roots.append(run.root)
            close_run(run)

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'report.md'
            junit = Path(directory) / 'report.xml'
            with mock.patch.object(validator, 'free_port', return_value=12345), \
                 mock.patch.object(validator.Run, 'workflow', workflow or passed), \
                 mock.patch.object(validator.Run, 'cleanup_remote', cleanup or (lambda run: None)), \
                 mock.patch.object(validator.Run, 'check_logs', log_scan or clean_logs), \
                 mock.patch.object(validator.Run, 'close', close_and_track), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = validator.main(['--build-dir', '/build', '--report', str(report), '--junit', str(junit)])
            for root in roots:
                shutil.rmtree(root, ignore_errors=True)
            return code, report.read_text(), ET.fromstring(junit.read_text())

    def test_clean_run_is_zero_and_diagnostics_reach_report(self):
        code, report, junit = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn('unrelated editor diagnostic', report)
        self.assertIn('editor-diagnostics.log', report)
        self.assertEqual(junit.attrib['failures'], '0')

    def test_failed_step_then_passing_log_scan_stays_failed(self):
        def workflow(run):
            with run.step('Publish closed portable-multi'):
                raise ValueError('broken JSON')

        code, report, junit = self.run_main(workflow=workflow)
        self.assertEqual(code, 1)
        self.assertIn('| Publish closed portable-multi | FAIL |', report)
        self.assertIn('| Studio and portal log scan | PASS |', report)
        self.assertEqual(junit.attrib['failures'], '1')

    def test_failed_row_without_exception_still_fails(self):
        def workflow(run):
            run.steps.append(dict(name='Failed row', status='FAIL', seconds=0., detail='failed'))

        self.assertEqual(self.run_main(workflow=workflow)[0], 1)

    def test_exceptions_outside_steps_return_nonzero_and_are_reported(self):
        def raises(run):
            raise RuntimeError('outside step')

        for phase in ('workflow', 'cleanup', 'log_scan'):
            with self.subTest(phase=phase):
                code, report, junit = self.run_main(**{phase: raises})
                self.assertEqual(code, 1)
                self.assertIn('outside step', report)
                self.assertEqual(junit.attrib['failures'], '1')

    def test_shutdown_exception_returns_nonzero_and_restores_signal(self):
        original_close = validator.Run.close
        original_handler = validator.signal.getsignal(validator.signal.SIGTERM)

        def close(run):
            original_close(run)
            raise OSError('shutdown failed')

        code, report, junit = self.run_main(close=close)
        self.assertEqual(code, 1)
        self.assertIn('| Shutdown | FAIL |', report)
        self.assertEqual(junit.attrib['failures'], '1')
        self.assertEqual(validator.signal.getsignal(validator.signal.SIGTERM), original_handler)

    def test_late_log_failure_returns_nonzero(self):
        def workflow(run):
            (run.artifacts / 'studio-0.log').write_text('[error] shutdown failure\n')

        code, report, junit = self.run_main(workflow=workflow)
        self.assertEqual(code, 1)
        self.assertIn('| Studio and portal log scan | FAIL |', report)
        self.assertEqual(junit.attrib['failures'], '1')

    def test_setup_exception_returns_nonzero(self):
        with mock.patch.object(validator, 'Run', side_effect=OSError('setup failed')), redirect_stderr(io.StringIO()):
            self.assertEqual(validator.main(['--build-dir', '/build']), 1)


if __name__ == '__main__':
    unittest.main()
