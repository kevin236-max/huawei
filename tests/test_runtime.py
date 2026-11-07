import contextlib
import http.client
import http.server
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from harness import gateway, pi, sandbox, seccomp


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__('localhost', timeout=2)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class RuntimeTests(unittest.TestCase):
    def test_persistent_main_keeps_process_memory_and_one_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / 'main'
            config = {'python': sys.executable, 'api': {'model': 'test', 'max_tokens': 10}}
            children = []
            stopped = []
            class Gateway:
                def stop(self):
                    stopped.append('gateway')
            def start(*args, **kwargs):
                child = subprocess.Popen([sys.executable, '-u', '-c', '''
import json, sys
history = []
for line in sys.stdin:
    history.append(json.loads(line)['message'])
    print(json.dumps({'type': 'message_end', 'message': {
        'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(history)}],
        'stopReason': 'stop'}}), flush=True)
    print(json.dumps({'type': 'agent_settled'}), flush=True)
'''], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                children.append(child)
                return child, object()
            def stop(child, identity):
                child.kill()
                child.wait()
                for stream in (child.stdin, child.stdout, child.stderr):
                    stream.close()
                stopped.append('process')
            with patch.object(gateway, 'start_gateway', return_value=Gateway()), \
                 patch.object(sandbox, 'start', side_effect=start), \
                 patch.object(sandbox, 'stop', side_effect=stop):
                with pi.PiSession(config, stage, Path(temporary), [0], time.monotonic()+5, role='main') as session:
                    first = session.prompt('make a plan')
                    second = session.prompt('worker result\n\nsubmit final result')
                    self.assertEqual(json.loads(first['text']), ['make a plan'])
                    self.assertEqual(json.loads(second['text']), ['make a plan', 'worker result\n\nsubmit final result'])
                    self.assertEqual(second['status'], 'finished')
                    self.assertEqual(len(children), 1)
                    self.assertIsNone(children[0].poll())
                    self.assertEqual(stopped, [])
                self.assertEqual(stopped, ['process', 'gateway'])
                # A closed session must never silently restart with an empty context.
                self.assertEqual(session.prompt('again')['status'], 'failed')
                self.assertEqual(len(children), 1)
            self.assertEqual((stage / 'prompt.txt').read_text(), 'make a plan')
            prompts = [json.loads(line)['message'] for line in (stage / 'prompts.jsonl').read_text().splitlines()]
            self.assertEqual(prompts, ['make a plan', 'worker result\n\nsubmit final result'])
            events = [json.loads(line) for line in (stage / 'trajectory.jsonl').read_text().splitlines()]
            self.assertEqual(len(events), 2)
            settings = json.loads((stage / 'home/.pi/settings.json').read_text())
            self.assertFalse(settings['compaction']['enabled'])

    def test_persistent_main_closes_on_exception_between_turns(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / 'main'
            config = {'python': sys.executable, 'api': {'model': 'test', 'max_tokens': 10}}
            child = subprocess.Popen([sys.executable, '-u', '-c',
                'import sys,json,time;sys.stdin.readline();print(json.dumps({"type":"agent_settled"}),flush=True);time.sleep(10)'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            stopped = []
            class Gateway:
                def stop(self): stopped.append('gateway')
            def stop(process, identity):
                process.kill()
                process.wait()
                for stream in (process.stdin, process.stdout, process.stderr): stream.close()
            try:
                with patch.object(gateway, 'start_gateway', return_value=Gateway()), \
                     patch.object(sandbox, 'start', return_value=(child, object())), \
                     patch.object(sandbox, 'stop', side_effect=stop):
                    with self.assertRaisesRegex(ValueError, 'invalid plan'):
                        with pi.PiSession(config, stage, Path(temporary), [0], time.monotonic()+5, role='main') as session:
                            self.assertEqual(session.prompt('plan')['status'], 'finished')
                            raise ValueError('invalid plan')
                self.assertIsNotNone(child.returncode)
                self.assertTrue(child.stdin.closed and child.stdout.closed and child.stderr.closed)
                self.assertEqual(stopped, ['gateway'])
            finally:
                if child.poll() is None: stop(child, None)

    def test_final_json_handles_fences_and_trailing_prose(self):
        self.assertEqual(pi.final_json('Result:\n```json\n{"path":"a.py", "nested":{"x":1}}\n```\nDone.'),
                         {'path': 'a.py', 'nested': {'x': 1}})
        self.assertEqual(pi.final_json('draft {bad} then {"ok":true}'), {'ok': True})
        with self.assertRaises(ValueError):
            pi.final_json('no JSON')

    def rpc(self, code, prompt='hello', duration=2, soft_after=None):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            process = subprocess.Popen([sys.executable, '-u', '-c', code], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            try:
                started = time.monotonic()
                options = {} if soft_after is None else dict(
                    soft_deadline=started+soft_after, finish_prompt='Finish the current candidate and report.')
                result = pi.consume_rpc(process, prompt, started + duration,
                                        directory / 'trajectory.jsonl', directory / 'stderr.log', **options)
                prompts = directory / 'prompts.jsonl'
                result['injected_prompts'] = [json.loads(line) for line in prompts.read_text().splitlines()] if prompts.exists() else []
                messages = [json.loads(line) for line in (directory / 'trajectory.jsonl').read_text().splitlines()]
                return result, messages, (directory / 'stderr.log').read_text()
            finally:
                process.kill()
                process.wait()
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()

    def test_rpc_drains_pipes_and_keeps_full_messages(self):
        result, messages, stderr = self.rpc('''
import json,sys,time
sys.stderr.write('e' * 100000)
sys.stderr.flush()
request = json.loads(sys.stdin.readline())
assert len(request['message']) == 200000
print('not-json')
for message in [
 {'role':'assistant','content':[{'type':'thinking','thinking':'think'},{'type':'toolCall','name':'bash','arguments':{'bad':'JSON'}}]},
 {'role':'toolResult','content':[{'type':'text','text':'tool output'}]},
 {'role':'assistant','content':[{'type':'text','text':'answer'}],'stopReason':'stop'}]:
 print(json.dumps({'type':'message_end','message':message}), flush=True)
print(json.dumps({'type':'agent_end'}), flush=True)
time.sleep(.05)
print(json.dumps({'type':'agent_settled'}), flush=True)
time.sleep(10)
''', prompt='p' * 200000)
        self.assertEqual(result['status'], 'finished')
        self.assertEqual(result['text'], 'answer')
        self.assertEqual(len(result['messages']), 3)
        self.assertEqual(messages[0]['message']['content'][0]['type'], 'thinking')
        self.assertEqual(len(stderr), 100000)

    def test_rpc_rejects_assistant_error_and_process_exit(self):
        result, _, _ = self.rpc('''
import sys,json
sys.stdin.readline()
print(json.dumps({'type':'message_end','message':{'role':'assistant','content':[], 'stopReason':'error','errorMessage':'bad model'}}))
print(json.dumps({'type':'agent_settled'}))
''')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['text'], '')
        result, _, _ = self.rpc("import sys;sys.stdin.readline()")
        self.assertEqual(result['status'], 'failed')

    def test_rpc_empty_final_message_does_not_return_intermediate_text(self):
        result, _, _ = self.rpc('''
import sys,json
sys.stdin.readline()
for message in [
 {'role':'assistant','content':[{'type':'text','text':'intermediate'}]},
 {'role':'assistant','content':[], 'stopReason':'stop'}]:
 print(json.dumps({'type':'message_end','message':message}))
print(json.dumps({'type':'agent_settled'}))
''')
        self.assertEqual(result['text'], '')

    def test_rpc_timeout_uses_shared_monotonic_deadline(self):
        started = time.monotonic()
        result, _, _ = self.rpc('import time; time.sleep(20)', duration=.12)
        self.assertEqual(result['status'], 'timeout')
        self.assertLess(time.monotonic() - started, 1)

    def test_soft_deadline_queues_one_wrapup_and_allows_current_work_to_finish(self):
        result, _, stderr = self.rpc('''
import json, sys, time, select
assert json.loads(sys.stdin.readline())['type'] == 'prompt'
# Work is already running when the soft deadline arrives.
time.sleep(.2)
request = json.loads(sys.stdin.readline())
assert request['type'] == 'steer'
assert request['message'] == 'Finish the current candidate and report.'
print(json.dumps({'type': 'response', 'id': request['id'], 'command': 'steer',
                  'success': True, 'data': {'disposition': 'queued'}}), flush=True)
time.sleep(.2)
assert not select.select([sys.stdin], [], [], .05)[0], 'duplicate wrapup'
sys.stderr.write('current work finished')
print(json.dumps({'type': 'message_end', 'message': {'role': 'assistant',
    'content': [{'type': 'text', 'text': 'candidate evaluated'}], 'stopReason': 'stop'}}), flush=True)
print(json.dumps({'type': 'agent_settled'}), flush=True)
''', soft_after=.05)
        self.assertEqual(result['status'], 'finished')
        self.assertEqual(result['text'], 'candidate evaluated')
        self.assertEqual(stderr, 'current work finished')
        self.assertEqual(len(result['injected_prompts']), 1)
        self.assertEqual(result['injected_prompts'][0]['type'], 'steer')

    def test_soft_deadline_does_not_delay_early_completion(self):
        result, _, _ = self.rpc('''
import json, sys
sys.stdin.readline()
print(json.dumps({'type': 'agent_settled'}), flush=True)
''', soft_after=1)
        self.assertEqual(result['status'], 'finished')
        self.assertEqual(result['injected_prompts'], [])

    def test_hard_deadline_still_limits_worker_after_wrapup(self):
        result, _, _ = self.rpc('''
import json, sys, time
sys.stdin.readline()
request = json.loads(sys.stdin.readline())
assert request['type'] == 'steer'
time.sleep(10)
''', duration=.3, soft_after=.05)
        self.assertEqual(result['status'], 'timeout')

    def test_sandbox_rejects_unknown_mode_and_symlink_mount(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / 'stage'
            stage.mkdir()
            (stage / 'workspace').symlink_to(Path(temporary), target_is_directory=True)
            config = {'python':sys.executable, 'bwrap':'/missing/bwrap'}
            with self.assertRaises(ValueError):
                sandbox.start(config, stage, Path(temporary), [0], 'candidate', ['/opt/test.py'])

    def test_main_rejects_symlink_input_mount_point(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / 'main'
            (stage / 'workspace').mkdir(parents=True)
            (stage / 'workspace/input').symlink_to(root, target_is_directory=True)
            config = {'python':sys.executable, 'bwrap':'/bwrap', 'node':sys.executable,
                      'pi_root':str(root)}
            with patch.object(sandbox, 'require_root'), patch.object(sandbox, 'Identity'):
                with self.assertRaisesRegex(ValueError, 'Input mount point cannot be a symlink'):
                    sandbox.start(config, stage, root, sorted(os.sched_getaffinity(0))[:1],
                                  'main', ['node', 'cli'], root)

    def test_sandbox_mounts_only_public_inputs_and_runtime_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / 'public'
            public.mkdir()
            pi_root = root / 'pi'
            dist = pi_root / 'packages/coding-agent/dist'
            dist.mkdir(parents=True)
            (pi_root / 'private.env').write_text('secret')
            config = {'python':sys.executable, 'bwrap':'/bwrap', 'node':sys.executable,
                      'pi_root':str(pi_root)}
            fake_identity = unittest.mock.Mock()
            def check_mount_points(arguments, *args, **kwargs):
                # bwrap cannot create a nested mount point inside a read-only bind.
                readonly = {}
                for index, option in enumerate(arguments):
                    if option not in ('--bind', '--ro-bind'):
                        continue
                    source, target = arguments[index+1:index+3]
                    for parent, host in readonly.items():
                        if Path(target).is_relative_to(parent) and target != parent:
                            mountpoint = Path(host) / Path(target).relative_to(parent)
                            self.assertTrue(mountpoint.is_dir(), f'Missing mount point: {mountpoint}')
                    if option == '--ro-bind':
                        readonly[target] = source
            fake_identity.launch.side_effect = check_mount_points
            with patch.object(sandbox, 'require_root'), \
                 patch.object(sandbox, 'Identity', return_value=fake_identity):
                available = sorted(os.sched_getaffinity(0))
                sandbox.start(config, root/'main', public, available[:2], 'main', ['node','cli'], root/'control')
                arguments = fake_identity.launch.call_args.args[0]
                mounts = [(arguments[i+1], arguments[i+2]) for i, option in enumerate(arguments) if option == '--ro-bind']
                self.assertIn((str(public), '/workspace/input'), mounts)
                self.assertIn((str(root/'main/workspace'), '/workspace'), mounts)
                self.assertIn((str(dist), '/opt/pi/packages/coding-agent/dist'), mounts)
                self.assertNotIn((str(pi_root), '/opt/pi'), mounts)
                self.assertNotIn('/opt/tools', arguments)
                self.assertEqual(arguments[arguments.index('OMP_NUM_THREADS')+1], '1')
                self.assertIn(','.join(map(str, available[:2])), arguments)
                config['native_threads'] = 1000
                sandbox.start(config, root/'worker', public, available[-2:], 'agent', ['node','cli'], root/'control')
                arguments = fake_identity.launch.call_args.args[0]
                self.assertIn('/opt/tools', arguments)
                self.assertIn((str(root/'worker/submissions'), '/workspace/submissions'),
                              [(arguments[i+1], arguments[i+2]) for i, option in enumerate(arguments) if option == '--ro-bind'])
                self.assertEqual(arguments[arguments.index('OMP_NUM_THREADS')+1], str(len(available[-2:])))
                self.assertIn(','.join(map(str, available[-2:])), arguments)

    def test_candidate_bootstrap_runs_requested_script_without_memory_limit(self):
        original_argv, original_path = sys.argv[:], sys.path[:]
        calls = []
        class Libc:
            def prctl(self, *_):
                return 0
            def capset(self, *_):
                return 0
        try:
            sys.argv = ['entry.py','candidate','0','/opt/forecasting.py','--serve','/workspace/selected.json']
            with patch.object(seccomp.ctypes, 'CDLL', return_value=Libc()), \
                 patch.object(seccomp.os, 'setresgid'), patch.object(seccomp.os, 'setresuid'), \
                 patch.object(seccomp.os, 'getgroups', return_value=[]), \
                 patch.object(seccomp.os, 'sched_setaffinity'), patch.object(seccomp.os, 'umask'), \
                 patch.object(seccomp.os, 'chdir'), patch.object(seccomp.os, 'listdir', return_value=[]), \
                 patch('builtins.open', return_value=io.StringIO('')), \
                 patch.object(seccomp.resource, 'setrlimit') as limits, \
                 patch.object(seccomp, 'install_filter') as install, \
                 patch.object(seccomp.runpy, 'run_path', side_effect=lambda *a, **k: calls.append((a,k,sys.argv[:]))):
                seccomp.main()
            self.assertEqual(calls[0][0], ('/opt/forecasting.py',))
            self.assertEqual(calls[0][2], ['/opt/forecasting.py','--serve','/workspace/selected.json'])
            install.assert_called_once_with(candidate=True)
            self.assertNotIn(seccomp.resource.RLIMIT_AS, [call.args[0] for call in limits.call_args_list])
        finally:
            sys.argv, sys.path = original_argv, original_path

    def test_session_timeout_reaps_process_and_closes_gateway(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / 'stage'
            public = Path(temporary) / 'public'
            public.mkdir()
            config = {'python':sys.executable, 'api':{'model':'test', 'max_tokens':10}}
            process = subprocess.Popen([sys.executable, '-u', '-c', 'import time;time.sleep(20)'],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            stopped = []
            class FakeGateway:
                def stop(self):
                    stopped.append('gateway')
            class Identity:
                def stop(self, child):
                    child.kill()
                    child.wait()
                    stopped.append('process')
            try:
                with patch.object(gateway, 'start_gateway', return_value=FakeGateway()), \
                     patch.object(sandbox, 'start', return_value=(process, Identity())):
                    result = pi.run_session(config, stage, public, [0], 'prompt', time.monotonic()+.1)
                self.assertEqual(result['status'], 'timeout')
                self.assertEqual(stopped, ['process','gateway'])
                self.assertIsNotNone(process.returncode)
                self.assertTrue(process.stdin.closed and process.stdout.closed and process.stderr.closed)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_session_startup_receives_shared_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / 'stage'
            config = {'python':sys.executable, 'api':{'model':'test', 'max_tokens':10}}
            deadline = time.monotonic() + .1
            class FakeGateway:
                def stop(self):
                    pass
            with patch.object(gateway, 'start_gateway', return_value=FakeGateway()), \
                 patch.object(sandbox, 'start', side_effect=RuntimeError('startup')) as launch:
                pi.run_session(config, stage, Path(temporary), [0], 'prompt', deadline)
            self.assertEqual(launch.call_args.kwargs['deadline'], deadline)

    def test_session_writes_no_real_credentials_and_main_tools_are_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / 'stage'
            public = Path(temporary) / 'public'
            public.mkdir()
            config = {'python':sys.executable, 'node':'/usr/bin/node', 'pi_root':'/pi', 'bwrap':'/bwrap',
                      'api':{'model':'test', 'api_key':'host-secret', 'base_url':'http://example.invalid/v1',
                             'max_tokens':10, 'reasoning_effort':'high'}}
            process = subprocess.Popen([sys.executable, '-u', '-c',
                'import sys,json,time;sys.stdin.readline();print(json.dumps({"type":"agent_settled"}),flush=True);time.sleep(10)'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            class FakeGateway:
                def stop(self):
                    pass
            def cleanup(child, identity):
                child.kill()
                child.wait()
                for stream in (child.stdin, child.stdout, child.stderr):
                    stream.close()
            with patch.object(gateway, 'start_gateway', return_value=FakeGateway()), \
                 patch.object(sandbox, 'start', return_value=(process, object())) as launch, \
                 patch.object(sandbox, 'stop', side_effect=cleanup):
                result = pi.run_session(config, stage, public, [0], 'the prompt', time.monotonic()+2, role='main')
            self.assertEqual(result['status'], 'finished')
            self.assertEqual((stage / 'prompt.txt').read_text(), 'the prompt')
            self.assertNotIn('host-secret', (stage / 'home/.pi/models.json').read_text())
            args = launch.call_args.args
            self.assertEqual(args[4], 'main')
            tools = args[5][args[5].index('--tools')+1]
            self.assertEqual(tools, 'read,ls')


class GatewayWithoutSocketsTests(unittest.TestCase):
    def test_request_auth_model_policy_and_streaming_through_http_handler(self):
        requests = []
        class LocalServer:
            def __init__(self, path, handler):
                self.handler = handler
            def serve_forever(self, **kwargs):
                pass
            def shutdown(self):
                pass
            def server_close(self):
                pass
        class Transport:
            def __init__(self, payload=b''):
                self.input = io.BytesIO(payload)
                self.output = io.BytesIO()
            def makefile(self, mode, *args):
                return self.input
            def sendall(self, data):
                self.output.write(data)
            def settimeout(self, timeout):
                pass
            def shutdown(self, *_):
                pass
        class Response:
            status = 200
            def __init__(self):
                self.chunks = iter([b'data: first\n\n', b'data: [DONE]\n\n', b''])
            def getheader(self, name, default=None):
                return 'text/event-stream'
            def read1(self, size):
                return next(self.chunks)
        class Upstream:
            def __init__(self, *args, **kwargs):
                self.sock = Transport()
            def connect(self):
                pass
            def request(self, method, path, payload, headers):
                requests.append((method, path, json.loads(payload), headers))
            def getresponse(self):
                return Response()
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(gateway.socketserver, 'ThreadingUnixStreamServer', LocalServer), \
             patch.object(gateway.http.client, 'HTTPConnection', Upstream), \
             patch.object(gateway, 'getproxies_environment', return_value={}):
            config = {'base_url':'http://local.invalid/v1', 'model':'fixed', 'api_key':'host-secret',
                      'reasoning_effort':'high', 'max_tokens':20}
            server = gateway.start_gateway(Path(temporary)/'api.sock', config, time.monotonic()+2)
            try:
                def request(body):
                    payload = json.dumps(body).encode()
                    client = Transport(b'POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\nContent-Length: '
                                       + str(len(payload)).encode()+b'\r\n\r\n'+payload)
                    server.handler(client, ('local', 0), server)
                    return client.output.getvalue()
                output = request({'model':'fixed', 'messages':[], 'stream':True})
                self.assertIn(b'200 OK', output)
                self.assertIn(b'data: first\n\ndata: [DONE]\n\n', output)
                self.assertNotIn(b'host-secret', output)
                self.assertEqual(requests[0][3]['Authorization'], 'Bearer host-secret')
                self.assertEqual(requests[0][2]['reasoning_effort'], 'high')
                self.assertEqual(requests[0][2]['max_tokens'], 20)
                self.assertIn(b'400', request({'model':'other', 'messages':[]}))
                self.assertIn(b'400', request({'model':'fixed', 'messages':[], 'max_tokens':21}))
                self.assertEqual(len(requests), 1)
            finally:
                server.stop()


def local_sockets_available():
    try:
        with socket.socket(socket.AF_INET) as stream:
            stream.bind(('127.0.0.1', 0))
        with tempfile.TemporaryDirectory() as temporary:
            with socket.socket(socket.AF_UNIX) as stream:
                stream.bind(str(Path(temporary) / 'probe.sock'))
        return True
    except PermissionError:
        return False


@unittest.skipUnless(local_sockets_available(), 'Environment denies local TCP/Unix sockets')
class GatewayTests(unittest.TestCase):
    def test_gateway_streams_with_host_auth_and_fixed_model(self):
        received = []
        release = threading.Event()
        class Upstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                received.append((self.path, self.headers.get('Authorization'),
                                 json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(b'data: first\n\n')
                self.wfile.flush()
                release.wait(2)
                self.wfile.write(b'data: [DONE]\n\n')
        upstream = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'api.sock'
            config = {'base_url': f'http://127.0.0.1:{upstream.server_port}/v1',
                      'model':'fixed', 'api_key':'secret', 'reasoning_effort':'high', 'max_tokens':20, 'api_timeout':3}
            server = gateway.start_gateway(path, config, time.monotonic()+4)
            connection = UnixHTTPConnection(path)
            try:
                connection.request('POST', '/v1/chat/completions', json.dumps({'model':'fixed', 'messages':[], 'stream':True}))
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.readline(), b'data: first\n')
                release.set()
                self.assertIn(b'[DONE]', response.read())
                self.assertEqual(received[0][0], '/v1/chat/completions')
                self.assertEqual(received[0][1], 'Bearer secret')
                self.assertEqual(received[0][2]['reasoning_effort'], 'high')
                self.assertEqual(received[0][2]['max_tokens'], 20)
                connection.close()
                connection = UnixHTTPConnection(path)
                connection.request('POST', '/v1/chat/completions', json.dumps({'model':'other', 'messages':[]}))
                self.assertEqual(connection.getresponse().status, 400)
            finally:
                release.set()
                connection.close()
                server.stop()
                upstream.shutdown()
                upstream.server_close()
            self.assertFalse(path.exists())

    def test_gateway_binds_under_long_session_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / ('long_method_' * 12) / 'control'
            directory.mkdir(parents=True)
            path = directory / 'api.sock'
            server = gateway.start_gateway(path, {'base_url':'http://127.0.0.1:1/v1','model':'m','max_tokens':20}, time.monotonic()-1)
            try:
                self.assertTrue(path.exists())
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    connection = UnixHTTPConnection(f'/proc/self/fd/{fd}/api.sock')
                    connection.request('POST','/v1/chat/completions', json.dumps({'model':'m', 'messages':[]}))
                    self.assertEqual(connection.getresponse().status, 403)
                    connection.close()
                finally:
                    os.close(fd)
            finally:
                server.stop()

    def test_gateway_refuses_expired_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'api.sock'
            server = gateway.start_gateway(path, {'base_url':'http://127.0.0.1:1/v1','model':'m','max_tokens':20}, time.monotonic()-1)
            try:
                connection = UnixHTTPConnection(path)
                connection.request('POST','/v1/chat/completions', json.dumps({'model':'m', 'messages':[]}))
                self.assertEqual(connection.getresponse().status, 403)
                connection.close()
            finally:
                server.stop()


if __name__ == '__main__':
    unittest.main()
