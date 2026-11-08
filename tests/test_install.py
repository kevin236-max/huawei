"""Exercise installer file publication with local fake dependency tools only."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE_TOOL = '''#!PYTHON
import json
import os
from pathlib import Path
import shutil
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['INSTALL_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps({'tool':name, 'args':args, 'cwd':os.getcwd(),
                            'env': {key: os.environ.get(key) for key in
                                    ('NODE_USE_ENV_PROXY', 'http_proxy', 'https_proxy', 'no_proxy',
                                     'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY')}})+'\\n')
if name == 'git':
    if args[-2:] == ['status', '--porcelain']:
        if os.environ.get('DIRTY_SOURCE'): print(' M package.json')
    elif args[-2:] == ['archive', 'HEAD']:
        sys.stdout.buffer.write(Path(os.environ['INSTALL_TEST_SOURCE']).read_bytes())
    else:
        raise SystemExit('Unexpected git command')
elif name == 'node':
    if args[0] == '-e':
        raise SystemExit(1 if os.environ.get('OLD_SYSTEM_NODE') and Path(sys.argv[0]).parent.name == 'fake-bin' else 0)
    if args[0] == '-':
        sys.stdin.read()
        file, node, pi_root, bwrap = args[1:]
        Path(file).write_text(json.dumps(dict(node=node, pi_root=pi_root, bwrap=bwrap)))
    else:
        assert Path(args[0]).is_file(), args[0]
        print('test-pi')
elif name == 'npm':
    if args == ['run', 'build']:
        if os.environ.get('FAIL_BUILD'): raise SystemExit(14)
        cli = Path('packages/coding-agent/dist/bundle/cli.js')
        cli.parent.mkdir(parents=True)
        cli.write_text('test build')
elif name == 'curl':
    shutil.copyfile(os.environ['INSTALL_TEST_NODE'], args[args.index('-o')+1])
elif name == 'bwrap':
    print('bubblewrap test')
elif name == 'id':
    print(os.environ.get('TEST_UID', '0'))
elif name == 'sudo':
    assert args[0].startswith('--preserve-env=')
    os.execvp(args[1], args[1:])
elif name == 'dpkg-query':
    print('install ok installed')
elif name == 'apt-get':
    if os.environ.get('FAIL_APT'): raise SystemExit(100)
    if 'install' in args:
        Path(os.environ['INSTALL_TEST_PACKAGES']).touch()
elif name == 'python':
    if args[0] == '-c' and os.environ.get('MISSING_SECCOMP'):
        raise SystemExit(0 if Path(os.environ['INSTALL_TEST_PACKAGES']).exists() else 1)
elif name != 'python':
    raise SystemExit('Unexpected dependency tool: '+name)
'''.replace('PYTHON', sys.executable, 1)


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.project = self.directory/'project'
        self.source = self.project/'packages/pi'
        self.source.mkdir(parents=True)
        (self.source/'package.json').write_text('{}')
        shutil.copyfile(ROOT/'install.sh', self.project/'install.sh')
        self.bin = self.directory/'fake-bin'
        self.bin.mkdir()
        for name in ('git', 'node', 'npm', 'python', 'bwrap', 'curl', 'apt-get', 'sudo', 'id', 'dpkg-query'):
            path = self.bin/name
            path.write_text(FAKE_TOOL)
            path.chmod(0o755)
        self.log = self.directory/'calls.jsonl'
        self.archive = self.directory/'source.tar'
        with tarfile.open(self.archive, 'w') as archive:
            archive.add(self.source/'package.json', arcname='package.json')
        self.node_archive = self.directory/'node.tar.xz'
        with tarfile.open(self.node_archive, 'w:xz') as archive:
            for name in ('node', 'npm'):
                content = FAKE_TOOL.encode()
                member = tarfile.TarInfo('node-v24-test/bin/'+name)
                member.size = len(content)
                member.mode = 0o755
                archive.addfile(member, io.BytesIO(content))
        self.runtime = self.project/'.runtime'
        clean_env = {key: value for key, value in os.environ.items()
                     if 'proxy' not in key.lower()}
        self.env = {**clean_env, 'PATH':str(self.bin)+':/usr/bin:/bin',
                    'HUAWEI_PYTHON':str(self.bin/'python'), 'PI_BWRAP':str(self.bin/'bwrap'),
                    'PI_RUNTIME_ROOT':str(self.runtime), 'INSTALL_TEST_LOG':str(self.log),
                    'INSTALL_TEST_SOURCE':str(self.archive), 'INSTALL_TEST_NODE':str(self.node_archive),
                    'INSTALL_TEST_PACKAGES':str(self.directory/'packages-installed')}

    def install(self, **environment):
        return subprocess.run(['/bin/bash', str(self.project/'install.sh')], cwd=self.directory,
                              env={**self.env, **environment}, text=True, capture_output=True, timeout=15)

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_builds_snapshot_in_runtime_and_publishes_current_cli(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        runtime = json.loads((self.runtime/'runtime.json').read_text())
        self.assertEqual(runtime, dict(node=str(self.bin/'node'), pi_root=str(self.runtime/'pi'),
                                       bwrap=str(self.bin/'bwrap')))
        self.assertEqual((self.runtime/'pi/packages/coding-agent/dist/bundle/cli.js').read_text(), 'test build')
        self.assertFalse((self.source/'node_modules').exists())
        self.assertFalse((self.source/'packages').exists())
        calls = self.calls()
        self.assertFalse(any(call['tool']=='curl' for call in calls))
        npm = [call for call in calls if call['tool']=='npm']
        self.assertEqual([call['args'] for call in npm], [['ci','--ignore-scripts'], ['run','build']])
        self.assertTrue(all(Path(call['cwd']).is_relative_to(self.runtime) for call in npm))
        self.assertFalse(list(self.runtime.glob('.install.*')))

    def test_failed_build_preserves_active_runtime(self):
        (self.runtime/'pi').mkdir(parents=True)
        (self.runtime/'pi/keep.py').write_text('previous build')
        (self.runtime/'runtime.json').write_text('{"previous":true}')
        result = self.install(FAIL_BUILD='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.runtime/'pi/keep.py').read_text(), 'previous build')
        self.assertEqual(json.loads((self.runtime/'runtime.json').read_text()), {'previous':True})
        self.assertFalse(list(self.runtime.glob('.install.*')))

    def test_old_system_node_downloads_and_uses_local_node(self):
        result = self.install(OLD_SYSTEM_NODE='1')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        runtime = json.loads((self.runtime/'runtime.json').read_text())
        self.assertEqual(runtime['node'], str(self.runtime/'node/bin/node'))
        downloads = [call for call in self.calls() if call['tool']=='curl']
        self.assertEqual(len(downloads), 1)
        self.assertTrue(any(argument.startswith('https://nodejs.org/dist/v24.') for argument in downloads[0]['args']))

    def test_node_without_npm_downloads_complete_toolchain(self):
        (self.bin/'npm').unlink()
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        runtime = json.loads((self.runtime/'runtime.json').read_text())
        self.assertEqual(runtime['node'], str(self.runtime/'node/bin/node'))
        self.assertTrue(any(call['tool']=='curl' for call in self.calls()))

    def test_rejects_runtime_that_would_replace_source(self):
        result = self.install(PI_RUNTIME_ROOT=str(self.project/'packages'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('dedicated runtime directory', result.stderr)
        self.assertEqual((self.source/'package.json').read_text(), '{}')
        self.assertFalse(any(call['tool']=='npm' for call in self.calls()))

    def test_invalid_explicit_bubblewrap_fails_before_installing_packages(self):
        result = self.install(PI_BWRAP=str(self.directory/'missing-bwrap'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('PI_BWRAP', result.stderr)
        self.assertFalse(any(call['tool']=='apt-get' for call in self.calls()))
        self.assertFalse(any(call['tool']=='npm' for call in self.calls()))

    def test_proxy_reaches_downloads_and_node_build(self):
        result = self.install(OLD_SYSTEM_NODE='1', HTTPS_PROXY='http://proxy.test:8080',
                              HTTP_PROXY='http://proxy.test:8080', NO_PROXY='localhost,127.0.0.1')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        for call in self.calls():
            if call['tool'] in ('curl', 'python', 'node', 'npm'):
                self.assertEqual(call['env']['NODE_USE_ENV_PROXY'], '1')
                self.assertEqual(call['env']['https_proxy'], 'http://proxy.test:8080')
                self.assertEqual(call['env']['http_proxy'], 'http://proxy.test:8080')
                self.assertEqual(call['env']['no_proxy'], 'localhost,127.0.0.1')

    def test_missing_system_dependency_is_installed_by_default(self):
        for uid in ('0', '1000'):
            with self.subTest(uid=uid):
                Path(self.env['INSTALL_TEST_PACKAGES']).unlink(missing_ok=True)
                self.log.unlink(missing_ok=True)
                result = self.install(MISSING_SECCOMP='1', TEST_UID=uid,
                                      https_proxy='http://proxy.test:8080')
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                apt = [call for call in self.calls() if call['tool']=='apt-get']
                self.assertEqual(apt[0]['args'], ['update'])
                self.assertIn('libseccomp2', apt[1]['args'])
                self.assertIn('ca-certificates', apt[1]['args'])
                self.assertEqual(apt[0]['env']['https_proxy'], 'http://proxy.test:8080')
                sudo = [call for call in self.calls() if call['tool']=='sudo']
                self.assertEqual(len(sudo), 2 if uid=='1000' else 0)
                if sudo:
                    self.assertIn('https_proxy', sudo[0]['args'][0])

    def test_apt_failure_stops_before_download_and_build(self):
        result = self.install(MISSING_SECCOMP='1', FAIL_APT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(any(call['tool']=='apt-get' for call in self.calls()))
        self.assertFalse(any(call['tool'] in ('curl', 'npm') for call in self.calls()))


if __name__ == '__main__':
    unittest.main()
