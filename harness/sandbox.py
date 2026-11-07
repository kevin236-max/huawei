"""Fail-closed bubblewrap isolation with distinct host UIDs and read-only proc."""
import os
from pathlib import Path
import subprocess

from harness.shared_proc import Identity, require_root

ROOT = Path(__file__).resolve().parent.parent


def start(config, stage, public, cpus, mode, command, control=None, *, deadline=None):
    """Start isolated command; caller must call stop(process, identity)."""
    if mode not in ('agent', 'main', 'candidate'):
        raise ValueError('Invalid sandbox mode')
    if not cpus or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError('Requested CPUs are unavailable')
    stage, public = Path(stage).absolute(), Path(public).resolve(strict=True)
    if stage.is_symlink() or stage != stage.resolve():
        raise ValueError('Stage must be a canonical directory')
    stage.mkdir(mode=0o700, parents=True, exist_ok=True)
    stage.chmod(0o700)
    for name in ('workspace', 'home', 'tmp'):
        path = stage / name
        if path.is_symlink():
            raise ValueError('Writable mount roots cannot be symlinks')
        path.mkdir(mode=0o700, exist_ok=True)
    if mode == 'main':
        # bwrap cannot create nested mount points after /workspace is read-only.
        input_mount = stage / 'workspace/input'
        if input_mount.is_symlink():
            raise ValueError('Input mount point cannot be a symlink')
        input_mount.mkdir(mode=0o755, exist_ok=True)
    require_root()
    python = str(Path(config['python']).absolute())
    args = [config['bwrap'], '--unshare-user', '--unshare-net', '--unshare-ipc', '--unshare-uts',
            '--die-with-parent', '--new-session', '--cap-drop', 'ALL', '--clearenv',
            '--perms', '0755', '--dir', '/opt', '--perms', '0755', '--dir', '/home',
            '--ro-bind', '/usr', '/usr', '--tmpfs', '/usr/local', '--remount-ro', '/usr/local']
    for name in ('bin', 'sbin', 'lib', 'lib64'):
        path = Path('/') / name
        if path.is_symlink():
            args += ['--symlink', os.readlink(path), str(path)]
        elif path.exists():
            args += ['--ro-bind', str(path), str(path)]
    args += ['--perms', '0755', '--dir', '/etc', '--perms', '0755', '--dir', '/etc/ssl']
    for source in ('/etc/ld.so.cache', '/etc/ssl/certs', '/etc/ssl/openssl.cnf', '/etc/alternatives'):
        if Path(source).exists():
            args += ['--ro-bind', source, source]

    def bind_runtime(source, target):
        for parent in reversed(Path(target).parents):
            if parent != Path('/'):
                args.extend(['--perms', '0755', '--dir', str(parent)])
        args.extend(['--ro-bind', str(source), str(target)])

    python_prefix = Path(python).parent.parent
    real_prefix = python_prefix.resolve()
    if real_prefix != Path('/usr'):
        bind_runtime(real_prefix, real_prefix)
    if python_prefix != real_prefix:
        bind_runtime(real_prefix, python_prefix)
    args += ['--ro-bind', '/proc', '/proc', '--dev', '/dev', '--tmpfs', '/dev/shm',
             '--ro-bind', str(ROOT / 'harness/seccomp.py'), '/opt/entry.py',
             '--ro-bind', str(ROOT / 'harness/forecasting.py'), '/opt/forecasting.py',
             '--ro-bind', str(ROOT / 'harness/analysis.py'), '/opt/analysis.py',
             '--ro-bind' if mode == 'main' else '--bind', str(stage / 'workspace'), '/workspace',
             '--ro-bind', str(public), '/workspace/input',
             '--bind', str(stage / 'home'), '/home/agent', '--bind', str(stage / 'tmp'), '/tmp']
    threads = str(max(1, min(config.get('native_threads', 1), len(cpus))))
    env = {'HOME':'/home/agent', 'TMPDIR':'/tmp', 'XDG_CACHE_HOME':'/tmp/cache',
           'MPLCONFIGDIR':'/tmp/matplotlib', 'LANG':'C.UTF-8', 'TZ':'UTC',
           'PATH':str(Path(python).parent)+':/usr/bin:/bin', 'PYTHONPATH':'/opt:/workspace',
           'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONNOUSERSITE':'1', 'PIP_NO_INDEX':'1',
           'OMP_NUM_THREADS':threads, 'OPENBLAS_NUM_THREADS':threads, 'MKL_NUM_THREADS':threads}
    if mode in ('agent', 'main'):
        if control is None:
            raise ValueError('Agent sandbox needs a control directory')
        pi_root = Path(config['pi_root']).resolve()
        for relative in ('packages/coding-agent/dist', 'packages/coding-agent/package.json',
                         'packages/coding-agent/README.md', 'packages/coding-agent/CHANGELOG.md',
                         'packages/coding-agent/docs', 'node_modules/@silvia-odwyer/photon-node',
                         'node_modules/jiti'):
            source = pi_root / relative
            if source.exists():
                bind_runtime(source.resolve(), '/opt/pi/' + relative)
        bind_runtime(Path(config['node']).resolve(), '/opt/node/bin/node')
        args += ['--ro-bind', str(ROOT / 'harness/relay.py'), '/opt/relay.py',
                 '--ro-bind', str(Path(control).resolve()), '/control']
        if mode == 'agent':
            submissions = stage / 'submissions'
            if submissions.is_symlink():
                raise ValueError('Submission export root cannot be a symlink')
            submissions.mkdir(mode=0o755, exist_ok=True)
            args += ['--ro-bind', str(submissions), '/workspace/submissions',
                     '--ro-bind', str(ROOT / 'tools'), '/opt/tools']
            env['PATH'] = '/opt/tools:' + env['PATH']
        env['PI_CODING_AGENT_DIR'] = '/home/agent/.pi'
        env['PATH'] = '/opt/node/bin:' + env['PATH']
    for key, value in env.items():
        args += ['--setenv', key, value]
    args += ['--remount-ro', '/', '--chdir', '/', '--', python, '-I', '-B', '/opt/entry.py',
             'candidate' if mode == 'candidate' else 'agent', ','.join(map(str, cpus)), *command]
    identity = Identity(stage, stage.parent)
    process = identity.launch(args, control, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
                              env={'PATH':'/usr/bin:/bin', 'LANG':'C.UTF-8'}, deadline=deadline)
    return process, identity


def stop(process, identity):
    """Kill every process belonging to this session, reap, and close all pipes."""
    try:
        identity.stop(process)
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
