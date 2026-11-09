"""Shared read-only proc and per-sandbox host UIDs, adapted from pipeline.

Only the new bwrap user namespace receives UID/GID maps. No host accounts,
cgroups, services, or system mount configuration are changed.
"""
import contextlib
import json
import os
from pathlib import Path
import secrets
import select
import signal
import stat
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
WRITABLE = ('workspace', 'home', 'tmp')
# Live sandbox identities, so an interrupt can kill every sandbox at once (see kill_all).
_LIVE = set()
_LIVE_LOCK = threading.RLock()  # reentrant: the interrupt handler runs on the main thread
_INTERRUPTED = threading.Event()


def kill_all():
    """On interrupt: refuse new sandboxes and SIGKILL every process of every live sandbox UID.

    Sandboxes run in their own sessions under their own UIDs, so killing the launcher alone
    would leave them running. Owners still call Identity.stop afterwards for the normal cleanup.
    """
    _INTERRUPTED.set()
    with _LIVE_LOCK:
        uids = [identity.uid for identity in _LIVE]
    for uid in uids:
        for pid in processes(uid):
            with contextlib.suppress(ProcessLookupError, FileNotFoundError, StopIteration):
                lines = Path(f'/proc/{pid}/status').read_text().splitlines()
                ids = next(line for line in lines if line.startswith('Uid:')).split()[1:]
                if ids == [str(uid)] * 4:
                    os.kill(pid, signal.SIGKILL)


def require_root():
    if os.geteuid() != 0:
        raise RuntimeError('Run the launcher as root, as with pipeline shared-proc isolation')
    fields = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    caps = int(fields['CapEff'].strip(), 16)
    if any(not caps & (1 << bit) for bit in (0, 1, 3, 5, 6, 7)):
        raise RuntimeError('Launcher needs CHOWN/DAC_OVERRIDE/FOWNER/KILL/SETGID/SETUID capabilities')
    for name in ('uid_map', 'gid_map'):
        ranges = [list(map(int, row.split())) for row in Path('/proc/self', name).read_text().splitlines()]
        if not any(start <= 1000000000 and start + length >= 2000000000 for start, _, length in ranges):
            raise RuntimeError('Launcher UID/GID mapping does not cover task identities')


def processes(uid):
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            lines = (path / 'status').read_text().splitlines()
            ids = next(line for line in lines if line.startswith('Uid:')).split()[1:]
            state = next(line for line in lines if line.startswith('State:')).split()[1]
            if str(uid) in ids and state not in ('Z', 'X'):
                result.append(int(path.name))
        except (OSError, StopIteration):
            pass
    return result


class Identity:
    def __init__(self, stage, run):
        self.stage = Path(stage)
        if self.stage.resolve(strict=True) != self.stage or not self.stage.is_relative_to(run):
            raise ValueError('Stage must be a canonical directory inside this run')
        info = self.stage.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError('Stage must be root-owned without group/other write access')
        reservations = ROOT / '.cache/task-uids'
        reservations.mkdir(mode=0o700, parents=True, exist_ok=True)
        for _ in range(100):
            uid = 1000000000 + secrets.randbelow(1000000000)
            try:
                with (reservations / str(uid)).open('x') as stream:
                    stream.write(str(self.stage))
            except FileExistsError:
                continue
            if not processes(uid):
                self.uid = uid
                self.reservation = reservations / str(uid)
                break
        else:
            raise RuntimeError('Cannot reserve a sandbox UID')
        (self.stage / 'identity.json').write_text(json.dumps({'host_uid': self.uid, 'sandbox_uid': 1000}))
        self.lock = threading.Lock()
        self.stopped = False

    def prepare(self, control):
        for name in WRITABLE:
            top = self.stage / name
            if top.is_symlink():
                raise ValueError('Writable root cannot be a symlink')
            os.chown(top, self.uid, self.uid)
            for _, dirs, files, fd in os.fwalk(top, follow_symlinks=False):
                for entry in dirs + files:
                    os.chown(entry, self.uid, self.uid, dir_fd=fd, follow_symlinks=False)
        if control is not None:
            control = Path(control)
            control.chmod(0o755)
            for item in control.iterdir():
                if stat.S_ISSOCK(item.lstat().st_mode):
                    os.chown(item, 0, self.uid)
                    item.chmod(0o660)

    def stop(self, process):
        with self.lock:
            if self.stopped:
                return
            if process is not None:
                if process.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            # A detached grandchild can outlive the original process group.
            for _ in range(40):
                found = processes(self.uid)
                if not found:
                    break
                for pid in found:
                    with contextlib.suppress(ProcessLookupError, FileNotFoundError):
                        lines = Path(f'/proc/{pid}/status').read_text().splitlines()
                        ids = next(line for line in lines if line.startswith('Uid:')).split()[1:]
                        # PR_SET_DUMPABLE=0 may make the proc directory root-owned.
                        if ids == [str(self.uid)] * 4:
                            os.kill(pid, signal.SIGKILL)
                time.sleep(0.05)
            else:
                raise RuntimeError('Sandbox processes remain after cleanup')
            for name in WRITABLE:
                top = self.stage / name
                if top.is_symlink():
                    raise ValueError('Writable root became a symlink')
                os.chown(top, 0, 0)
                top.chmod(0o700)
                for _, dirs, files, fd in os.fwalk(top, follow_symlinks=False):
                    for entry in dirs + files:
                        info = os.stat(entry, dir_fd=fd, follow_symlinks=False)
                        os.chown(entry, 0, 0, dir_fd=fd, follow_symlinks=False)
                        if stat.S_ISDIR(info.st_mode):
                            os.chmod(entry, 0o700, dir_fd=fd)
                        elif stat.S_ISREG(info.st_mode):
                            os.chmod(entry, (stat.S_IMODE(info.st_mode) & 0o777) | 0o600, dir_fd=fd)
            self.reservation.unlink(missing_ok=True)
            self.stopped = True
            with _LIVE_LOCK:
                _LIVE.discard(self)

    def launch(self, argv, control, *, deadline=None, **kwargs):
        if _INTERRUPTED.is_set():
            raise RuntimeError('运行已中断，不再启动新的沙箱')
        with _LIVE_LOCK:
            _LIVE.add(self)
        fds = []
        process = None
        try:
            self.prepare(control)
            block_r, block_w = os.pipe()
            fds += [block_r, block_w]
            info_r, info_w = os.pipe()
            fds += [info_r, info_w]
            inherited = [block_r, info_w]
            transformed = []
            i = 0
            while i < len(argv):
                option = argv[i]
                if option == '--':
                    transformed += ['--cap-add', 'CAP_SETUID', '--cap-add', 'CAP_SETGID',
                                    '--cap-add', 'CAP_SETPCAP', '--userns-block-fd', str(block_r),
                                    '--info-fd', str(info_w), *argv[i:]]
                    break
                if option in ('--ro-bind', '--bind'):
                    fd = os.open(argv[i + 1], os.O_PATH | os.O_NOFOLLOW)
                    fds.append(fd)
                    inherited.append(fd)
                    transformed += [option + '-fd', str(fd), argv[i + 2]]
                    i += 3
                else:
                    transformed.append(option)
                    i += 1
            process = subprocess.Popen(transformed, pass_fds=inherited, extra_groups=[], **kwargs)
            for fd in (block_r, info_w):
                os.close(fd)
                fds.remove(fd)
            data = b''
            deadline = min(deadline, time.monotonic() + 15) if deadline is not None else time.monotonic() + 15
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([info_r], [], [], remaining)[0]:
                    raise RuntimeError('Timed out waiting for bwrap user namespace')
                chunk = os.read(info_r, 8192)
                if not chunk:
                    break
                data += chunk
                if len(data) > 65536:
                    raise RuntimeError('Invalid bwrap namespace information')
            if not data:
                process.wait(timeout=5)
                detail = process.stderr.read(2000).decode(errors='replace')
                raise RuntimeError('bwrap could not create its user namespace: ' + detail)
            pid = json.loads(data)['child-pid']
            Path(f'/proc/{pid}/uid_map').write_text(f'0 0 1\n1000 {self.uid} 1\n')
            Path(f'/proc/{pid}/setgroups').write_text('deny\n')
            Path(f'/proc/{pid}/gid_map').write_text(f'0 0 1\n1000 {self.uid} 1\n')
            if _INTERRUPTED.is_set():
                # An interrupt arrived while this sandbox was starting; kill_all may have missed it.
                raise RuntimeError('运行已中断，不再启动新的沙箱')
            return process
        except BaseException:
            self.stop(process)
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            raise
        finally:
            # Closing block_w releases bwrap only AFTER both maps are ready.
            for fd in fds:
                os.close(fd)
