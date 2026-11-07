"""Trusted bootstrap inside bwrap; never import code from the workspace here.

Classic filters adapted from pipeline/sandbox/entry.py. Filesystem authority is
bwrap's mount list; no seccomp NOTIFY supervisor is needed.
"""
import ctypes
import errno
import os
import resource
import runpy
import sys


def install_filter(candidate=False):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0):
        raise RuntimeError('Cannot set no_new_privs')
    sec = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    context = sec.seccomp_init(0x7fff0000)
    if not context:
        raise RuntimeError('Cannot allocate seccomp filter')
    try:
        blocked = ('ptrace', 'process_vm_readv', 'process_vm_writev', 'pidfd_getfd',
                   'mount', 'umount2', 'pivot_root', 'chroot', 'setns', 'unshare',
                   'open_by_handle_at', 'name_to_handle_at', 'open_tree', 'move_mount',
                   'fsopen', 'fsconfig', 'fsmount', 'mount_setattr', 'bpf',
                   'perf_event_open', 'keyctl', 'add_key', 'request_key',
                   'io_uring_setup', 'userfaultfd', 'sched_setaffinity')
        if candidate:
            blocked += ('execve', 'execveat', 'socket', 'socketpair', 'connect', 'bind',
                        'listen', 'accept', 'accept4', 'sendto', 'recvfrom')
        for name in blocked:
            number = sec.seccomp_syscall_resolve_name(name.encode())
            # Optional library probes must fail without killing the interpreter:
            # PyTorch probes CPU affinity; TF dependencies probe IPv6 and uname.
            # These calls remain denied; libraries can handle PermissionError.
            if name in ('sched_setaffinity', 'socket', 'execve'):
                action = 0x50000 | errno.EPERM
            elif name == 'io_uring_setup' and not candidate:
                # Node/libuv falls back when this optional interface is absent.
                action = 0x50000 | errno.ENOSYS
            else:
                action = 0x80000000
            if number >= 0 and sec.seccomp_rule_add(context, action, number, 0):
                raise RuntimeError('Cannot add seccomp rule')

        class Compare(ctypes.Structure):
            _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_int),
                        ('mask', ctypes.c_uint64), ('value', ctypes.c_uint64)]

        sec.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                               ctypes.c_uint, ctypes.POINTER(Compare)]
        clone = sec.seccomp_syscall_resolve_name(b'clone')
        for flag in (0x00020000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000):
            comparison = Compare(0, 7, flag, flag)
            if sec.seccomp_rule_add_array(context, 0x80000000, clone, 1, ctypes.byref(comparison)):
                raise RuntimeError('Cannot block namespace creation')
        number = sec.seccomp_syscall_resolve_name(b'clone3')
        if number >= 0 and sec.seccomp_rule_add(context, 0x50000 | errno.ENOSYS, number, 0):
            raise RuntimeError('Cannot filter clone3')
        if sec.seccomp_load(context):
            raise RuntimeError('Cannot load seccomp policy')
    finally:
        sec.seccomp_release(context)


def main():
    mode = sys.argv[1]
    libc = ctypes.CDLL(None, use_errno=True)
    for cap in range(64):
        if libc.prctl(24, cap, 0, 0, 0) and ctypes.get_errno() != errno.EINVAL:
            raise RuntimeError('Cannot clear capability bounding set')
    os.setresgid(1000, 1000, 1000)
    os.setresuid(1000, 1000, 1000)

    class Header(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]

    class Caps(ctypes.Structure):
        _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]

    if libc.capset(ctypes.byref(Header(0x20080522, 0)), (Caps * 2)()):
        raise RuntimeError('Cannot clear process capabilities')
    if os.getgroups():
        raise RuntimeError('Unexpected supplementary groups')
    for line in open('/proc/self/status'):
        if line.startswith(('CapEff:', 'CapPrm:', 'CapInh:', 'CapAmb:', 'CapBnd:')) and int(line.split()[1], 16):
            raise RuntimeError('Sandbox retained capabilities')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    os.sched_setaffinity(0, [int(cpu) for cpu in sys.argv[2].split(',')])
    os.umask(0o077)
    os.chdir('/workspace')
    for name in os.listdir('/proc/self/fd'):
        if int(name) > 2:
            try:
                os.close(int(name))
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise
    install_filter(candidate=mode == 'candidate')
    if mode == 'candidate':
        sys.path[:0] = ['/opt', '/workspace']
        sys.argv = sys.argv[3:]
        if not sys.argv or not sys.argv[0].endswith('.py'):
            raise ValueError('Candidate command must be a Python script and its arguments')
        runpy.run_path(sys.argv[0], run_name='__main__')
    elif mode == 'agent':
        os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
    else:
        raise ValueError('Invalid sandbox mode')


if __name__ == '__main__':
    main()
