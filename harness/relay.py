"""In-sandbox loopback HTTP to the one host-owned Unix API socket."""
import contextlib
import socket
import socketserver
import subprocess
import sys
import threading


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.socket(socket.AF_UNIX) as upstream:
            upstream.connect('/control/api.sock')

            def copy(source, target):
                try:
                    while chunk := source.recv(65536):
                        target.sendall(chunk)
                except OSError:
                    pass
                finally:
                    with contextlib.suppress(OSError):
                        target.shutdown(socket.SHUT_WR)

            thread = threading.Thread(target=copy, args=(self.request, upstream), daemon=True)
            thread.start()
            copy(upstream, self.request)
            thread.join()


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == '__main__':
    with Server(('127.0.0.1', 18080), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            code = subprocess.call(sys.argv[1:])
        finally:
            server.shutdown()
    raise SystemExit(code)
