"""One host-owned model endpoint per Pi session; real credentials stay here."""
import base64
import contextlib
import http.client
import http.server
import json
import os
from pathlib import Path
import socket
import socketserver
import ssl
import threading
import time
from urllib.parse import unquote, urlsplit
from urllib.request import getproxies_environment, proxy_bypass_environment


def start_gateway(path, config, deadline):
    """Serve OpenAI chat completions on a Unix socket; call returned server.stop()."""
    path = Path(path)
    url = urlsplit(config['base_url'])
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.query or url.fragment:
        raise ValueError('base_url must be an HTTP(S) endpoint')
    proxies = getproxies_environment()
    proxy_url = None if proxy_bypass_environment(url.netloc, proxies) else proxies.get(url.scheme)
    proxy = urlsplit(proxy_url) if proxy_url else None
    if proxy and (proxy.scheme != 'http' or not proxy.hostname or proxy.path not in ('', '/')):
        raise ValueError('Expected an HTTP proxy URL')
    host, port = (proxy.hostname, proxy.port or 80) if proxy else (url.hostname, url.port)
    proxy_headers = {}
    if proxy and proxy.username is not None:
        credential = f'{unquote(proxy.username)}:{unquote(proxy.password or "")}'
        proxy_headers['Proxy-Authorization'] = 'Basic ' + base64.b64encode(credential.encode()).decode()
    target = url.path.rstrip('/')
    if not target.endswith('/chat/completions'):
        target += '/chat/completions'
    if proxy and url.scheme == 'http':
        target = f'{url.scheme}://{url.netloc}{target}'
    headers = dict(config.get('headers', {}))
    if config.get('api_key'):
        headers['Authorization'] = 'Bearer ' + config['api_key']
    headers.update({'Content-Type':'application/json', 'Accept':'text/event-stream'})
    if proxy and url.scheme == 'http':
        headers.update(proxy_headers)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def do_POST(self):
            self.close_connection = True
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.server.stopped.is_set():
                self.send_error(403, 'Session ended')
                return
            self.connection.settimeout(min(remaining, 30))
            lengths = self.headers.get_all('Content-Length', [])
            if (self.path != '/v1/chat/completions' or len(lengths) != 1
                    or not lengths[0].isascii() or not lengths[0].isdigit() or len(lengths[0]) > 9
                    or self.headers.get('Transfer-Encoding') or self.headers.get('Content-Encoding')):
                self.send_error(400, 'Unsupported request')
                return
            length = int(lengths[0])
            if length > 32 * 1024**2:
                self.send_error(413, 'Request too large')
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError('Incomplete body')
                body = json.loads(raw)
                allowed = {'model', 'messages', 'tools', 'tool_choice', 'parallel_tool_calls',
                           'stream', 'stream_options', 'max_tokens', 'max_completion_tokens',
                           'temperature', 'top_p', 'stop', 'reasoning_effort', 'thinking', 'service_tier'}
                if (not isinstance(body, dict) or set(body) - allowed or body.get('model') != config['model']
                        or not isinstance(body.get('messages'), list)):
                    raise ValueError('Invalid model request')
                for key in ('max_tokens', 'max_completion_tokens'):
                    if key in body and (type(body[key]) is not int or not 1 <= body[key] <= config['max_tokens']):
                        raise ValueError('Token limit exceeded')
                if not any(key in body for key in ('max_tokens', 'max_completion_tokens')):
                    body['max_tokens'] = config['max_tokens']
                effort = config.get('reasoning_effort')
                if effort and effort != 'off':
                    body['reasoning_effort'] = effort
                else:
                    body.pop('reasoning_effort', None)
                body.pop('thinking', None)
                payload = json.dumps(body, allow_nan=False).encode()
            except (ValueError, TypeError, RecursionError, OSError):
                self.send_error(400, 'Invalid model request')
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.server.stopped.is_set():
                self.send_error(403, 'Session ended')
                return
            timeout = min(remaining, config.get('api_timeout', 120))
            if url.scheme == 'https':
                upstream = http.client.HTTPSConnection(host, port, timeout=timeout,
                            context=ssl.create_default_context(cafile=config.get('ca_file')))
                if proxy:
                    upstream.set_tunnel(url.hostname, url.port or 443, headers=proxy_headers)
            else:
                upstream = http.client.HTTPConnection(host, port, timeout=timeout)
            started = False
            upstream_socket = None
            try:
                upstream.connect()
                with self.server.lock:
                    if self.server.stopped.is_set() or time.monotonic() >= deadline:
                        return
                    upstream_socket = upstream.sock
                    self.server.connections.add(upstream_socket)
                upstream.request('POST', target, payload, headers)
                response = upstream.getresponse()
                if response.status != 200:
                    self.send_error(502, f'Upstream returned HTTP {response.status}')
                    return
                self.send_response(200)
                self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
                self.send_header('Connection', 'close')
                self.end_headers()
                started = True
                size = 0
                while time.monotonic() < deadline and not self.server.stopped.is_set():
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 64 * 1024**2:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (OSError, http.client.HTTPException):
                if not started:
                    with contextlib.suppress(OSError):
                        self.send_error(502, 'Upstream connection failed')
            finally:
                with self.server.lock:
                    self.server.connections.discard(upstream_socket)
                upstream.close()

    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True

        def __init__(self):
            self.stopped = threading.Event()
            self.lock = threading.Lock()
            self.connections = set()
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                super().__init__(f'/proc/self/fd/{fd}/{path.name}', Handler)
            finally:
                os.close(fd)

        def process_request_thread(self, request, address):
            with self.lock:
                self.connections.add(request)
            try:
                super().process_request_thread(request, address)
            finally:
                with self.lock:
                    self.connections.discard(request)

        def stop(self):
            self.stopped.set()
            with self.lock:
                for connection in self.connections:
                    with contextlib.suppress(OSError):
                        connection.shutdown(socket.SHUT_RDWR)
            self.shutdown()
            self.server_close()
            path.unlink(missing_ok=True)

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    server = Server()
    threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05}, daemon=True).start()
    return server
