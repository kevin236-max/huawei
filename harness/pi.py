"""Pi RPC sessions with persistent context, message logs, and one search deadline."""
import json
import os
from pathlib import Path
import selectors
import time

from harness import gateway, sandbox


def json_problem(text):
    """Short hint for the outermost JSON-looking span, used in retry messages."""
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        return '回复中没有 JSON 对象'
    try:
        json.loads(text[start:end+1])
    except ValueError as error:
        detail = getattr(error, 'msg', str(error))
        return f'JSON 格式有误（{detail}），可能是括号或引号不完整'
    return 'JSON 中缺少约定的字段'


def final_json(text, keys=None):
    """Read the last complete JSON object, including fenced/prose-wrapped output.

    With ``keys``, only top-level dicts containing one of them qualify, so a nested
    object left over from a truncated outer object is not mistaken for the answer.
    """
    decoder = json.JSONDecoder()
    objects = []
    position = 0
    while position < len(text):
        start = text.find('{', position)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            position = start + 1
            continue
        objects.append(value)
        position = end
    if keys is not None:
        objects = [value for value in objects if isinstance(value, dict) and any(key in value for key in keys)]
    if not objects:
        raise ValueError(json_problem(text))
    return objects[-1]


def needs_continuation(result):
    """A settled turn with no answer text, e.g. the output limit was spent on thinking."""
    if result['status'] != 'finished' or result['text'].strip():
        return None
    assistant = [m for m in result['messages'] if isinstance(m, dict) and m.get('role') == 'assistant']
    if not assistant:
        return None
    if assistant[-1].get('stopReason') == 'length':
        return '上一条回复达到了单次输出的长度上限，还没有形成结果。请接着当前进展继续。'
    return '上一条回复没有包含结果。请接着当前进展继续，完成后按约定格式给出结果。'


def model_config(api):
    model = api['model']
    return {'providers': {'session': {
        'baseUrl':'http://127.0.0.1:18080/v1', 'api':'openai-completions', 'apiKey':'sandbox',
        'models':[{'id':model, 'name':model, 'input':['text'],
                   'reasoning':bool(api.get('reasoning_effort') and api['reasoning_effort'] != 'off'),
                   'contextWindow':api['context_window'], 'maxTokens':api['max_tokens'],
                   'thinkingLevelMap':{key:key for key in ('minimal','low','medium','high','xhigh','max')},
                   'compat':{'supportsStore':False, 'supportsDeveloperRole':False,
                             'maxTokensField':'max_completion_tokens' if 'gpt-' in model.lower() else 'max_tokens',
                             'thinkingFormat':'openai', 'supportsReasoningEffort':True,
                             'requiresReasoningContentOnAssistantMessages':'deepseek' in model.lower()}}]}}}


# One-line role descriptions; task rules and data stay in the user prompts.
SYSTEM_PROMPTS = {
    'main': '你负责规划月度径流预测的方法搜索：分批向并行的 subagent 下达实验任务，根据返回结果安排后续实验，'
            '最后选出最终候选；你只读文件、不写代码，每次回复以一个 JSON 对象结束。',
    'subagent': '你负责按给定的实验指令，在 /workspace 中实现单文件 Python 月度径流预测候选：用内部验证调试和比较参数，'
                '用 bench 完成外部验证，按指令执行完调参步骤后，用一个 JSON 对象报告最佳已评估结果。',
}


class PiSession:
    """One Pi process across sequential prompts; never restart a lost context."""

    def __init__(self, config, stage, public, cpus, deadline, role='subagent'):
        self.config, self.stage = config, Path(stage)
        self.public, self.cpus, self.deadline, self.role = public, cpus, deadline, role
        self.process = self.identity = self.server = None
        self.closed = False
        self.finish_sent = False
        self.buffer = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _start(self):
        api = self.config.get('api', self.config)
        home = self.stage / 'home/.pi'
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        (home / 'models.json').write_text(json.dumps(model_config(api), indent=2))
        (home / 'settings.json').write_text(json.dumps({
            'retry': {'enabled': False}, 'compaction': {'enabled': self.role != 'main'}}))
        command = [self.config['python'], '-I', '-B', '/opt/relay.py', '/opt/node/bin/node',
                   '/opt/pi/packages/coding-agent/dist/bundle/cli.js', '--mode', 'rpc', '--no-session', '--offline',
                   '--provider', 'session', '--model', api['model'], '--tools',
                   'read,ls' if self.role == 'main' else 'read,bash,edit,write',
                   '--no-extensions', '--no-mcp', '--no-skills', '--no-prompt-templates',
                   '--no-context-files', '--no-approve', '--thinking', api.get('reasoning_effort') or 'off',
                   '--system-prompt', SYSTEM_PROMPTS['main' if self.role == 'main' else 'subagent']]
        self.server = gateway.start_gateway(self.stage / 'control/api.sock', api, self.deadline)
        self.process, self.identity = sandbox.start(
            self.config, self.stage, self.public, self.cpus, 'main' if self.role == 'main' else 'agent',
            command, self.stage / 'control', deadline=self.deadline)

    def prompt(self, prompt, *, soft_deadline=None, finish_prompt=None, continuations=1):
        """Send one prompt; if the turn settles without answer text, nudge it to continue."""
        result = self._send(prompt, soft_deadline, finish_prompt)
        for _ in range(continuations):
            message = needs_continuation(result)
            if message is None:
                break
            later = self._send(message, soft_deadline, finish_prompt)
            later['messages'] = result['messages'] + later['messages']
            result = later
        return result

    def _send(self, prompt, soft_deadline, finish_prompt):
        if self.closed:
            return {'text': '', 'status': 'failed', 'messages': []}
        self.stage.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            if self.process is None:
                (self.stage / 'prompt.txt').write_text(prompt)
            with (self.stage / 'prompts.jsonl').open('a') as stream:
                stream.write(json.dumps({'type': 'prompt', 'message': prompt}, ensure_ascii=False)+'\n')
            if time.monotonic() >= self.deadline:
                result = {'text': '', 'status': 'timeout', 'messages': []}
            else:
                if self.process is None:
                    self._start()
                # The wrap-up steer is sent at most once per session, across follow-up prompts.
                result = consume_rpc(self.process, prompt, self.deadline, self.stage / 'trajectory.jsonl',
                                     self.stage / 'stderr.log', buffer=self.buffer, soft_deadline=soft_deadline,
                                     finish_prompt=None if self.finish_sent else finish_prompt)
                self.finish_sent = self.finish_sent or result.get('finish_sent', False)
        except Exception as error:
            with (self.stage / 'stderr.log').open('a') as stream:
                stream.write(f'Runtime error: {type(error).__name__}: {error}\n')
            result = {'text': '', 'status': 'timeout' if time.monotonic() >= self.deadline else 'failed', 'messages': []}
        if result['status'] != 'finished':
            self.close()
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process is not None:
                sandbox.stop(self.process, self.identity)
        finally:
            if self.server is not None:
                self.server.stop()


def run_session(config, stage, public, cpus, prompt, deadline, role='subagent', *,
                soft_deadline=None, finish_prompt=None, check=None):
    """Run one independent worker session, closing it after its response.

    ``check(text)`` may return a short follow-up (e.g. the report JSON did not parse);
    it is sent once in the same session. A failed follow-up keeps the earlier answer.
    """
    with PiSession(config, stage, public, cpus, deadline, role) as session:
        result = session.prompt(prompt, soft_deadline=soft_deadline, finish_prompt=finish_prompt)
        message = check(result['text']) if check and result['status'] == 'finished' else None
        if message:
            later = session.prompt(message, soft_deadline=soft_deadline, finish_prompt=finish_prompt)
            if later['status'] == 'finished' and later['text'].strip():
                later['messages'] = result['messages'] + later['messages']
                result = later
        return result


def consume_rpc(process, prompt, deadline, trajectory, stderr_path, *, buffer=None,
                soft_deadline=None, finish_prompt=None):
    """Drain stdin/stdout/stderr without blocking; only agent_settled completes RPC."""
    messages = []
    final_text = ''
    failed = False
    status = 'failed'
    finish_sent = False
    buffer = bytearray() if buffer is None else buffer
    pending = memoryview((json.dumps({'type':'prompt', 'message':prompt}, ensure_ascii=False)+'\n').encode())
    selector = selectors.DefaultSelector()
    with Path(trajectory).open('a') as log, Path(stderr_path).open('ab') as errors:
        for stream in (process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE)

        def record(line):
            nonlocal final_text, failed
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                return False
            if not isinstance(event, dict):
                return False
            if event.get('type') == 'message_end' and isinstance(event.get('message'), dict):
                message = event['message']
                log.write(json.dumps(event, ensure_ascii=False)+'\n')
                log.flush()
                messages.append(message)
                if message.get('role') == 'assistant':
                    if message.get('stopReason') in ('error', 'aborted') or message.get('errorMessage'):
                        failed = True
                    content = message.get('content', [])
                    if isinstance(content, list):
                        text = '\n'.join(block['text'] for block in content
                                         if isinstance(block, dict) and block.get('type') == 'text'
                                         and isinstance(block.get('text'), str))
                        if not any(isinstance(block, dict) and block.get('type') == 'toolCall' for block in content):
                            final_text = text
            if event.get('type') == 'response' and event.get('success') is False:
                failed = True
                return True
            return event.get('type') == 'agent_settled'

        try:
            settled = False
            while not settled:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = 'timeout'
                    break
                if not selector.get_map():
                    break
                if (not finish_sent and soft_deadline is not None and finish_prompt
                        and time.monotonic() >= soft_deadline):
                    # Steering is queued by Pi until the current response/tools finish.
                    # Keep the session, gateway and evaluator alive through the hard deadline.
                    command = {'id':'worker_finish', 'type':'steer', 'message':finish_prompt}
                    pending = memoryview(bytes(pending) + (json.dumps(command, ensure_ascii=False)+'\n').encode())
                    if process.stdin.fileno() not in selector.get_map():
                        selector.register(process.stdin, selectors.EVENT_WRITE)
                    with (Path(trajectory).parent/'prompts.jsonl').open('a') as prompts:
                        prompts.write(json.dumps(command, ensure_ascii=False)+'\n')
                    finish_sent = True
                for key, _ in selector.select(min(remaining, .1)):
                    stream = key.fileobj
                    if stream is process.stdin:
                        try:
                            size = os.write(stream.fileno(), pending[:65536])
                            pending = pending[size:]
                        except (BrokenPipeError, OSError):
                            selector.unregister(stream)
                            failed = True
                            continue
                        if not pending:
                            selector.unregister(stream)
                        continue
                    try:
                        chunk = os.read(stream.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        if stream is process.stdout and buffer:
                            settled = record(bytes(buffer)) or settled
                            buffer.clear()
                        continue
                    if stream is process.stderr:
                        errors.write(chunk)
                        errors.flush()
                    else:
                        buffer.extend(chunk)
                        while b'\n' in buffer:
                            line, _, tail = buffer.partition(b'\n')
                            buffer[:] = tail
                            settled = record(line) or settled
                        if len(buffer) > 64 * 1024**2:
                            failed = True
                            settled = True
                if settled:
                    status = 'failed' if failed else 'finished'
                elif process.poll() is not None and not any(key.fileobj is process.stdout for key in selector.get_map().values()):
                    break
        finally:
            selector.close()
    return {'text':final_text, 'status':status, 'messages':messages, 'finish_sent':finish_sent}
