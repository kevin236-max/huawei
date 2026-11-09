"""Research only: render pi's captured requests the way vLLM 0.26 does and compare with recorded usage.

For every assistant turn k of a trajectory:
  * prompt check: tokens of the rendered request (add_generation_prompt) vs usage.input + usage.cacheRead.
  * output check: tokens the next request's re-rendering assigns to assistant k vs usage.output.
  * prefix check: whether request k+1's prompt tokens start with request k's prompt tokens plus
    assistant k's re-rendered tokens, i.e. whether one sequence can stand for the whole trajectory.

Usage: python render_check.py <trajectory.jsonl> <requests.jsonl> <model dir> [--effort medium]
"""
import argparse
import json

from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def vllm_conversation(request):
    """Mirror vLLM chat_utils parsing for text-only messages (reasoning rename, JSON-parsed tool args)."""
    conversation = []
    for message in request.messages:
        role = message['role']
        content = message.get('content')
        if content is None:
            text = ''
        elif isinstance(content, str):
            text = content
        else:
            text = '\n'.join(part['text'] for part in content if part.get('type') == 'text')
        item = {'role': role, 'content': text}
        if role == 'assistant':
            if message.get('tool_calls'):
                calls = []
                for call in message['tool_calls']:
                    function = dict(call['function'])
                    function['arguments'] = json.loads(function['arguments']) if function.get('arguments') else {}
                    calls.append({**call, 'function': function})
                item['tool_calls'] = calls
            if message.get('reasoning') is not None:
                item['reasoning'] = item['reasoning_content'] = message['reasoning']
        elif role == 'tool':
            item['tool_call_id'] = message.get('tool_call_id')
        conversation.append(item)
    return conversation


def render(tokenizer, body, effort, generation_prompt=True):
    request = ChatCompletionRequest(**body)
    tools = [tool.model_dump() for tool in request.tools] if request.tools else None
    kwargs = {'add_generation_prompt': generation_prompt}
    if effort is not None:
        kwargs.update(reasoning_effort=effort, enable_thinking=effort != 'none')
    text = tokenizer.apply_chat_template(vllm_conversation(request), tools=tools, tokenize=False, **kwargs)
    return text, tokenizer(text, add_special_tokens=False)['input_ids']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trajectory')
    parser.add_argument('requests')
    parser.add_argument('model')
    parser.add_argument('--effort', default='medium', help="reasoning_effort sent by the gateway; 'unset' to omit")
    args = parser.parse_args()
    effort = None if args.effort == 'unset' else args.effort
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [json.loads(line)['message'] for line in open(args.trajectory)]
    requests = [json.loads(line) for line in open(args.requests)]

    rendered = [render(tokenizer, r['body'], effort) for r in requests]
    first_text = rendered[0][0]
    print('system block of first request:')
    print(first_text[:first_text.index('<|im_end|>')][:400].replace('\n', '\\n'), '...\n')

    stats = {'prompt_ok': 0, 'output_ok': 0, 'prefix_ok': 0, 'turns': len(requests)}
    for n, (req, (text, ids)) in enumerate(zip(requests, rendered)):
        usage = messages[req['index']]['usage']
        recorded_prompt = usage['input'] + usage['cacheRead']
        line = f"turn {req['index']:3d} prompt rendered={len(ids):6d} recorded={recorded_prompt:6d} diff={len(ids) - recorded_prompt:+5d}"
        stats['prompt_ok'] += len(ids) == recorded_prompt
        if n + 1 < len(rendered):
            next_text, next_ids = rendered[n + 1]
            # Assistant k as re-rendered inside request k+1: from the end of prompt k to its <|im_end|>.
            if next_text.startswith(text):
                end = next_text.index('<|im_end|>', len(text)) + len('<|im_end|>')
                output_ids = tokenizer(text + next_text[len(text):end], add_special_tokens=False)['input_ids'][len(ids):]
                line += f" | output rerendered={len(output_ids):5d} recorded={usage['output']:5d} diff={len(output_ids) - usage['output']:+4d}"
                stats['output_ok'] += len(output_ids) == usage['output']
                token_prefix = next_ids[:len(ids)] == ids
                line += f" | token prefix {'ok' if token_prefix else 'BROKEN'}"
                stats['prefix_ok'] += token_prefix
            else:
                at = next((i for i, (a, b) in enumerate(zip(text, next_text)) if a != b), min(len(text), len(next_text)))
                line += f" | TEXT PREFIX BROKEN at char {at}: {text[at-60:at+60]!r} vs {next_text[at-60:at+60]!r}"
        print(line)
    print(stats)


if __name__ == '__main__':
    main()
