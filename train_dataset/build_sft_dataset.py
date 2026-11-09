"""Convert pi agent trajectories (runs/<run>/**/trajectory.jsonl) into a torchtitan
qwen35_kling_omni SFT LMDB.

One trajectory.jsonl (main or worker) -> one training sample. Each file is one linear
conversation (only `message_end` events, no compaction), and the Qwen3.8 chat template
keeps every turn's thinking (preserve_thinking defaults to true), so every assistant turn
is a prefix-consistent continuation and all of them are supervised in one sample.

Message reconstruction mirrors what pi actually sent to the model
(packages/pi/packages/ai/src/api/openai-completions.ts + transform-messages.ts):
  - system  = content + sections joined by "\\n\\n"; tools = system.toolsAdded
  - assistant with stopReason error/aborted is dropped
  - assistant: text blocks (non-blank) joined "", thinking joined "\\n" -> reasoning_content,
    toolCall -> tool_calls; an assistant with neither text nor tool calls is dropped
  - toolResult -> role "tool", text joined "\\n", "(no tool output)" when empty
  - tool calls without a result get a synthetic "No result provided" result
The conversation is rendered with the vanilla Qwen3.8-27B chat template
(reasoning_effort from the assistant's thinkingLevel).

Loss mask: the framework (Qwen3DataCollator.create_label_and_mask_and_position_ids) splits
`text` on <question_answer_split_token> into alternating question/answer segments and drops
the split tokens. Every assistant turn's body, from "<think>" through "<|im_end|>", is wrapped
in a pair of split tokens; everything else (system, user, tool results, "<|im_start|>assistant\\n")
is masked. Span boundaries come from rendering message prefixes, not string search, and are
checked on every sample.

Usage (needs transformers + lmdb, e.g. the shared training env):
  python build_sft_dataset.py --run /kling_test/gen4/tangjiafu/huawei/runs/task10_20261109_030319
"""
import argparse
import json
import os
import pickle
import statistics
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
VANILLA_MODEL = "/kling_test/gen4/tangjiafu/model_list/Qwen3.8-27B"
TRAIN_TOKENIZER = "/kling_test/gen4/tangjiafu/huawei/torchtitan/tjf/model_assets/Qwen3.8-27B"
SPLIT = "<question_answer_split_token>"
ASSISTANT_HEADER = "<|im_start|>assistant\n"
TEMPLATE_EFFORTS = ("xhigh", "medium", "low")

_render_tok = None
_train_tok = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append", required=True,
                    help="pi run dir; repeatable. Every */trajectory.jsonl below it is used.")
    ap.add_argument("--name", default=None, help="dataset / lmdb name (default: tjf_sft_<run dir name>)")
    ap.add_argument("--out-root", type=Path, default=HERE / "datasets")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=128000, help="only used for the stats report")
    args = ap.parse_args()

    runs = [r.resolve() for r in args.run]
    name = args.name or "tjf_sft_" + "_".join(r.name for r in runs)
    out = (args.out_root / name).resolve()
    lmdb_dir = out / "lmdb"
    if lmdb_dir.exists():
        sys.exit(f"{lmdb_dir} already exists; pick another --name or remove it first")
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(str(p) for r in runs for p in r.rglob("trajectory.jsonl"))
    if not files:
        sys.exit("no trajectory.jsonl found")
    print(f"[build] {len(files)} trajectories from {[str(r) for r in runs]}", flush=True)

    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker) as pool:
        results = pool.map(_convert_one, [(f, name, runs) for f in files], chunksize=1)
    print(f"[build] converted in {time.time() - t0:.0f}s", flush=True)

    failed = [r for r in results if "error" in r]
    for r in failed:
        print(f"[build] FAILED {r['source']}: {r['error']}", flush=True)
    skipped = [r for r in results if "skipped" in r]
    for r in skipped:
        print(f"[build] skipped {r['source']}: {r['skipped']}", flush=True)
    samples = [r for r in results if "text" in r]
    if failed:
        sys.exit(f"{len(failed)} trajectories failed; refusing to write a partial dataset")

    _write_jsonl(out / "samples.jsonl", samples)
    _write_lmdb(lmdb_dir, samples)
    (out / "lmdb_config.json").write_text(json.dumps({name: {"lmdb_path": str(lmdb_dir)}}, indent=2) + "\n")
    stats = _stats(samples, files, runs, args.max_seq_len)
    stats["skipped"] = [{"path": r["source"], "reason": r["skipped"]} for r in skipped]
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[build] wrote {out}")


def _init_worker():
    global _render_tok, _train_tok
    from transformers import AutoTokenizer
    _render_tok = AutoTokenizer.from_pretrained(VANILLA_MODEL)
    _train_tok = AutoTokenizer.from_pretrained(TRAIN_TOKENIZER)


def _convert_one(job):
    path, name, runs = job
    rel = _relpath(path, runs)
    try:
        messages, tools, effort, notes = load_conversation(path)
        if messages is None:
            return {"source": rel, "skipped": notes}
        text, n_answers = render_with_splits(messages, tools, effort)
        ids = _train_tok(text, add_special_tokens=False)["input_ids"]
        split_id = _train_tok.convert_tokens_to_ids(SPLIT)
        if ids.count(split_id) != 2 * n_answers:
            raise ValueError(f"split token count {ids.count(split_id)} != 2*{n_answers}")
        plain_ids = _train_tok(text.replace(SPLIT, ""), add_special_tokens=False)["input_ids"]
        if [t for t in ids if t != split_id] != plain_ids:
            raise ValueError("inserting split tokens changed the tokenization")
        answer_tokens, inside = 0, False
        for t in ids:
            if t == split_id:
                inside = not inside
            elif inside:
                answer_tokens += 1
        parts = Path(rel).parts
        return {
            "text": text,
            "num_text_tokens": len(ids),
            "name": name,
            "source_info": {
                "path": rel,
                "role": "main" if "main" in parts else "worker",
                "task": next((p for p in parts if p.startswith("task_")), None),
                "repeat": next((p for p in parts if p.startswith("repeat_")), None),
                "worker": "/".join(parts[parts.index("workers") + 1:-1]) if "workers" in parts else None,
                "assistant_turns": n_answers,
                "answer_tokens": answer_tokens,
                "reasoning_effort": effort,
                "notes": notes,
            },
        }
    except Exception as e:  # reported and fatal in main(); keeps the pool alive
        return {"source": rel, "error": f"{type(e).__name__}: {e}"}


def load_conversation(path):
    """pi trajectory -> (HF messages, tools, reasoning_effort, notes).

    Returns (None, None, None, reason) when the file has no trainable assistant turn.
    """
    events = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    raw = [e["message"] for e in events if e.get("type") == "message_end"]
    notes = Counter()

    systems = [m for m in raw if m["role"] == "system"]
    if len(systems) != 1 or raw[0]["role"] != "system":
        raise ValueError(f"expected exactly one leading system message, got {len(systems)}")
    system = systems[0]
    tools = [{"type": "function",
              "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
             for t in system.get("toolsAdded") or []]
    sys_parts = [_content_text(system.get("content"))] + [v for v in (system.get("sections") or {}).values() if v is not None]
    messages = [{"role": "system", "content": "\n\n".join(p for p in sys_parts if p)}]

    efforts = set()
    pending = []      # tool calls of the last assistant turn
    answered = set()

    def close_pending():
        for tc in pending:
            if tc["id"] not in answered:
                messages.append({"role": "tool", "content": "No result provided"})
                notes["synthetic_tool_result"] += 1
        pending.clear()
        answered.clear()

    for m in raw[1:]:
        role = m["role"]
        if role == "assistant":
            close_pending()
            if m.get("stopReason") in ("error", "aborted"):
                notes["dropped_error_assistant"] += 1
                continue
            content = m["content"]
            text = "".join(b["text"] for b in content if b["type"] == "text" and b["text"].strip())
            thinking = [b["thinking"] for b in content if b["type"] == "thinking" and b["thinking"].strip()]
            calls = [b for b in content if b["type"] == "toolCall"]
            if not text and not calls:
                notes["dropped_empty_assistant"] += 1
                continue
            msg = {"role": "assistant", "content": text}
            if thinking:
                msg["reasoning_content"] = "\n".join(thinking)
            if calls:
                msg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for c in calls]
                pending.extend(calls)
            if m.get("thinkingLevel"):
                efforts.add(m["thinkingLevel"])
            messages.append(msg)
        elif role == "toolResult":
            answered.add(m["toolCallId"])
            text = "\n".join(b["text"] for b in m["content"] if b["type"] == "text")
            if any(b["type"] != "text" for b in m["content"]):
                raise ValueError("non-text tool result content is not supported")
            messages.append({"role": "tool", "content": text if text.strip() else "(no tool output)"})
        elif role == "user":
            close_pending()
            content = m["content"]
            if isinstance(content, list):
                if any(b["type"] != "text" for b in content):
                    raise ValueError("non-text user content is not supported")
                content = "".join(b["text"] for b in content if b["text"])
            if content:
                messages.append({"role": "user", "content": content})
        elif role == "system":
            raise ValueError("mid-conversation system message is not supported")
        else:
            raise ValueError(f"unexpected role {role!r}")
    close_pending()

    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
        notes["dropped_trailing_non_assistant"] += 1
    if not any(m["role"] == "assistant" for m in messages):
        # e.g. the only request failed upstream (502): nothing to train on.
        return None, None, None, f"no usable assistant turn ({dict(notes)})"
    if len(efforts) > 1:
        raise ValueError(f"mixed thinkingLevel {efforts}")
    effort = efforts.pop() if efforts else "medium"
    if effort not in TEMPLATE_EFFORTS:
        raise ValueError(f"thinkingLevel {effort!r} has no chat-template equivalent {TEMPLATE_EFFORTS}")
    return messages, tools, effort, dict(notes)


def render_with_splits(messages, tools, effort):
    """Render with the chat template and wrap each assistant body in split tokens."""
    def render(msgs):
        return _render_tok.apply_chat_template(msgs, tools=tools or None, tokenize=False,
                                               add_generation_prompt=False, reasoning_effort=effort)

    full = render(messages)
    if SPLIT in full:
        raise ValueError("rendered text already contains the split token")
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        pre, post = render(messages[:i]), render(messages[:i + 1])
        if not (post.startswith(pre) and full.startswith(post)):
            raise ValueError(f"template is not prefix-stable at message {i}")
        turn = post[len(pre):]
        if not (turn.startswith(ASSISTANT_HEADER + "<think>") and turn.endswith("<|im_end|>\n")):
            raise ValueError(f"unexpected assistant rendering at message {i}: {turn[:60]!r}...{turn[-30:]!r}")
        spans.append((len(pre) + len(ASSISTANT_HEADER), len(post) - 1))  # body incl. <|im_end|>, excl. "\n"
    pieces, cur = [], 0
    for s, e in spans:
        pieces += [full[cur:s], SPLIT, full[s:e], SPLIT]
        cur = e
    pieces.append(full[cur:])
    return "".join(pieces), len(spans)


def _content_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


def _relpath(path, runs):
    for r in runs:
        try:
            return str(Path(path).relative_to(r.parent))
        except ValueError:
            pass
    return path


def _write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _write_lmdb(lmdb_dir, samples):
    """Same layout as datasets/qwen3_omni/1_jsonl2lmdb.py::write_to_lmdb: "{i}" -> sample, "{i}.meta" -> meta."""
    import lmdb
    lmdb_dir.mkdir(parents=True)
    total = sum(len(pickle.dumps(s)) for s in samples)
    env = lmdb.open(str(lmdb_dir), map_size=max(4 * total, 1 << 30))
    with env.begin(write=True) as txn:
        for i, s in enumerate(samples):
            txn.put(str(i).encode(), pickle.dumps(s))
            txn.put(f"{i}.meta".encode(), pickle.dumps({"name": s["name"], "num_text_tokens": s["num_text_tokens"]}))
    env.sync()
    env.close()


def _stats(samples, files, runs, max_seq_len):
    def dist(xs):
        xs = sorted(xs)
        return {"min": xs[0], "median": int(statistics.median(xs)), "p90": xs[int(0.9 * (len(xs) - 1))],
                "max": xs[-1], "sum": sum(xs)}
    tok = [s["num_text_tokens"] for s in samples]
    ans = [s["source_info"]["answer_tokens"] for s in samples]
    notes = Counter()
    for s in samples:
        notes.update(s["source_info"]["notes"])
    return {
        "runs": [str(r) for r in runs],
        "num_trajectory_files": len(files),
        "num_samples": len(samples),
        "by_role": dict(Counter(s["source_info"]["role"] for s in samples)),
        "by_task": dict(sorted(Counter(s["source_info"]["task"] for s in samples).items())),
        "num_text_tokens": dist(tok),
        "answer_tokens": dist(ans),
        "answer_token_share": round(sum(ans) / sum(tok), 4),
        f"samples_over_{max_seq_len}": sum(t > max_seq_len for t in tok),
        "assistant_turns": dist([s["source_info"]["assistant_turns"] for s in samples]),
        "reasoning_effort": dict(Counter(s["source_info"]["reasoning_effort"] for s in samples)),
        "conversion_notes": dict(notes),
        "tokenizer_for_counts": TRAIN_TOKENIZER,
        "chat_template_from": VANILLA_MODEL,
    }


if __name__ == "__main__":
    main()
