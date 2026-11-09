#!/usr/bin/env python3
"""Run tasks x repeats method searches concurrently, each pinned to its own CPU group.

Every job is one `run.py run --run-dir ...` process. A job needs workers * cpus_per_worker
CPUs from the config; `parallel` jobs run at once on the first parallel * that many CPUs
of this process's affinity, and further jobs wait for a free group. Defaults come from the
config's `batch` section; command-line options override them.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.run import summarize  # noqa: E402

DEFAULT_CONFIG = ROOT / 'config/config_qwen3.8-27B.yaml'


def resolve_tasks(names):
    """Task names under data/ or explicit directories; default is every data/* with benchmark.yaml."""
    if not names:
        return sorted(p for p in (ROOT / 'data').iterdir() if (p / 'benchmark.yaml').is_file())
    tasks = []
    for name in names:
        path = Path(name)
        path = path if path.is_absolute() or path.exists() else ROOT / 'data' / name
        if not (path / 'benchmark.yaml').is_file():
            raise SystemExit(f'找不到任务 {name}：{path} 下没有 benchmark.yaml')
        tasks.append(path.resolve())
    if len({t.name for t in tasks}) != len(tasks):
        raise SystemExit('任务目录名不能重复')
    return tasks


def positive(value, name, allow_zero=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 or (value == 0 and not allow_zero):
        raise SystemExit(f'{name} 须为{"非负数" if allow_zero else "正数"}，当前为 {value!r}')
    return value


def read_result(job):
    try:
        return json.loads((job['dir'] / 'result.json').read_text())
    except (OSError, ValueError):
        return None


def write_summaries(output, tasks, jobs, repeat):
    overall = {}
    for task in tasks:
        results = [r for j in jobs if j['task'] == task.name and (r := read_result(j))]
        overall[task.name] = summarize(results, repeat)
        (output / task.name).mkdir(parents=True, exist_ok=True)
        (output / task.name / 'summary.json').write_text(json.dumps(overall[task.name], indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(overall, ensure_ascii=False, indent=2) + '\n')
    return overall


def git_state():
    try:
        commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain'], capture_output=True, text=True).stdout.strip())
        return {'commit': commit or None, 'dirty': dirty}
    except OSError:
        return {'commit': None, 'dirty': None}


def child_setup(cpus):
    """Pin the job and restore SIGINT, which nohup/& launches ignore, so run.py can clean up."""
    def setup():
        os.sched_setaffinity(0, cpus)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    return setup


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--tasks', help='Comma-separated task names under data/ or directories')
    parser.add_argument('--repeat', type=int, help='Searches per task')
    parser.add_argument('--parallel', type=int, help='Jobs running at the same time')
    parser.add_argument('--task-stagger', type=float, help='Seconds between the first jobs of consecutive tasks')
    parser.add_argument('--job-stagger', type=float, help='Seconds between jobs of the same task')
    parser.add_argument('--name', default='parallel', help='Prefix of the experiment directory under runs/')
    parser.add_argument('--output', type=Path, help='Experiment directory (required with --resume)')
    parser.add_argument('--resume', action='store_true', help='Skip completed jobs in --output and rerun the rest')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan without starting anything')
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    batch = config.get('batch') or {}
    pick = lambda value, key, default: value if value is not None else batch.get(key, default)
    task_names = args.tasks.split(',') if args.tasks else batch.get('tasks')
    tasks = resolve_tasks([t.strip() for t in task_names if t.strip()] if task_names else None)
    repeat = int(positive(pick(args.repeat, 'repeat', 1), 'repeat'))
    parallel = int(positive(pick(args.parallel, 'parallel', len(tasks) * repeat), 'parallel'))
    task_stagger = positive(pick(args.task_stagger, 'task_stagger', 2), 'task_stagger', allow_zero=True)
    job_stagger = positive(pick(args.job_stagger, 'job_stagger', 0), 'job_stagger', allow_zero=True)
    workers, per_worker = config['workers'], config['cpus_per_worker']
    per_job = int(positive(workers, 'workers')) * int(positive(per_worker, 'cpus_per_worker'))
    prompts = config['prompts']
    for key in ('main', 'subagent', 'subagent_finish'):
        if not (ROOT / prompts[key]).is_file():
            raise SystemExit(f'找不到提示词文件 prompts.{key}：{ROOT / prompts[key]}')

    total = len(tasks) * repeat
    parallel = min(parallel, total)
    available = sorted(os.sched_getaffinity(0))
    need = parallel * per_job
    if len(available) < need:
        raise SystemExit(f'需要 {need} 个 CPU（并发 {parallel} × {workers} 个 subagent × {per_worker} 核），'
                         f'当前进程只能使用 {len(available)} 个')
    groups = [available[i * per_job:(i + 1) * per_job] for i in range(parallel)]

    if args.resume:
        if args.output is None or not args.output.is_dir():
            raise SystemExit('--resume 需要用 --output 指定已有的实验目录')
        output = args.output.resolve()
    else:
        output = (args.output or ROOT / 'runs' / f'{args.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}').resolve()
        if output.exists() and not args.dry_run:
            raise SystemExit(f'实验目录已存在：{output}')

    jobs = [dict(task=t.name, task_dir=t, number=n, dir=output / t.name / f'repeat_{n:03d}',
                 log=output / 'logs' / f'{t.name}_r{n:03d}.log')
            for t in tasks for n in range(1, repeat + 1)]
    pending = []
    for job in jobs:
        result = read_result(job) if args.resume else None
        if result and result.get('status') == 'complete':
            job['status'] = 'skipped'
            continue
        pending.append(job)

    print(f'Experiment: {output}')
    print(f'Config: {config_path}')
    print(f'Main prompt: {prompts["main"]}')
    print(f'Tasks: {", ".join(t.name for t in tasks)}; repeat {repeat}; {len(pending)} of {total} jobs to run')
    print(f'Parallel {parallel}; {per_job} CPUs per job ({workers} subagents x {per_worker}); '
          f'CPUs {groups[0][0]}..{groups[-1][-1]} of {len(available)} available')
    print(f'Stagger: {task_stagger}s between tasks, {job_stagger}s within a task')
    if args.dry_run:
        for slot, job in enumerate(pending[:parallel]):
            print(f'  slot {slot:2d} CPUs {groups[slot][0]}-{groups[slot][-1]}: {job["task"]} repeat_{job["number"]:03d}')
        if len(pending) > parallel:
            print(f'  {len(pending) - parallel} more jobs wait for a free slot')
        return 0
    if not pending:
        print('Nothing to run.')
        write_summaries(output, tasks, jobs, repeat)
        return 0

    (output / 'logs').mkdir(parents=True, exist_ok=True)
    for job in pending:
        if job['dir'].exists():
            # An unfinished earlier attempt is kept aside; run.py refuses to reuse a directory.
            job['dir'].rename(job['dir'].with_name(f'{job["dir"].name}.stale-{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
    manifest = dict(started=datetime.now().astimezone().isoformat(timespec='seconds'), argv=sys.argv,
                    config_path=str(config_path), config=config_path.read_text(), git=git_state(),
                    tasks=[str(t) for t in tasks], repeat=repeat, parallel=parallel,
                    cpus_per_worker=per_worker, workers=workers, cpus_per_job=per_job,
                    task_stagger=task_stagger, job_stagger=job_stagger, resume=args.resume,
                    jobs=[])
    manifest_path = output / ('manifest.json' if not args.resume else
                              f'manifest.resume-{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    def save_manifest():
        manifest['jobs'] = [{k: (str(v) if isinstance(v, Path) else v) for k, v in j.items()
                             if k not in ('process', 'log_file')}
                            for j in jobs]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')

    free = list(range(parallel))
    running = []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)

    def reap():
        for job in list(running):
            code = job['process'].poll()
            if code is None:
                continue
            running.remove(job)
            free.append(job['slot'])
            job['log_file'].close()
            result = read_result(job) or {}
            status = result.get('status')
            if status in (None, 'running'):
                # run.py never wrote a final status: it was killed (second Ctrl-C) or crashed.
                status = 'killed' if code < 0 else 'failed'
            job.update(returncode=code, status=status,
                       finished=datetime.now().astimezone().isoformat(timespec='seconds'))
            print(f'[done] {job["task"]} repeat_{job["number"]:03d}: {job["status"]} '
                  f'(validation {result.get("validation_score")}, test {result.get("test_score")})', flush=True)
            write_summaries(output, tasks, jobs, repeat)
            save_manifest()

    previous_task = None
    try:
        for index, job in enumerate(pending):
            if index:
                time.sleep(task_stagger if job['task'] != previous_task else job_stagger)
            while not free and not stopping:
                reap()
                time.sleep(1)
            if stopping:
                break
            previous_task = job['task']
            slot = free.pop(0)
            cpus = groups[slot]
            job['dir'].parent.mkdir(parents=True, exist_ok=True)
            job['log_file'] = job['log'].open('w')
            command = [sys.executable, str(ROOT / 'run.py'), 'run', '--config', str(config_path),
                       '--task', str(job['task_dir']), '--run-dir', str(job['dir'])]
            job['process'] = subprocess.Popen(command, cwd=ROOT, stdout=job['log_file'], stderr=subprocess.STDOUT,
                                              start_new_session=True, preexec_fn=child_setup(cpus))
            job.update(slot=slot, cpus=f'{cpus[0]}-{cpus[-1]}', status='running',
                       started=datetime.now().astimezone().isoformat(timespec='seconds'))
            running.append(job)
            print(f'[start] {job["task"]} repeat_{job["number"]:03d} on CPUs {job["cpus"]}', flush=True)
            save_manifest()
        while running and not stopping:
            reap()
            time.sleep(2)
    except KeyboardInterrupt:
        stopping = True
    if stopping:
        print('Stopping: interrupting running jobs (press Ctrl-C again to kill them immediately)...', flush=True)
        for job in running:
            try:
                os.killpg(job['process'].pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        # Each run.py kills its sandboxes at once on SIGINT and then records `interrupted`;
        # a second Ctrl-C (or the time limit) skips the wait and kills the jobs outright.
        try:
            deadline = time.monotonic() + 120
            while running and time.monotonic() < deadline:
                reap()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print('Killing running jobs now.', flush=True)
        for job in list(running):
            try:
                os.killpg(job['process'].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            job['process'].wait()
        # Everything is stopped; finish the short summary writing without further interrupts.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        reap()
    for job in jobs:
        job.setdefault('status', 'not started')
    overall = write_summaries(output, tasks, jobs, repeat)
    save_manifest()
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    complete = all(j['status'] in ('complete', 'skipped') for j in jobs)
    return 0 if complete and not stopping else 1


if __name__ == '__main__':
    raise SystemExit(main())
