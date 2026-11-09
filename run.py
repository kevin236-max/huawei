#!/usr/bin/env python3
"""Run independent Pi method searches on monthly runoff tasks."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import yaml
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parent
DEFAULT_CONFIG=ROOT/'config/config_qwen3.8-27B.yaml'
# Chat Completions backends selectable through LLM_BACKEND; each reads <PREFIX>_* from .env.
BACKENDS=('openai','novita')


def load_api(api, values):
    """Resolve the selected backend; config file values override the backend's .env entries."""
    name=api.get('backend') or values.get('LLM_BACKEND') or 'openai'
    if name not in BACKENDS:
        raise ValueError(f'LLM_BACKEND 须为 {"、".join(BACKENDS)} 之一，当前为 {name}')
    prefix=name.upper()
    api['backend']=name
    for key,suffix in [('base_url','BASE_URL'),('model','MODEL'),('reasoning_effort','REASONING_EFFORT')]:
        api[key]=api.get(key) or values.get(f'{prefix}_{suffix}')
    # A local vLLM without authentication accepts the EMPTY placeholder; hosted backends need a key.
    api['api_key']=values.get(f'{prefix}_API_KEY') or ('EMPTY' if name=='openai' else None)
    return api


def load_config(path):
    config=yaml.safe_load(Path(path).read_text())
    config['task']=str((ROOT/config['task']).resolve())
    values={**dotenv_values(ROOT/'.env'),**os.environ}
    load_api(config['api'],values)
    for key,lower,upper in [('rounds',1,3),('workers',1,3)]:
        if type(config[key]) is not int or not lower<=config[key]<=upper:
            raise ValueError(f'{key} 须在 {lower} 至 {upper} 之间')
    if type(config['cpus_per_worker']) is not int or config['cpus_per_worker']<1:
        raise ValueError('cpus_per_worker 须为正整数')
    if type(config['native_threads']) is not int or config['native_threads']<1:
        raise ValueError('native_threads 须为正整数')
    for key in ('worker_timeout','worker_finish_timeout','search_timeout','candidate_timeout'):
        if config[key]<=0: raise ValueError(f'{key} 须为正数')
    for key in ('max_tokens','context_window','api_timeout'):
        if config['api'][key]<=0: raise ValueError(f'api.{key} 须为正数')
    for key in ('main','subagent','subagent_finish'):
        if not (ROOT/config['prompts'][key]).is_file():
            raise ValueError(f'找不到提示词文件 prompts.{key}：{ROOT/config["prompts"][key]}')
    runtime=(ROOT/values.get('PI_RUNTIME_ROOT','.runtime'))/'runtime.json'
    if runtime.exists(): config.update(json.loads(runtime.read_text()))
    return config


def doctor(config):
    from harness.shared_proc import require_root
    from harness import sandbox
    import tempfile
    require_root()
    for key in ('node','pi_root','bwrap'):
        if key not in config: raise RuntimeError('请先运行 bash install.sh')
    per_worker=config['cpus_per_worker']
    if len(os.sched_getaffinity(0))<config['workers']*per_worker:
        raise RuntimeError(f'CPU 数量不足以为每个 subagent 分配 {per_worker} 核')
    with tempfile.TemporaryDirectory(prefix='runoff-doctor-',dir=ROOT) as temp:
        stage=Path(temp)/'probe'
        public=Path(temp)/'input';public.mkdir()
        p,identity=sandbox.start(config,stage,public,sorted(os.sched_getaffinity(0))[:per_worker],
            'candidate',['/opt/forecasting.py','--help'],deadline=time.monotonic()+15)
        try:
            output,error=p.communicate(timeout=15)
            if p.returncode: raise RuntimeError(error.decode(errors='replace'))
        finally: sandbox.stop(p,identity)
    return {'status':'ready'}


def install_interrupt_handler():
    """Ctrl-C/SIGTERM: kill every sandbox at once, then unwind so results record `interrupted`.

    Sandboxes live in their own sessions and UIDs, so without this they would keep running
    while worker threads wait for them. Later signals only repeat the kill and let cleanup finish.
    """
    from harness.shared_proc import kill_all
    state = {'raised': False}

    def handler(signum, frame):
        kill_all()
        if not state['raised']:
            state['raised'] = True
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','verify-data','doctor','evaluate'],nargs='?',default='run')
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--task',help='Task directory, with benchmark.yaml/public/private')
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--repeat',type=int,help='Number of independent searches in one batch (default: 1)')
    parser.add_argument('--run-dir',type=Path,help='Run exactly one search into this new directory (no batch wrapper)')
    args=parser.parse_args()
    if args.repeat is not None and (args.command!='run' or args.repeat<1):
        parser.error('--repeat 须为正整数，且仅用于 run')
    if args.run_dir is not None and (args.command!='run' or args.repeat not in (None,1)):
        parser.error('--run-dir 仅用于 run，且每次只执行一次搜索')
    config=load_config(args.config)
    if args.task: config['task']=str(Path(args.task).resolve())
    from harness.forecasting import load_task,task_context
    if args.command=='verify-data':
        print(task_context(load_task(config['task'],config['horizon'])))
        return 0
    if args.command=='doctor':
        print(json.dumps(doctor(config)));return 0
    for key in ('node','pi_root','bwrap'):
        if key not in config: raise RuntimeError('请先运行 bash install.sh')
    install_interrupt_handler()
    try:
        result=launch(args,config,parser)
    except KeyboardInterrupt:
        print('Interrupted: sandboxes killed; partial results keep status "interrupted".',file=sys.stderr,flush=True)
        return 130
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.command=='run' and args.run_dir is None:
        return 0 if result['completed']==result['repeat'] else 1
    return 0 if result.get('status','complete')=='complete' else 1


def launch(args, config, parser):
    from harness.forecasting import load_task
    if args.command=='evaluate':
        from harness.evaluate import evaluate
        from datetime import datetime
        if not args.candidate: parser.error('evaluate requires --candidate')
        return evaluate(config,load_task(config['task'],config['horizon']),args.candidate.resolve(),
            ROOT/'runs'/('evaluate_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')),
            sorted(os.sched_getaffinity(0))[:config['cpus_per_worker']],time.monotonic()+config['candidate_timeout'])
    api=config['api']
    prefix=api['backend'].upper()
    if not api.get('base_url') or not api.get('model'):
        raise ValueError(f'请在 .env 配置 {prefix}_BASE_URL 和 {prefix}_MODEL（当前 LLM_BACKEND={api["backend"]}）')
    if not api.get('api_key'):
        raise ValueError(f'请在 .env 配置 {prefix}_API_KEY（当前 LLM_BACKEND={api["backend"]}）')
    if args.run_dir is not None:
        from harness.run import execute
        return execute(config,args.run_dir.resolve())
    from harness.run import execute_batch
    return execute_batch(config,repeat=args.repeat if args.repeat is not None else 1)

if __name__=='__main__':
    raise SystemExit(main())
