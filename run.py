#!/usr/bin/env python3
"""Run independent Pi method searches on monthly runoff tasks."""
import argparse
import json
import os
from pathlib import Path
import time

import yaml
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parent


def load_config(path):
    config=yaml.safe_load(Path(path).read_text())
    config['task']=str((ROOT/config['task']).resolve())
    values={**dotenv_values(ROOT/'.env'),**os.environ}
    api=config['api']
    for key,variable in [('base_url','OPENAI_BASE_URL'),('model','OPENAI_MODEL'),('reasoning_effort','OPENAI_REASONING_EFFORT')]:
        api[key]=api.get(key) or values.get(variable)
    api['api_key']=values.get('OPENAI_API_KEY','EMPTY')
    for key,lower,upper in [('rounds',1,3),('workers',1,3),('cpus_per_worker',12,12)]:
        if type(config[key]) is not int or not lower<=config[key]<=upper:
            raise ValueError(f'{key} 须在 {lower} 至 {upper} 之间')
    config.setdefault('worker_finish_timeout',180)
    for key in ('worker_timeout','worker_finish_timeout','search_timeout','candidate_timeout'):
        if config[key]<=0: raise ValueError(f'{key} 须为正数')
    config['worker_timeout']=min(config['worker_timeout'],480)
    config['search_timeout']=min(config['search_timeout'],3600)
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
    if len(os.sched_getaffinity(0))<config['workers']*12:
        raise RuntimeError('CPU 数量不足以为每个 subagent 分配 12 核')
    with tempfile.TemporaryDirectory(prefix='runoff-doctor-',dir=ROOT) as temp:
        stage=Path(temp)/'probe'
        public=Path(temp)/'input';public.mkdir()
        p,identity=sandbox.start(config,stage,public,sorted(os.sched_getaffinity(0))[:12],
            'candidate',['/opt/forecasting.py','--help'],deadline=time.monotonic()+15)
        try:
            output,error=p.communicate(timeout=15)
            if p.returncode: raise RuntimeError(error.decode(errors='replace'))
        finally: sandbox.stop(p,identity)
    return {'status':'ready'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','verify-data','doctor','evaluate'],nargs='?',default='run')
    parser.add_argument('--config',default=str(ROOT/'config.yaml'))
    parser.add_argument('--task',help='Task directory, with benchmark.yaml/public/private')
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--repeat',type=int,help='Number of independent searches in one batch (default: 1)')
    args=parser.parse_args()
    if args.repeat is not None and (args.command!='run' or args.repeat<1):
        parser.error('--repeat 须为正整数，且仅用于 run')
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
    if args.command=='evaluate':
        from harness.evaluate import evaluate
        from datetime import datetime
        if not args.candidate: parser.error('evaluate requires --candidate')
        result=evaluate(config,load_task(config['task'],config['horizon']),args.candidate.resolve(),
            ROOT/'runs'/('evaluate_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')),
            sorted(os.sched_getaffinity(0))[:12],time.monotonic()+config['candidate_timeout'])
    else:
        if not config['api'].get('base_url') or not config['api'].get('model'):
            raise ValueError('请在 .env 或 config.yaml 配置模型地址和名称')
        from harness.run import execute_batch
        result=execute_batch(config,repeat=args.repeat if args.repeat is not None else 1)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.command=='run':
        return 0 if result['completed']==result['repeat'] else 1
    return 0 if result.get('status','complete')=='complete' else 1

if __name__=='__main__':
    raise SystemExit(main())
