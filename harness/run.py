"""Three rounds of independent method workspaces and ordinary file handoffs."""
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness.evaluate import read_candidate


def within(root, name):
    name=Path(name)
    if name.is_absolute() and name.is_relative_to('/workspace'):
        name=name.relative_to('/workspace')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('文件必须位于当前工作区内')
    return path


def copy_references(main, workspace, files):
    references = []
    for name in files:
        source = within(main, name)
        relative = Path('references') / source.relative_to(main.resolve())
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        references.append(str(relative))
    return references


def submissions(root):
    result = []
    for file in sorted(root.glob('evaluations/*/result.json'), key=lambda p: int(p.parent.name)):
        record = json.loads(file.read_text())
        if 'score' in record:
            record['file'] = str(file.parent / record['file'])
            result.append(record)
    return result


def recover(worker, status):
    records = submissions(worker)
    report = dict(file=None, validation_score=None, best_params=None,
                  internal_score=None, summary='超时，没有得到验证结果' if status=='timeout' else '没有得到验证结果')
    if records:
        best = max(records, key=lambda r: r['score'])
        report.update(file=best['file'], validation_score=best['score'], best_params=best['best_params'],
                      summary='超时，返回已提交的最佳验证结果' if status=='timeout' else '返回已提交的最佳验证结果')
    else:
        files = [p for p in (worker/'workspace/solution').glob('*.py') if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to((worker/'workspace').resolve())]
        if files:
            latest=max(files, key=lambda p: p.stat().st_mtime_ns)
            if latest.stat().st_size<=2*1024**2:
                report['file'] = str(latest)
    return report


def collect(worker, main, round_number, name, report):
    folder = main / 'results' / f'round{round_number}' / name
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(report)
    if report.get('file'):
        source = Path(report['file'])
        target = folder/source.name
        target.write_bytes(read_candidate(source))
        report['file'] = str(target.relative_to(main))
    (folder/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    # Main keeps its sandbox UID for the entire search. Newly published host files
    # must be readable through its live read-only workspace mount, even at umask 077.
    for directory in (main/'results', folder.parent, folder):
        directory.chmod(0o755)
    for path in folder.iterdir():
        path.chmod(0o644)
    return report


def choose_final(run, selection):
    records = [r for stage in sorted((run/'workers').glob('round*/*')) for r in submissions(stage)]
    if not records:
        return None
    if selection and selection.get('file'):
        main = run/'main/workspace'
        try:
            selected_path=Path(selection['file'])
            selected = within(main, selected_path)
            # Compare the actual small source file, not a digest or mutable path label.
            code = read_candidate(selected)
            matching = [r for r in records if Path(r['file']).read_bytes()==code
                        and (selection.get('best_params') is None or r['best_params']==selection['best_params'])]
            if matching:
                return max(matching,key=lambda r:r['score'])
        except (OSError, ValueError):
            pass
    return max(records,key=lambda r:r['score'])


def run_rounds(decide, worker, *, rounds, workers, deadline):
    """Count worker batches; after the last batch permit one selection-only turn."""
    results = []
    selection = None
    for number in range(1, rounds+1):
        if time.monotonic() >= deadline:
            break
        plan = decide(number, results, False)
        if plan.get('file'):
            selection = plan
            break
        tasks = plan.get('tasks', [])
        if not tasks or time.monotonic() >= deadline:
            break
        if len(tasks)>workers:
            raise ValueError(f'每轮最多 {workers} 个任务')
        names=[t['name'] for t in tasks]
        if len(set(names))!=len(names) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',n) for n in names):
            raise ValueError('任务名称须为不重复的英文方法名')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(worker,t,number,slot,deadline) for slot,t in enumerate(tasks)]
            results.extend(f.result() for f in futures)
    else:
        if time.monotonic()<deadline:
            final=decide(rounds,results,True)
            selection=final if final.get('file') else None
    return results,selection


def report_result(worker, response):
    """Prefer a worker's chosen evaluated file; never attach stale scores to a draft."""
    from harness.pi import final_json
    if response['status']!='finished':
        return recover(worker,response['status'])
    try:
        report=final_json(response['text'])
        if report.get('file') is None:
            result=recover(worker,'finished')
            if report.get('summary'):
                result['summary']=str(report['summary'])[:1500]
            return result
        relative=Path(report['file'])
        if relative.suffix!='.py': raise ValueError('交付文件须为 Python 候选')
        if relative.is_absolute(): relative=relative.relative_to('/workspace')
        source=within(worker/'submissions',Path(*relative.parts[1:])) if relative.parts[0]=='submissions' else within(worker/'workspace',relative)
        code=read_candidate(source)
        records=[r for r in submissions(worker) if Path(r['file']).read_bytes()==code
                 and (report.get('best_params') is None or report['best_params']==r['best_params'])]
        result=dict(file=str(source),validation_score=None,best_params=None,internal_score=None,
                    summary=str(report.get('summary',''))[:1500])
        if records:
            best=max(records,key=lambda r:r['score'])
            result.update(file=best['file'],validation_score=best['score'],best_params=best['best_params'],
                          internal_score=report.get('internal_score'))
        return result
    except (ValueError,KeyError,TypeError,OSError):
        return recover(worker,'failed')


def execute(config):
    import os
    from datetime import datetime
    from string import Template
    from harness.forecasting import load_task, public_settings, task_context
    from harness.pi import PiSession, run_session, final_json
    from harness.evaluate import start_server, evaluate

    root=Path(__file__).resolve().parent.parent
    task=load_task(config['task'],config['horizon'])
    available=sorted(os.sched_getaffinity(0))
    count=config['workers']*config['cpus_per_worker']
    if len(available)<count:
        raise ValueError(f'需要 {count} 个可用 CPU，实际只有 {len(available)}')
    run=root/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    main=run/'main/workspace'
    main.mkdir(parents=True)
    public=run/'input'
    public.mkdir()
    task['train'].to_csv(public/'train.csv',index=False)
    (public/'task.json').write_text(json.dumps(public_settings(task)))
    public.chmod(0o755)
    for path in public.iterdir(): path.chmod(0o644)
    # Private API keys are never included in run files or prompts.
    saved={k:v for k,v in config.items() if k!='api'}
    saved['api']={k:v for k,v in config['api'].items() if k!='api_key'}
    (run/'config.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2))
    print(f'Run: {run}',flush=True)
    deadline=time.monotonic()+config['search_timeout']
    context=task_context(task)
    main_template=Template((root/'prompts/main.md').read_text())
    worker_template=Template((root/'prompts/subagent.md').read_text())
    finish_prompt=(root/'prompts/subagent_finish.md').read_text()
    delivered=0
    main_session=PiSession(config,run/'main',public,available[:config['cpus_per_worker']],deadline,role='main')

    def decide(number,results,final_only):
        nonlocal delivered
        instruction=('请综合各轮实验结果，提交最终结果：候选文件路径和对应参数。' if final_only else
                     f'请制定第 {number} 轮 subagent 的实验计划，每批最多 {config["workers"]} 个任务。')
        if number == 1 and not final_only:
            prompt=main_template.substitute(task_context=context,round_instruction=instruction)
        else:
            # Append only new worker results to the existing conversation. The final
            # instruction intentionally follows the results and exactly two newlines.
            prompt=json.dumps({'results':results[delivered:]},ensure_ascii=False)+'\n\n'+instruction
        delivered=len(results)
        response=main_session.prompt(prompt)
        if response['status']!='finished': return {'tasks':[]}
        try:
            plan=final_json(response['text'])
            if not isinstance(plan,dict): raise ValueError('主 agent 输出应为 JSON 对象')
            if not final_only and not plan.get('file'):
                tasks=plan.get('tasks',[])
                if (not isinstance(tasks,list) or len(tasks)>config['workers']
                        or any(not isinstance(t,dict) or not isinstance(t.get('name'),str)
                               or not isinstance(t.get('instruction'),str)
                               or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',t['name']) for t in tasks)
                        or len({t['name'] for t in tasks})!=len(tasks)):
                    raise ValueError('主 agent 任务列表无效')
        except ValueError:
            return {'tasks':[]}
        return plan

    def worker(spec,number,slot,global_deadline):
        stage=run/'workers'/f'round{number}'/spec['name']
        workspace=stage/'workspace'
        (workspace/'solution').mkdir(parents=True)
        soft_deadline=min(global_deadline,time.monotonic()+config['worker_timeout'])
        worker_deadline=min(global_deadline,soft_deadline+config.get('worker_finish_timeout',600))
        cpus=available[slot*config['cpus_per_worker']:(slot+1)*config['cpus_per_worker']]
        server=None
        try:
            references=copy_references(main,workspace,spec.get('files',[]))
            # Supply the concise result next to each referenced method automatically.
            for file in spec.get('files',[]):
                sibling=str(Path(file).with_name('result.json'))
                if sibling not in spec.get('files',[]) and sibling not in [str(Path(p).relative_to('references')) for p in references] and within(main,sibling).is_file():
                    references+=copy_references(main,workspace,[sibling])
            prompt=worker_template.substitute(task_context=context,instruction=spec['instruction'],
                        references='\n'.join('- '+p for p in dict.fromkeys(references)) or '- 本次没有参考文件。')
            server=start_server(config,task,stage,cpus,worker_deadline)
            response=run_session(config,stage,public,cpus,prompt,worker_deadline,
                                 soft_deadline=soft_deadline,finish_prompt=finish_prompt)
        except (ValueError,KeyError,OSError,RuntimeError) as error:
            (stage/'error.log').write_text(f'{type(error).__name__}: {error}\n')
            response={'text':'','status':'failed'}
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
        report=report_result(stage,response)
        return collect(stage,main,number,spec['name'],report)

    with main_session:
        results,selection=run_rounds(decide,worker,rounds=config['rounds'],workers=config['workers'],deadline=deadline)
    best=choose_final(run,selection)
    result={'status':'no_valid_candidate','results':results}
    if best:
        final=run/'final'
        try:
            tested=evaluate(config,task,Path(best['file']),final,available[:config['cpus_per_worker']],
                            time.monotonic()+config['candidate_timeout'],split='test',params=best['best_params'])
            result.update(status='complete',file=str((final/tested['file']).relative_to(run)),
                          validation_score=best['score'],best_params=best['best_params'],test_score=tested['score'])
        except (OSError,ValueError,RuntimeError,TimeoutError) as error:
            result.update(status='test_failed',file=str(Path(best['file']).relative_to(run)),
                          validation_score=best['score'],best_params=best['best_params'],error=str(error))
    (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
