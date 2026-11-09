"""Three rounds of independent method workspaces and ordinary file handoffs."""
import json
import re
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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


REPORT_KEYS = ('file', 'summary', 'validation_score', 'best_params')


def last_progress(worker, limit=600):
    """Tail of the worker's last assistant text, for handoffs without a usable report."""
    try:
        lines = (worker/'trajectory.jsonl').read_text().splitlines()
    except OSError:
        return ''
    for line in reversed(lines):
        try:
            message = json.loads(line)['message']
            if message.get('role') != 'assistant':
                continue
            text = '\n'.join(block['text'] for block in message.get('content', [])
                             if isinstance(block, dict) and block.get('type') == 'text'
                             and isinstance(block.get('text'), str)).strip()
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        if text:
            return text[-limit:]
    return ''


def recover(worker, status, note=''):
    records = submissions(worker)
    prefix = '超时，' if status == 'timeout' else ''
    report = dict(file=None, validation_score=None, best_params=None,
                  internal_score=None, summary=prefix+'没有得到验证结果')
    if records:
        best = max(records, key=lambda r: r['score'])
        report.update(file=best['file'], validation_score=best['score'], best_params=best['best_params'],
                      internal_score=best.get('internal_score'), summary=prefix+'返回已提交的最佳验证结果')
    else:
        files = [p for p in (worker/'workspace/solution').glob('*.py') if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to((worker/'workspace').resolve())]
        if files:
            latest=max(files, key=lambda p: p.stat().st_mtime_ns)
            if latest.stat().st_size<=2*1024**2:
                report['file'] = str(latest)
    if note:
        # Without this the next plan only sees a fixed phrase, not what went wrong.
        report['summary'] = (report['summary'] + '。最后的说明：' + note)[:1500]
    return report


def collect(worker, main, round_number, name, report):
    folder = main / 'results' / f'round{round_number}' / name
    folder.mkdir(parents=True, exist_ok=True)
    # Name first: a result without a file must still be attributable to its task.
    report = {'name': name, **{k: v for k, v in report.items() if k != 'name'}}
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
                return {**max(matching,key=lambda r:r['score']),'selected_by':'main'}
        except (OSError, ValueError):
            pass
    return {**max(records,key=lambda r:r['score']),'selected_by':'fallback'}


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


def reported_source(worker, name):
    relative=Path(name)
    if relative.suffix!='.py': raise ValueError('交付文件须为 Python 候选')
    if relative.is_absolute(): relative=relative.relative_to('/workspace')
    if relative.parts[0]=='submissions':
        return within(worker/'submissions',Path(*relative.parts[1:]))
    return within(worker/'workspace',relative)


def report_result(worker, response):
    """Prefer a worker's chosen evaluated file; never attach stale scores to a draft.

    Scores, parameters and the internal score always come from host records.
    """
    from harness.pi import final_json
    if response['status']!='finished':
        return recover(worker,response['status'],last_progress(worker))
    text=response.get('text') or ''
    try:
        report=final_json(text,REPORT_KEYS)
    except ValueError:
        # A malformed report still explains the work; keep its text for the next plan.
        return recover(worker,'finished',('最终报告未能解析，原文：'+text.strip()[-1200:]) if text.strip() else last_progress(worker))
    summary=str(report.get('summary') or '')[:1500]
    if report.get('file') is None:
        result=recover(worker,'finished')
        if summary: result['summary']=summary
        return result
    try:
        source=reported_source(worker,report['file'])
        code=read_candidate(source)
    except (ValueError,TypeError,OSError,IndexError):
        source=code=None
    records=submissions(worker)
    matching=[r for r in records if code is not None and Path(r['file']).read_bytes()==code
              and (report.get('best_params') is None or report['best_params']==r['best_params'])]
    if matching:
        best=max(matching,key=lambda r:r['score'])
        return dict(file=best['file'],validation_score=best['score'],best_params=best['best_params'],
                    internal_score=best.get('internal_score'),summary=summary)
    if records:
        # The reported file or parameters match no evaluation; hand back the best evaluated version.
        result=recover(worker,'finished')
        result['summary']=(f'报告的 {report["file"]} 及参数与已评测版本不一致，这里给出该实验外部验证最高的版本。'+summary)[:1500]
        return result
    if source is not None:
        return dict(file=str(source),validation_score=None,best_params=None,internal_score=None,summary=summary)
    result=recover(worker,'finished')
    if summary: result['summary']=summary
    return result


def summarize(results, repeat):
    completed = [r for r in results if r['status'] == 'complete']
    summary = {'repeat': repeat, 'completed': len(completed)}
    for name in ('validation', 'test'):
        scores = [r[name+'_score'] for r in completed]
        summary[name+'_mean'] = statistics.mean(scores) if scores else None
        summary[name+'_std'] = statistics.stdev(scores) if len(scores) > 1 else None
    return summary


def execute_batch(config, repeat=1):
    if type(repeat) is not int or repeat < 1:
        raise ValueError('repeat 须为正整数')
    root = Path(__file__).resolve().parent.parent
    batch = root/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S')
    batch.mkdir(parents=True)  # Refuse a same-second collision instead of mixing batches.
    print(f'Batch: {batch}', flush=True)
    results = []
    summary = summarize(results, repeat)
    (batch/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for number in range(1, repeat+1):
        results.append(execute(config, batch/f'repeat_{number:03d}'))
        summary = summarize(results, repeat)
        (batch/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def execute(config, run):
    start = datetime.now().astimezone().isoformat(timespec='seconds')
    started = time.monotonic()
    run = Path(run)
    run.mkdir(parents=True)
    result = {'status': 'running', 'start': start, 'finish': None, 'duration': None}
    (run/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    try:
        result = _execute(config, run)
    except KeyboardInterrupt:
        result = {'status': 'interrupted'}
        raise
    except Exception as error:
        result = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
    finally:
        result.update(start=start, finish=datetime.now().astimezone().isoformat(timespec='seconds'),
                      duration=round(time.monotonic()-started, 3))
        (run/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def _execute(config, run):
    import os
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
    # Prompt files come from the config (paths relative to the project); see run.py load_config.
    prompts=config['prompts']
    main_template=Template((root/prompts['main']).read_text())
    worker_template=Template((root/prompts['subagent']).read_text())
    finish_prompt=(root/prompts['subagent_finish']).read_text()
    delivered=0
    main_session=PiSession(config,run/'main',public,available[:config['cpus_per_worker']],deadline,role='main')

    def parse_plan(text,final_only):
        """Validate main's JSON; the ValueError message is shown to main for one retry."""
        plan=final_json(text,('file',) if final_only else ('tasks','file'))
        if final_only or plan.get('file'):
            if not isinstance(plan.get('file'),str): raise ValueError('需要给出 file（候选文件路径）和 best_params')
            return plan
        tasks=plan.get('tasks')
        if not isinstance(tasks,list): raise ValueError('tasks 应为任务列表')
        if len(tasks)>config['workers']: raise ValueError(f'每轮最多 {config["workers"]} 个任务')
        for t in tasks:
            if not isinstance(t,dict) or not isinstance(t.get('name'),str) or not isinstance(t.get('instruction'),str):
                raise ValueError('每个任务需要 name 和 instruction 两个字符串字段')
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',t['name']):
                raise ValueError(f'任务名称 {t["name"]!r} 只能使用英文字母、数字、下划线和连字符')
            if not isinstance(t.get('files',[]),list) or not all(isinstance(f,str) for f in t.get('files',[])):
                raise ValueError('files 应为文件路径列表')
        if len({t['name'] for t in tasks})!=len(tasks): raise ValueError('同一轮的任务名称不能重复')
        return plan

    def decide(number,results,final_only):
        nonlocal delivered
        if final_only:
            instruction='请综合各轮实验结果，提交最终结果：候选文件路径和对应参数。'
        else:
            instruction=f'请制定第 {number} 轮（共 {config["rounds"]} 轮）subagent 的实验计划，每批最多 {config["workers"]} 个任务。'
            if number==config['rounds']:
                instruction+='这是最后一批实验，结果返回后将请你提交最终选择。'
        if number == 1 and not final_only:
            prompt=main_template.substitute(task_context=context,round_instruction=instruction)
        else:
            # Append only new worker results to the existing conversation. The final
            # instruction intentionally follows the results and exactly two newlines.
            prompt=json.dumps({'results':results[delivered:]},ensure_ascii=False)+'\n\n'+instruction
        delivered=len(results)
        response=main_session.prompt(prompt)
        for retry in (True,False):
            if response['status']!='finished': return {'tasks':[]}
            try:
                return parse_plan(response['text'],final_only)
            except ValueError as error:
                if not retry: return {'tasks':[]}
                # One plain follow-up in the same conversation; the plan itself stays main's.
                kept='选择' if final_only else '方案内容'
                response=main_session.prompt(f'上一条回复中的 JSON 没有读取成功：{error}。'
                                             f'{kept}不需要改动，请按约定格式重新输出完整的 JSON 对象。')

    def check_report(text):
        try:
            final_json(text,REPORT_KEYS)
        except ValueError as error:
            return (f'最终报告的 JSON 没有读取成功：{error}。'
                    '内容不需要修改，请按约定字段重新输出这个 JSON 对象。')
        return None

    def available_reference(name):
        try:
            return within(main,name).is_file()
        except ValueError:
            return False

    def worker(spec,number,slot,global_deadline):
        stage=run/'workers'/f'round{number}'/spec['name']
        workspace=stage/'workspace'
        (workspace/'solution').mkdir(parents=True)
        soft_deadline=min(global_deadline,time.monotonic()+config['worker_timeout'])
        worker_deadline=min(global_deadline,soft_deadline+config['worker_finish_timeout'])
        cpus=available[slot*config['cpus_per_worker']:(slot+1)*config['cpus_per_worker']]
        server=None
        setup_error=None
        try:
            # A mistyped reference path should not cost the whole experiment slot.
            files=[f for f in spec.get('files',[]) if available_reference(f)]
            skipped=[f for f in spec.get('files',[]) if f not in files]
            references=copy_references(main,workspace,files)
            # Supply the concise result next to each referenced method automatically.
            for file in files:
                sibling=str(Path(file).with_name('result.json'))
                if sibling not in files and sibling not in [str(Path(p).relative_to('references')) for p in references] and within(main,sibling).is_file():
                    references+=copy_references(main,workspace,[sibling])
            lines=['- '+p for p in dict.fromkeys(references)]+[f'- {f}（未找到该文件，已跳过）' for f in skipped]
            prompt=worker_template.substitute(task_context=context,instruction=spec['instruction'],
                        references='\n'.join(lines) or '- 本次没有参考文件。')
            server=start_server(config,task,stage,cpus,worker_deadline)
            response=run_session(config,stage,public,cpus,prompt,worker_deadline,
                                 soft_deadline=soft_deadline,finish_prompt=finish_prompt,check=check_report)
        except (ValueError,KeyError,OSError,RuntimeError) as error:
            setup_error=f'{type(error).__name__}: {error}'
            (stage/'error.log').write_text(setup_error+'\n')
            response={'text':'','status':'failed'}
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
        report=report_result(stage,response)
        if setup_error and not last_progress(stage):
            report['summary']=f'{report["summary"]}（实验未能启动：{setup_error}）'[:1500]
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
                          validation_score=best['score'],best_params=best['best_params'],test_score=tested['score'],
                          selected_by=best['selected_by'])
        except (OSError,ValueError,RuntimeError,TimeoutError) as error:
            result.update(status='test_failed',file=str(Path(best['file']).relative_to(run)),
                          validation_score=best['score'],best_params=best['best_params'],
                          selected_by=best['selected_by'],error=str(error))
    return result
