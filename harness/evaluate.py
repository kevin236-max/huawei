"""Evaluate a copied candidate in isolation; the host owns private observations."""
import json
import os
from pathlib import Path
import selectors
import shutil
import stat
import socketserver
import threading
import time

from harness import sandbox
from harness.forecasting import INTERNAL_VALIDATION_MONTHS, windows, score


def read_candidate(source):
    # A FIFO or a file being replaced must not hang the host beyond the task deadline.
    source=Path(source).absolute()
    directory=os.open('/',os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in source.parts[1:-1]:
            child=os.open(part,os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,dir_fd=directory)
            os.close(directory)
            directory=child
        fd=os.open(source.name,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>2*1024**2:
            raise ValueError('候选须为普通 Python 文件，大小不超过 2 MiB')
        code=stream.read(2*1024**2+1)
    if len(code)>2*1024**2: raise ValueError('候选文件过大')
    return code


def copy_candidate(source, destination):
    Path(destination).write_bytes(read_candidate(source))


def split_data(task, split):
    """Return (fit rows, rolling source rows, target bounds) for one scoring split.

    ``internal`` mirrors analysis.calculate_internal_validation_score on the public table.
    """
    if split == 'internal':
        data = task['train']
        months = INTERNAL_VALIDATION_MONTHS
        start, end = data.month.iloc[-months], data.month.iloc[-1]
        return data.iloc[:-months], data, (start, end)
    splits = task['splits']
    fit_end = splits['train_observations' if split == 'validation' else 'final_fit_observations'][1]
    history = task['history']
    return history.loc[history.month <= fit_end], history, splits[split+'_targets']


def evaluate(config, task, candidate, destination, cpus, deadline, split='validation', params=None):
    destination=Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    candidate=Path(candidate)
    saved=destination/candidate.name
    copy_candidate(candidate,saved)
    stage=destination/'sandbox'
    workspace=stage/'workspace'
    workspace.mkdir(parents=True)
    shutil.copyfile(saved,workspace/candidate.name)
    public=stage/'input'
    public.mkdir()
    # No private files mounted: only legal history prefixes cross stdin.
    command=['/opt/forecasting.py','--serve','/workspace/'+candidate.name]
    process,identity=sandbox.start(config,stage,public,cpus,'candidate',command,deadline=deadline)
    selector=selectors.DefaultSelector()
    buffer=bytearray()
    for stream in (process.stdin,process.stdout,process.stderr):
        os.set_blocking(stream.fileno(),False)
    selector.register(process.stdout,selectors.EVENT_READ)
    selector.register(process.stderr,selectors.EVENT_READ)
    with (destination/'stderr.log').open('wb') as errors:
        def exchange(request):
            pending=memoryview((json.dumps(request,allow_nan=False)+'\n').encode())
            selector.register(process.stdin,selectors.EVENT_WRITE)
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise TimeoutError('候选执行超时')
                for key,_ in selector.select(min(remaining,.1)):
                    stream=key.fileobj
                    if stream is process.stdin:
                        pending=pending[os.write(stream.fileno(),pending[:65536]):]
                        if not pending: selector.unregister(stream)
                    else:
                        chunk=os.read(stream.fileno(),65536)
                        if not chunk:
                            selector.unregister(stream)
                            if stream is process.stdout: raise RuntimeError('候选执行失败；请使用公开数据检查代码')
                        elif stream is process.stderr: errors.write(chunk)
                        else:
                            buffer.extend(chunk)
                            if len(buffer)>64*1024**2: raise ValueError('候选输出过大')
                if not pending and b'\n' in buffer:
                    line,_,rest=buffer.partition(b'\n'); buffer[:]=rest
                    return json.loads(line)
        try:
            train, source, bounds = split_data(task, split)
            meta=exchange(dict(train=train.to_dict('records'),horizon=task['horizon'],params=None if params is None else [params]))
            parameters=meta['candidates']
            if not parameters or not all(isinstance(p,dict) for p in parameters):
                raise ValueError('候选参数列表无效')
            if params is not None and parameters!=[params]: raise ValueError('候选未使用选定参数')
            predictions=[]; truths=[]
            for history,targets,truth in windows(source,*bounds,task['horizon']):
                response=exchange(dict(history=history.to_dict('records'),targets=targets))
                values=response['predictions']
                if len(values)!=len(parameters): raise ValueError('预测方案数量不一致')
                predictions.append(values); truths.append(truth)
            scores=score(predictions,truths)
            best=max(range(len(scores)),key=scores.__getitem__)
            result=dict(file=saved.name,score=scores[best],best_params=parameters[best])
            (destination/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            return result
        finally:
            selector.close()
            sandbox.stop(process,identity)


def start_server(config, task, worker, cpus, deadline):
    """Each worker has one sequential evaluator; no shared queue or quota."""
    workspace=worker/'workspace'
    control=worker/'control'
    control.mkdir(parents=True,exist_ok=True)
    evaluations=worker/'evaluations'
    evaluations.mkdir(exist_ok=True)
    visible=worker/'submissions'
    visible.mkdir(exist_ok=True)
    visible.chmod(0o755)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.request.settimeout(5)
            try:
                result=self.submit()
            except Exception as error:
                result={'error': '验证执行失败；请使用公开内部验证检查候选'}
                self.log(error)
            try: self.wfile.write((json.dumps(result,ensure_ascii=False)+'\n').encode())
            except OSError: pass

        def submit(self):
            from harness.run import within
            request=json.loads(self.rfile.readline(65536))
            name=str(request['candidate'])
            # Request problems carry no private information, so report them plainly.
            try:
                relative=Path(name)
                if relative.is_absolute(): relative=relative.relative_to('/workspace')
                source=within(worker/'submissions',Path(*relative.parts[1:])) if relative.parts[0]=='submissions' else within(workspace,relative)
            except (ValueError,IndexError):
                return {'error': f'候选文件须位于 /workspace 内：{name}'}
            if source.suffix!='.py': return {'error': f'请提交 Python 候选文件：{name}'}
            if not source.is_file(): return {'error': f'找不到候选文件：{name}'}
            if time.monotonic()>=deadline: return {'error': '本次任务已结束'}
            folder=evaluations/str(len(list(evaluations.iterdir()))+1)
            try:
                result=evaluate(config,task,source,folder,cpus,
                                min(deadline,time.monotonic()+config['candidate_timeout']))
            except TimeoutError as error:
                self.log(error)
                return {'error': '外部验证执行超时；可以先用内部验证检查运行耗时'}
            except (ValueError,RuntimeError) as error:
                # Host-side contract messages (shape, finiteness, parameter count) are generic.
                self.log(error)
                return {'error': f'外部验证未完成：{error}'}
            result['internal_score']=self.internal(folder/result['file'],folder,result['best_params'])
            (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            target=visible/folder.name
            target.mkdir()
            shutil.copyfile(folder/result['file'],target/result['file'])
            result={**result,'file':str(Path('submissions')/folder.name/result['file'])}
            (target/'result.json').write_text(json.dumps(result,ensure_ascii=False))
            # New host-written files must remain readable to the live sandbox UID.
            target.chmod(0o755)
            for path in target.iterdir(): path.chmod(0o644)
            return result

        def internal(self, saved, folder, params):
            """Host-computed internal score of the reported parameters; never fatal."""
            try:
                return evaluate(config,task,saved,folder/'internal',cpus,
                                min(deadline,time.monotonic()+config['candidate_timeout']),
                                split='internal',params=params)['score']
            except Exception as error:
                self.log(RuntimeError(f'internal score: {type(error).__name__}: {error}'))
                return None

        def log(self, error):
            with (worker/'evaluation_error.log').open('a') as log:
                log.write(f'{type(error).__name__}: {error}\n')

    fd=os.open(control,os.O_RDONLY | os.O_DIRECTORY)
    try:
        server=socketserver.UnixStreamServer(f'/proc/self/fd/{fd}/evaluation.sock',Handler)
    finally:
        os.close(fd)
    threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.1},daemon=True).start()
    return server
