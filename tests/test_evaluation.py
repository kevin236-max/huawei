import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from harness.evaluate import evaluate, copy_candidate

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.candidate=self.root/'seasonal.py'
        self.candidate.write_text('''import numpy as np
candidates=[{"value":5},{"value":10}]
def fit(train_data,num_months,candidates):
    assert train_data.month.iloc[-1]=="2000-03"
    return candidates
def predict(state,history_data,target_months):
    return [np.full((len(target_months),1),p['value']) for p in state]
''')
        data=pd.DataFrame({'month':['2000-01','2000-02','2000-03','2000-04','2000-05','2000-06'],'s':[10.]*6})
        self.task=dict(train=data.iloc[:3],history=data,horizon=2,stations=['s'],splits={'train_observations':['2000-01','2000-03'],'validation_targets':['2000-04','2000-06'],'final_fit_observations':['2000-01','2000-03'],'test_targets':['2000-04','2000-06']})
    def tearDown(self): self.tmp.cleanup()
    def start(self,config,stage,public,cpus,mode,command,control=None,**kwargs):
        path=Path(__file__).resolve().parents[1]/'harness/forecasting.py'
        proc=subprocess.Popen([sys.executable,str(path),'--serve',str(stage/'workspace'/self.candidate.name)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        return proc,None
    def stop(self,p,identity):
        if p.poll() is None: p.kill()
        p.wait()
        for s in (p.stdin,p.stdout,p.stderr): s.close()
    def test_scores_saved_copy_and_final_uses_selected_parameters(self):
        with patch('harness.evaluate.sandbox.start',self.start),patch('harness.evaluate.sandbox.stop',self.stop):
            result=evaluate({},self.task,self.candidate,self.root/'evaluation',[0],time.monotonic()+20)
            self.assertEqual(result['score'],1.)
            self.assertEqual(result['best_params'],{'value':10})
            self.candidate.write_text('changed')
            self.assertIn('def fit',(self.root/'evaluation/seasonal.py').read_text())
            selected=evaluate({},self.task,self.root/'evaluation/seasonal.py',self.root/'final',[0],time.monotonic()+20,split='test',params={'value':5})
            self.assertEqual(selected['score'],.5)
    def test_snapshot_rejects_pipe_and_symlink_without_waiting(self):
        pipe=self.root/'pipe.py'
        os.mkfifo(pipe)
        with self.assertRaises(ValueError): copy_candidate(pipe,self.root/'copy.py')
        link=self.root/'link.py'
        link.symlink_to(self.candidate)
        with self.assertRaises(OSError): copy_candidate(link,self.root/'copy.py')

    def test_timeout_does_not_leave_candidate_running(self):
        self.candidate.write_text('import time\ntime.sleep(60)')
        procs=[]
        def start(*args,**kwargs):
            p,i=self.start(*args,**kwargs);procs.append(p);return p,i
        with patch('harness.evaluate.sandbox.start',start),patch('harness.evaluate.sandbox.stop',self.stop):
            with self.assertRaises(TimeoutError):
                evaluate({},self.task,self.candidate,self.root/'slow',[0],time.monotonic()+.2)
        self.assertIsNotNone(procs[0].poll())


ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/wujiang'


def serve_named(config,stage,public,cpus,mode,command,control=None,**kwargs):
    """Run the requested workspace candidate without isolation (test stand-in)."""
    path=stage/'workspace'/Path(command[2]).name
    proc=subprocess.Popen([sys.executable,str(ROOT/'harness/forecasting.py'),'--serve',str(path)],
                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return proc,None


def stop_named(p,identity):
    if p.poll() is None: p.kill()
    p.wait()
    for s in (p.stdin,p.stdout,p.stderr): s.close()


@unittest.skipUnless(DATA.exists(),'Example task package not present')
class HostInternalTests(unittest.TestCase):
    def setUp(self):
        from harness.forecasting import load_task
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.task=load_task(DATA)

    def tearDown(self): self.tmp.cleanup()

    def test_host_internal_split_matches_public_analysis(self):
        from harness.analysis import calculate_internal_validation_score
        from harness.forecasting import public_settings
        params={'season_weight':1.0}
        with patch('harness.evaluate.sandbox.start',serve_named),patch('harness.evaluate.sandbox.stop',stop_named):
            host=evaluate({},self.task,ROOT/'examples/seasonal_mean.py',self.root/'internal',[0],
                          time.monotonic()+60,split='internal',params=params)
        (self.root/'input').mkdir()
        self.task['train'].to_csv(self.root/'input/train.csv',index=False)
        (self.root/'input/task.json').write_text(json.dumps(public_settings(self.task)))
        previous=os.getcwd()
        os.chdir(self.root)
        try:
            public=calculate_internal_validation_score(str(ROOT/'examples/seasonal_mean.py'),params)
        finally:
            os.chdir(previous)
        self.assertAlmostEqual(host['score'],public['best_score'],places=12)

    def test_evaluator_reports_request_problems_and_host_internal_score(self):
        import shutil, socket
        from harness.evaluate import start_server
        worker=self.root/'worker'
        (worker/'workspace/solution').mkdir(parents=True)
        shutil.copyfile(ROOT/'examples/seasonal_mean.py',worker/'workspace/solution/seasonal_mean.py')
        config={'candidate_timeout':60}
        with patch('harness.evaluate.sandbox.start',serve_named),patch('harness.evaluate.sandbox.stop',stop_named):
            try:
                server=start_server(config,self.task,worker,[0],time.monotonic()+120)
            except OSError as error:
                self.skipTest(f'local sockets unavailable: {error}')
            def ask(name):
                with socket.socket(socket.AF_UNIX) as connection:
                    connection.settimeout(120)
                    connection.connect(str(worker/'control/evaluation.sock'))
                    connection.sendall((json.dumps({'candidate':name})+'\n').encode())
                    with connection.makefile('rb') as stream:
                        return json.loads(stream.readline())
            try:
                self.assertIn('找不到候选文件', ask('solution/missing.py')['error'])
                result=ask('solution/seasonal_mean.py')
            finally:
                server.shutdown(); server.server_close()
        self.assertIsInstance(result['internal_score'],float)
        record=json.loads((worker/'evaluations/1/result.json').read_text())
        self.assertEqual(record['internal_score'],result['internal_score'])
        self.assertTrue((worker/'submissions/1/seasonal_mean.py').is_file())

if __name__=='__main__': unittest.main()
