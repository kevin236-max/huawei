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

if __name__=='__main__': unittest.main()
