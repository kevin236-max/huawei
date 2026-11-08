"""Real orchestration/evaluation/files; only Pi and sandbox launch are local stand-ins."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from harness.run import execute
from harness.evaluate import evaluate
from harness.forecasting import load_task

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/wujiang'

class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(DATA.exists(),'Example task package not present')
    def test_three_round_file_handoff_and_final_single_parameter_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'harness').mkdir()
            shutil.copytree(ROOT/'prompts',root/'prompts')
            task=load_task(DATA)
            active=[]
            controls=[]
            main_processes=[]
            class Server:
                def shutdown(self): pass
                def server_close(self): pass
                def stop(self): pass
            def server(config,task,stage,cpus,deadline):
                controls.append((stage,cpus,deadline))
                return Server()
            def start(config,stage,public,cpus,mode,command,control=None,**kwargs):
                if mode == 'main':
                    # Keep protocol state in a real child: restarting it loses the round count.
                    p=subprocess.Popen([sys.executable, '-u', '-c', r'''
import json, sys
from pathlib import Path
turn = 0
for line in sys.stdin:
    prompt = json.loads(line)['message']
    if prompt.startswith('上一条回复中的 JSON'):
        # Retry of the truncated first plan, in the same conversation.
        assert turn == 1 and '括号' in prompt
        payload = {'tasks': [{'name': 'seasonal', 'instruction': 'Compare seasonal forecast and persistence.', 'files': []}]}
        print(json.dumps({'type': 'message_end', 'message': {'role': 'user', 'content': prompt}}), flush=True)
        print(json.dumps({'type': 'message_end', 'message': {'role': 'assistant', 'content': [
            {'type': 'text', 'text': json.dumps(payload)}], 'stopReason': 'stop'}}), flush=True)
        print(json.dumps({'type': 'agent_settled'}), flush=True)
        continue
    turn += 1
    if turn > 1:
        results, instruction = prompt.rsplit('\n\n', 1)
        batch = json.loads(results)['results']
        assert len(batch) == 1
        assert Path(batch[0]['file']).is_file()
        assert f'results/round{turn-1}/' in batch[0]['file']
    if turn == 4:
        assert '提交最终结果' in instruction
        reported = json.loads(Path('results/round1/seasonal/result.json').read_text())
        payload = {'file': 'results/round1/seasonal/seasonal_mean.py', 'best_params': reported['best_params']}
    else:
        assert turn <= 3
        files = [] if turn == 1 else [f'results/round{turn-1}/seasonal/seasonal_mean.py']
        payload = {'tasks': [{'name': 'seasonal', 'instruction': 'Compare seasonal forecast and persistence.', 'files': files}]}
    text = json.dumps(payload)
    if turn == 1:
        text = text[:-1]  # first plan loses its closing brace, as in a real run
    print(json.dumps({'type': 'message_end', 'message': {'role': 'user', 'content': prompt}}), flush=True)
    print(json.dumps({'type': 'message_end', 'message': {'role': 'assistant', 'content': [
        {'type': 'text', 'text': text}], 'stopReason': 'stop'}}), flush=True)
    print(json.dumps({'type': 'agent_settled'}), flush=True)
'''], cwd=stage/'workspace', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                    main_processes.append(p)
                    return p, None
                path=stage/'workspace'/Path(command[2]).name
                p=subprocess.Popen([sys.executable,str(ROOT/'harness/forecasting.py'),'--serve',str(path)],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                return p,None
            def stop(p,identity):
                if p.poll() is None: p.kill()
                p.wait()
                for s in (p.stdin,p.stdout,p.stderr): s.close()
            def session(config,stage,public,cpus,prompt,deadline,role='subagent', *,
                        soft_deadline=None,finish_prompt=None,check=None):
                (stage/'prompt.txt').write_text(prompt)
                self.assertEqual(role, 'subagent')
                self.assertLess(soft_deadline, deadline)
                self.assertEqual(controls[-1][2], deadline)
                self.assertTrue(finish_prompt)
                self.assertEqual(len(main_processes), 1)
                self.assertIsNone(main_processes[0].poll())
                active.append(stage)
                if 'round1' not in str(stage):
                    refs=list((stage/'workspace/references').rglob('seasonal_mean.py'))
                    self.assertEqual(len(refs),1)
                    self.assertTrue(any((stage/'workspace/references').rglob('result.json')))
                    self.assertIn(str(refs[0].relative_to(stage/'workspace')),prompt)
                candidate=stage/'workspace/solution/seasonal_mean.py'
                shutil.copyfile(ROOT/'examples/seasonal_mean.py',candidate)
                # A candidate submitted after the soft deadline still has its finishing budget.
                time.sleep(max(0, soft_deadline-time.monotonic())+.01)
                result=evaluate(config,task,candidate,stage/'evaluations/1',cpus,deadline)
                # It is valid to return the original file while it still matches the evaluation.
                payload=dict(file='solution/seasonal_mean.py',validation_score=-999,
                    best_params=result['best_params'],internal_score=None,summary='Seasonal reference tested.')
                return {'text':json.dumps(payload),'status':'finished'}
            config=dict(task=str(DATA),horizon=3,workers=3,rounds=3,cpus_per_worker=12,
                        worker_timeout=.05,worker_finish_timeout=30,search_timeout=120,candidate_timeout=30,
                        python=sys.executable,api={'api_key':'private','model':'test','max_tokens':10})
            with patch('harness.run.__file__',str(root/'harness/run.py')),patch('harness.pi.run_session',session),\
                 patch('harness.pi.gateway.start_gateway',return_value=Server()),\
                 patch('harness.evaluate.start_server',server),patch('harness.evaluate.sandbox.start',start),\
                 patch('harness.evaluate.sandbox.stop',stop),patch('os.sched_getaffinity',return_value=set(range(36))):
                run=root/'runs/20261107_120000/repeat_001'
                result=execute(config,run)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(active),3)
            self.assertEqual(len(set(active)),3)
            self.assertTrue(all(len(cpus)==12 for _,cpus,_ in controls))
            self.assertTrue(all(r['validation_score']!=-999 for r in result['results']))
            self.assertGreater(result['duration'], 0)
            self.assertEqual(json.loads((run/'result.json').read_text()), result)
            self.assertNotIn('private',(run/'config.json').read_text())
            self.assertTrue((run/result['file']).exists())
            self.assertIn('test_score',result)
            self.assertEqual(result['selected_by'],'main')
            self.assertTrue(all(r['name']=='seasonal' for r in result['results']))
            self.assertEqual(len(main_processes), 1)
            self.assertIsNotNone(main_processes[0].returncode)
            self.assertTrue(all(stream.closed for stream in (main_processes[0].stdin, main_processes[0].stdout, main_processes[0].stderr)))
            self.assertEqual(list((run/'main').rglob('trajectory.jsonl')), [run/'main/trajectory.jsonl'])
            trajectory=[json.loads(line) for line in (run/'main/trajectory.jsonl').read_text().splitlines()]
            self.assertEqual(len([event for event in trajectory if event['message']['role']=='assistant']), 5)
            prompts=[json.loads(line)['message'] for line in (run/'main/prompts.jsonl').read_text().splitlines()]
            self.assertEqual(len(prompts), 5)
            self.assertIn('（共 3 轮）', prompts[0])
            self.assertIn('最后一批', prompts[3])
            self.assertIn('results/round3/seasonal/seasonal_mean.py', prompts[-1])
            self.assertIn('\n\n', prompts[-1])
            self.assertIn('提交最终结果', prompts[-1].rsplit('\n\n', 1)[1])

if __name__=='__main__': unittest.main()
