import json
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from harness.run import copy_references, recover, collect, choose_final, run_rounds, report_result, last_progress

class OrchestrationTests(unittest.TestCase):
    def test_three_worker_batches_then_final_decision_cannot_launch_fourth(self):
        decisions = []
        batches = []
        def decide(number, results, final_only):
            decisions.append((number, list(results), final_only))
            return {'tasks': [{'name': 'ridge', 'instruction': 'try', 'files': []}]}
        def worker(task, number, slot, deadline):
            batches.append(number)
            return {'round': number}
        results, selection = run_rounds(decide, worker, rounds=3, workers=3, deadline=time.monotonic()+5)
        self.assertEqual(batches, [1, 2, 3])
        self.assertEqual([(n, final) for n, _, final in decisions], [(1, False), (2, False), (3, False), (3, True)])
        self.assertEqual(decisions[-1][1], [{'round': 1}, {'round': 2}, {'round': 3}])
        self.assertIsNone(selection)

    def test_early_selection_stops_before_next_worker_batch(self):
        calls = []
        def decide(number, results, final_only):
            return {'tasks': [{'name': 'ridge'}]} if number == 1 else {'file': 'results/ridge.py'}
        def worker(task, number, slot, deadline):
            calls.append(number)
            return {'score': .6}
        results, selection = run_rounds(decide, worker, rounds=3, workers=3, deadline=time.monotonic()+5)
        self.assertEqual(calls, [1])
        self.assertEqual(selection, {'file': 'results/ridge.py'})

    def test_expired_search_after_workers_skips_final_decision(self):
        decisions = []
        def decide(number, results, final_only):
            decisions.append((number, final_only))
            return {'tasks': [{'name': 'ridge'}]}
        with patch('harness.run.time.monotonic', side_effect=[0, 0, 2]):
            results, selection = run_rounds(decide, lambda *args: {'score': .6}, rounds=1, workers=1, deadline=1)
        self.assertEqual(decisions, [(1, False)])
        self.assertEqual(results, [{'score': .6}])
        self.assertIsNone(selection)

    def test_deliveries_are_readable_by_the_already_running_main_uid(self):
        import os
        self.evaluated()
        previous = os.umask(0o077)
        try:
            result = collect(self.worker, self.main, 1, 'ridge', recover(self.worker, 'timeout'))
        finally:
            os.umask(previous)
        for path in [self.main/'results', self.main/'results/round1', self.main/'results/round1/ridge']:
            self.assertEqual(path.stat().st_mode & 0o005, 0o005)
        for path in [self.main/result['file'], (self.main/result['file']).with_name('result.json')]:
            self.assertTrue(path.stat().st_mode & 0o004)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.main = self.root / 'main/workspace'
        self.main.mkdir(parents=True)
        self.worker = self.root / 'workers/round1/ridge'
        (self.worker / 'workspace/solution').mkdir(parents=True)
    def tearDown(self):
        self.tmp.cleanup()
    def evaluated(self, name='ridge.py', score=.6, code='old = True', number=1):
        folder = self.worker / 'evaluations' / str(number)
        folder.mkdir(parents=True)
        (folder / name).write_text(code)
        (folder / 'result.json').write_text(json.dumps(dict(file=name, score=score, best_params={'alpha':2})))
        return folder / name
    def test_report_resolves_readonly_submission_export(self):
        saved=self.evaluated()
        exported=self.worker/'submissions/1/ridge.py'
        exported.parent.mkdir(parents=True)
        exported.write_bytes(saved.read_bytes())
        response={'status':'finished','text':json.dumps({'file':'submissions/1/ridge.py','best_params':{'alpha':2},'summary':'specific finding'})}
        report=report_result(self.worker,response)
        self.assertEqual(report['summary'],'specific finding')
        self.assertEqual(report['validation_score'],.6)

    def test_plan_only_report_reaches_main_without_invented_scores(self):
        summary = '采用季节比值回归，尚未完成代码；下一步实现特征转换。'
        response = {'status': 'finished', 'text': json.dumps(
            {'file': None, 'summary': summary, 'validation_score': 99, 'best_params': {'alpha': 1}})}
        report = report_result(self.worker, response)
        delivered = collect(self.worker, self.main, 1, 'ridge', report)
        self.assertIsNone(delivered['file'])
        self.assertIsNone(delivered['validation_score'])
        self.assertIsNone(delivered['best_params'])
        self.assertEqual(delivered['summary'], summary)
        self.assertEqual(json.loads((self.main/'results/round1/ridge/result.json').read_text())['summary'], summary)

    def test_plan_only_report_keeps_existing_best_evaluation_and_handoff(self):
        saved = self.evaluated()
        response = {'status': 'finished', 'text': json.dumps(
            {'file': None, 'summary': '后续变体尚未完成，已有版本可继续使用。'})}
        report = report_result(self.worker, response)
        self.assertEqual(report['file'], str(saved))
        self.assertEqual(report['validation_score'], .6)
        self.assertIn('后续变体尚未完成', report['summary'])

    def test_unscored_draft_handoff_preserves_code_and_explanation(self):
        draft = self.worker/'workspace/solution/draft.py'
        draft.write_text('draft = True')
        response = {'status': 'finished', 'text': json.dumps(
            {'file': 'solution/draft.py', 'summary': '已实现特征，预测形状检查失败。'})}
        report = report_result(self.worker, response)
        self.assertEqual(report['file'], str(draft))
        self.assertIsNone(report['validation_score'])
        self.assertEqual(report['summary'], '已实现特征，预测形状检查失败。')

    def test_report_cannot_read_private_evaluation_or_fifo(self):
        import os
        saved=self.evaluated()
        secret=self.worker/'evaluations/1/private.py'
        secret.write_text('private')
        response={'status':'finished','text':json.dumps({'file':'submissions/../evaluations/1/private.py'})}
        self.assertEqual(report_result(self.worker,response)['file'],str(saved))
        os.mkfifo(self.worker/'workspace/solution/wait.py')
        response['text']=json.dumps({'file':'solution/wait.py'})
        self.assertEqual(report_result(self.worker,response)['file'],str(saved))

    def test_timeout_returns_evaluated_code_not_overwritten_draft(self):
        self.evaluated()
        (self.worker / 'workspace/solution/ridge.py').write_text('broken draft')
        report = recover(self.worker, 'timeout')
        self.assertEqual(Path(report['file']).read_text(), 'old = True')
        self.assertEqual(report['validation_score'], .6)
    def test_timeout_without_submission_uses_latest_solution(self):
        p = self.worker / 'workspace/solution/seasonal_ridge.py'
        p.write_text('draft')
        report = recover(self.worker, 'timeout')
        self.assertEqual(Path(report['file']), p)
        self.assertIsNone(report['validation_score'])
        self.assertIn('超时', report['summary'])
        p.unlink()
        self.assertIsNone(recover(self.worker, 'timeout')['file'])
    def test_file_transfer_preserves_method_name_and_is_independent(self):
        self.evaluated(name='seasonal_ridge.py')
        result = collect(self.worker, self.main, 1, 'ridge', recover(self.worker,'timeout'))
        target = self.main / result['file']
        self.assertEqual(target.name, 'seasonal_ridge.py')
        workspace = self.root / 'next/workspace'
        refs = copy_references(self.main, workspace, [result['file']])
        copied = workspace / refs[0]
        self.assertEqual(copied.read_text(), target.read_text())
        copied.write_text('changed')
        self.assertEqual(target.read_text(), 'old = True')
    def test_final_selected_must_match_successful_submission_and_parameters(self):
        a = self.evaluated(score=.6)
        b = self.evaluated(name='tree.py',score=.8,number=2,code='tree = True')
        self.assertEqual(choose_final(self.root,None)['file'],str(b))
        copy=self.main/'ridge.py'
        copy.write_bytes(a.read_bytes())
        self.assertEqual(choose_final(self.root,{'file':'ridge.py','best_params':{'alpha':2}})['file'],str(a))
        self.assertEqual(choose_final(self.root,{'file':str(a),'best_params':{'alpha':99}})['file'],str(b))
        self.assertEqual(choose_final(self.root,{'file':'ridge.py','best_params':{'alpha':2}})['selected_by'],'main')
        self.assertEqual(choose_final(self.root,None)['selected_by'],'fallback')
    def test_worker_written_results_cannot_enter_host_selection(self):
        real=self.evaluated(score=.6)
        fake=self.worker/'workspace/evaluations/1'
        fake.mkdir(parents=True)
        (fake/'result.json').write_text(json.dumps({'file':'fake.py','score':999,'best_params':{}}))
        (fake/'fake.py').write_text('fake')
        self.assertEqual(choose_final(self.root,None)['file'],str(real))
        self.assertEqual(recover(self.worker,'timeout')['validation_score'],.6)

    def test_references_cannot_escape_main_workspace(self):
        (self.root/'secret.py').write_text('private')
        with self.assertRaises(ValueError):
            copy_references(self.main,self.root/'next',['../../secret.py'])
    def test_three_rounds_workers_concurrent(self):
        import threading
        active = peak = 0
        seen = []
        lock = threading.Lock()
        def decide(round_number, results, final_only):
            return {'tasks': [] if final_only else [{'name': n,'instruction':'try','files':[]} for n in ['ridge','tree','blend']]}
        def worker(task, round_number, slot, deadline):
            nonlocal active,peak
            with lock:
                active+=1
                peak=max(peak,active)
                seen.append((round_number,task['name'],slot))
            time.sleep(.03)
            with lock:
                active-=1
            return {'file':None,'summary':'done'}
        results, selection=run_rounds(decide,worker,rounds=3,workers=3,deadline=time.monotonic()+5)
        self.assertEqual(len(seen),9)
        self.assertEqual(len(set((r,n) for r,n,s in seen)),9)
        self.assertEqual(peak,3)
        self.assertEqual(len(results),9)

if __name__=='__main__': unittest.main()


class ReportFallbackTests(unittest.TestCase):
    """Handoffs keep host scores and the worker's own explanation when its report is imperfect."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.worker = Path(self.tmp.name) / 'workers/round1/ridge'
        (self.worker / 'workspace/solution').mkdir(parents=True)
    def tearDown(self):
        self.tmp.cleanup()
    def evaluated(self, score=.6, internal=.5, code='old = True', number=1, params=None):
        folder = self.worker / 'evaluations' / str(number)
        folder.mkdir(parents=True)
        (folder / 'ridge.py').write_text(code)
        (folder / 'result.json').write_text(json.dumps(dict(
            file='ridge.py', score=score, best_params=params or {'alpha': 2}, internal_score=internal)))
        return folder / 'ridge.py'
    def trajectory(self, *texts):
        with (self.worker / 'trajectory.jsonl').open('w') as stream:
            for text in texts:
                stream.write(json.dumps({'type': 'message_end', 'message': {
                    'role': 'assistant', 'content': [{'type': 'text', 'text': text}]}}) + '\n')

    def test_malformed_report_keeps_text_and_host_scores(self):
        self.evaluated()
        text = '{"file":"submissions/1/ridge.py","best_params":{"lr":0.05,"seed=0"},"summary":"跨站特征无增益"}'
        report = report_result(self.worker, {'status': 'finished', 'text': text})
        self.assertEqual(report['validation_score'], .6)
        self.assertEqual(report['internal_score'], .5)
        self.assertIn('跨站特征无增益', report['summary'])

    def test_internal_score_comes_from_host_record(self):
        saved = self.evaluated(internal=.42)
        exported = self.worker / 'workspace/solution/ridge.py'
        exported.write_bytes(saved.read_bytes())
        report = report_result(self.worker, {'status': 'finished', 'text': json.dumps(
            {'file': 'solution/ridge.py', 'best_params': {'alpha': 2}, 'internal_score': 0.99, 'summary': 's'})})
        self.assertEqual(report['internal_score'], .42)
        self.assertEqual(report['validation_score'], .6)

    def test_mismatched_report_returns_best_evaluated_version(self):
        best = self.evaluated(score=.7)
        (self.worker / 'workspace/solution/ridge.py').write_text('edited after evaluation')
        report = report_result(self.worker, {'status': 'finished', 'text': json.dumps(
            {'file': 'solution/ridge.py', 'best_params': {'alpha': 2}, 'summary': '改了特征'})})
        self.assertEqual(report['file'], str(best))
        self.assertEqual(report['validation_score'], .7)
        self.assertIn('改了特征', report['summary'])
        self.assertIn('不一致', report['summary'])

    def test_timeout_handoff_includes_last_progress(self):
        self.trajectory('写好了特征', '第二次提交前遇到 KeyError: 2009-10，正在修复月份索引')
        report = recover(self.worker, 'timeout', last_progress(self.worker))
        self.assertIn('超时', report['summary'])
        self.assertIn('KeyError: 2009-10', report['summary'])
        report = report_result(self.worker, {'status': 'timeout', 'text': ''})
        self.assertIn('正在修复月份索引', report['summary'])
