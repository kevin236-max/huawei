"""Batch layout, timing, and statistics without launching model searches."""
import json
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness import run


class RepeatTests(unittest.TestCase):
    def test_batch_keeps_independent_results_and_excludes_failed_searches(self):
        self.assertTrue(hasattr(run, 'execute_batch'), 'batch execution is missing')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outcomes = iter([
                {'status': 'complete', 'validation_score': .4, 'test_score': .6, 'results': []},
                RuntimeError('search failed'),
                {'status': 'complete', 'validation_score': .6, 'test_score': .8, 'results': []},
                {'status': 'test_failed', 'validation_score': .9},
                {'status': 'no_valid_candidate', 'results': []},
            ])

            def search(config, directory):
                outcome = next(outcomes)
                if isinstance(outcome, Exception):
                    raise outcome
                (directory/'candidate.py').write_text('original candidate')
                return outcome

            with patch.object(run, '__file__', str(root/'harness/run.py')), \
                 patch.object(run, '_execute', search):
                summary = run.execute_batch({}, repeat=5)
            batch, = (root/'runs').iterdir()
            self.assertRegex(batch.name, r'^\d{8}_\d{6}$')
            self.assertEqual(sorted(p.name for p in batch.iterdir()),
                             [f'repeat_{i:03d}' for i in range(1, 6)] + ['summary.json'])
            self.assertEqual(summary['repeat'], 5)
            self.assertEqual(summary['completed'], 2)
            self.assertAlmostEqual(summary['validation_mean'], .5)
            self.assertAlmostEqual(summary['test_mean'], .7)
            self.assertAlmostEqual(summary['test_std'], 2 ** .5 / 10)
            self.assertEqual(json.loads((batch/'summary.json').read_text()), summary)
            for i in range(1, 6):
                result = json.loads((batch/f'repeat_{i:03d}/result.json').read_text())
                start, finish = (datetime.fromisoformat(result[key]) for key in ('start', 'finish'))
                self.assertIsNotNone(start.tzinfo)
                self.assertGreaterEqual(finish, start)
                self.assertGreaterEqual(result['duration'], 0)
            self.assertEqual(json.loads((batch/'repeat_001/result.json').read_text())['results'], [])
            self.assertEqual(json.loads((batch/'repeat_002/result.json').read_text())['status'], 'failed')

    def test_interruption_saves_timing_and_stops_batch(self):
        self.assertTrue(hasattr(run, 'execute_batch'), 'batch execution is missing')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(run, '__file__', str(root/'harness/run.py')), \
                 patch.object(run, '_execute', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run.execute_batch({}, repeat=5)
            batch, = (root/'runs').iterdir()
            result = json.loads((batch/'repeat_001/result.json').read_text())
            self.assertEqual(result['status'], 'interrupted')
            self.assertIsNotNone(result['finish'])
            self.assertFalse((batch/'repeat_002').exists())
            summary = json.loads((batch/'summary.json').read_text())
            self.assertEqual(summary['completed'], 0)
            self.assertIsNone(summary['test_mean'])
            self.assertIsNone(summary['test_std'])

    def test_single_sample_std_and_invalid_repeat(self):
        self.assertTrue(hasattr(run, 'summarize'), 'batch statistics are missing')
        summary = run.summarize([{'status': 'complete', 'validation_score': .4, 'test_score': .6}], 1)
        self.assertEqual(summary['test_mean'], .6)
        self.assertIsNone(summary['test_std'])
        for repeat in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                run.execute_batch({}, repeat=repeat)

    def test_same_second_batch_collision_preserves_existing_results(self):
        self.assertTrue(hasattr(run, 'execute_batch'), 'batch execution is missing')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(run, '__file__', str(root/'harness/run.py')), \
                 patch.object(run, 'datetime') as clock, \
                 patch.object(run, '_execute', return_value={'status': 'no_valid_candidate'}):
                clock.now.return_value = datetime(2026, 11, 7, 15, 0).astimezone()
                run.execute_batch({}, repeat=1)
                result = root/'runs/20261107_150000/repeat_001/result.json'
                original = result.read_bytes()
                with self.assertRaises(FileExistsError):
                    run.execute_batch({}, repeat=1)
                self.assertEqual(result.read_bytes(), original)
