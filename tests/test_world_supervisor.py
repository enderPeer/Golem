"""Owned process shutdown, frozen selection and lossless stage retention."""
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from unittest.mock import Mock, patch

from ops import world_supervisor as supervisor


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'base.toml').write_text('seed = 7\nbatch_size = 2\n', encoding='utf-8')
        (self.root / 'data.json').write_text('{}', encoding='utf-8')

    def instance(self, **kwargs):
        options = dict(project_root=self.root, stage_seconds=60, self_replay=False,
                       min_free_gib=.000001, memory_limit_mib=9999)
        options.update(kwargs)
        deadline = options.pop('deadline', datetime.fromtimestamp(time.time() + 3600, timezone.utc).isoformat())
        return supervisor.WorldSupervisor('base.toml', 'data.json', 'run', deadline, **options)

    def stage(self, number, owner='owner', state=True):
        path = self.root / 'run' / 'stages' / f'stage-{number:06d}'
        (path / 'run').mkdir(parents=True)
        supervisor.atomic_json(path / 'ownership.json', {'supervisor_id': owner, 'stage': path.name})
        supervisor.atomic_json(path / 'summary.json', {'status': 'complete'})
        (path / 'trainer.log').write_bytes(b'failure evidence\n' * 10)
        (path / 'run' / 'report.json').write_bytes(b'{"all_rollouts": [1,2,3]}\n')
        (path / 'run' / 'population.json').write_bytes(b'{"births": [1,2]}\n')
        if state:
            (path / 'run' / 'semantic-training.pt').write_bytes(b'resumable state')
            (path / 'run' / 'best.pt').write_bytes(b'weights')
        return path

    def test_explicit_utc_deadline_and_containment(self):
        self.assertEqual(supervisor.parse_deadline('2026-10-12T08:00:00Z').hour, 8)
        for invalid in ('2026-10-12T08:00:00', '2026-10-12T10:00:00+02:00'):
            with self.assertRaises(ValueError):
                supervisor.parse_deadline(invalid)
        with self.assertRaises(ValueError):
            supervisor.contained(self.root, '../foreign.pt')
        with self.assertRaises(ValueError):
            self.instance(stage_seconds=float('nan'))

    def test_expired_deadline_never_launches_child(self):
        worker = self.instance(deadline='2000-01-01T00:00:00Z')
        with patch.object(supervisor.subprocess, 'Popen') as popen:
            result = worker.run()
        popen.assert_not_called()
        self.assertEqual(result['status'], 'deadline_reached')
        self.assertIsNone(result['active_child'])
        self.assertEqual(result['stages'], [])

    def test_stage_training_seed_changes_but_validation_is_frozen(self):
        worker = self.instance()
        first, _ = worker.stage_config(self.root / 'unused-stage', 1)
        second, _ = worker.stage_config(self.root / 'unused-stage', 2)
        self.assertNotEqual(first['seed'], second['seed'])
        for key, value in {'validation_seed': 40097, 'validation_nll_seed': 30097,
                           'validation_batch_size': 8, 'evaluate_test_on_complete': False}.items():
            self.assertEqual(first[key], value)
            self.assertEqual(second[key], value)
        first['model'] = {'hidden_size': 16, 'dropout': .1}
        config_path = self.root / 'roundtrip.toml'
        supervisor.write_toml(config_path, first)
        with config_path.open('rb') as stream:
            self.assertEqual(tomllib.load(stream), first)

    def test_pruning_retains_exact_evidence_and_only_owned_tensors(self):
        stages = [self.stage(number) for number in range(1, 5)]
        foreign = self.stage(0, owner='other-island')
        original = (stages[0] / 'run' / 'report.json').read_bytes()
        removed = supervisor.prune_stages(self.root / 'run', 'owner')
        self.assertTrue(removed)
        for old in stages[:2]:
            self.assertFalse((old / 'run' / 'semantic-training.pt').exists())
            self.assertEqual(gzip.decompress((old / 'run' / 'report.json.gz').read_bytes()), original)
            self.assertIn(b'failure evidence', gzip.decompress((old / 'trainer.log.gz').read_bytes()))
            self.assertTrue((old / 'run' / 'population.json.gz').is_file())
            self.assertEqual(len(json.loads((old / 'summary.json').read_text())['archived_evidence']), 3)
        for retained in [*stages[2:], foreign]:
            self.assertTrue((retained / 'run' / 'semantic-training.pt').exists())
            self.assertTrue((retained / 'run' / 'report.json').exists())
        self.assertEqual(supervisor.prune_stages(self.root / 'run', 'owner'), [])

    def test_failed_stages_cannot_evict_last_two_actual_full_states(self):
        first, second = self.stage(1), self.stage(2)
        self.stage(3, state=False)
        self.stage(4, state=False)
        supervisor.prune_stages(self.root / 'run', 'owner')
        for retained in (first, second):
            self.assertTrue((retained / 'run' / 'semantic-training.pt').exists())
            self.assertTrue((retained / 'run' / 'report.json.gz').is_file())

    def test_three_consecutive_failures_stop_without_unbounded_retries(self):
        worker = self.instance()
        clock = [time.time()]
        def advance(seconds):
            clock[0] += seconds
        with patch.object(worker, 'resource_check', side_effect=RuntimeError('storage reserve')), \
             patch.object(supervisor.time, 'time', side_effect=lambda: clock[0]), \
             patch.object(supervisor.time, 'sleep', side_effect=advance):
            result = worker.run()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['consecutive_failures'], 3)
        self.assertEqual(len(result['stages']), 3)
        self.assertIsNone(result['active_child'])

    def test_resume_refuses_live_preexisting_child_without_signalling_it(self):
        worker = self.instance()
        worker.state['active_child'] = {'pid': 7654, 'start_ticks': '100'}
        worker.save()
        with patch.object(supervisor, 'process_start', return_value='100'), \
             patch.object(supervisor.os, 'killpg', create=True) as killpg:
            with self.assertRaisesRegex(RuntimeError, 'previous owned child'):
                self.instance(deadline=worker.state['deadline_utc'], resume=True)
        killpg.assert_not_called()

    def test_shutdown_signals_only_owned_process_group_then_reaps(self):
        child = Mock(pid=5432, returncode=-9)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('owned', 2), -9]
        with patch.object(supervisor.os, 'killpg', create=True) as killpg, \
             patch.object(supervisor.signal, 'SIGKILL', 9, create=True):
            supervisor.terminate_owned(child, time.time() + 2, grace=15)
        self.assertEqual([call.args[0] for call in killpg.call_args_list], [5432, 5432])
        self.assertLessEqual(child.wait.call_args_list[0].kwargs['timeout'], 2)
        self.assertEqual(child.wait.call_args_list[-1].kwargs['timeout'], 5)

    def test_resume_journals_crashed_stage_only_after_run_begins(self):
        worker = self.instance(deadline='2000-01-01T00:00:00Z')
        stage = self.stage(1, owner=worker.state['supervisor_id'])
        (stage / 'summary.json').unlink()
        worker.state.update(active_stage=stage.name, next_stage=2)
        worker.save()
        resumed = self.instance(deadline=worker.state['deadline_utc'], resume=True)
        self.assertFalse((stage / 'summary.json').exists())
        result = resumed.run()
        self.assertEqual(result['status'], 'deadline_reached')
        self.assertEqual(len(result['stages']), 1)
        self.assertEqual(result['stages'][0]['status'], 'interrupted')
        self.assertEqual(result['stages'][0]['pending_final_candidate']['sha256'],
                         supervisor.sha256(stage / 'run' / 'best.pt'))
        self.assertTrue((stage / 'summary.json').exists())

    def test_parent_imports_no_torch_and_helpers_are_isolated_sessions(self):
        check = subprocess.run([sys.executable, '-B', '-c',
                                "import sys; import ops.world_supervisor; assert 'torch' not in sys.modules"],
                               cwd=supervisor.PROJECT, capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)
        worker = self.instance()
        child = Mock(pid=4321, returncode=0)
        child.poll.return_value = 0
        child.wait.return_value = 0
        with patch.object(supervisor.subprocess, 'Popen', return_value=child) as popen:
            worker.run_child(['python', 'helper'], self.root / 'helper.log', time.time() + 60, 'seed', helper=True)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        for name in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
            self.assertEqual(popen.call_args.kwargs['env'][name], '')
        self.assertIsNone(worker.child)

    def test_protocol_change_is_rejected_before_global_champion_write(self):
        worker = self.instance()
        stage = self.root / 'candidate'
        (stage / 'run').mkdir(parents=True)
        (stage / 'run' / 'best.pt').write_bytes(b'new')
        worker.state['global_best'] = {'validation_protocol_hash': 'a' * 64}
        with patch.object(worker, 'helper', return_value=({'validation_protocol_hash': 'b' * 64}, {})):
            with self.assertRaisesRegex(ValueError, 'frozen validation protocol'):
                worker.promote(stage)
        self.assertFalse((worker.run_dir / 'global_best.pt').exists())

    def test_only_owned_budget_sigterm_with_committed_clean_report_is_acceptable(self):
        phase = {'reason': 'budget_stop', 'returncode': -supervisor.signal.SIGTERM}
        report = {'schema': 'golem-semantic-training-v1', 'status': 'time_limit', 'error': None}
        self.assertTrue(supervisor.committed_training_phase(phase, report))
        for changed_phase, changed_report in (
                ({**phase, 'reason': 'exited'}, report),
                ({**phase, 'reason': 'memory budget'}, report),
                ({**phase, 'returncode': -9}, report),
                ({**phase, 'returncode': 1}, report),
                (phase, {**report, 'status': 'running'}),
                (phase, {**report, 'status': 'complete'}),
                (phase, {**report, 'error': {'message': 'failed'}}),
                (phase, {})):
            with self.subTest(phase=changed_phase, report=changed_report):
                self.assertFalse(supervisor.committed_training_phase(changed_phase, changed_report))

    def test_budget_sigterm_stage_still_validates_and_promotes_committed_checkpoint(self):
        worker = self.instance(max_stages=1)
        protocol = 'd' * 64
        def stopped_trainer(command, log_path, hard_deadline, phase, helper=False):
            stage = Path(log_path).parent
            (stage / 'run').mkdir()
            (stage / 'run' / 'best.pt').write_bytes(b'committed candidate')
            supervisor.atomic_json(stage / 'run' / 'report.json', {
                'schema': 'golem-semantic-training-v1', 'status': 'time_limit', 'error': None,
                'validation_protocol_hash': protocol})
            return {'phase': 'training', 'reason': 'budget_stop', 'returncode': -supervisor.signal.SIGTERM}
        def inspect(kind, stage, payload, seconds):
            self.assertEqual(kind, 'inspect')
            return {'checkpoint_sha256': supervisor.sha256(stage / 'run' / 'best.pt'),
                    'validation_protocol_hash': protocol,
                    'quality': {'false_memory_claims': 0, 'score': .2,
                                'whole_episode_pass_rate': 0., 'reply_loss': 1.}}, {'phase': 'inspect'}
        with patch.object(worker, 'run_child', side_effect=stopped_trainer), \
             patch.object(worker, 'helper', side_effect=inspect), patch.object(worker, 'resource_check'):
            result = worker.run()
        self.assertEqual(result['status'], 'stage_limit_reached')
        self.assertEqual(result['stages'][0]['status'], 'complete')
        self.assertIn('SIGTERM', result['stages'][0]['shutdown_note'])
        self.assertEqual(result['consecutive_failures'], 0)
        self.assertEqual((worker.run_dir / 'global_best.pt').read_bytes(), b'committed candidate')

    def test_resume_recovers_journaled_budget_exit_and_pending_stage_not_unrecorded_files(self):
        worker = self.instance()
        worker.resume_requested = True
        one, two, unrecorded = [self.stage(n, owner=worker.state['supervisor_id']) for n in (1, 2, 3)]
        protocol = 'c' * 64
        for stage in (one, two, unrecorded):
            supervisor.atomic_json(stage / 'run' / 'report.json', {
                'schema': 'golem-semantic-training-v1', 'status': 'time_limit', 'error': None,
                'validation_protocol_hash': protocol})
        worker.state['stages'] = [
            {'stage': one.name, 'status': 'failed', 'trainer_status': 'time_limit',
             'error': {'message': 'training stage failed or did not publish a complete status report'},
             'phases': [{'phase': 'training', 'reason': 'budget_stop', 'returncode': -supervisor.signal.SIGTERM}]},
            {'stage': two.name, 'status': 'interrupted',
             'pending_final_candidate': {'path': str((two / 'run' / 'best.pt').relative_to(worker.run_dir)),
                                         'sha256': supervisor.sha256(two / 'run' / 'best.pt')}}]
        worker.state['consecutive_failures'] = 2
        inspected = []
        def inspect(kind, stage, payload, seconds):
            inspected.append(stage.name)
            return {'checkpoint_sha256': supervisor.sha256(stage / 'run' / 'best.pt'),
                    'validation_protocol_hash': protocol,
                    'quality': {'false_memory_claims': 0, 'score': .1 if stage == one else .2,
                                'whole_episode_pass_rate': 0., 'reply_loss': 1.}}, {'phase': 'inspect'}
        with patch.object(worker, 'helper', side_effect=inspect), patch.object(worker, 'resource_check'):
            worker.recover_recorded_candidates()
            worker.recover_recorded_candidates()  # Already validated identities are not reloaded.
        self.assertEqual(inspected, [one.name, two.name])
        self.assertEqual(worker.state['global_best']['quality']['score'], .2)
        self.assertEqual(worker.state['consecutive_failures'], 0)
        self.assertEqual(len(worker.state['candidate_recovery']), 2)
        self.assertTrue(all(row['status'] == 'validated' for row in worker.state['candidate_recovery']))
        self.assertEqual(worker.state['stages'][0]['status'], 'failed')  # Preserve original audit record.

    def test_recovery_requires_pending_sha_and_never_promotes_after_deadline(self):
        worker = self.instance()
        worker.resume_requested = True
        stage = self.stage(1, owner=worker.state['supervisor_id'])
        worker.state['stages'] = [{'stage': stage.name, 'status': 'interrupted',
            'pending_final_candidate': {'path': str((stage / 'run' / 'best.pt').relative_to(worker.run_dir)),
                                        'sha256': '0' * 64}}]
        with patch.object(worker, 'helper') as helper:
            worker.recover_recorded_candidates()
            self.assertEqual(worker.state['candidate_recovery'][-1]['status'], 'rejected')
            worker.deadline = time.time() - 1
            worker.recover_recorded_candidates()
        helper.assert_not_called()
        self.assertIsNone(worker.state['global_best'])

    def test_flat_seed_preserves_weights_and_source_bytes_without_nested_metadata(self):
        import torch
        from english_world.neural_model import NeuralConfig
        from english_world.semantic_model import SemanticGRU, save_semantic_checkpoint, load_semantic_checkpoint
        torch.set_num_threads(1)
        model = SemanticGRU(NeuralConfig(embedding_dim=8, hidden_size=12, num_layers=1))
        source, target = self.root / 'original.pt', self.root / 'seed.pt'
        metadata = {'selection_split': 'validation', 'validation_protocol_hash': 'a' * 64,
                    'validation': {'rollout': {'split': 'validation', 'score': .5,
                        'false_memory_claims': 0, 'whole_episode_pass_rate': .25},
                        'teacher_forced': {'reply_loss': 1.5}},
                    'initialization': {'source_metadata': {'source_metadata': {'large': 'x' * 3000}}}}
        save_semantic_checkpoint(source, model, metadata)
        before = source.read_bytes()
        summary = supervisor.candidate_summary(source)
        result = supervisor.condensed_seed(source, target, {'parent_run': 'stage-1', 'quality': summary['quality']})
        restored, flat = load_semantic_checkpoint(target)
        self.assertEqual(source.read_bytes(), before)
        self.assertLess(result['metadata_bytes'], 1000)
        self.assertNotIn('initialization', flat)
        self.assertEqual(flat['parent_checkpoint_sha256'], summary['checkpoint_sha256'])
        for name, weight in model.state_dict().items():
            self.assertTrue(torch.equal(weight, restored.state_dict()[name]))
        self.assertGreater(supervisor.quality_key(summary), supervisor.quality_key({
            'quality': {**summary['quality'], 'false_memory_claims': 1, 'score': 1.}}))


if __name__ == '__main__':
    unittest.main()
