"""Synthetic stopped-run fixtures only: no remote hosts, tensor loads or TEST."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ops import world_final_collect as collector


class FinalCollectorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / 'golem'
        self.root.mkdir()
        self.data = self.root / 'data/conversation-world/dataset.json'
        self.data.parent.mkdir(parents=True)
        # Invalid JSON proves that the collector never parses corpus examples.
        self.data.write_bytes(b'opaque frozen corpus, not parsed or evaluated\x00')
        license_path = self.data.parent / 'ORIGINAL-DATA-LICENSE.txt'
        license_path.write_bytes(b'fixture license')
        snapshot = self.data.parent / 'source_snapshot/simulation_spec.json'
        self.write(snapshot, {'fixture': True})
        self.write(self.data.parent / 'manifest.json', {'content_sha256': 'a' * 64,
                   'licenses': {license_path.name: collector.digest(license_path)},
                   'snapshots': {snapshot.name: collector.digest(snapshot)}})
        self.manifest = {'schema': 'golem-semantic-world-sweep-v1', 'mode': 'supervised_until_deadline',
                         'deadline_utc': '2026-10-12T08:00:00Z', 'jobs': [], 'suspended_jobs': [], 'retired_jobs': []}
        self.path = self.base / 'manifest.json'
        self.now = datetime(2026, 10, 12, 8, 1, tzinfo=timezone.utc)
        self.services = Mock(return_value={'LoadState': 'loaded', 'ActiveState': 'inactive', 'SubState': 'dead',
                                           'MainPID': '0', 'ControlPID': '0', 'cgroup_pids': []})
        self.patch_roots = patch.dict(collector.HOST_ROOTS, {'specht32': str(self.root)}, clear=True)
        self.patch_roots.start()
        self.addCleanup(self.patch_roots.stop)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def job(self, suffix='cpu-0', category='jobs'):
        identity = 'specht32-world-' + suffix
        job = {'id': identity, 'host': 'specht32', 'root': str(self.root),
               'run_dir': 'runs/world-long/' + identity,
               'config': 'runs/world-long-configs/' + identity + '.toml'}
        self.manifest[category].append(job)
        run = self.root / job['run_dir']
        run.mkdir(parents=True)
        config = self.root / job['config']
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text('seed = 7\n', encoding='utf-8')
        receipt = self.root / 'runs/world-long-launches' / (identity + '.unit')
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text('golem-world-long-fixture-' + suffix + '.service\n', encoding='utf-8')
        (run / 'global_best.pt').write_bytes(('opaque weights ' + identity).encode())
        self.write(run / 'global_best.json', {'checkpoint_sha256': collector.digest(run / 'global_best.pt')})
        self.write(run / 'supervisor-state.json', {'schema': 'golem-world-supervisor-v1',
            'supervisor_id': identity, 'deadline_utc': self.manifest['deadline_utc'],
            'base_data_sha256': collector.digest(self.data), 'base_config_sha256': collector.digest(config),
            'status': 'deadline_reached', 'active_child': None, 'active_stage': None, 'stages': []})
        self.write(run / 'status.json', {'status': 'deadline_reached'})
        return job, run

    def stage(self, job, run, number=1, pending=True):
        name = f'stage-{number:06d}'
        stage = run / 'stages' / name
        (stage / 'run').mkdir(parents=True)
        self.write(stage / 'ownership.json', {'supervisor_id': job['id'], 'stage': name})
        (stage / 'warm-seed.pt').write_bytes(b'opaque warm seed')
        (stage / 'repair.json').write_bytes(b'{"opaque_train_overlay":true}')
        (stage / 'config.toml').write_text('seed = 8\ninit_checkpoint = ' +
            json.dumps((stage / 'warm-seed.pt').as_posix()) + '\nrepair_overlay = ' +
            json.dumps((stage / 'repair.json').as_posix()) + '\n', encoding='utf-8')
        for file in ('semantic-training.pt', 'best.pt', 'ordinary_champion.pt', 'evolution_champion.pt'):
            (stage / 'run' / file).write_bytes(('opaque ' + file).encode())
        self.write(stage / 'run/report.json', {'schema': 'golem-semantic-training-v1', 'status': 'interrupted', 'error': None})
        entry = {'stage': name, 'status': 'deadline_checkpoint'}
        if pending:
            entry['pending_final_candidate'] = {'path': f'stages/{name}/run/best.pt',
                                               'sha256': collector.digest(stage / 'run/best.pt')}
        self.write(stage / 'summary.json', entry)
        state = json.loads((run / 'supervisor-state.json').read_text())
        state['stages'].append(entry)
        self.write(run / 'supervisor-state.json', state)
        return stage

    def capture(self, output='snapshot', **kwargs):
        self.write(self.path, self.manifest)
        options = {'now': self.now, 'service_inspector': self.services, 'child_check': lambda _: False}
        options.update(kwargs)
        return collector.capture(self.path, 'specht32', self.root, self.base / output, **options)

    def test_premature_capture_refused_before_service_or_output_actions(self):
        self.job()
        with self.assertRaisesRegex(RuntimeError, 'before.*deadline'):
            self.capture(now=datetime(2026, 10, 12, 7, 59, tzinfo=timezone.utc))
        self.services.assert_not_called()
        self.assertFalse((self.base / 'snapshot').exists())
        self.assertFalse(list(self.base.glob('snapshot.partial-*')))

    def test_active_unit_cgroup_or_owned_child_refused(self):
        self.job()
        for values in ({'ActiveState': 'active'}, {'MainPID': '123'}, {'ControlPID': '123'}, {'cgroup_pids': [123]}):
            with self.subTest(values=values):
                self.services.return_value = {'ActiveState': 'inactive', 'MainPID': '0', 'ControlPID': '0', 'cgroup_pids': [], **values}
                with self.assertRaisesRegex(RuntimeError, 'still active'):
                    self.capture()
        self.services.return_value = {'ActiveState': 'inactive', 'MainPID': '0', 'ControlPID': '0', 'cgroup_pids': []}
        with self.assertRaisesRegex(RuntimeError, 'still active'):
            self.capture(child_check=lambda _: True)
        self.assertFalse((self.base / 'snapshot').exists())

    def test_unknown_service_state_fails_closed_and_child_identity_handles_pid_reuse(self):
        with patch.object(collector.subprocess, 'run', return_value=Mock(returncode=1, stdout='')):
            with self.assertRaisesRegex(RuntimeError, 'cannot establish'):
                collector.inspect_unit('golem-world-long-fixture.service')
        fields = ['S'] + ['0'] * 18 + ['999']
        with patch.object(Path, 'read_text', return_value='123 (owned worker) ' + ' '.join(fields)):
            self.assertTrue(collector.child_exists({'pid': 123, 'start_ticks': 999}))
            self.assertTrue(collector.child_exists({'pid': 123, 'start_ticks': None}))
            self.assertFalse(collector.child_exists({'pid': 123, 'start_ticks': 888}))

    def test_union_recovery_dependencies_candidates_and_exact_hashes(self):
        for index, category in enumerate(('jobs', 'suspended_jobs', 'retired_jobs')):
            job, run = self.job(f'cpu-{index}', category)
            self.stage(job, run)
        result = self.capture()
        self.assertEqual(result['runs'], 3)
        self.assertTrue(result['verified'])
        output = self.base / 'snapshot'
        manifest = json.loads((output / 'snapshot-manifest.json').read_text())
        self.assertEqual({r['category'] for r in manifest['runs']}, {'jobs', 'suspended_jobs', 'retired_jobs'})
        self.assertEqual(sum(name.endswith('semantic-training.pt') for name in manifest['files']), 3)
        self.assertEqual(sum(name.endswith('warm-seed.pt') for name in manifest['files']), 3)
        self.assertEqual(sum(':journaled_pending:' in c['origin'] for c in manifest['candidates']), 3)
        self.assertEqual(sum(':latest_committed:' in c['origin'] for c in manifest['candidates']), 3)
        self.assertEqual(sum(':strategy:' in c['origin'] for c in manifest['candidates']), 6)
        self.assertEqual(collector.verify(output), {k: result[k] for k in collector.verify(output)})
        self.assertEqual(manifest['files']['data/conversation-world/dataset.json']['sha256'], collector.digest(self.data))
        self.assertEqual(json.loads((output / 'allocation-manifest.json').read_text()), self.manifest)
        with self.assertRaisesRegex(ValueError, 'fresh dedicated'):
            self.capture()

    def test_duplicate_identity_and_manifest_escape_refused(self):
        job, _ = self.job()
        self.manifest['retired_jobs'].append(dict(job))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.capture()
        self.manifest['retired_jobs'].clear()
        job['run_dir'] = 'runs/world-long/../../foreign'
        with self.assertRaisesRegex(ValueError, 'root/run'):
            self.capture()

    def test_dependency_path_escape_and_symlink_refused(self):
        job, run = self.job()
        stage = self.stage(job, run)
        (stage / 'config.toml').write_text('init_checkpoint = "../foreign.pt"\n')
        with self.assertRaisesRegex(ValueError, 'unsafe path'):
            self.capture()
        (stage / 'config.toml').write_text('seed = 8\n')
        foreign = self.base / 'foreign'
        foreign.write_bytes(b'unrelated data')
        link = stage / 'escape.pt'
        try:
            link.symlink_to(foreign)
        except (OSError, NotImplementedError):
            self.skipTest('host cannot create symlinks')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.capture()

    def test_global_and_pending_hash_mismatch_refused(self):
        job, run = self.job()
        stage = self.stage(job, run)
        (run / 'global_best.pt').write_bytes(b'tampered global')
        with self.assertRaisesRegex(ValueError, 'candidate hash mismatch'):
            self.capture()
        self.write(run / 'global_best.json', {'checkpoint_sha256': collector.digest(run / 'global_best.pt')})
        (stage / 'run/best.pt').write_bytes(b'tampered pending')
        with self.assertRaisesRegex(ValueError, 'candidate hash mismatch'):
            self.capture()

    def test_missing_pruned_pending_is_reported_not_substituted(self):
        job, run = self.job()
        stage = self.stage(job, run)
        (stage / 'run/best.pt').unlink()
        self.capture()
        record = json.loads((self.base / 'snapshot/snapshot-manifest.json').read_text())
        missing = [r for r in record['missing_historical_artifacts'] if ':journaled_pending:' in r['role']]
        self.assertEqual(len(missing), 1)
        self.assertIn('no replacement', missing[0]['reason'])
        self.assertFalse(any(':journaled_pending:' in c['origin'] for c in record['candidates']))

    def test_archive_and_per_file_integrity_checked(self):
        self.job()
        self.capture()
        output = self.base / 'snapshot'
        path = output / 'snapshot-manifest.json'
        record = json.loads(path.read_text())
        key = next(iter(record['files']))
        record['files'][key]['sha256'] = '0' * 64
        self.write(path, record)
        with self.assertRaisesRegex(ValueError, 'member hash'):
            collector.verify(output)
        with (output / 'snapshot.tar.gz').open('ab') as handle:
            handle.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'archive identity'):
            collector.verify(output)

    def test_summary_only_pending_survives_interrupted_ledger_commit(self):
        job, run = self.job()
        self.stage(job, run)
        state = json.loads((run / 'supervisor-state.json').read_text())
        state['active_stage'] = state['stages'][0]['stage']
        state['stages'] = []
        self.write(run / 'supervisor-state.json', state)
        self.capture()
        record = json.loads((self.base / 'snapshot/snapshot-manifest.json').read_text())
        self.assertEqual(sum(':journaled_pending:' in c['origin'] for c in record['candidates']), 1)
        self.assertEqual(sum(':latest_committed:' in c['origin'] for c in record['candidates']), 1)

    def test_latest_committed_candidate_found_with_retention_compressed_report(self):
        job, run = self.job()
        stage = self.stage(job, run, pending=False)
        report = stage / 'run/report.json'
        (stage / 'run/report.json.gz').write_bytes(gzip.compress(report.read_bytes()))
        report.unlink()
        self.capture()
        record = json.loads((self.base / 'snapshot/snapshot-manifest.json').read_text())
        self.assertEqual(sum(':latest_committed:' in c['origin'] for c in record['candidates']), 1)

    def test_owned_orphan_models_are_preserved_but_not_candidates(self):
        job, run = self.job()
        self.stage(job, run)
        state = json.loads((run / 'supervisor-state.json').read_text())
        state['stages'] = []
        self.write(run / 'supervisor-state.json', state)
        self.capture()
        record = json.loads((self.base / 'snapshot/snapshot-manifest.json').read_text())
        self.assertEqual(record['runs'][0]['owned_orphan_stage_ids'], ['stage-000001'])
        self.assertEqual(record['runs'][0]['ledger_stage_ids'], [])
        self.assertTrue(any(name.endswith('ordinary_champion.pt') for name in record['files']))
        self.assertEqual([c['origin'] for c in record['candidates']], [job['id'] + ':global'])

    def test_declared_license_and_source_snapshot_are_required_and_pinned(self):
        self.job()
        snapshot = self.data.parent / 'source_snapshot/simulation_spec.json'
        snapshot.write_text('{"modified":true}')
        with self.assertRaisesRegex(ValueError, 'snapshot identity mismatch'):
            self.capture()
        (self.data.parent / 'ORIGINAL-DATA-LICENSE.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'required artifact missing'):
            self.capture()

    def test_symlink_components_refused_without_platform_privilege(self):
        self.job()
        original = Path.is_symlink
        def flagged(path):
            return path == self.root / 'runs/world-long' or original(path)
        with patch.object(Path, 'is_symlink', flagged):
            with self.assertRaisesRegex(ValueError, 'symlink'):
                self.capture()

    def test_metadata_change_at_final_stopped_check_refuses_publication(self):
        _, run = self.job()
        count = 0
        stopped = self.services.return_value
        def inspect(_):
            nonlocal count
            count += 1
            if count == 2:
                state_path = run / 'supervisor-state.json'
                state = json.loads(state_path.read_text())
                state['status'] = 'manually_changed'
                self.write(state_path, state)
            return stopped
        with self.assertRaisesRegex(ValueError, 'identity changed before publication'):
            self.capture(service_inspector=inspect)
        self.assertFalse((self.base / 'snapshot').exists())

    def test_restart_during_capture_prevents_atomic_publication(self):
        self.job()
        stopped = {'ActiveState': 'inactive', 'MainPID': '0', 'ControlPID': '0', 'cgroup_pids': []}
        self.services.side_effect = [stopped, {**stopped, 'ActiveState': 'active', 'MainPID': '45'}]
        with self.assertRaisesRegex(RuntimeError, 'still active'):
            self.capture()
        self.assertFalse((self.base / 'snapshot').exists())
        self.assertEqual(len(list(self.base.glob('snapshot.partial-*'))), 1)


if __name__ == '__main__':
    unittest.main()
