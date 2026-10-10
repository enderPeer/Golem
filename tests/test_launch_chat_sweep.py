"""Admission for partial launches and truthful service receipt handling."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from ops import launch_chat_sweep as launch


class LaunchAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.jobs = [self.job('cpu-0', 'cpu', 2048), self.job('cpu-1', 'cpu', 2048),
                     self.job('gpu-0', 'gpu', 4096)]
        self.manifest = {'schema': 'golem-semantic-world-sweep-v1',
                         'mode': 'supervised_until_deadline', 'jobs': self.jobs}

    def job(self, name, kind, memory):
        return {'id': name, 'host': 'host', 'root': str(self.root), 'kind': kind,
                'threads': 2 if kind == 'cpu' else 1, 'memory_mib': memory,
                'config': 'configs/' + name + '.toml', 'run_dir': 'runs/jobs/' + name}

    def receipt(self, job):
        path = self.root / 'runs/world-long-launches' / (job['id'] + '.unit')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('golem-world-long-20261011T120000-123.service\n', encoding='utf-8')
        return path

    def status(self, job, status):
        path = self.root / job['run_dir'] / 'status.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'status': status}), encoding='utf-8')

    def test_partial_launch_counts_only_pending_selected_workers(self):
        self.receipt(self.jobs[0])
        inspect = Mock(return_value={'id': 'cpu-0', 'receipt_verified': True})
        jobs, pending, existing = launch.plan_launch(self.manifest, self.root, 'host', 'cpu',
                                                     receipt_inspector=inspect)
        self.assertEqual([job['id'] for job in pending], ['cpu-1'])
        self.assertEqual(len(existing), 1)
        inspect.assert_called_once()
        # Old behavior required 8 GiB + reserve although this call adds only 2.
        launch.check_admission(jobs, pending, 8, available_memory=5 * 2**30)
        with self.assertRaisesRegex(ValueError, 'New selected jobs'):
            launch.check_admission(jobs, pending, 8, available_memory=3 * 2**30)

    def test_other_device_kind_receipts_are_not_mistaken_for_requested_runs(self):
        self.receipt(self.jobs[0])
        inspect = Mock(side_effect=AssertionError('CPU receipt should not be inspected for GPU launch'))
        jobs, pending, existing = launch.plan_launch(self.manifest, self.root, 'host', 'gpu',
                                                     receipt_inspector=inspect)
        self.assertEqual([job['id'] for job in pending], ['gpu-0'])
        self.assertEqual(existing, [])
        launch.check_admission(jobs, pending, 8, available_memory=7 * 2**30)

    def test_all_running_needs_no_second_memory_reservation(self):
        for job in self.jobs:
            self.receipt(job)
        jobs, pending, existing = launch.plan_launch(self.manifest, self.root, 'host', 'all',
            receipt_inspector=lambda root, job, path: {'id': job['id'], 'receipt_verified': True})
        self.assertEqual(pending, [])
        self.assertEqual(len(existing), 3)
        launch.check_admission(jobs, pending, 8, available_memory=1 * 2**30)

    def test_live_receipt_requires_actual_active_service_and_nonterminal_report(self):
        job = self.jobs[0]
        receipt = self.receipt(job)
        self.status(job, 'running')
        running = subprocess.CompletedProcess([], 0,
            'LoadState=loaded\nActiveState=active\nSubState=running\nResult=success\nExecMainStatus=0\n', '')
        with patch.object(launch.subprocess, 'run', return_value=running) as command:
            result = launch.inspect_receipt(self.root, job, receipt)
        self.assertTrue(result['receipt_verified'])
        self.assertEqual(command.call_args.args[0][:3], ['systemctl', '--user', 'show'])
        self.status(job, 'complete')
        with patch.object(launch.subprocess, 'run', return_value=running):
            with self.assertRaisesRegex(RuntimeError, 'No automatic restart'):
                launch.inspect_receipt(self.root, job, receipt)

    def test_dead_or_failed_receipt_does_not_hide_behind_stale_running_report(self):
        job = self.jobs[0]
        receipt = self.receipt(job)
        self.status(job, 'running')
        failed = subprocess.CompletedProcess([], 0,
            'LoadState=loaded\nActiveState=failed\nSubState=failed\nResult=oom-kill\nExecMainStatus=9\n', '')
        with patch.object(launch.subprocess, 'run', return_value=failed):
            with self.assertRaisesRegex(RuntimeError, 'oom-kill'):
                launch.plan_launch(self.manifest, self.root, 'host', 'cpu')

    def test_unreceipted_existing_run_and_overallocated_host_are_rejected(self):
        self.status(self.jobs[0], 'complete')
        with self.assertRaisesRegex(RuntimeError, 'Unreceipted existing run'):
            launch.plan_launch(self.manifest, self.root, 'host', 'all')
        with self.assertRaisesRegex(ValueError, 'logical CPU'):
            launch.check_admission(self.jobs, [], 4, available_memory=100 * 2**30)


if __name__ == '__main__':
    unittest.main()
