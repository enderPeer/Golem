"""Admission for partial launches and truthful service receipt handling."""
import json
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
                'physical_gpu': 0 if kind == 'gpu' else None,
                'config': 'configs/' + name + '.toml', 'run_dir': 'runs/jobs/' + name}

    def test_device_environment_overrides_ambient_filters_without_mutating_inputs(self):
        ambient = {'CUDA_VISIBLE_DEVICES': '0,1,2', 'HIP_VISIBLE_DEVICES': '0,1,2',
                   'ROCR_VISIBLE_DEVICES': '2', 'GPU_DEVICE_ORDINAL': '2',
                   'PATH': '/reviewed/runtime/bin', 'GOLEM_THREADS': '99'}
        original = dict(ambient)
        for kind, backend, physical, expected in (
                ('cpu', 'cuda', 7, ('', '')),
                ('gpu', 'cuda', 2, ('2', '')),
                ('gpu', 'hip', 1, ('', '1')),
                ('gpu', None, 0, ('', '0'))):
            job = {**self.job('island', kind, 2048), 'physical_gpu': physical,
                   'python': '/reviewed/python', 'minimum_free_gib': 50}
            if backend is not None:
                job['gpu_backend'] = backend
            before = dict(job)
            with self.subTest(kind=kind, backend=backend):
                env = launch.job_environment(job, ambient)
                self.assertEqual((env['CUDA_VISIBLE_DEVICES'], env['HIP_VISIBLE_DEVICES']), expected)
                self.assertNotIn('ROCR_VISIBLE_DEVICES', env)
                self.assertNotIn('GPU_DEVICE_ORDINAL', env)
                self.assertEqual(env['GOLEM_THREADS'], str(job['threads']))
                self.assertEqual(env['PATH'], ambient['PATH'])
                self.assertEqual(job, before)
                self.assertEqual(ambient, original)

    def test_invalid_gpu_admission_is_rejected_before_any_receipt_inspection(self):
        for field, values in (
                ('gpu_backend', (None, '', 'CUDA', 'rocm', 'cuda,hip', 0, True)),
                ('physical_gpu', (None, -1, 0., '0', '0,1', True, False))):
            for value in values:
                self.jobs[2] = {**self.job('gpu-0', 'gpu', 4096), field: value}
                inspect = Mock()
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    launch.plan_launch(self.manifest, self.root, 'host', 'gpu', receipt_inspector=inspect)
                inspect.assert_not_called()
        self.jobs[2] = self.job('gpu-0', 'gpu', 4096)
        self.jobs[2].pop('physical_gpu')
        with self.assertRaisesRegex(ValueError, 'physical_gpu'):
            launch.plan_launch(self.manifest, self.root, 'host', 'gpu')

    def test_cuda_uuid_overrides_ordinal_but_still_requires_inventory_index(self):
        uuid = 'GPU-5a30c18a-c75e-c22a-7663-74f2eb24a91d'
        job = {**self.job('gpu-0', 'gpu', 4096), 'gpu_backend': 'cuda',
               'gpu_uuid': uuid, 'physical_gpu': 2}
        self.assertEqual(launch.device_visibility(job),
                         {'CUDA_VISIBLE_DEVICES': uuid, 'HIP_VISIBLE_DEVICES': ''})
        for physical in (None, -1, '2', True):
            with self.subTest(physical=physical), self.assertRaisesRegex(ValueError, 'physical_gpu'):
                launch.device_visibility({**job, 'physical_gpu': physical})

    def test_invalid_uuid_or_uuid_on_non_cuda_job_is_rejected_during_planning(self):
        uuid = 'GPU-5a30c18a-c75e-c22a-7663-74f2eb24a91d'
        for value in (None, True, 0, '', 'GPU-5a30c18a', uuid.lower(), uuid + ',0',
                      uuid + '\n', uuid[:-1] + 'z', [uuid]):
            self.jobs[2] = {**self.job('gpu-0', 'gpu', 4096), 'gpu_backend': 'cuda', 'gpu_uuid': value}
            with self.subTest(uuid=value), self.assertRaisesRegex(ValueError, 'gpu_uuid'):
                launch.plan_launch(self.manifest, self.root, 'host', 'gpu')
        for kind, backend in (('gpu', 'hip'), ('gpu', None), ('cpu', 'cuda')):
            job = {**self.job('island', kind, 2048), 'gpu_uuid': uuid}
            if backend is not None:
                job['gpu_backend'] = backend
            with self.subTest(kind=kind, backend=backend), self.assertRaisesRegex(ValueError, 'gpu_uuid'):
                launch.device_visibility(job)

    def test_main_forwards_only_admitted_cuda_uuid_to_the_worker_wrapper(self):
        uuid = 'GPU-5a30c18a-c75e-c22a-7663-74f2eb24a91d'
        self.jobs[:] = [{**self.job('gpu-0', 'gpu', 4096), 'gpu_backend': 'cuda',
                        'physical_gpu': 1, 'gpu_uuid': uuid, 'python': '/reviewed/python', 'minimum_free_gib': 50}]
        self.manifest['deadline_utc'] = '2026-10-12T08:00:00Z'
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(self.manifest), encoding='utf-8')
        real_read = Path.read_text
        def read(instance, *args, **kwargs):
            if str(instance).replace('\\', '/') == '/proc/meminfo':
                return 'MemAvailable: 33554432 kB\n'
            return real_read(instance, *args, **kwargs)
        with patch.object(launch, '__file__', str(self.root / 'ops' / 'launch_chat_sweep.py')), \
             patch.object(launch.socket, 'gethostname', return_value='host'), \
             patch.object(launch.os, 'cpu_count', return_value=8), \
             patch.object(Path, 'read_text', read), \
             patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,2', 'HIP_VISIBLE_DEVICES': '3',
                                     'ROCR_VISIBLE_DEVICES': '2', 'GPU_DEVICE_ORDINAL': '2'}), \
             patch.object(sys, 'argv', ['launch_chat_sweep.py', '--manifest', str(path), '--kind', 'gpu']), \
             patch.object(launch.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            launch.main()
        run.assert_called_once()
        env = run.call_args.kwargs['env']
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], uuid)
        self.assertEqual(env['HIP_VISIBLE_DEVICES'], '')
        self.assertNotIn('ROCR_VISIBLE_DEVICES', env)
        self.assertNotIn('GPU_DEVICE_ORDINAL', env)
        self.assertEqual(run.call_args.args[0], ['bash', 'ops/run_world_long.sh',
            self.jobs[0]['config'], self.jobs[0]['run_dir'], self.manifest['deadline_utc']])

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

    def test_optional_manifest_stage_seconds_requires_bounded_integer(self):
        for value in (30, 1800, 7200):
            self.jobs[0]['stage_seconds'] = value
            launch.plan_launch(self.manifest, self.root, 'host', 'cpu')
        for value in (29, 7201, 30., True, None, '7200', float('nan')):
            self.jobs[0]['stage_seconds'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'stage_seconds'):
                launch.plan_launch(self.manifest, self.root, 'host', 'cpu')

    def test_manifest_stage_override_and_default_are_forwarded_without_deadline_change(self):
        self.jobs[0]['stage_seconds'] = 7200
        for job in self.jobs:
            job.update(python='/usr/bin/python3', minimum_free_gib=50, physical_gpu=0)
        self.manifest['deadline_utc'] = '2026-10-12T08:00:00Z'
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(self.manifest), encoding='utf-8')
        real_read = Path.read_text
        def read(instance, *args, **kwargs):
            if str(instance).replace('\\', '/') == '/proc/meminfo':
                return 'MemAvailable: 33554432 kB\n'
            return real_read(instance, *args, **kwargs)
        with patch.object(launch, '__file__', str(self.root / 'ops' / 'launch_chat_sweep.py')), \
             patch.object(launch.socket, 'gethostname', return_value='host'), \
             patch.object(launch.os, 'cpu_count', return_value=8), \
             patch.object(Path, 'read_text', read), \
             patch.dict(os.environ, {'GOLEM_STAGE_SECONDS': '9999'}), \
             patch.object(sys, 'argv', ['launch_chat_sweep.py', '--manifest', str(path), '--kind', 'cpu']), \
             patch.object(launch.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            launch.main()
        self.assertEqual(run.call_count, 2)
        self.assertEqual([call.kwargs['env']['GOLEM_STAGE_SECONDS'] for call in run.call_args_list], ['7200', '1800'])
        for call in run.call_args_list:
            self.assertEqual(call.args[0][1], 'ops/run_world_long.sh')
            self.assertEqual(call.args[0][-1], self.manifest['deadline_utc'])

    def test_shell_stage_validation_executes_without_launching_any_service(self):
        git_bash = Path('C:/Program Files/Git/bin/bash.exe')
        bash = str(git_bash) if os.name == 'nt' and git_bash.is_file() else shutil.which('bash')
        if not bash:
            self.skipTest('Bash unavailable for the launcher validation check')
        source = Path(launch.__file__).resolve().parent / 'run_world_long.sh'
        script = source.read_text(encoding='utf-8')
        syntax = subprocess.run([bash, '-n', source.as_posix()], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        # Execute the real argument/env validation prefix only. The deadline
        # calculation and all service/file-launch side effects are excluded.
        prefix = script.split('# Start shutdown', 1)[0]
        isolated = self.root / 'ops' / 'validation-only.sh'
        isolated.parent.mkdir()
        isolated.write_text(prefix + '\nprintf "%s\\n" "$STAGE_SECONDS"\n', encoding='utf-8', newline='\n')
        base = dict(os.environ, GOLEM_PYTHON='/usr/bin/true', GOLEM_THREADS='2', GOLEM_MEMORY_MIB='1024', GOLEM_RESUME='0')
        base.pop('GOLEM_STAGE_SECONDS', None)
        for value, expected in ((None, '1800'), ('30', '30'), ('7200', '7200'), ('0030', '30'),
                                ('29', None), ('7201', None), ('30.5', None), ('nan', None),
                                ('-30', None), ('999999999999999999', None)):
            environment = dict(base)
            if value is not None:
                environment['GOLEM_STAGE_SECONDS'] = value
            result = subprocess.run([bash, isolated.as_posix(), 'config', 'run', '2026-10-12T08:00:00Z'],
                                    env=environment, capture_output=True, text=True)
            with self.subTest(value=value):
                if expected is not None:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)
                else:
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('GOLEM_STAGE_SECONDS', result.stderr)


if __name__ == '__main__':
    unittest.main()
