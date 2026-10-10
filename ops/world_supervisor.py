"""Supervise one owned Golem island until an explicit UTC deadline.

The parent stays stdlib-only. Torch inspection, warm-seed condensation and TRAIN
repair generation run in short-lived helper processes that finish before the
trainer starts. Only process groups launched by this supervisor are signalled.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import uuid

PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = 'golem-world-supervisor-v1'
GIB = 1024 ** 3
HEAVY_FILES = ('semantic-training.pt', 'best.pt', 'ordinary_champion.pt',
               'independent_champion.pt', 'evolution_champion.pt')


def parse_deadline(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError('deadline must explicitly specify UTC, for example 2026-10-12T08:00:00Z')
    return parsed.astimezone(timezone.utc)


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def contained(root, raw, must_exist=False):
    root = Path(root).resolve()
    path = Path(raw)
    path = (path if path.is_absolute() else root / path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'path must remain inside the dedicated workspace: {raw}')
    if must_exist and not path.is_file():
        raise ValueError(f'required file missing: {path}')
    return path


def write_toml(path, values):
    def scalar(value):
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if type(value) is bool:
            return 'true' if value else 'false'
        if type(value) in (int, float) and math.isfinite(value):
            return repr(value)
        if isinstance(value, list):
            return '[' + ', '.join(scalar(item) for item in value) + ']'
        raise ValueError('stage configuration contains an unsupported TOML value')
    lines = []
    def table(value, prefix=''):
        for key in value:
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
                raise ValueError('invalid TOML key')
        for key, item in value.items():
            if item is not None and not isinstance(item, dict):
                lines.append(key + ' = ' + scalar(item))
        for key, item in value.items():
            if isinstance(item, dict):
                name = prefix + key
                lines.extend(['', '[' + name + ']'])
                table(item, name + '.')
    table(values)
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def tree_bytes(root):
    total = 0
    for base, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(base) / name).is_symlink()]
        for name in files:
            path = Path(base) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
    return total


def process_rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0


def owned_memory_bytes(child_pid=None):
    # In a dedicated systemd service this includes all descendants. Never use a
    # shared root cgroup as if it were the island's own resource usage.
    if os.environ.get('INVOCATION_ID'):
        try:
            line = next(row for row in Path('/proc/self/cgroup').read_text().splitlines() if row.startswith('0::'))
            relative = line[3:].lstrip('/')
            if relative:
                group = (Path('/sys/fs/cgroup') / relative).resolve()
                if group.is_relative_to(Path('/sys/fs/cgroup')):
                    return int((group / 'memory.current').read_text()), 'dedicated_systemd_cgroup'
        except (FileNotFoundError, PermissionError, ValueError, StopIteration):
            pass
    return process_rss(os.getpid()) + (process_rss(child_pid) if child_pid else 0), 'supervisor_and_direct_child_rss'


def process_start(pid):
    try:
        # comm may contain spaces or parentheses; fields after the final ')' are
        # state (field 3) onward. Start time is field 22.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, PermissionError, IndexError):
        return None


def terminate_owned(child, hard_deadline, grace=15.):
    """Signal exactly the session/process group created by our Popen call."""
    if child.poll() is not None:
        return child.returncode
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return child.returncode
    remaining = max(0., min(15., grace, hard_deadline - time.time()))
    try:
        return child.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return child.wait(timeout=5)


def quality_key(summary):
    quality = summary['quality']
    return (quality['false_memory_claims'] == 0, quality['score'],
            quality['whole_episode_pass_rate'], -quality['reply_loss'])


def candidate_summary(checkpoint):
    """Helper-only Torch operation. The supervising process never calls this."""
    import torch
    from english_world.semantic_model import load_semantic_checkpoint
    torch.set_num_threads(1)
    path = Path(checkpoint)
    if not 0 < path.stat().st_size <= 512 * 1024 * 1024:
        raise ValueError('semantic candidate checkpoint exceeds the 512 MiB limit')
    model, metadata = load_semantic_checkpoint(path, map_location='cpu')
    if metadata.get('selection_split') != 'validation':
        raise ValueError('candidate must be selected exclusively on validation')
    protocol = metadata.get('validation_protocol_hash')
    if not isinstance(protocol, str) or not re.fullmatch(r'[0-9a-f]{64}', protocol):
        raise ValueError('candidate lacks a frozen validation protocol hash')
    validation = metadata['validation']
    rollout, teacher = validation['rollout'], validation['teacher_forced']
    if rollout.get('split') != 'validation':
        raise ValueError('candidate rollout is not validation')
    quality = {'false_memory_claims': rollout['false_memory_claims'], 'score': rollout['score'],
               'whole_episode_pass_rate': rollout['whole_episode_pass_rate'], 'reply_loss': teacher['reply_loss']}
    if (type(quality['false_memory_claims']) is not int or quality['false_memory_claims'] < 0 or
            any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in quality.values()) or
            not 0 <= quality['score'] <= 1 or not 0 <= quality['whole_episode_pass_rate'] <= 1 or quality['reply_loss'] < 0):
        raise ValueError('invalid candidate quality metrics')
    return {'checkpoint_sha256': sha256(path), 'bytes': path.stat().st_size,
            'schema': 'golem-semantic-gru-v1', 'quality': quality,
            'validation_protocol_hash': protocol, 'strategy': metadata.get('strategy'),
            'updates': metadata.get('global_updates'), 'recipe': metadata.get('recipe', {}),
            'model_config': model.config.to_dict(), 'parameter_count': model.parameter_count}


def condensed_seed(source, destination, provenance, seed=1):
    """Write a small, flat warm-start artifact without recursive source metadata."""
    import torch
    from english_world.semantic_model import warm_start_semantic, save_semantic_checkpoint
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(seed)
    before = sha256(source)
    model, _ = warm_start_semantic(source)
    metadata = {'schema': 'golem-flat-warm-seed-v1', 'parent_checkpoint_sha256': before,
                'parent_run': str(provenance.get('parent_run', 'initial_seed')),
                'parent_quality': provenance.get('quality'), 'optimizer_reset': True,
                'note': 'Only flat parent identity is inherited; full parent metadata remains in its original artifact.'}
    save_semantic_checkpoint(destination, model, metadata)
    if sha256(source) != before:
        raise ValueError('warm-start source changed while condensing')
    return {'path': str(Path(destination).resolve()), 'sha256': sha256(destination),
            'parent_checkpoint_sha256': before, 'metadata_bytes': len(json.dumps(metadata).encode())}


def prune_stages(run_dir, supervisor_id, keep=2):
    """Unlink ONLY recognized files in our marked old stages; never recursive rm."""
    root = Path(run_dir).resolve()
    stages = root / 'stages'
    if not stages.is_dir() or stages.is_symlink():
        return []
    owned = []
    for directory in stages.iterdir():
        if not re.fullmatch(r'stage-[0-9]{6}', directory.name) or directory.is_symlink() or not directory.is_dir():
            continue
        resolved = directory.resolve()
        if not resolved.is_relative_to(stages.resolve()):
            raise ValueError('stage escaped supervisor workspace')
        marker = directory / 'ownership.json'
        if marker.is_symlink() or not marker.is_file():
            continue
        identity = json.loads(marker.read_text(encoding='utf-8'))
        if identity.get('supervisor_id') == supervisor_id and identity.get('stage') == directory.name:
            owned.append(directory)
    # A failed stage may contain no trainer state. Do not let two such failures
    # evict the last two actual resumable states from earlier stages.
    full_states = [stage for stage in sorted(owned)
                   if (stage / 'run' / 'semantic-training.pt').is_file()
                   and not (stage / 'run' / 'semantic-training.pt').is_symlink()]
    retained_states = set(full_states[-keep:]) if keep else set()
    removed = []
    for stage in sorted(owned)[:-keep] if keep else sorted(owned):
        if (stage / 'summary.json').is_symlink() or not (stage / 'summary.json').is_file():
            continue  # Never prune an active/uncommitted stage.
        # Keep research evidence and failures losslessly compressed; these are
        # not equivalent to the compact supervisor summary.
        archived = []
        for relative in ('run/report.json', 'run/status.json', 'run/population.json', 'trainer.log'):
            source = stage / relative
            if source.is_symlink() or not source.is_file():
                continue
            if not source.resolve().is_relative_to(stage.resolve()):
                raise ValueError('refusing unsafe stage report archival')
            destination = source.with_name(source.name + '.gz')
            if destination.is_symlink():
                raise ValueError('refusing to replace a symlink during report archival')
            digest = sha256(source)
            temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
            with source.open('rb') as incoming, temporary.open('xb') as raw:
                with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
                    shutil.copyfileobj(incoming, compressed, 1048576)
                raw.flush()
                os.fsync(raw.fileno())
            recovered = hashlib.sha256()
            with gzip.open(temporary, 'rb') as check:
                for block in iter(lambda: check.read(1048576), b''):
                    recovered.update(block)
            if recovered.hexdigest() != digest:
                raise ValueError('compressed stage evidence failed checksum verification')
            os.replace(temporary, destination)
            archived.append({'path': str(destination.relative_to(stage)), 'uncompressed_sha256': digest,
                             'uncompressed_bytes': source.stat().st_size, 'compressed_bytes': destination.stat().st_size})
            source.unlink()
        if archived:
            summary_path = stage / 'summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            summary.setdefault('archived_evidence', []).extend(archived)
            atomic_json(summary_path, summary)
        if stage in retained_states:
            continue
        names = [stage / 'warm-seed.pt']
        names += [stage / 'run' / name for name in HEAVY_FILES]
        names += [Path(str(path) + '.tmp') for path in list(names)]
        for path in names:
            if path.is_symlink():
                continue
            if path.exists():
                if not path.resolve().is_relative_to(stage.resolve()) or not path.is_file():
                    raise ValueError('refusing unsafe stage artifact cleanup')
                path.unlink()
                removed.append(str(path.relative_to(root)))
    return removed


@contextmanager
def supervisor_lock(path):
    import fcntl
    with Path(path).open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another supervisor already owns this run') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class WorldSupervisor:
    def __init__(self, config, data, run_dir, deadline, *, project_root=PROJECT,
                 stage_seconds=1800., shutdown_grace=15., memory_limit_mib=2048,
                 min_free_gib=50., run_budget_gib=1., project_budget_gib=20.,
                 max_stages=1000, self_replay=True, replay_seconds=120., replay_episodes=32,
                 replay_samples=256, resume=False):
        self.root = Path(project_root).resolve()
        self.config_path = contained(self.root, config, True)
        self.data_path = contained(self.root, data, True)
        self.run_dir = contained(self.root, run_dir)
        self.deadline = parse_deadline(deadline).timestamp()
        for key, value in (('stage_seconds', stage_seconds), ('shutdown_grace', shutdown_grace),
                           ('memory_limit_mib', memory_limit_mib), ('min_free_gib', min_free_gib),
                           ('run_budget_gib', run_budget_gib), ('project_budget_gib', project_budget_gib),
                           ('replay_seconds', replay_seconds)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if not 30 <= stage_seconds <= 1800 or shutdown_grace > 15 or not 1 <= max_stages <= 1000:
            raise ValueError('stage_seconds must be 30..1800, grace <=15, and max_stages 1..1000')
        if not 1 <= replay_episodes <= 32 or not 1 <= replay_samples <= 512 or replay_seconds > 300:
            raise ValueError('replay bounds exceed the small TRAIN-only overlay budget')
        self.stage_seconds, self.grace = float(stage_seconds), float(shutdown_grace)
        self.memory_limit = int(memory_limit_mib * 1024 * 1024)
        self.min_free = int(min_free_gib * GIB)
        self.run_budget, self.project_budget = int(run_budget_gib * GIB), int(project_budget_gib * GIB)
        self.max_stages, self.self_replay = int(max_stages), bool(self_replay)
        self.replay_seconds, self.replay_episodes, self.replay_samples = replay_seconds, replay_episodes, replay_samples
        self.stop_requested, self.child = False, None
        self.interrupted_stage = None
        self.last_project_check, self.last_status = 0., 0.
        with self.config_path.open('rb') as handle:
            self.base = tomllib.load(handle)
        base_seed = int(self.base.get('seed', 97))
        self.validation_seed = int(self.base.get('validation_seed', 40097))
        self.validation_nll_seed = int(self.base.get('validation_nll_seed', 30097))
        self.validation_batch_size = int(self.base.get('validation_batch_size', 8))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.run_dir / 'supervisor-state.json'
        identity = {'base_config_sha256': sha256(self.config_path), 'base_data_sha256': sha256(self.data_path),
                    'deadline_utc': parse_deadline(deadline).isoformat().replace('+00:00', 'Z')}
        if resume:
            self.state = json.loads(state_path.read_text(encoding='utf-8'))
            if self.state.get('schema') != SCHEMA or any(self.state.get(key) != value for key, value in identity.items()):
                raise ValueError('supervisor resume identity/deadline mismatch')
            active = self.state.get('active_child')
            if active:
                current_start = process_start(active['pid'])
                if current_start is not None and (not active.get('start_ticks') or current_start == active['start_ticks']):
                    raise RuntimeError('previous owned child still exists; stop its managed unit before resuming')
            self.interrupted_stage = self.state.get('active_stage')
            self.state['active_child'] = None
            self.state['active_stage'] = None
        else:
            if any(self.run_dir.iterdir()):
                raise ValueError('supervisor run directory must be empty; use --resume explicitly')
            self.state = {'schema': SCHEMA, 'supervisor_id': uuid.uuid4().hex, **identity,
                          'started_utc': utc_now(), 'next_stage': 1, 'stages': [], 'global_best': None,
                          'consecutive_failures': 0, 'active_child': None, 'active_stage': None,
                          'helper_wall_seconds': 0.}
        self.state['status'] = 'running'

    def recover_interrupted_stage(self):
        """Journal a crashed stage only after the caller holds the run lock."""
        name = self.interrupted_stage
        if name is None:
            return
        if not isinstance(name, str) or not re.fullmatch(r'stage-[0-9]{6}', name):
            raise ValueError('invalid interrupted-stage identity')
        stage = contained(self.run_dir, Path('stages') / name)
        if stage.is_symlink() or not stage.is_dir():
            raise ValueError('interrupted stage is missing or unsafe')
        marker = stage / 'ownership.json'
        if marker.is_symlink() or not marker.is_file():
            raise ValueError('interrupted stage lacks an ownership marker')
        identity = json.loads(marker.read_text(encoding='utf-8'))
        if identity != {'supervisor_id': self.state['supervisor_id'], 'stage': name}:
            raise ValueError('interrupted stage does not belong to this supervisor')
        summary_path = stage / 'summary.json'
        if summary_path.is_symlink():
            raise ValueError('unsafe interrupted-stage summary')
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        else:
            summary = {'stage': name, 'status': 'interrupted', 'finished_utc': utc_now(), 'phases': [],
                       'recovery_note': 'Previous supervisor ended without a committed stage summary; its child no longer exists.'}
            pending = stage / 'run' / 'best.pt'
            if pending.is_file() and not pending.is_symlink() and pending.resolve().is_relative_to(stage):
                summary['pending_final_candidate'] = {'path': str(pending.relative_to(self.run_dir)),
                    'sha256': sha256(pending), 'note': 'Unpromoted candidate retained for explicit collection.'}
            atomic_json(summary_path, summary)
        if not any(row['stage'] == name for row in self.state['stages']):
            self.state['stages'].append(summary)
        self.interrupted_stage = None

    def stopping(self):
        return self.stop_requested or time.time() >= self.deadline - self.grace

    def save(self):
        self.state['updated_utc'] = utc_now()
        atomic_json(self.run_dir / 'supervisor-state.json', self.state)
        self.status()

    def status(self):
        value = {key: self.state.get(key) for key in ('schema', 'supervisor_id', 'status', 'deadline_utc',
                   'active_stage', 'active_child', 'global_best', 'consecutive_failures', 'helper_wall_seconds')}
        value.update(updated_utc=utc_now(), completed_stages=len(self.state['stages']),
                     seconds_remaining=max(0., self.deadline - time.time()),
                     latest_stage=self.state['stages'][-1] if self.state['stages'] else None,
                     final_test_policy='No stage uses test labels; final test is deferred until after the global deadline.')
        atomic_json(self.run_dir / 'status.json', value)
        self.last_status = time.monotonic()

    def resource_check(self):
        free = shutil.disk_usage(self.root).free
        used = tree_bytes(self.run_dir)
        if free < self.min_free or used >= self.run_budget:
            prune_stages(self.run_dir, self.state['supervisor_id'], keep=2)
            free, used = shutil.disk_usage(self.root).free, tree_bytes(self.run_dir)
        if free < self.min_free:
            raise RuntimeError('free disk fell below the configured reserve')
        if used >= self.run_budget:
            raise RuntimeError('supervisor run exceeded its storage budget')
        now = time.monotonic()
        if now - self.last_project_check >= 60:
            if tree_bytes(self.root) >= self.project_budget:
                raise RuntimeError('dedicated Golem project reached its global storage budget')
            self.last_project_check = now
        memory, scope = owned_memory_bytes(self.child.pid if self.child else None)
        if memory > self.memory_limit:
            raise RuntimeError(f'owned memory exceeded configured limit ({scope})')
        return {'free_disk_bytes': free, 'run_bytes': used, 'owned_memory_bytes': memory, 'memory_scope': scope}

    def run_child(self, command, log_path, hard_deadline, phase, helper=False):
        if self.stopping():
            raise InterruptedError('supervisor deadline/shutdown reached')
        environment = dict(os.environ)
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        if helper:
            environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                               HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
        started, peak = time.monotonic(), 0
        reason = 'exited'
        with Path(log_path).open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=self.root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            self.child = child
            self.state['active_child'] = {'pid': child.pid, 'start_ticks': process_start(child.pid), 'phase': phase}
            self.save()
            try:
                while child.poll() is None:
                    if self.stop_requested or time.time() >= min(hard_deadline, self.deadline) - self.grace:
                        reason = 'interrupted' if self.stop_requested and time.time() < self.deadline - self.grace else 'budget_stop'
                        terminate_owned(child, min(hard_deadline, self.deadline), self.grace)
                        break
                    try:
                        sample = self.resource_check()
                        peak = max(peak, sample['owned_memory_bytes'])
                    except RuntimeError as exc:
                        reason = str(exc)
                        terminate_owned(child, min(hard_deadline, self.deadline), self.grace)
                        break
                    if time.monotonic() - self.last_status >= 10:
                        self.status()
                    try:
                        child.wait(timeout=min(2., max(.01, hard_deadline - self.grace - time.time())))
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if child.poll() is None:
                    terminate_owned(child, min(hard_deadline, self.deadline), self.grace)
                child.wait()
                self.child = None
                self.state['active_child'] = None
                elapsed = time.monotonic() - started
                if helper:
                    self.state['helper_wall_seconds'] += elapsed
                self.save()
        return {'phase': phase, 'returncode': child.returncode, 'reason': reason,
                'wall_seconds': elapsed, 'peak_owned_memory_bytes': peak}

    def helper(self, kind, stage, payload, seconds=120):
        request = stage / (kind + '-request.json')
        result = stage / (kind + '-result.json')
        hard = min(self.deadline, time.time() + seconds)
        payload = {**payload, 'result': str(result), 'deadline_epoch': hard - self.grace}
        atomic_json(request, payload)
        phase = self.run_child([sys.executable, '-B', str(Path(__file__).resolve()), '--helper', kind,
                                '--request', str(request)], stage / (kind + '.log'), hard, kind, helper=True)
        if phase['returncode'] != 0 or phase['reason'] not in ('exited', 'budget_stop') or not result.is_file():
            if self.stopping():
                raise InterruptedError('helper stopped at supervisor deadline')
            raise RuntimeError(f'{kind} helper failed: {phase}')
        value = json.loads(result.read_text(encoding='utf-8'))
        return value, phase

    def promote(self, stage):
        candidate = stage / 'run' / 'best.pt'
        if not candidate.is_file():
            return None, None
        summary, phase = self.helper('inspect', stage, {'checkpoint': str(candidate)}, seconds=120)
        previous = self.state['global_best']
        if previous and previous['validation_protocol_hash'] != summary['validation_protocol_hash']:
            raise ValueError('stage changed the frozen validation protocol; candidate rejected')
        if previous is None or quality_key(summary) > quality_key(previous):
            destination = self.run_dir / 'global_best.pt'
            temporary = self.run_dir / ('.global-best-' + uuid.uuid4().hex + '.pt')
            shutil.copyfile(candidate, temporary)
            if sha256(temporary) != summary['checkpoint_sha256']:
                temporary.unlink(missing_ok=True)
                raise ValueError('stage champion changed while being promoted')
            os.replace(temporary, destination)
            summary.update(parent_run=str(stage.relative_to(self.run_dir)), promoted_utc=utc_now())
            self.state['global_best'] = summary
            atomic_json(self.run_dir / 'global_best.json', summary)
            self.save()
            return summary, phase
        return {**summary, 'promoted': False}, phase

    def stage_config(self, stage, number):
        config = json.loads(json.dumps(self.base))
        config.update(seed=int(self.base.get('seed', 97)) + 1009 * number,
                      validation_seed=self.validation_seed, validation_nll_seed=self.validation_nll_seed,
                      validation_batch_size=self.validation_batch_size, evaluate_test_on_complete=False)
        config.pop('init_checkpoint_sha256', None)
        config.pop('repair_overlay', None)
        phases = []
        parent = self.run_dir / 'global_best.pt' if self.state['global_best'] else None
        if parent is None and config.get('init_checkpoint'):
            parent = contained(self.root, config['init_checkpoint'], True)
        if parent is not None:
            provenance = self.state['global_best'] or {'parent_run': str(parent.relative_to(self.root)), 'quality': None}
            seed, phase = self.helper('seed', stage, {'checkpoint': str(parent), 'destination': str(stage / 'warm-seed.pt'),
                    'provenance': provenance, 'seed': config['seed']}, seconds=120)
            phases.append(phase)
            config['init_checkpoint'] = str(contained(stage, seed['path'], True))
            if self.state['global_best']:
                recipe = self.state['global_best'].get('recipe', {})
                for key in ('learning_rate', 'dropout', 'intent_loss_weight', 'first_response_weight'):
                    if key in recipe:
                        config[key] = recipe[key]
        if self.self_replay and self.state['global_best']:
            overlay, phase = self.helper('repair', stage, {'checkpoint': config['init_checkpoint'], 'data': str(self.data_path),
                'output': str(stage / 'repair.json'), 'seed': config['seed'],
                'max_episodes': self.replay_episodes, 'max_samples': self.replay_samples}, seconds=self.replay_seconds)
            phases.append(phase)
            config['repair_overlay'] = str(contained(stage, overlay['path'], True))
        return config, phases

    def run(self):
        try:
            self.recover_interrupted_stage()
            self.save()
            while self.state['next_stage'] <= self.max_stages:
                if self.stopping() or self.deadline - time.time() <= self.grace + 20:
                    self.state['status'] = 'interrupted' if self.stop_requested and time.time() < self.deadline - self.grace else 'deadline_reached'
                    break
                number = self.state['next_stage']
                name = f'stage-{number:06d}'
                stage = self.run_dir / 'stages' / name
                stage.mkdir(parents=True, exist_ok=False)
                atomic_json(stage / 'ownership.json', {'supervisor_id': self.state['supervisor_id'], 'stage': name})
                self.state['next_stage'] += 1
                self.state['active_stage'] = name
                self.save()
                summary = {'stage': name, 'started_utc': utc_now(), 'phases': [], 'status': 'running'}
                try:
                    self.resource_check()
                    config, phases = self.stage_config(stage, number)
                    summary['phases'].extend(phases)
                    if self.stopping():
                        raise InterruptedError('deadline reached during stage preparation')
                    hard = min(self.deadline, time.time() + self.stage_seconds)
                    config['max_seconds'] = max(1., hard - time.time() - self.grace - 5.)
                    configuration = stage / 'config.toml'
                    write_toml(configuration, config)
                    phase = self.run_child([sys.executable, '-B', '-m', 'english_world', 'neural', 'world-train',
                        '--data', str(self.data_path), '--config', str(configuration), '--run-dir', str(stage / 'run')],
                        stage / 'trainer.log', hard, 'training')
                    summary['phases'].append(phase)
                    if self.stopping():
                        summary['status'] = 'deadline_checkpoint' if not self.stop_requested else 'interrupted'
                        raise InterruptedError('supervisor stopped after saving child checkpoint')
                    report_path = stage / 'run' / 'report.json'
                    child_report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else {}
                    summary['trainer_status'] = child_report.get('status')
                    if phase['returncode'] != 0 or phase['reason'] not in ('exited', 'budget_stop') or child_report.get('status') not in ('complete', 'time_limit', 'interrupted'):
                        raise RuntimeError('training stage failed or did not publish a complete status report')
                    candidate, promotion_phase = self.promote(stage)
                    if promotion_phase:
                        summary['phases'].append(promotion_phase)
                    summary['candidate'] = candidate
                    if candidate is None:
                        raise RuntimeError('training stage produced no validation-selected champion')
                    summary['status'] = 'complete'
                    self.state['consecutive_failures'] = 0
                except InterruptedError:
                    summary['status'] = 'deadline_checkpoint' if time.time() >= self.deadline - self.grace else 'interrupted'
                    self.state['status'] = 'deadline_reached' if time.time() >= self.deadline - self.grace else 'interrupted'
                    pending = stage / 'run' / 'best.pt'
                    if pending.is_file() and not pending.is_symlink():
                        summary['pending_final_candidate'] = {'path': str(pending.relative_to(self.run_dir)),
                            'sha256': sha256(pending), 'note': 'Preserved for post-deadline collection; no Torch helper started inside shutdown reserve.'}
                except Exception as exc:
                    summary.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)[:2000]})
                    self.state['consecutive_failures'] += 1
                summary['finished_utc'] = utc_now()
                atomic_json(stage / 'summary.json', summary)
                self.state['stages'].append(summary)
                self.state['active_stage'] = None
                prune_stages(self.run_dir, self.state['supervisor_id'], keep=2)
                self.save()
                if self.state['status'] in ('deadline_reached', 'interrupted'):
                    break
                if self.state['consecutive_failures'] >= 3:
                    self.state['status'] = 'failed'
                    break
                if summary['status'] == 'failed':
                    until = min(time.time() + 5 * 2 ** (self.state['consecutive_failures'] - 1), self.deadline - self.grace)
                    while time.time() < until and not self.stop_requested:
                        time.sleep(min(1., until - time.time()))
            else:
                self.state['status'] = 'stage_limit_reached'
        except Exception as exc:
            self.state.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)[:2000]})
        finally:
            if self.child is not None:
                terminate_owned(self.child, self.deadline, self.grace)
                self.child = None
            self.state['active_child'] = None
            self.state['finished_utc'] = utc_now()
            self.save()
        return self.state


def helper_main(kind, request):
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    payload = json.loads(Path(request).read_text(encoding='utf-8'))
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    def should_stop():
        return stopping[0] or time.time() >= payload['deadline_epoch']
    if should_stop():
        return 2
    if kind == 'inspect':
        result = candidate_summary(payload['checkpoint'])
    elif kind == 'seed':
        result = condensed_seed(payload['checkpoint'], payload['destination'], payload['provenance'], payload['seed'])
    else:
        from english_world.world_replay import build_repair_overlay
        result = build_repair_overlay(payload['checkpoint'], payload['data'], payload['output'],
                    seed=payload['seed'], max_episodes=payload['max_episodes'], max_samples=payload['max_samples'],
                    should_stop=should_stop)
    if should_stop():
        return 2
    atomic_json(payload['result'], result)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper', choices=('inspect', 'seed', 'repair'))
    parser.add_argument('--request')
    parser.add_argument('--deadline', help='Explicit UTC deadline, e.g. 2026-10-12T08:00:00Z')
    parser.add_argument('--config', '--config-base', dest='config')
    parser.add_argument('--data')
    parser.add_argument('--run-dir')
    parser.add_argument('--stage-seconds', type=float, default=1800.)
    parser.add_argument('--shutdown-grace', type=float, default=15.)
    parser.add_argument('--memory-limit-mib', type=float, default=2048.)
    parser.add_argument('--min-free-gib', type=float, default=50.)
    parser.add_argument('--run-budget-gib', type=float, default=1.)
    parser.add_argument('--project-budget-gib', type=float, default=20.)
    parser.add_argument('--max-stages', type=int, default=1000)
    parser.add_argument('--self-replay', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--replay-seconds', type=float, default=120.)
    parser.add_argument('--replay-episodes', type=int, default=32)
    parser.add_argument('--replay-samples', type=int, default=256)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.helper:
        if not args.request:
            parser.error('--helper requires --request')
        return helper_main(args.helper, args.request)
    if os.name != 'posix':
        parser.error('the owned-process-group supervisor runs on Linux/POSIX')
    if not all((args.deadline, args.config, args.data, args.run_dir)):
        parser.error('--deadline, --config, --data and --run-dir are required')
    supervisor = WorldSupervisor(args.config, args.data, args.run_dir, args.deadline,
        stage_seconds=args.stage_seconds, shutdown_grace=args.shutdown_grace,
        memory_limit_mib=args.memory_limit_mib, min_free_gib=args.min_free_gib,
        run_budget_gib=args.run_budget_gib, project_budget_gib=args.project_budget_gib,
        max_stages=args.max_stages, self_replay=args.self_replay, replay_seconds=args.replay_seconds,
        replay_episodes=args.replay_episodes, replay_samples=args.replay_samples, resume=args.resume)
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda *_: setattr(supervisor, 'stop_requested', True))
    try:
        with supervisor_lock(supervisor.run_dir / 'supervisor.lock'):
            state = supervisor.run()
        print(json.dumps({'status': state['status'], 'stages': len(state['stages']),
                          'deadline_utc': state['deadline_utc'], 'active_child': state['active_child']}, indent=2))
        return 1 if state['status'] == 'failed' else 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
