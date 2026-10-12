"""Capture stopped semantic-world runs; never select, train, or evaluate a model.

Host-local JSON interfaces (stdlib only)::

    python -B ops/world_final_collect.py capture --manifest manifest.json \
        --host specht32 --root /home/ender/golem --output /safe/new-directory
    python -B ops/world_final_collect.py verify --output /safe/new-directory

Capture requires the manifest UTC deadline to have passed and every local owned
receipt unit, cgroup and recorded child to be stopped. There is no override flag.
The fresh output is atomically published with snapshot.tar.gz, its per-file
snapshot-manifest.json, allocation-manifest.json and verification.json. Corpus bytes are copied opaquely;
no corpus examples or tensor checkpoints are parsed. Missing historical files
are reported, never replaced. Original candidate hashes are retained for a
separate, validation-only selector. No source cleanup or service actions occur.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import tarfile
import tomllib
import uuid

HOST_ROOTS = {name: '/home/ender/golem' for name in ('specht32', 'falke64', 'adler40')}
HOST_ROOTS['knecht24'] = '/tmp/golem-chat-20261010'
SCHEMA = 'golem-world-final-snapshot-v1'
STAGE = re.compile(r'stage-[0-9]{6}')
UNIT = re.compile(r'golem-world-long-[A-Za-z0-9_.@-]+\.service')
MAX_BYTES = 32 * 1024**3


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def deadline(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError('manifest deadline must explicitly be UTC')
    return parsed


def no_links(path):
    path = Path(os.path.abspath(path))
    for component in [*reversed(path.parents), path]:
        if component.is_symlink() or getattr(component, 'is_junction', lambda: False)():
            raise ValueError(f'symlink/junction forbidden: {component}')
    return path


def inside(root, raw):
    """Reject lexical traversal as well as resolved escapes, including parents."""
    raw = raw.as_posix() if isinstance(raw, Path) else str(raw)
    if '\\' in raw or '..' in PurePosixPath(raw).parts:
        raise ValueError(f'unsafe path: {raw}')
    path = Path(raw)
    path = no_links(path if path.is_absolute() else root / path)
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'path escapes owned project: {raw}')
    return path


def walk_error(error):
    raise error  # A permission-denied subtree is not evidence of emptiness.


def read_json(path):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError(f'oversized metadata: {path}')
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as handle:
            raw = handle.read(64 * 1024**2 + 1)
        if len(raw) > 64 * 1024**2:
            raise ValueError(f'oversized decompressed metadata: {path}')
        value = json.loads(raw)
    else:
        value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'metadata must be a JSON object: {path}')
    return value


def validate_manifest(manifest, host, root):
    if manifest.get('schema') != 'golem-semantic-world-sweep-v1' or manifest.get('mode') != 'supervised_until_deadline':
        raise ValueError('expected supervised semantic-world allocation manifest')
    deadline(manifest.get('deadline_utc'))
    if host not in HOST_ROOTS or root != no_links(HOST_ROOTS[host]):
        raise ValueError('host/project root is not an admitted Golem location')
    result, identities = [], set()
    for category in ('jobs', 'suspended_jobs', 'retired_jobs'):
        jobs = manifest.get(category, [])
        if not isinstance(jobs, list):
            raise ValueError('manifest categories must be job lists')
        for job in jobs:
            if not isinstance(job, dict):
                raise ValueError('manifest job must be an object')
            identity, name = job.get('id'), job.get('host')
            if (name not in HOST_ROOTS or not isinstance(identity, str) or
                    not re.fullmatch(re.escape(name) + r'-world-[a-z0-9-]{1,96}', identity) or identity in identities):
                raise ValueError('invalid or duplicate manifest identity')
            identities.add(identity)
            if job.get('root') != HOST_ROOTS[name] or job.get('run_dir') != f'runs/world-long/{identity}':
                raise ValueError('manifest root/run identity mismatch')
            if job.get('config') != f'runs/world-long-configs/{identity}.toml':
                raise ValueError('manifest config identity mismatch')
            if name == host:
                inside(root, job['run_dir'])
                inside(root, job['config'])
                result.append({**job, 'category': category})
    if not result:
        raise ValueError('manifest has no owned jobs on this host')
    return result


def inspect_unit(unit):
    """Read only this receipt's systemd identity and owned cgroup membership."""
    if not UNIT.fullmatch(unit):
        raise ValueError('invalid owned service receipt')
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '--property=LoadState,ActiveState,SubState,MainPID,ControlPID,ControlGroup'],
        text=True, capture_output=True, timeout=15, check=False)
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not all(k in values for k in ('LoadState', 'ActiveState', 'MainPID', 'ControlPID')):
        raise RuntimeError(f'cannot establish stopped service state: {unit}')
    if result.returncode and values['LoadState'] != 'not-found':
        raise RuntimeError(f'service inspection failed: {unit}')
    pids = []
    cgroup = values.get('ControlGroup')
    if cgroup:
        if not cgroup.startswith('/') or '..' in PurePosixPath(cgroup).parts or PurePosixPath(cgroup).name != unit:
            raise ValueError('unexpected owned cgroup identity')
        base = inside(Path('/sys/fs/cgroup'), cgroup.lstrip('/'))
        if base.exists():
            for directory, children, files in os.walk(base, followlinks=False, onerror=walk_error):
                no_links(directory)
                for child in children:
                    no_links(Path(directory) / child)
                if 'cgroup.procs' in files:
                    pids.extend(int(value) for value in (Path(directory) / 'cgroup.procs').read_text().split())
    values['cgroup_pids'] = sorted(set(pids))
    return values


def child_exists(child):
    if not child:
        return False
    pid = child.get('pid')
    if type(pid) is not int or pid <= 0:
        raise ValueError('invalid recorded child PID')
    path = Path('/proc') / str(pid) / 'stat'
    try:
        current = path.read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return False
    # Missing start identity must fail closed if that PID exists.
    return not child.get('start_ticks') or str(child['start_ticks']) == current


def stopped_jobs(root, jobs, inspect=inspect_unit, child_check=child_exists):
    rows = []
    for job in jobs:
        run = inside(root, job['run_dir'])
        state = read_json(inside(root, run / 'supervisor-state.json'))
        receipt = inside(root, f"runs/world-long-launches/{job['id']}.unit")
        unit = receipt.read_text(encoding='utf-8').strip()
        if not UNIT.fullmatch(unit) or job.get('last_owned_unit', unit) != unit:
            raise ValueError('receipt disagrees with owned manifest service')
        service = inspect(unit)
        if (service.get('ActiveState') not in ('inactive', 'failed') or
                str(service.get('MainPID')) != '0' or str(service.get('ControlPID')) != '0' or
                service.get('cgroup_pids') or child_check(state.get('active_child'))):
            raise RuntimeError(f"owned worker still active or unverified: {job['id']}")
        rows.append({'id': job['id'], 'unit': unit, 'service': service,
                     'recorded_child': state.get('active_child'), 'recorded_child_absent': True})
    return rows


class Inventory:
    def __init__(self, root):
        self.root, self.files, self.missing, self.candidates = root, {}, [], []
        self.observed = {}

    def add(self, raw, role, required=True):
        path = inside(self.root, raw)
        if not path.exists():
            if required:
                raise ValueError(f'required artifact missing: {path}')
            missing = {'path': path.relative_to(self.root).as_posix(), 'role': role,
                       'reason': 'missing_or_pruned; no replacement made'}
            if missing not in self.missing:
                self.missing.append(missing)
            return None
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f'non-regular artifact: {path}')
        name = path.relative_to(self.root).as_posix()
        self.files.setdefault(name, {'path': path, 'roles': set()})['roles'].add(role)
        return path

    def json(self, raw, role):
        path = self.add(raw, role)
        before = digest(path)
        value = read_json(path)
        if digest(path) != before:
            raise ValueError('metadata changed while reading')
        self.observed[path.relative_to(self.root).as_posix()] = before
        return value

    def tree(self, raw, role):
        base = inside(self.root, raw)
        for directory, children, files in os.walk(base, followlinks=False, onerror=walk_error):
            no_links(directory)
            for child in children:
                no_links(Path(directory) / child)
            for name in files:
                self.add(Path(directory) / name, role)

    def config(self, raw, required=True):
        path = self.add(raw, 'config', required)
        if path is None:
            return
        before = digest(path)
        config = tomllib.loads(path.read_text(encoding='utf-8'))
        self.observed[path.relative_to(self.root).as_posix()] = before
        for field in ('init_checkpoint', 'repair_overlay', 'teacher_overlay'):
            if config.get(field):
                dependency = self.add(config[field], field, required=required)
                expected = config.get(field + '_sha256')
                if dependency and expected and digest(dependency) != expected:
                    raise ValueError(f'{field} dependency hash mismatch')

    def candidate(self, raw, origin, expected=None, required=False, *, stage=None, stage_journaled=None):
        path = self.add(raw, 'candidate:' + origin, required)
        if path is None:
            return
        sha = digest(path)
        if expected is not None and (not re.fullmatch(r'[0-9a-f]{64}', str(expected)) or sha != expected):
            raise ValueError(f'candidate hash mismatch: {path}')
        self.observed[path.relative_to(self.root).as_posix()] = sha
        entry = {'path': path.relative_to(self.root).as_posix(), 'origin': origin,
                 'sha256': sha, 'stage': stage, 'stage_journaled': stage_journaled,
                 'model_validation': 'not_performed; separate weights_only semantic inspection required'}
        if entry not in self.candidates:
            self.candidates.append(entry)


def build_inventory(root, jobs, manifest, data):
    inv = Inventory(root)
    data_path = inv.add(data, 'frozen_corpus_opaque')
    data_sha = digest(data_path)
    inv.observed[data_path.relative_to(root).as_posix()] = data_sha
    provenance = inv.json(data_path.parent / 'manifest.json', 'corpus_provenance')
    for field, folder in (('licenses', data_path.parent), ('snapshots', data_path.parent / 'source_snapshot')):
        declared = provenance.get(field)
        if not isinstance(declared, dict) or not declared:
            raise ValueError('corpus manifest must declare licenses and source snapshots')
        for name, expected in declared.items():
            if not isinstance(name, str) or PurePosixPath(name).name != name or '\\' in name:
                raise ValueError('unsafe corpus provenance filename')
            path = inv.add(folder / name, 'corpus_' + field)
            if digest(path) != expected:
                raise ValueError('corpus license/source snapshot identity mismatch')
            inv.observed[path.relative_to(root).as_posix()] = expected
    inv.corpus_identity = {'path': data_path.relative_to(root).as_posix(), 'file_sha256': data_sha,
                          'manifest_path': (data_path.parent / 'manifest.json').relative_to(root).as_posix(),
                          'canonical_content_sha256': provenance.get('content_sha256')}
    summaries = []
    for job in jobs:
        run = inside(root, job['run_dir'])
        state = inv.json(run / 'supervisor-state.json', 'supervisor_state')
        if (state.get('schema') != 'golem-world-supervisor-v1' or
                deadline(state.get('deadline_utc')) != deadline(manifest['deadline_utc']) or
                state.get('base_data_sha256') != data_sha):
            raise ValueError('supervisor schema/deadline/corpus identity mismatch')
        config_path = inside(root, job['config'])
        if digest(config_path) != state.get('base_config_sha256'):
            raise ValueError('base config identity mismatch')
        inv.config(job['config'])
        inv.add(f"runs/world-long-launches/{job['id']}.unit", 'service_receipt')
        for name in ('status.json', 'supervisor.log'):
            inv.add(run / name, 'supervisor_audit', required=False)
        if (run / 'global_best.json').exists() or (run / 'global_best.pt').exists():
            best = inv.json(run / 'global_best.json', 'global_metadata')
            expected = best.get('checkpoint_sha256')
            if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
                raise ValueError('global metadata lacks an exact checkpoint hash')
            inv.candidate(run / 'global_best.pt', job['id'] + ':global', expected, True)
        else:
            inv.missing.append({'path': str(run.relative_to(root) / 'global_best.pt'), 'role': 'global', 'reason': 'no global checkpoint published'})
        ledger = state.get('stages', [])
        if not isinstance(ledger, list):
            raise ValueError('invalid stage ledger')
        names = [entry.get('stage') for entry in ledger]
        if state.get('active_stage'):
            names.append(state['active_stage'])
        if any(not isinstance(name, str) or not STAGE.fullmatch(name) for name in names):
            raise ValueError('unsafe stage ledger identity')
        journaled = set(names)
        # Owned orphan stages are preserved as recovery evidence, never silently selected.
        base = inside(root, run / 'stages')
        for path in base.iterdir() if base.exists() else []:
            no_links(path)
            if not path.is_dir() or not STAGE.fullmatch(path.name):
                raise ValueError(f'unexpected entry in owned stages: {path}')
            names.append(path.name)
        pending_entries = list(ledger)
        committed = []
        for name in sorted(set(names)):
            stage = inside(root, base / name)
            if not stage.exists():
                inv.missing.append({'path': stage.relative_to(root).as_posix(), 'role': 'historical_stage', 'reason': 'missing; ledger retained'})
                continue
            owner = inv.json(stage / 'ownership.json', 'stage_ownership')
            if owner.get('supervisor_id') != state.get('supervisor_id') or owner.get('stage') != name:
                raise ValueError('stage ownership mismatch')
            inv.tree(stage, 'owned_stage_artifact')
            if (stage / 'summary.json').exists():
                summary = inv.json(stage / 'summary.json', 'stage_summary')
                if summary.get('stage', name) != name:
                    raise ValueError('summary/stage identity mismatch')
                if summary.get('pending_final_candidate') and name in journaled:
                    pending_entries.append({**summary, 'stage': name})
            resumable = (stage / 'run' / 'semantic-training.pt').is_file()
            if (stage / 'config.toml').exists():
                inv.config(stage / 'config.toml', required=resumable)
            elif resumable:
                raise ValueError('resumable state lacks its stage config')
            for strategy in ('ordinary', 'evolution', 'independent'):
                checkpoint = stage / 'run' / (strategy + '_champion.pt')
                if checkpoint.exists() and name in journaled:
                    inv.candidate(checkpoint, job['id'] + ':strategy:' + strategy + ':' + name,
                                  stage=name, stage_journaled=True)
            report_path = inside(root, stage / 'run' / 'report.json')
            if not report_path.exists():
                report_path = inside(root, stage / 'run' / 'report.json.gz')
            if name in journaled and report_path.exists():
                committed.append((name, stage, report_path))
        # Only journalled/active stages may supply a latest candidate. Orphan
        # files remain recovery evidence. Stop at the newest committed report,
        # explicitly recording a missing best instead of selecting an older one.
        for name, stage, report_path in sorted(committed, reverse=True):
            report = inv.json(report_path, 'committed_stage_report')
            if (report.get('schema') == 'golem-semantic-training-v1' and report.get('error') is None and
                    report.get('status') in ('complete', 'time_limit', 'interrupted')):
                inv.candidate(stage / 'run' / 'best.pt', job['id'] + ':latest_committed:' + name,
                              stage=name, stage_journaled=True)
                break
        for entry in pending_entries:
            pending = entry.get('pending_final_candidate')
            if pending:
                expected_path = f"stages/{entry['stage']}/run/best.pt"
                if pending.get('path') != expected_path or not pending.get('sha256'):
                    raise ValueError('unsafe or unhashed pending candidate')
                inv.candidate(run / expected_path, job['id'] + ':journaled_pending:' + entry['stage'], pending['sha256'],
                              stage=entry['stage'], stage_journaled=True)
        summaries.append({'id': job['id'], 'category': job['category'], 'supervisor_id': state['supervisor_id'],
                          'status': state.get('status'), 'ledger_stages': len(ledger), 'captured_stage_directories': len(set(names)),
                          'active_stage_at_stop': state.get('active_stage'), 'ledger_stage_ids': [e['stage'] for e in ledger],
                          'owned_orphan_stage_ids': sorted(set(names) - journaled)})
    for directory in ('english_world', 'ops'):
        base = inside(root, directory)
        if base.exists():
            for path in base.iterdir():
                if path.suffix in ('.py', '.sh'):
                    inv.add(path, 'current_source_snapshot')
    for name in ('pyproject.toml', 'LICENSE', '.gitattributes'):
        if (root / name).exists():
            inv.add(name, 'current_source_identity')
    return inv, summaries


class HashReader:
    def __init__(self, stream):
        self.stream, self.sha = stream, hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.sha.update(value)
        return value


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def verify(output):
    output = no_links(output)
    record = read_json(no_links(output / 'snapshot-manifest.json'))
    archive = no_links(output / 'snapshot.tar.gz')
    allocation = no_links(output / 'allocation-manifest.json')
    if digest(allocation) != record['allocation_manifest_sha256'] or read_json(allocation) != record['allocation_manifest']:
        raise ValueError('allocation manifest identity mismatch')
    expected = record['files']
    if (not isinstance(expected, dict) or len(expected) > 100000 or
            any(type(v.get('bytes')) is not int or v['bytes'] < 0 for v in expected.values()) or
            sum(v['bytes'] for v in expected.values()) > MAX_BYTES):
        raise ValueError('invalid or oversized archive file inventory')
    if record.get('schema') != SCHEMA or digest(archive) != record['archive_sha256']:
        raise ValueError('archive identity mismatch')
    seen = set()
    with tarfile.open(archive, 'r:gz') as handle:
        for member in handle:
            if not member.isfile() or member.name in seen or member.name not in expected:
                raise ValueError('unexpected/duplicate/nonregular archive member')
            if PurePosixPath(member.name).is_absolute() or '..' in PurePosixPath(member.name).parts or '\\' in member.name:
                raise ValueError('unsafe archive member')
            wanted = expected[member.name]
            if member.size != wanted['bytes']:
                raise ValueError('archive member size mismatch')
            sha = hashlib.sha256()
            with handle.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024**2), b''):
                    sha.update(block)
            if sha.hexdigest() != wanted['sha256']:
                raise ValueError('archive member hash mismatch')
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('archive members missing')
    return {'schema': 'golem-world-final-verification-v1', 'verified': True, 'files': len(seen),
            'source_bytes': sum(f['bytes'] for f in expected.values()), 'archive_bytes': archive.stat().st_size,
            'archive_sha256': record['archive_sha256'], 'manifest_sha256': digest(output / 'snapshot-manifest.json')}


def capture(manifest_path, host, root, output, *, data='data/conversation-world/dataset.json',
            now=None, service_inspector=inspect_unit, child_check=child_exists):
    root = no_links(root)
    manifest_path = no_links(manifest_path)
    manifest_sha = digest(manifest_path)
    manifest = read_json(manifest_path)
    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None or current < deadline(manifest.get('deadline_utc')):
        raise RuntimeError('final capture refused before the explicit manifest deadline')
    jobs = validate_manifest(manifest, host, root)
    services = stopped_jobs(root, jobs, service_inspector, child_check)
    inv, runs = build_inventory(root, jobs, manifest, data)
    output = no_links(output)
    if output.exists() or not output.parent.is_dir() or output.is_relative_to(root / 'runs/world-long'):
        raise ValueError('output must be a fresh dedicated directory with an existing safe parent')
    if sum(v['path'].stat().st_size for v in inv.files.values()) > MAX_BYTES or len(inv.files) > 100000:
        raise ValueError('snapshot exceeds bounded capture budget')
    temporary = output.with_name(output.name + '.partial-' + uuid.uuid4().hex)
    temporary.mkdir()
    # Failed private staging is intentionally retained; no deletion is performed.
    archive = temporary / 'snapshot.tar.gz'
    files = {}
    with archive.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=1) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for name, item in sorted(inv.files.items()):
                    path = no_links(item['path'])
                    before = path.stat()
                    info = tar.gettarinfo(str(path), arcname=name)
                    if not info.isfile():
                        raise ValueError('source ceased to be a regular file')
                    with path.open('rb') as stream:
                        hashed = HashReader(stream)
                        tar.addfile(info, hashed)
                    sha = hashed.sha.hexdigest()
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                        raise ValueError('source changed during archive capture')
                    if digest(path) != sha or inv.observed.get(name, sha) != sha:
                        raise ValueError('source hash changed during capture')
                    files[name] = {'bytes': info.size, 'sha256': sha, 'roles': sorted(item['roles'])}
        raw.flush()
        os.fsync(raw.fileno())
    if digest(manifest_path) != manifest_sha:
        raise ValueError('allocation manifest changed during capture')
    stopped_jobs(root, jobs, service_inspector, child_check)
    for name, expected_sha in inv.observed.items():
        if digest(inside(root, name)) != expected_sha:
            raise ValueError('checkpoint/metadata identity changed before publication')
    record = {'schema': SCHEMA, 'host': host, 'root': str(root), 'deadline_utc': manifest['deadline_utc'],
              'captured_utc': datetime.now(timezone.utc).isoformat(), 'allocation_manifest_sha256': manifest_sha,
              'allocation_manifest': manifest, 'runs': runs, 'service_checks': services,
              'corpus_identity': inv.corpus_identity,
              'candidates': inv.candidates, 'missing_historical_artifacts': inv.missing,
              'files': files, 'archive_sha256': digest(archive),
              'policy': 'Original bytes only; no TEST access by evaluation, no model selection, no source deletion. Source snapshot identifies code at capture, not necessarily every historical training version.'}
    with (temporary / 'allocation-manifest.json').open('xb') as handle:
        handle.write(manifest_path.read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    write_json(temporary / 'snapshot-manifest.json', record)
    verified = verify(temporary)
    write_json(temporary / 'verification.json', verified)
    if output.exists():
        raise ValueError('output appeared during capture; refusing overwrite')
    os.rename(temporary, output)
    return {**verified, 'output': str(output), 'runs': len(runs), 'candidate_records': len(inv.candidates),
            'missing_historical_artifacts': len(inv.missing)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    cap = commands.add_parser('capture')
    cap.add_argument('--manifest', required=True)
    cap.add_argument('--host', default=socket.gethostname())
    cap.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    cap.add_argument('--output', required=True)
    cap.add_argument('--data', default='data/conversation-world/dataset.json')
    check = commands.add_parser('verify')
    check.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'capture' and args.host != socket.gethostname().split('.')[0]:
            raise ValueError('--host must match this local hostname')
        result = verify(args.output) if args.command == 'verify' else capture(
            args.manifest, args.host, args.root, args.output, data=args.data)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError, tarfile.TarError) as exc:
        print(json.dumps({'status': 'refused', 'error': type(exc).__name__, 'message': str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
