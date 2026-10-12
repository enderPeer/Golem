"""Freeze validation-only selection, then explicitly evaluate the frozen roster.

The ``freeze`` command never opens the corpus. ``evaluate`` checks the deadline,
roster seal, checkpoint bytes and exact corpus identity before loading examples.
Both commands are offline: no SSH, service control, training or promotion.
Snapshot inputs are host-local ``world_final_collect`` bundles, not live runs.

After the deadline (repeat --snapshot for every host)::

    python -B ops/world_final_evaluate.py freeze --manifest allocation.json \
        --snapshot snapshots/specht32 --snapshot snapshots/falke64 \
        --snapshot snapshots/adler40 --snapshot snapshots/knecht24 --output frozen
    python -B ops/world_final_evaluate.py evaluate --roster frozen/roster.json \
        --data data/conversation-world/dataset.json --output final-results

No generation settings, corpus substitution, clock override or promotion flags
are exposed. A partial evaluation directory is evidence, never a resume input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA = 'golem-world-final-roster-v1'
EVALUATION = {'device': 'cpu', 'threads': 1, 'seed': 40097, 'temperature': 0.,
              'max_new_tokens': 120, 'max_context_tokens': 512,
              'test_episodes': 72, 'test_targets': 256, 'nll_batch_size': 8,
              'max_reply_tokens': 128}
CORPUS_SHA256 = '478b61e93933bf67313893fb5ae6478fdf218a9949734c9ae125a5a8606c8c1a'
CORPUS_CONTENT_SHA256 = 'e4215a3b8e4428c867c436dda27c12609fbb4619f930c3f3304ecbb834eb47d9'
EXPECTED_VALIDATION_PROTOCOL_SHA256 = '8224d561771c5a63a6e1685ce339deaee4653d0145e238a4916b92e2a424780f'
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with _safe(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _read(path):
    return json.loads(_safe(path).read_text(encoding='utf-8'))


def _write(path, value):
    with _safe(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def _safe(path):
    # Check all lexical parents BEFORE resolving; resolving first hides links.
    from ops.world_final_collect import no_links
    return no_links(path)


def _utc(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('deadline/timestamps must include a timezone')
    return stamp.astimezone(timezone.utc)


def require_deadline(deadline, now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.astimezone(timezone.utc) < _utc(deadline):
        raise ValueError('final selection and TEST are forbidden before the manifest deadline')


def _relative(value):
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value or
            path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError('unsafe snapshot-relative path')
    return path.as_posix()


def _file(root, relative):
    root = _safe(root)
    path = _safe(root / _relative(relative))
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('missing or unsafe frozen file')
    return path


def run_union(manifest):
    """Include retired and deliberately suspended workers, deduplicated by path."""
    result, identities = {}, {}
    for category in ('jobs', 'retired_jobs', 'suspended_jobs'):
        for job in manifest.get(category, []):
            key = (job['host'], job['run_dir'])
            if key in result and result[key]['id'] != job['id']:
                raise ValueError('conflicting run identities in manifest')
            if job['id'] in identities and identities[job['id']] != key:
                raise ValueError('conflicting run paths for one manifest identity')
            identities[job['id']] = key
            result[key] = {'id': job['id'], 'host': key[0], 'run_dir': key[1]}
    if not result:
        raise ValueError('manifest has no runs')
    return [result[key] for key in sorted(result)]


def quality_key(candidate):
    """Ascending sort: original validation rule, then SHA ascending exact ties."""
    q = candidate['quality']
    return (-(q['false_memory_claims'] == 0), -q['score'],
            -q['whole_episode_pass_rate'], q['reply_loss'], candidate['checkpoint_sha256'])


def inspect_candidate(path):
    """Safe Torch loader validates schema, finite tensors and strict dimensions."""
    from english_world.semantic_model import load_semantic_checkpoint
    path = _safe(path)
    if not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError('checkpoint exceeds its byte limit')
    model, metadata = load_semantic_checkpoint(path, map_location='cpu')
    if metadata.get('selection_split') != 'validation':
        raise ValueError('candidate was not selected on validation')
    protocol = metadata.get('validation_protocol')
    protocol_hash = metadata.get('validation_protocol_hash')
    if (not isinstance(protocol, dict) or protocol.get('schema') != 'golem-semantic-validation-protocol-v1'
            or digest(protocol) != protocol_hash):
        raise ValueError('missing or inconsistent recorded validation protocol')
    if protocol_hash != EXPECTED_VALIDATION_PROTOCOL_SHA256:
        raise ValueError('candidate does not match the expected recorded validation protocol')
    rollout = metadata['validation']['rollout']
    teacher = metadata['validation']['teacher_forced']
    if rollout.get('split') != 'validation':
        raise ValueError('candidate quality is not validation')
    quality = {key: rollout[key] for key in ('false_memory_claims', 'score', 'whole_episode_pass_rate')}
    quality['reply_loss'] = teacher['reply_loss']
    if (type(quality['false_memory_claims']) is not int or quality['false_memory_claims'] < 0 or
            any(type(value) not in (int, float) or not math.isfinite(value) for value in quality.values()) or
            not 0 <= quality['score'] <= 1 or not 0 <= quality['whole_episode_pass_rate'] <= 1 or quality['reply_loss'] < 0):
        raise ValueError('invalid recorded validation quality')
    result = {'checkpoint_sha256': sha256(path), 'quality': quality,
              'validation_protocol': protocol, 'validation_protocol_hash': protocol_hash,
              'validation_sha256': digest(metadata['validation']),
              'strategy': metadata.get('strategy'), 'updates': metadata.get('global_updates'),
              'initialization': metadata.get('initialization'),
              'parameter_count': model.parameter_count, 'meaning_condition_gain': model.meaning_condition_gain,
              'model_config': model.config.to_dict()}
    return result


def completed_pair(report, candidates):
    """Return actual work only for a fully completed, retained, matched pair."""
    if (report.get('schema') != 'golem-semantic-training-v1' or report.get('status') != 'complete'
            or report.get('error') is not None or set(candidates) != {'ordinary', 'evolution'}):
        return None
    config, initialization = report.get('config', {}), report.get('initialization', {})
    updates, rounds = config.get('max_total_updates'), config.get('rounds')
    if (type(updates) is not int or updates <= 0 or type(rounds) is not int or rounds <= 0 or
            config.get('evaluate_test_on_complete') is not False or
            initialization.get('shared_semantic_initialization') is not True or
            not re.fullmatch('[0-9a-f]{64}', initialization.get('weight_sha256', ''))):
        return None
    work = {}
    for name, candidate in candidates.items():
        row = report.get('strategies', {}).get(name, {})
        if (row.get('status') != 'complete_test_deferred' or row.get('updates') != updates or
                row.get('validation_calls') != 4 * rounds or row.get('test') is not None or
                candidate['strategy'] != name or candidate['initialization'] != initialization or
                candidate['validation_protocol_hash'] != report.get('validation_protocol_hash') or
                candidate['validation_sha256'] != digest(row.get('best_validation'))):
            return None
        work[name] = {key: row.get(key) for key in ('updates', 'validation_calls', 'reply_tokens',
                       'prefix_tokens', 'cpu_seconds', 'wall_seconds', 'births')}
        for key in ('updates', 'validation_calls', 'reply_tokens', 'prefix_tokens', 'cpu_seconds', 'wall_seconds'):
            if type(work[name][key]) not in (int, float) or not math.isfinite(work[name][key]) or work[name][key] < 0:
                return None
    return {'planned_updates_per_strategy': updates, 'planned_validation_calls_per_strategy': 4 * rounds,
            'shared_initialization_weight_sha256': initialization['weight_sha256'], 'actual_work': work,
            'training_data_fingerprint': report.get('data_fingerprint'),
            'limitation': 'Latest AVAILABLE fully completed pair, not a fresh experiment. Evolution counts population-total updates; token lengths and actual CPU/wall costs differ. Earlier tensors may be pruned.'}


def masked_nll(logits, targets, mask):
    """Unweighted sum and count, including END but excluding padded positions."""
    import torch
    errors = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                               targets.reshape(-1), reduction='none').reshape_as(targets)
    values = errors[mask]
    if not len(values) or not torch.isfinite(values).all():
        raise ValueError('empty/nonfinite reply likelihood')
    return float(values.double().sum()), int(mask.sum())


def exhaustive_reply_nll(model, samples, batch_size=8):
    """Every supplied example exactly once; no act, prefix, or name weighting."""
    import torch
    from english_world.semantic_training import encode_samples, make_batch
    encoded = encode_samples({'samples': {'evaluation': samples}}, 'evaluation',
                             {'max_context_tokens': 512, 'max_reply_tokens': 128})
    total, tokens = 0., 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch = make_batch(encoded[start:start + batch_size], 'cpu')
            logits, _ = model(batch['prefix'], batch['prefix_lengths'], batch['reply_inputs'])
            loss, count = masked_nll(logits, batch['targets'], batch['mask'])
            total += loss
            tokens += count
    return {'reply_loss': total / tokens, 'bits_per_reply_token': total / tokens / math.log(2),
            'total_negative_log_likelihood': total, 'reply_tokens': tokens, 'examples': len(encoded),
            'sample_ids_sha256': digest([row['id'] for row in encoded]),
            'scope': 'Exhaustive token-weighted current reference reply UTF-8 bytes plus END; no sampled rows, history-token loss, act loss, name weighting or first-response weighting.'}


def _cpu_only():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def _snapshot_candidates(record, jobs):
    """Use only the collector index; arbitrary archived tensors are evidence."""
    found = {}
    for row in record['candidates']:
        name = _relative(row['path'])
        run_id = row['origin'].split(':', 1)[0]
        if run_id not in jobs or not name.startswith(jobs[run_id]['run_dir'] + '/'):
            raise ValueError('candidate lies outside its declared run')
        if row['sha256'] != record['files'][name]['sha256']:
            raise ValueError('candidate-index hash mismatch')
        found.setdefault(name, {'run_id': run_id, 'origins': []})['origins'].append(row['origin'])
    return found


def primary_eligible(source):
    """Strategy-only and orphan artifacts never silently change primary selection."""
    prefix = re.escape(source['run_id'])
    return any(re.fullmatch(prefix + r':(?:global|(?:journaled_pending|latest_committed):stage-[0-9]{6})',
                            origin) for origin in source['origins'])


def _extract_subset(bundle, record, wanted, destination):
    """Only regular listed members; never tar.extract/all or corpus parsing."""
    with tarfile.open(Path(bundle) / 'snapshot.tar.gz', 'r|gz') as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            path = destination / _relative(member.name)
            if not member.isfile() or member.size != record['files'][member.name]['bytes']:
                raise ValueError('unsafe or changed snapshot member')
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target)
            if sha256(path) != record['files'][member.name]['sha256']:
                raise ValueError('snapshot member changed after verification')


def freeze_selection(manifest_path, bundles, output, *, now=None):
    """Consume all host bundles and atomically publish an immutable roster.

    ``now`` is a unit-test clock, absent from the CLI. The corpus identity is
    fixed in this tool; a changed corpus requires a separately reviewed protocol.
    """
    from ops.world_final_collect import verify
    manifest = _read(manifest_path)
    require_deadline(manifest['deadline_utc'], now)
    if manifest.get('schema') != 'golem-semantic-world-sweep-v1':
        raise ValueError('expected a semantic-world allocation manifest')
    expected = run_union(manifest)
    allocation_sha = sha256(manifest_path)
    output = _safe(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError('freeze output must be a fresh directory with an existing parent')
    _cpu_only()
    records, seen = [], set()
    for bundle in bundles:
        bundle = _safe(bundle)
        record = _read(bundle / 'snapshot-manifest.json')
        if (record.get('schema') != 'golem-world-final-snapshot-v1' or
                record['allocation_manifest_sha256'] != allocation_sha or record['allocation_manifest'] != manifest or
                record['deadline_utc'] != manifest['deadline_utc'] or
                _utc(record['captured_utc']) < _utc(manifest['deadline_utc'])):
            raise ValueError('snapshot allocation/deadline identity mismatch')
        verification = verify(bundle)
        if _read(bundle / 'verification.json') != verification:
            raise ValueError('snapshot verification seal mismatch')
        jobs = {row['id']: row for row in expected if row['host'] == record['host']}
        if not jobs or {row['id'] for row in record['runs']} != set(jobs):
            raise ValueError('snapshot run union mismatch')
        keys = {(row['host'], row['run_dir']) for row in jobs.values()}
        if seen & keys:
            raise ValueError('duplicate host/run snapshot')
        seen.update(keys)
        corpus = [entry['sha256'] for entry in record['files'].values()
                  if 'frozen_corpus_opaque' in entry.get('roles', [])]
        identity = record.get('corpus_identity', {})
        if (corpus != [CORPUS_SHA256] or identity.get('file_sha256') != CORPUS_SHA256 or
                identity.get('canonical_content_sha256') != CORPUS_CONTENT_SHA256):
            raise ValueError('snapshot corpus differs from the frozen final-evaluation corpus')
        records.append((bundle, record, verification, jobs))
    if seen != {(row['host'], row['run_dir']) for row in expected}:
        raise ValueError('missing active, suspended or retired run snapshot')

    candidates, associations, pairs, snapshots = {}, {}, {}, []
    protocol_hash = None
    # This controlled temporary directory is outside live training directories.
    with tempfile.TemporaryDirectory(prefix=output.name + '.staging-', dir=output.parent) as temporary:
        staging = Path(temporary)
        published = staging / 'frozen'
        (published / 'checkpoints').mkdir(parents=True)
        for bundle, record, verification, jobs in records:
            found = _snapshot_candidates(record, jobs)
            wanted = set(found) | {job['run_dir'] + '/supervisor-state.json' for job in jobs.values()}
            if not wanted <= set(record['files']):
                raise ValueError('snapshot lacks required supervisor provenance')
            for name in found:
                if name.endswith('_champion.pt'):
                    report = str(PurePosixPath(name).parent / 'report.json')
                    wanted.update(path for path in (report, report + '.gz') if path in record['files'])
            extracted = staging / record['host']
            _extract_subset(bundle, record, wanted, extracted)
            inspected = {}
            for name in sorted(found):
                item = inspect_candidate(extracted / name)
                if item['checkpoint_sha256'] != record['files'][name]['sha256']:
                    raise ValueError('inspected candidate hash mismatch')
                if protocol_hash is None:
                    protocol_hash = item['validation_protocol_hash']
                if item['validation_protocol_hash'] != protocol_hash:
                    raise ValueError('candidates do not share the common recorded validation protocol')
                checkpoint_sha = item['checkpoint_sha256']
                inspected[name] = item
                if checkpoint_sha not in candidates:
                    item['checkpoint_path'] = f'checkpoints/{checkpoint_sha}.pt'
                    shutil.copyfile(extracted / name, published / item['checkpoint_path'])
                    candidates[checkpoint_sha] = item
                associations.setdefault(checkpoint_sha, []).append({
                    'host': record['host'], 'path': name, **found[name]})
            for run_id, job in jobs.items():
                state = _read(extracted / (job['run_dir'] + '/supervisor-state.json'))
                if state.get('schema') != 'golem-world-supervisor-v1':
                    raise ValueError('invalid captured supervisor provenance')
                ledger_stages = {row['stage'] for row in state['stages']}
                stage_names = set()
                for name in found:
                    match = re.fullmatch(re.escape(job['run_dir']) +
                        r'/stages/(stage-[0-9]{6})/run/(ordinary|evolution)_champion\.pt', name)
                    if match:
                        stage_names.add(match[1])
                pair = None
                rejected = []
                for stage in sorted(stage_names, reverse=True):
                    if stage not in ledger_stages:
                        rejected.append({'stage': stage, 'reason': 'owned orphan/uncommitted stage is recovery evidence only; absent from supervisor ledger'})
                        continue
                    prefix = job['run_dir'] + '/stages/' + stage + '/run/'
                    members = {kind: inspected[prefix + kind + '_champion.pt']
                               for kind in ('ordinary', 'evolution') if prefix + kind + '_champion.pt' in inspected}
                    report_path = extracted / (prefix + 'report.json')
                    if report_path.is_file():
                        report = _read(report_path)
                    elif report_path.with_suffix('.json.gz').is_file():
                        report = json.loads(gzip.decompress(report_path.with_suffix('.json.gz').read_bytes()))
                    else:
                        rejected.append({'stage': stage, 'reason': 'retained champions lack a committed report'})
                        continue
                    work = completed_pair(report, members)
                    if work is not None:
                        pair = {'run_id': run_id, 'host': record['host'], 'stage': stage,
                                'members': {kind: member['checkpoint_sha256'] for kind, member in members.items()},
                                'report_sha256': digest(report), **work}
                        break
                    rejected.append({'stage': stage, 'reason': 'incomplete, unmatched, unretained or inconsistent pair'})
                pairs[run_id] = {'pair': pair, 'newer_retained_stages_not_eligible': rejected,
                                 'retention_scope': 'Only retained champion tensors can be evaluated; missing older tensors cannot be reconstructed from JSON.'}
            snapshots.append({'host': record['host'], **verification,
                              'missing_historical_artifacts': record.get('missing_historical_artifacts', [])})

        winners = {}
        for job in expected:
            eligible = [item for checkpoint_sha, item in candidates.items()
                        if any(row['run_id'] == job['id'] and primary_eligible(row)
                               for row in associations[checkpoint_sha])]
            if not eligible:
                raise ValueError('run has no verified available validation candidate: ' + job['id'])
            winners[job['id']] = min(eligible, key=quality_key)['checkpoint_sha256']
        primary = min((candidates[checkpoint_sha] for checkpoint_sha in set(winners.values())),
                      key=quality_key)['checkpoint_sha256']
        roster_hashes = set(winners.values()) | {primary}
        for row in pairs.values():
            if row['pair']:
                roster_hashes.update(row['pair']['members'].values())
        roster = {'schema': SCHEMA, 'deadline_utc': manifest['deadline_utc'],
                  'frozen_utc': (now or datetime.now(timezone.utc)).isoformat(),
                  'allocation_manifest_sha256': allocation_sha, 'runs': expected, 'snapshots': snapshots,
                  'allocation_manifest_content_sha256': digest(manifest),
                  'corpus_sha256': CORPUS_SHA256, 'corpus_content_sha256': CORPUS_CONTENT_SHA256,
                  'evaluation_protocol': EVALUATION, 'evaluation_protocol_sha256': digest(EVALUATION),
                  'validation_protocol_hash': protocol_hash, 'primary_checkpoint_sha256': primary,
                  'per_run_winners': winners, 'latest_available_completed_pairs': pairs,
                  'candidates': candidates, 'candidate_sources': associations,
                  'evaluation_roster': sorted(roster_hashes),
                  'selection': 'Recorded validation only: zero observed legacy false-memory claims, rollout score, whole-episode pass, negative unweighted reply NLL; SHA ascending exact ties.',
                  'primary_candidate_scope': 'Only explicit global, journaled_pending and latest_committed origins. Strategy-only tensors are comparison/recovery evidence and cannot displace the primary.',
                  'no_reranking': 'TEST, known regressions and independent name audits never change primary, per-run winners or roster. No automatic promotion.',
                  'claims': 'Repeated validation-selected runs share ancestors and data; these are not independent statistical trials. Heuristic dialogue scores are not proof of semantic quality.'}
        roster['roster_sha256'] = digest(roster)
        _write(published / 'roster.json', roster)
        _write(published / 'allocation-manifest.json', manifest)
        os.replace(published, output)
    return output / 'roster.json'


def load_frozen_roster(path, *, now=None):
    path = _safe(path)
    roster = _read(path)
    require_deadline(roster['deadline_utc'], now)
    if (roster.get('schema') != SCHEMA or
            roster.get('roster_sha256') != digest({key: value for key, value in roster.items() if key != 'roster_sha256'}) or
            roster.get('evaluation_protocol') != EVALUATION or roster.get('evaluation_protocol_sha256') != digest(EVALUATION) or
            roster.get('validation_protocol_hash') != EXPECTED_VALIDATION_PROTOCOL_SHA256 or
            roster.get('corpus_sha256') != CORPUS_SHA256 or roster.get('corpus_content_sha256') != CORPUS_CONTENT_SHA256):
        raise ValueError('frozen roster/protocol/corpus seal mismatch')
    allocation = _read(_file(path.parent, 'allocation-manifest.json'))
    if (digest(allocation) != roster['allocation_manifest_content_sha256'] or
            allocation['deadline_utc'] != roster['deadline_utc'] or run_union(allocation) != roster['runs'] or
            _utc(roster['frozen_utc']) < _utc(roster['deadline_utc'])):
        raise ValueError('frozen allocation, union or deadline changed')
    if roster['primary_checkpoint_sha256'] not in roster['evaluation_roster']:
        raise ValueError('primary is absent from frozen evaluation roster')
    for checkpoint_sha in roster['evaluation_roster']:
        item = roster['candidates'][checkpoint_sha]
        checkpoint = _file(path.parent, item['checkpoint_path'])
        if sha256(checkpoint) != checkpoint_sha:
            raise ValueError('frozen checkpoint bytes changed')
    return roster


def evaluate_frozen(roster_path, data_path, output):
    """Explicit final-only entry point. Never call before the authorized deadline."""
    roster_path = _safe(roster_path)
    roster = load_frozen_roster(roster_path)  # MUST precede corpus loading/import.
    data_path = _safe(data_path)
    if sha256(data_path) != roster['corpus_sha256']:
        raise ValueError('final corpus byte identity mismatch')
    output = _safe(output)
    if output.exists():
        raise ValueError('evaluation output must be fresh; never silently rerun TEST')
    _cpu_only()
    from english_world.conversation_sim import load_conversation_sim
    from english_world.conversation_eval import evaluate_conversations
    from english_world.conversation_audit import audit_report_file
    from english_world.semantic_model import load_semantic_checkpoint, generate_semantic
    data = load_conversation_sim(data_path)
    if (data['manifest']['content_sha256'] != roster['corpus_content_sha256'] or
            len(data['episodes']['test']) != 72 or len(data['samples']['test']) != 256):
        raise ValueError('final corpus content/count mismatch')
    output.mkdir(parents=True)
    _write(output / 'frozen-roster.json', roster)  # Published before any generation.
    results = []
    for checkpoint_sha in roster['evaluation_roster']:
        checkpoint = _file(roster_path.parent, roster['candidates'][checkpoint_sha]['checkpoint_path'])
        if sha256(checkpoint) != checkpoint_sha:
            raise ValueError('checkpoint changed since roster verification')
        model, metadata = load_semantic_checkpoint(checkpoint, map_location='cpu')
        if metadata.get('validation_protocol_hash') != roster['validation_protocol_hash']:
            raise ValueError('candidate validation protocol changed')
        folder = output / checkpoint_sha
        folder.mkdir()
        wall_start, cpu_start = time.monotonic(), time.process_time()
        def generate(history, **kwargs):
            return generate_semantic(model, history, max_context_tokens=512, **kwargs)
        reports = {}
        for split in ('test', 'regression'):
            count = 72 if split == 'test' else len(data['episodes']['regression'])
            report = evaluate_conversations(generate, data, split=split, max_episodes=count,
                                             max_new_tokens=120, seed=40097, temperature=0.)
            expected_ids = {row['id'] for row in data['episodes'][split]}
            if {row['id'] for row in report['rows']} != expected_ids or report['episodes'] != count:
                raise ValueError('final rollout did not cover every declared episode')
            report.update(checkpoint_sha256=checkpoint_sha, frozen_roster_sha256=roster['roster_sha256'],
                          final_evaluation_protocol=EVALUATION, selection_effect='none; roster already frozen')
            path = folder / (split + '-rollout.json')
            _write(path, report)
            audit = audit_report_file(path, folder / (split + '-name-audit.json'))
            reports[split] = {'report_sha256': sha256(path), 'name_audit_sha256': sha256(folder / (split + '-name-audit.json')),
                             'score': report['score'], 'episodes': report['episodes'], 'turns': report['turns'],
                             'whole_episode_pass_rate': report['whole_episode_pass_rate'],
                             'legacy_false_memory_claims': report['false_memory_claims'],
                             'independent_name_flags': audit['flags']}
        nll = exhaustive_reply_nll(model, data['samples']['test'])
        nll.update(split='test', checkpoint_sha256=checkpoint_sha, frozen_roster_sha256=roster['roster_sha256'])
        _write(folder / 'test-exhaustive-reply-nll.json', nll)
        result = {'checkpoint_sha256': checkpoint_sha, 'reports': reports, 'exhaustive_reply_nll': nll,
                  'cpu_seconds': time.process_time() - cpu_start, 'wall_seconds': time.monotonic() - wall_start}
        _write(folder / 'summary.json', result)
        results.append(result)
        del model
    summary = {'schema': 'golem-world-final-evaluation-v1', 'frozen_roster_sha256': roster['roster_sha256'],
               'primary_checkpoint_sha256': roster['primary_checkpoint_sha256'], 'results': results,
               'selection_changed': False, 'automatic_promotion': False,
               'limitation': 'Known regressions are disclosed separately. Independent name audit and dialogue rubrics are heuristic checks, not proof of general conversation quality.'}
    _write(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze', help='after deadline: verify all snapshots and freeze validation selection')
    freeze.add_argument('--manifest', required=True)
    freeze.add_argument('--snapshot', action='append', required=True, help='collector output directory; repeat for every host')
    freeze.add_argument('--output', required=True)
    evaluate = sub.add_parser('evaluate', help='after deadline ONLY: evaluate every already-frozen roster model')
    evaluate.add_argument('--roster', required=True)
    evaluate.add_argument('--data', required=True)
    evaluate.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'freeze':
        print(freeze_selection(args.manifest, args.snapshot, args.output))
    else:
        result = evaluate_frozen(args.roster, args.data, args.output)
        print(json.dumps({'evaluated_checkpoints': len(result['results']), 'selection_changed': False}))


if __name__ == '__main__':
    main()
