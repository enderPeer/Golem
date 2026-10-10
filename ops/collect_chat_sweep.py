"""Collect and optionally publish chat-island champions using validation ONLY.

No remote command runs at import. Sources are restricted to the named Golem
hosts/workspaces. Checkpoints are loaded on CPU with weights_only=True and their
fixed greedy validation protocols must agree. Global ties use reference loss on
the SAME generated-chat cases, not island-specific sampled validation batches.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
ALLOWED_HOSTS = frozenset(('specht32', 'falke64', 'adler40', 'knecht24'))
PUBLISH_DIRECTORY = '/home/ender/golem/runs/chat-live'
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
SCHEMA = 'golem-chat-collector-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
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


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get('schema') != 'golem-chat-cluster-sweep-v1':
        raise ValueError('unsupported chat sweep manifest')
    jobs = manifest.get('jobs')
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 256:
        raise ValueError('manifest must contain 1..256 jobs')
    result, seen = [], set()
    for job in jobs:
        identity, host, root, run = (job.get(key) for key in ('id', 'host', 'root', 'run_dir'))
        if not isinstance(identity, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', identity) or identity in seen:
            raise ValueError('invalid or duplicate island ID')
        if host not in ALLOWED_HOSTS:
            raise ValueError('island host is not allowlisted')
        if root != '/home/ender/golem' and not (host == 'knecht24' and isinstance(root, str)
                                               and re.fullmatch(r'/tmp/golem-chat-[0-9]{8}', root)):
            raise ValueError('island workspace is not allowlisted')
        if not isinstance(run, str) or run != f'runs/chat-sweep/{identity}':
            raise ValueError('run path must be its own contained chat-sweep island directory')
        path = str(PurePosixPath(root) / run / 'best.pt')
        result.append({'id': identity, 'host': host, 'root': root, 'run_dir': run, 'checkpoint': path})
        seen.add(identity)
    return result


def validation_info(metadata, expected_cases=26, reference_loss_override=None):
    """Plain-Python validation contract; shared verbatim with remote publication."""
    if not isinstance(metadata, dict) or metadata.get('training_schema') != 'golem-chat-training-v1':
        raise ValueError('checkpoint is not a chat-training artifact')
    if metadata.get('selection_split') != 'validation':
        raise ValueError('checkpoint was not selected exclusively on validation')
    validation = metadata.get('validation', {})
    chat = validation.get('chat', {})
    if chat.get('split') != 'validation' or chat.get('temperature') != 0:
        raise ValueError('global selection requires greedy held-out validation')
    rows = chat.get('rows')
    if chat.get('cases') != expected_cases or not isinstance(rows, list) or len(rows) != expected_cases:
        raise ValueError('checkpoint does not use the required fixed validation case count')
    score = chat.get('score')
    loss = reference_loss_override if reference_loss_override is not None else chat.get('reference_loss')
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('invalid generated-validation score')
    if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
        raise ValueError('fixed-case reference loss is required; sampled island NLL is not a global tie-break')
    protocol_rows, ids = [], set()
    for row in rows:
        identity = row.get('id')
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('validation case IDs must be unique')
        ids.add(identity)
        protocol_rows.append({key: row.get(key) for key in
                              ('id', 'intent', 'turns', 'reference', 'rubric', 'generation_budget_bytes')})
    protocol = {'schema': chat.get('schema'), 'temperature': 0, 'cases': expected_cases,
                'max_new_tokens': chat.get('max_new_tokens'),
                'reference_policy': {'max_reference_bytes': 256, 'include_end_token': True, 'max_prompt_bytes': 2048},
                'rows': sorted(protocol_rows, key=lambda row: row['id'])}
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=True,
                                      separators=(',', ':')).encode()).hexdigest()
    strategy = metadata.get('strategy')
    step = metadata.get('global_updates')
    if strategy not in ('ordinary', 'independent', 'evolution') or type(step) is not int or step < 0:
        raise ValueError('missing or invalid source strategy/update provenance')
    return {'score': float(score), 'reference_loss': float(loss), 'protocol_sha256': digest,
            'reference_loss_source': 'collector_cpu_fixed_case_recomputation' if reference_loss_override is not None else 'checkpoint_metadata',
            'cases': expected_cases, 'temperature': 0, 'strategy': strategy, 'updates': step,
            'sampled_validation_loss_not_used': validation.get('nll', {}).get('loss'),
            'selection_key': [float(score), -float(loss)]}


def fixed_case_reference_loss(model, metadata, expected_cases=26):
    """Score stored held-out references on CPU; never alter checkpoint metadata."""
    import torch
    from english_world.chat_eval import _reference_score, prompt_tokens
    chat = metadata.get('validation', {}).get('chat', {})
    rows = chat.get('rows')
    if chat.get('split') != 'validation' or not isinstance(rows, list) or len(rows) != expected_cases:
        raise ValueError('fixed-case reference scoring requires all expected validation rows')
    total, count = 0., 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for row in sorted(rows, key=lambda item: item['id']):
                turns, reference = row.get('turns'), row.get('reference')
                if not isinstance(turns, list) or not 1 <= len(turns) <= 64 or not isinstance(reference, str) or not reference:
                    raise ValueError('validation row lacks bounded conversation/reference text')
                prefix = prompt_tokens(turns, max_prompt_bytes=2048)
                scored = _reference_score(model, prefix, reference, max_reference_bytes=256)
                total += scored['reference_nll']
                count += scored['reference_tokens']
    finally:
        model.train(was_training)
    loss = total / count if count else math.inf
    if not math.isfinite(loss):
        raise ValueError('fixed-case reference evaluation produced nonfinite loss')
    return loss


def checkpoint_validation_info(model, metadata, expected_cases=26):
    existing = metadata.get('validation', {}).get('chat', {}).get('reference_loss')
    if existing is None:
        loss = fixed_case_reference_loss(model, metadata, expected_cases)
        return validation_info(metadata, expected_cases, reference_loss_override=loss)
    return validation_info(metadata, expected_cases)


def validate_checkpoint(path, expected_cases=26):
    import torch
    from english_world.neural_model import NeuralConfig, load_neural_checkpoint
    torch.set_num_threads(1)
    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES:
        raise ValueError('checkpoint size exceeds collector limit')
    # Bound architecture allocation BEFORE constructing a model from metadata.
    raw = torch.load(path, map_location='cpu', weights_only=True)
    config = NeuralConfig.from_dict(raw.get('config'))
    h, width, layers, vocab = config.hidden_size, config.embedding_dim, config.num_layers, config.vocab_size
    parameters = vocab * width + 3 * h * (width + h + 2) + max(0, layers - 1) * 3 * h * (2 * h + 2) + h * vocab + vocab
    if parameters > 10000000:
        raise ValueError('checkpoint architecture exceeds the ten-million-parameter collector limit')
    del raw
    model, metadata = load_neural_checkpoint(path, map_location='cpu')
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('checkpoint contains nonfinite weights')
    quality = checkpoint_validation_info(model, metadata, expected_cases)
    # A bounded forward catches incompatible/corrupt tensor loading before publish.
    with torch.inference_mode():
        logits, hidden = model(torch.tensor([[257, 104, 105, 259, 258]], dtype=torch.long))
        if not torch.isfinite(logits).all() or not torch.isfinite(hidden).all():
            raise ValueError('checkpoint fails bounded CPU inference validation')
    return {**quality, 'sha256': sha256(path), 'bytes': path.stat().st_size,
            'parameters': parameters, 'model_config': config.to_dict()}


def _run(argv, *, stdin=None, timeout=90):
    result = subprocess.run(argv, input=stdin, text=True, encoding='utf-8', errors='replace',
                            capture_output=True, timeout=timeout, shell=False,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(f'{argv[0]} failed ({result.returncode}): {result.stderr[-1600:]}')
    return result.stdout


def _ssh_json(host, script, python='python3', timeout=90):
    if host not in ALLOWED_HOSTS:
        raise ValueError('remote host is not allowlisted')
    command = shlex.join([python, '-'])
    output = _run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, command],
                  stdin=script, timeout=timeout)
    return json.loads(output)


_INVENTORY = r'''
import hashlib,json,os
from pathlib import Path
rows=[]
for item in payload:
    row={'id':item['id']}
    try:
        root=Path(item['root']).resolve(strict=True)
        run=(root/item['run_dir']).resolve(strict=True)
        path=Path(item['checkpoint']).resolve(strict=True)
        if not run.is_relative_to(root) or not path.is_relative_to(run) or path.name!='best.pt':
            raise ValueError('checkpoint escaped allowed workspace')
        with path.open('rb') as source:
            size=os.fstat(source.fileno()).st_size
            if not 0<size<=268435456: raise ValueError('invalid checkpoint size')
            digest=hashlib.sha256()
            for block in iter(lambda:source.read(1048576),b''): digest.update(block)
        row.update(available=True,sha256=digest.hexdigest(),bytes=size)
    except FileNotFoundError:
        row.update(available=False,reason='checkpoint not published yet')
    except Exception as exc:
        row.update(available=False,reason=str(exc))
    rows.append(row)
print(json.dumps(rows))
'''


def inventory_host(host, jobs):
    return _ssh_json(host, 'payload=' + repr(jobs) + '\n' + _INVENTORY)


def _download(job, inventory, directory, expected_cases):
    path = Path(directory) / (job['id'] + '.pt')
    for attempt in range(2):
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            _run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                  job['host'] + ':' + job['checkpoint'], str(temporary.resolve())], timeout=120)
            actual = sha256(temporary)
            if actual != inventory['sha256']:
                refreshed = inventory_host(job['host'], [job])[0]
                if not refreshed.get('available') or actual != refreshed['sha256']:
                    inventory = refreshed
                    if attempt == 0:
                        continue
                    raise ValueError('checkpoint changed during collection; retry on the next collection')
                inventory = refreshed
            quality = validate_checkpoint(temporary, expected_cases)
            if quality['sha256'] != inventory['sha256']:
                raise ValueError('downloaded checkpoint failed source SHA verification')
            os.replace(temporary, path)
            return {**job, **quality, 'local_path': str(path.resolve()), 'source_sha256': inventory['sha256']}
        finally:
            temporary.unlink(missing_ok=True)
    raise ValueError('checkpoint was not stable enough to collect')


def choose_candidate(candidates, protocol_sha256=None):
    if not candidates:
        return None
    protocols = {candidate['protocol_sha256'] for candidate in candidates}
    if protocol_sha256 is None and len(protocols) != 1:
        raise ValueError('validation protocols differ; do not compare or publish these islands together')
    eligible = [candidate for candidate in candidates if protocol_sha256 is None or candidate['protocol_sha256'] == protocol_sha256]
    return max(eligible, key=lambda row: (row['score'], -row['reference_loss'], row['id'])) if eligible else None


def _preserve_or_publish_local(candidate, output, expected_cases):
    output = Path(output)
    champion = output / 'champion.pt'
    if champion.exists():
        previous = validate_checkpoint(champion, expected_cases)
        if previous['protocol_sha256'] != candidate['protocol_sha256']:
            raise ValueError('local incumbent uses a different validation protocol')
        if (previous['score'], -previous['reference_loss']) >= (candidate['score'], -candidate['reference_loss']):
            return {'published': False, 'reason': 'incumbent is equal or better', 'sha256': previous['sha256'],
                    'local_path': str(champion.resolve())}
    temporary = output / ('.champion-' + uuid.uuid4().hex + '.pt')
    try:
        shutil.copyfile(candidate['local_path'], temporary)
        if sha256(temporary) != candidate['sha256']:
            raise ValueError('local candidate changed before publication')
        os.replace(temporary, champion)
        _json(output / 'provenance.json', candidate)
    finally:
        temporary.unlink(missing_ok=True)
    return {'published': True, 'sha256': candidate['sha256'], 'local_path': str(champion.resolve())}


_PUBLISH = r'''
import fcntl,hashlib,json,math,os,shutil,sys
from pathlib import Path
import torch
torch.set_num_threads(1)
sys.path.insert(0,'/home/ender/golem')
from english_world.neural_model import load_neural_checkpoint
directory=Path('/home/ender/golem/runs/chat-live')
root=Path('/home/ender/golem').resolve(strict=True)
if not directory.resolve().is_relative_to(root): raise ValueError('publish directory escaped workspace')
stage=directory/payload['stage']
stage_info=directory/payload['stage_info']
target=directory/'champion.pt'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
with (directory/'.publish.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if not stage.resolve().is_relative_to(directory.resolve()) or not stage_info.resolve().is_relative_to(directory.resolve()):
        raise ValueError('staging path escaped publication directory')
    if digest(stage)!=payload['sha256']: raise ValueError('staged checkpoint checksum mismatch')
    info=json.loads(stage_info.read_text())
    if info.get('sha256')!=payload['sha256']: raise ValueError('staged provenance checksum mismatch')
    gate=info.get('release_gate',{})
    if (gate.get('schema')!='golem-chat-release-gate-v1' or gate.get('passed') is not True or
        gate.get('pass_count')!=7 or gate.get('required_checks')!=7 or
        gate.get('checkpoint_sha256')!=payload['sha256']):
        raise ValueError('staged checkpoint lacks a SHA-bound all-pass release gate')
    incoming_model,incoming_metadata=load_neural_checkpoint(stage,map_location='cpu')
    new=checkpoint_validation_info(incoming_model,incoming_metadata,payload['expected_cases'])
    if new['protocol_sha256']!=payload['protocol_sha256']: raise ValueError('staged validation protocol mismatch')
    publish=True
    if target.exists():
        if target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError('incumbent checkpoint escaped publication directory')
        current_model,current_metadata=load_neural_checkpoint(target,map_location='cpu')
        old=checkpoint_validation_info(current_model,current_metadata,payload['expected_cases'])
        if old['protocol_sha256']!=new['protocol_sha256']: raise ValueError('incumbent validation protocol differs')
        publish=(new['score'],-new['reference_loss'])>(old['score'],-old['reference_loss'])
    if publish:
        os.replace(stage,target)
        os.replace(stage_info,directory/'provenance.json')
    else:
        stage.unlink()
        stage_info.unlink()
    print(json.dumps({'published':publish,'host':'specht32','checkpoint':str(target),
                      'sha256':digest(target),'reason':'improved validation' if publish else 'incumbent equal or better'}))
'''


def require_release_gate(candidate):
    """Verify a gate attestation or perform the local gate before any network write."""
    path = Path(candidate['local_path']).resolve(strict=True)
    digest = sha256(path)
    if digest != candidate.get('sha256'):
        raise ValueError('checkpoint SHA changed before release verification')
    attestation = candidate.get('release_gate', {})
    valid = (attestation.get('schema') == 'golem-chat-release-gate-v1' and
             attestation.get('passed') is True and attestation.get('pass_count') == 7 and
             attestation.get('required_checks') == 7 and attestation.get('checkpoint_sha256') == digest)
    if valid:
        return dict(candidate)
    import torch
    from english_world.neural_model import load_neural_checkpoint
    from ops.chat_release_gate import evaluate_release
    torch.set_num_threads(1)
    model, _ = load_neural_checkpoint(path, map_location='cpu')
    result = evaluate_release(model)
    if sha256(path) != digest:
        raise ValueError('checkpoint changed during release verification')
    if not result.get('passed') or result.get('pass_count') != 7 or result.get('required_checks') != 7:
        raise ValueError('checkpoint failed the known-task release gate; live model preserved')
    checked = dict(candidate)
    checked['release_gate'] = {'schema': 'golem-chat-release-gate-v1', 'passed': True,
                               'pass_count': 7, 'required_checks': 7, 'checkpoint_sha256': digest}
    return checked


def publish_candidate(candidate, host='specht32', expected_cases=26):
    if host != 'specht32':
        raise ValueError('publication is restricted to specht32 chat-live')
    candidate = require_release_gate(candidate)
    token = uuid.uuid4().hex
    stage, stage_info = '.incoming-' + token + '.pt', '.incoming-' + token + '.json'
    prepare = "from pathlib import Path\nimport json\nroot=Path('/home/ender/golem').resolve(strict=True)\np=Path(" + repr(PUBLISH_DIRECTORY) + ")\nif not p.resolve().is_relative_to(root): raise ValueError('publish path escaped workspace')\np.mkdir(parents=True,exist_ok=True)\nprint(json.dumps({'ready':True}))\n"
    _ssh_json(host, prepare)
    provenance = Path(candidate['local_path']).with_name('.publish-' + token + '.json')
    try:
        _json(provenance, candidate)
        _run(['scp', '-q', '-o', 'BatchMode=yes', str(Path(candidate['local_path']).resolve()),
              host + ':' + PUBLISH_DIRECTORY + '/' + stage], timeout=120)
        _run(['scp', '-q', '-o', 'BatchMode=yes', str(provenance.resolve()),
              host + ':' + PUBLISH_DIRECTORY + '/' + stage_info], timeout=60)
        payload = {'stage': stage, 'stage_info': stage_info, 'sha256': candidate['sha256'],
                   'protocol_sha256': candidate['protocol_sha256'], 'expected_cases': expected_cases}
        script = ('import hashlib,json,math\n' + inspect.getsource(validation_info) + '\n'
                  + inspect.getsource(fixed_case_reference_loss) + '\n' + inspect.getsource(checkpoint_validation_info)
                  + '\npayload=' + repr(payload) + '\n' + _PUBLISH)
        return _ssh_json(host, script, python='/home/ender/golem/.venv-neural/bin/python', timeout=120)
    finally:
        provenance.unlink(missing_ok=True)


def collect(manifest_path, output_dir, publish_host=None, expected_cases=26, max_workers=4):
    manifest_path, output = Path(manifest_path), Path(output_dir)
    jobs = validate_manifest(json.loads(manifest_path.read_text(encoding='utf-8')))
    if type(expected_cases) is not int or not 1 <= expected_cases <= 512:
        raise ValueError('expected_cases must be in 1..512')
    if type(max_workers) is not int or not 1 <= max_workers <= 4:
        raise ValueError('max_workers must be in 1..4')
    if publish_host not in (None, 'specht32'):
        raise ValueError('publication is restricted to specht32')
    collection = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    snapshot = output / 'snapshots' / collection
    snapshot.mkdir(parents=True, exist_ok=False)
    grouped = {}
    for job in jobs:
        grouped.setdefault(job['host'], []).append(job)
    available, errors, pending = [], [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(inventory_host, host, host_jobs): (host, host_jobs) for host, host_jobs in grouped.items()}
        for future in as_completed(futures):
            host, host_jobs = futures[future]
            try:
                inventory = {row['id']: row for row in future.result()}
                for job in host_jobs:
                    row = inventory.get(job['id'], {})
                    if row.get('available'):
                        available.append((job, row))
                    else:
                        pending.append({'id': job['id'], 'host': host, 'reason': row.get('reason', 'missing inventory result')})
            except Exception as exc:
                errors.append({'host': host, 'stage': 'inventory', 'error': str(exc)[:2000]})
    candidates = []
    # Torch checkpoint validation stays serial and uses one CPU thread. Network
    # inventory is concurrent, but model/RNG validation cannot race in threads.
    for job, inventory in sorted(available, key=lambda item: item[0]['id']):
        try:
            candidates.append(_download(job, inventory, snapshot, expected_cases))
        except Exception as exc:
            errors.append({'id': job['id'], 'host': job['host'], 'stage': 'download_or_validate', 'error': str(exc)[:2000]})
    selected = choose_candidate(candidates)
    local = remote = publication_selected = None
    if selected:
        selected.update(collection=collection, manifest_sha256=sha256(manifest_path),
                        selection_rule='validation chat macro score, then LOWER reference loss on the identical fixed chat cases; no test metrics')
        local = _preserve_or_publish_local(selected, output, expected_cases)
    report = {'schema': SCHEMA, 'collection': collection, 'manifest_sha256': sha256(manifest_path),
              'jobs': len(jobs), 'collected': len(candidates), 'pending': pending, 'errors': errors,
              'candidates': sorted(candidates, key=lambda row: (-row['score'], row['reference_loss'], row['id'])),
              'selected': selected, 'local_publication': local, 'remote_publication': remote,
              'publication_selected': publication_selected,
              'selection_policy': 'Validation only. Same greedy case protocol required. Global NLL tie-break uses fixed-case reference_loss, NOT sampled island nll.loss.',
              'comparison_note': 'Different island hardware/training budgets are not an equal-compute comparison. Rubric scores are heuristic, not semantic accuracy.'}
    if publish_host:
        from ops.chat_release_gate import run_release_gate
        # The gate reads the same assembled candidate list. Keep the raw leader
        # and local raw checkpoint for analysis; neither authorizes live release.
        _json(output / 'report.json', report)
        release = run_release_gate(output / 'report.json', output / 'release-report.json')
        publication_selected = release['chosen']
        report['publication_selected'] = publication_selected
        report['release_report'] = str((output / 'release-report.json').resolve())
        report['publication_policy'] = 'Only an all-pass, SHA-bound known-task release candidate may replace chat-live.'
        if publication_selected is not None:
            report['remote_publication'] = publish_candidate(publication_selected, publish_host, expected_cases)
        else:
            report['remote_publication'] = {'published': False, 'reason': 'no candidate passed all release checks; live model preserved'}
    _json(snapshot / 'report.json', report)
    _json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='runs/chat-cluster-20261010/manifest.json')
    parser.add_argument('--output', default='runs/chat-collector')
    parser.add_argument('--publish-host', choices=['specht32'])
    parser.add_argument('--expected-cases', type=int, default=26)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    try:
        result = collect(args.manifest, args.output, args.publish_host, args.expected_cases, args.workers)
    except Exception as exc:
        print(json.dumps({'schema': SCHEMA, 'error': type(exc).__name__, 'message': str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({'collected': result['collected'], 'jobs': result['jobs'], 'selected': result['selected'],
                      'publication_selected': result['publication_selected'],
                      'remote_publication': result['remote_publication'], 'errors': result['errors'],
                      'report': str((Path(args.output) / 'report.json').resolve())}, indent=2))
    return 0 if result['selected'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
