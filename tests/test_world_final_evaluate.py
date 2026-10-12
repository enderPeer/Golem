"""Final-selection controls use synthetic archives/models; never the real corpus."""
import copy
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch

from english_world.neural_model import END, NeuralConfig
from english_world.semantic_model import SemanticGRU, generate_semantic, save_semantic_checkpoint
from ops import world_final_evaluate as final
from ops.world_final_collect import verify

NOW = datetime(2030, 1, 2, tzinfo=timezone.utc)
PROTOCOL = {'schema': 'golem-semantic-validation-protocol-v1', 'fixture': 'synthetic-only'}
INITIALIZATION = {'shared_semantic_initialization': True, 'weight_sha256': 'd' * 64}


def metadata(strategy='ordinary', score=.4):
    return {'selection_split': 'validation', 'strategy': strategy, 'global_updates': 4,
            'initialization': INITIALIZATION, 'validation_protocol': PROTOCOL,
            'validation_protocol_hash': final.digest(PROTOCOL),
            'validation': {'rollout': {'split': 'validation', 'false_memory_claims': 0,
                           'score': score, 'whole_episode_pass_rate': .2},
                           'teacher_forced': {'reply_loss': .8}}}


def training_report():
    return {'schema': 'golem-semantic-training-v1', 'status': 'complete', 'error': None,
            'config': {'max_total_updates': 4, 'rounds': 1, 'evaluate_test_on_complete': False},
            'initialization': INITIALIZATION, 'validation_protocol_hash': final.digest(PROTOCOL),
            'strategies': {kind: {'status': 'complete_test_deferred', 'updates': 4,
                 'validation_calls': 4, 'reply_tokens': 80, 'prefix_tokens': 160,
                 'cpu_seconds': 1., 'wall_seconds': 2., 'births': 0, 'test': None,
                 'best_validation': metadata(kind)['validation']}
                 for kind in ('ordinary', 'evolution')}}


def fixture_bundle(root, manifest_path, *, divergent_protocol=False, high_orphan=False,
                   captured_at=NOW, corpus_bytes=None, stop_immediately=False):
    manifest = json.loads(manifest_path.read_text())
    bundle = root / 'bundle'
    bundle.mkdir()
    (bundle / 'allocation-manifest.json').write_bytes(manifest_path.read_bytes())
    data = {'data/conversation-world/dataset.json': corpus_bytes if corpus_bytes is not None else
            b'synthetic opaque corpus; no TEST examples'}
    candidates, roles = [], {'data/conversation-world/dataset.json': ['frozen_corpus_opaque']}
    model = SemanticGRU(NeuralConfig(embedding_dim=4, hidden_size=6, num_layers=1, dropout=0.))
    if stop_immediately:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.core.output.bias[END] = 4.
    for kind in ('ordinary', 'evolution'):
        checkpoint = root / (kind + '.pt')
        meta = metadata(kind)
        if divergent_protocol and kind == 'evolution':
            meta['validation_protocol'] = {**PROTOCOL, 'fixture': 'different-but-valid-hash'}
            meta['validation_protocol_hash'] = final.digest(meta['validation_protocol'])
        save_semantic_checkpoint(checkpoint, model, meta)
        prefix = 'runs/world-long/fixture/stages/stage-000001/run/'
        name = prefix + kind + '_champion.pt'
        data[name] = checkpoint.read_bytes()
        candidates.append({'path': name, 'sha256': final.sha256(checkpoint),
                           'origin': 'fixture:strategy:' + kind + ':stage-000001'})
        if kind == 'ordinary':
            global_name = 'runs/world-long/fixture/global_best.pt'
            data[global_name] = checkpoint.read_bytes()
            candidates.append({'path': global_name, 'sha256': final.sha256(checkpoint), 'origin': 'fixture:global'})
    data[prefix + 'report.json'] = json.dumps(training_report()).encode()
    data['runs/world-long/fixture/supervisor-state.json'] = json.dumps({
        'schema': 'golem-world-supervisor-v1', 'stages': [{'stage': 'stage-000001'}]}).encode()
    if high_orphan:
        for kind in ('ordinary', 'evolution'):
            checkpoint = root / ('orphan-' + kind + '.pt')
            save_semantic_checkpoint(checkpoint, model, metadata(kind, score=.99))
            name = 'runs/world-long/fixture/stages/stage-000002/run/' + kind + '_champion.pt'
            data[name] = checkpoint.read_bytes()
            candidates.append({'path': name, 'sha256': final.sha256(checkpoint),
                               'origin': 'fixture:strategy:' + kind + ':stage-000002'})
        data['runs/world-long/fixture/stages/stage-000002/run/report.json'] = json.dumps(training_report()).encode()
    with tarfile.open(bundle / 'snapshot.tar.gz', 'w:gz') as archive:
        for name, content in data.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    files = {name: {'bytes': len(content), 'sha256': final.hashlib.sha256(content).hexdigest(),
                    'roles': roles.get(name, ['fixture'])} for name, content in data.items()}
    record = {'schema': 'golem-world-final-snapshot-v1', 'host': 'fixturehost',
              'deadline_utc': manifest['deadline_utc'], 'captured_utc': captured_at.isoformat(),
              'allocation_manifest_sha256': final.sha256(manifest_path), 'allocation_manifest': manifest,
              'files': files, 'candidates': candidates, 'runs': [{'id': 'fixture'}],
              'corpus_identity': {'file_sha256': files['data/conversation-world/dataset.json']['sha256'],
                                  'canonical_content_sha256': final.CORPUS_CONTENT_SHA256},
              'archive_sha256': final.sha256(bundle / 'snapshot.tar.gz')}
    (bundle / 'snapshot-manifest.json').write_text(json.dumps(record))
    (bundle / 'verification.json').write_text(json.dumps(verify(bundle)))
    return bundle, files['data/conversation-world/dataset.json']['sha256']


class FinalSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        final._cpu_only()

    def setUp(self):
        protocol = patch.object(final, 'EXPECTED_VALIDATION_PROTOCOL_SHA256', final.digest(PROTOCOL))
        protocol.start()
        self.addCleanup(protocol.stop)

    def manifest(self, root, deadline='2030-01-01T08:00:00Z'):
        path = root / 'manifest.json'
        path.write_text(json.dumps({'schema': 'golem-semantic-world-sweep-v1',
            'deadline_utc': deadline,
            'jobs': [{'id': 'fixture', 'host': 'fixturehost', 'run_dir': 'runs/world-long/fixture'}]}))
        return path

    def test_deadline_guard_precedes_snapshot_and_corpus_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            with patch('ops.world_final_collect.verify', side_effect=AssertionError('must not verify')):
                with self.assertRaisesRegex(ValueError, 'before'):
                    final.freeze_selection(manifest, ['missing'], root / 'out',
                        now=datetime(2029, 1, 1, tzinfo=timezone.utc))
            roster = root / 'roster.json'
            roster.write_text(json.dumps({'deadline_utc': '2999-01-01T00:00:00Z'}))
            with patch('english_world.conversation_sim.load_conversation_sim', side_effect=AssertionError('TEST opened')):
                with self.assertRaisesRegex(ValueError, 'before'):
                    final.evaluate_frozen(roster, 'missing-corpus', root / 'eval')
            self.assertFalse((root / 'eval').exists())

    def test_manifest_union_includes_suspended_retired_and_deduplicates_path(self):
        jobs = [{'id': str(i), 'host': 'host', 'run_dir': f'run/{i}'} for i in range(3)]
        value = {'jobs': [jobs[0]], 'retired_jobs': [jobs[1]], 'suspended_jobs': [jobs[2], jobs[0]]}
        self.assertEqual(len(final.run_union(value)), 3)
        value['retired_jobs'].append({**jobs[0], 'id': 'conflict'})
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            final.run_union(value)

    def test_validation_rank_precedence_and_sha_ties(self):
        def row(sha, false=0, score=.4, episodes=.2, nll=.8):
            return {'checkpoint_sha256': sha, 'quality': {'false_memory_claims': false,
                'score': score, 'whole_episode_pass_rate': episodes, 'reply_loss': nll}}
        items = [row('z', score=.9, false=1), row('b'), row('a'), row('c', score=.3),
                 row('d', episodes=.1), row('e', nll=.9)]
        self.assertEqual([r['checkpoint_sha256'] for r in sorted(items, key=final.quality_key)],
                         ['a', 'b', 'e', 'd', 'c', 'z'])

    def test_snapshot_freeze_positive_integrity_and_no_test_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            bundle, corpus_sha = fixture_bundle(root, manifest)
            with patch.object(final, 'CORPUS_SHA256', corpus_sha), patch(
                    'english_world.conversation_sim.load_conversation_sim', side_effect=AssertionError('TEST opened')):
                path = final.freeze_selection(manifest, [bundle], root / 'selection', now=NOW)
                roster = final.load_frozen_roster(path, now=NOW)
                self.assertEqual(len(roster['runs']), 1)
                self.assertEqual(len(roster['evaluation_roster']), 2)
                pair = roster['latest_available_completed_pairs']['fixture']['pair']
                self.assertEqual(pair['actual_work']['ordinary']['updates'], 4)
                winner = roster['primary_checkpoint_sha256']
                self.assertEqual(roster['candidates'][winner]['strategy'], 'ordinary')
                checkpoint = path.parent / roster['candidates'][winner]['checkpoint_path']
                with checkpoint.open('ab') as stream:
                    stream.write(b'tamper')
                with self.assertRaisesRegex(ValueError, 'checkpoint bytes'):
                    final.load_frozen_roster(path, now=NOW)

    def test_missing_union_archive_and_roster_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            with self.assertRaisesRegex(ValueError, 'missing active'):
                final.freeze_selection(manifest, [], root / 'none', now=NOW)
            bundle, corpus_sha = fixture_bundle(root, manifest)
            with patch.object(final, 'CORPUS_SHA256', corpus_sha):
                roster_path = final.freeze_selection(manifest, [bundle], root / 'selection', now=NOW)
                roster = json.loads(roster_path.read_text())
                roster['evaluation_protocol']['seed'] = 9
                roster_path.write_text(json.dumps(roster))
                with self.assertRaisesRegex(ValueError, 'seal'):
                    final.load_frozen_roster(roster_path, now=NOW)
                with (bundle / 'snapshot.tar.gz').open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'archive identity'):
                    final.freeze_selection(manifest, [bundle], root / 'changed', now=NOW)

    def test_semantic_finite_weights_and_protocol_hash_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.pt'
            model = SemanticGRU(NeuralConfig(embedding_dim=4, hidden_size=6, num_layers=1, dropout=0.))
            bad_metadata = metadata()
            bad_metadata['validation_protocol_hash'] = 'a' * 64
            save_semantic_checkpoint(path, model, bad_metadata)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                final.inspect_candidate(path)
            save_semantic_checkpoint(path, model, metadata())
            artifact = torch.load(path, weights_only=True)
            next(iter(artifact['model_state'].values())).fill_(float('nan'))
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                final.inspect_candidate(path)

    def test_valid_but_different_common_protocol_and_candidate_index_hash_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            bundle, corpus_sha = fixture_bundle(root, manifest, divergent_protocol=True)
            with patch.object(final, 'CORPUS_SHA256', corpus_sha):
                with self.assertRaisesRegex(ValueError, 'expected recorded validation'):
                    final.freeze_selection(manifest, [bundle], root / 'different', now=NOW)
                path = bundle / 'snapshot-manifest.json'
                record = json.loads(path.read_text())
                record['candidates'][0]['sha256'] = '0' * 64
                path.write_text(json.dumps(record))
                (bundle / 'verification.json').write_text(json.dumps(verify(bundle)))
                with self.assertRaisesRegex(ValueError, 'candidate-index hash'):
                    final.freeze_selection(manifest, [bundle], root / 'bad-index', now=NOW)

    def test_all_candidates_consistent_but_wrong_pinned_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            bundle, corpus_sha = fixture_bundle(root, manifest)
            with patch.object(final, 'CORPUS_SHA256', corpus_sha), patch.object(
                    final, 'EXPECTED_VALIDATION_PROTOCOL_SHA256', 'b' * 64), patch(
                    'english_world.conversation_sim.load_conversation_sim', side_effect=AssertionError('real TEST opened')):
                with self.assertRaisesRegex(ValueError, 'expected recorded validation'):
                    final.freeze_selection(manifest, [bundle], root / 'wrong-protocol', now=NOW)
                self.assertFalse((root / 'wrong-protocol').exists())

    def test_high_scoring_orphan_strategy_cannot_displace_primary_or_completed_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            bundle, corpus_sha = fixture_bundle(root, manifest, high_orphan=True)
            with patch.object(final, 'CORPUS_SHA256', corpus_sha):
                path = final.freeze_selection(manifest, [bundle], root / 'selected', now=NOW)
                roster = final.load_frozen_roster(path, now=NOW)
                primary = roster['candidates'][roster['primary_checkpoint_sha256']]
                self.assertEqual(primary['quality']['score'], .4)
                self.assertEqual(roster['latest_available_completed_pairs']['fixture']['pair']['stage'], 'stage-000001')
                self.assertEqual(len(roster['evaluation_roster']), 2)
                self.assertTrue(any(item['quality']['score'] == .99 for item in roster['candidates'].values()))

    def test_symlink_or_junction_parent_refused_before_resolve(self):
        original = Path.is_symlink
        with patch.object(Path, 'is_symlink', lambda path: path.name == 'linked-parent' or original(path)):
            for operation in (lambda: final._read('linked-parent/snapshot-manifest.json'),
                              lambda: final.load_frozen_roster('linked-parent/roster.json'),
                              lambda: final.inspect_candidate('linked-parent/checkpoint.pt'),
                              lambda: final._write('linked-parent/output.json', {})):
                with self.assertRaisesRegex(ValueError, 'symlink/junction'):
                    operation()
        with patch.object(Path, 'is_junction', lambda path: path.name == 'junction-parent', create=True):
            with self.assertRaisesRegex(ValueError, 'symlink/junction'):
                final._safe('junction-parent/child/output')

    def test_pair_rejects_partial_work_unshared_init_or_prior_test(self):
        report = training_report()
        members = {kind: {'strategy': kind, 'initialization': INITIALIZATION,
                         'validation_protocol_hash': final.digest(PROTOCOL),
                         'validation_sha256': final.digest(metadata(kind)['validation'])}
                   for kind in ('ordinary', 'evolution')}
        self.assertIsNotNone(final.completed_pair(report, members))
        for key, value in [('updates', 3), ('validation_calls', 3), ('test', {'score': 1.}),
                           ('status', 'interrupted'), ('cpu_seconds', float('nan'))]:
            changed = copy.deepcopy(report)
            changed['strategies']['ordinary'][key] = value
            self.assertIsNone(final.completed_pair(changed, members), key)
        changed = copy.deepcopy(members)
        changed['ordinary']['initialization'] = {**INITIALIZATION, 'weight_sha256': 'b' * 64}
        self.assertIsNone(final.completed_pair(report, changed))
        changed = copy.deepcopy(report)
        changed['strategies']['ordinary']['best_validation']['rollout']['score'] = 1.
        self.assertIsNone(final.completed_pair(changed, members))

    def test_masked_nll_uses_global_token_denominator_not_example_mean(self):
        probabilities = torch.tensor([[[.5, .5], [1e-9, 1. - 1e-9], [1e-9, 1. - 1e-9]],
                                      [[.25, .75], [.25, .75], [.25, .75]]], dtype=torch.float64)
        targets = torch.zeros((2, 3), dtype=torch.long)
        mask = torch.tensor([[True, False, False], [True, True, True]])
        total, tokens = final.masked_nll(probabilities.log(), targets, mask)
        self.assertEqual(tokens, 4)
        self.assertAlmostEqual(total / tokens / math.log(2), 1.75, places=12)

    def test_exhaustive_synthetic_nll_every_reply_byte_plus_end_once(self):
        model = SemanticGRU(NeuralConfig(embedding_dim=4, hidden_size=6, num_layers=1, dropout=0.))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        samples = [{'id': str(i), 'act': 'greeting', 'reply': reply,
                    'turns': [{'role': 'user', 'content': 'Hello.'}]} for i, reply in enumerate(('a', 'long'))]
        result = final.exhaustive_reply_nll(model, samples, batch_size=1)
        self.assertEqual((result['examples'], result['reply_tokens']), (2, 7))
        self.assertAlmostEqual(result['reply_loss'], math.log(260), places=6)

    def test_evaluate_frozen_positive_end_to_end_synthetic_only(self):
        """Real GRU/rollout/audit/NLL APIs; replace only the corpus-source loader."""
        def episode(split, index):
            identity = f'synthetic-{split}-{index}'
            frame = {'turn_index': 1, 'case_id': identity, 'act': 'greeting', 'meaning_id': 0,
                     'user_name': None, 'source_user_turn': None,
                     'rubric': {'required_any': ['hello'], 'required_all': [], 'forbidden': [],
                                'max_response_bytes': 120}}
            return {'id': identity, 'family_id': identity,
                    'turns': [{'role': 'user', 'content': f'Hello, synthetic fixture {index}.'},
                              {'role': 'assistant', 'content': 'Hello.'}], 'target_frames': [frame]}

        synthetic = {'episodes': {'test': [episode('test', i) for i in range(72)],
                                 'regression': [episode('regression', i) for i in range(4)]},
                     'samples': {'test': [{'id': f'synthetic-target-{i}', 'act': 'greeting', 'reply': 'Hello.',
                         'turns': [{'role': 'user', 'content': 'Hello, synthetic fixture.'}]} for i in range(256)]}}
        content_sha = final.digest(synthetic)
        synthetic['manifest'] = {'content_sha256': content_sha}
        corpus_bytes = json.dumps(synthetic).encode('utf-8')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root, deadline='2000-01-01T00:00:00Z')
            synthetic_clock = datetime(2000, 1, 2, tzinfo=timezone.utc)
            with patch.object(final, 'CORPUS_CONTENT_SHA256', content_sha):
                bundle, corpus_sha = fixture_bundle(root, manifest, captured_at=synthetic_clock,
                    corpus_bytes=corpus_bytes, stop_immediately=True)
                corpus = root / 'synthetic-corpus.json'
                corpus.write_bytes(corpus_bytes)
                output = root / 'evaluation'
                with patch.object(final, 'CORPUS_SHA256', corpus_sha):
                    roster_path = final.freeze_selection(manifest, [bundle], root / 'frozen', now=synthetic_clock)
                    roster = final.load_frozen_roster(roster_path)
                    roster_file_sha = final.sha256(roster_path)
                    calls = []

                    def synthetic_loader(path):
                        self.assertEqual(Path(path), corpus)
                        return json.loads(corpus.read_text(encoding='utf-8'))

                    def checked_generation(model, history, **kwargs):
                        self.assertTrue((output / 'frozen-roster.json').is_file())
                        self.assertEqual(final.sha256(roster_path), roster_file_sha)
                        self.assertEqual(next(model.parameters()).device.type, 'cpu')
                        self.assertEqual(torch.get_num_threads(), 1)
                        self.assertEqual(kwargs['temperature'], 0.)
                        self.assertEqual(kwargs['max_context_tokens'], 512)
                        self.assertEqual(kwargs['max_new_tokens'], 120)
                        calls.append(history)
                        return generate_semantic(model, history, **kwargs)

                    with patch('english_world.conversation_sim.load_conversation_sim', side_effect=synthetic_loader) as loader, \
                         patch('english_world.semantic_model.generate_semantic', side_effect=checked_generation):
                        summary = final.evaluate_frozen(roster_path, corpus, output)
                    loader.assert_called_once_with(corpus)
                    self.assertEqual(len(calls), 2 * (72 + 4))
                    self.assertEqual(summary['primary_checkpoint_sha256'], roster['primary_checkpoint_sha256'])
                    self.assertFalse(summary['selection_changed'])
                    self.assertFalse(summary['automatic_promotion'])
                    self.assertEqual(final.sha256(roster_path), roster_file_sha)
                    self.assertEqual(len(summary['results']), 2)
                    for result in summary['results']:
                        folder = output / result['checkpoint_sha256']
                        for split, count in [('test', 72), ('regression', 4)]:
                            report_path = folder / (split + '-rollout.json')
                            report = json.loads(report_path.read_text())
                            audit_path = folder / (split + '-name-audit.json')
                            audit = json.loads(audit_path.read_text())
                            self.assertEqual(report['episodes'], count)
                            self.assertEqual(report['seed'], 40097)
                            self.assertEqual(report['frozen_roster_sha256'], roster['roster_sha256'])
                            self.assertEqual(audit['source_report_sha256'], final.sha256(report_path))
                            self.assertEqual(result['reports'][split]['name_audit_sha256'], final.sha256(audit_path))
                            self.assertIn('actual_history', report['rows'][0])
                            self.assertIn('input_history', report['rows'][0]['rows'][0])
                        nll = result['exhaustive_reply_nll']
                        self.assertEqual((nll['examples'], nll['reply_tokens']), (256, 256 * 7))
                        self.assertAlmostEqual(nll['reply_loss'], math.log(259 + math.exp(4)) - 4 / 7, places=6)


if __name__ == '__main__':
    unittest.main()
