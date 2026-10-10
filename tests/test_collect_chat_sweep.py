import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from english_world.neural_model import ByteGRULM, NeuralConfig, save_neural_checkpoint

SPEC = importlib.util.spec_from_file_location('collect_chat_sweep', Path(__file__).resolve().parents[1] / 'ops/collect_chat_sweep.py')
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def metadata(score=.5, reference_loss=2., sampled_loss=99., cases=26):
    rows = [{'id': f'case-{i}', 'intent': 'greeting', 'turns': [{'role': 'user', 'content': f'Hello {i}'}],
             'reference': 'Hello!', 'rubric': {'required_any': ['hello'], 'max_response_bytes': 96},
             'generation_budget_bytes': 96, 'reference_tokens': 7, 'reference_truncated': False}
            for i in range(cases)]
    return {'training_schema': 'golem-chat-training-v1', 'selection_split': 'validation',
            'strategy': 'ordinary', 'global_updates': 100,
            'validation': {'chat': {'schema': 'golem-generated-chat-eval-v1', 'split': 'validation',
                                    'temperature': 0., 'cases': cases, 'max_new_tokens': 96,
                                    'score': score, 'reference_loss': reference_loss, 'rows': rows},
                           'nll': {'loss': sampled_loss}}}


def job(identity='specht32-cpu-0'):
    return {'id': identity, 'host': 'specht32', 'root': '/home/ender/golem',
            'run_dir': 'runs/chat-sweep/' + identity}


class CollectorTests(unittest.TestCase):
    def test_manifest_rejects_untrusted_hosts_paths_and_shell_fragments(self):
        base = {'schema': 'golem-chat-cluster-sweep-v1', 'jobs': [job()]}
        result = collector.validate_manifest(base)
        self.assertEqual(result[0]['checkpoint'], '/home/ender/golem/runs/chat-sweep/specht32-cpu-0/best.pt')
        for key, value in [('host', 'specht32; echo bad'), ('root', '/home/ender/other-project'),
                           ('run_dir', 'runs/chat-sweep/../../secret'), ('id', 'bad;command')]:
            changed = copy.deepcopy(base)
            changed['jobs'][0][key] = value
            with self.assertRaises(ValueError):
                collector.validate_manifest(changed)

    def test_selection_uses_common_reference_loss_not_sampled_nll_or_test(self):
        left = collector.validation_info(metadata(reference_loss=2., sampled_loss=.01))
        right = collector.validation_info(metadata(reference_loss=1., sampled_loss=100.))
        left.update(id='a')
        right.update(id='b')
        self.assertEqual(collector.choose_candidate([left, right])['id'], 'b')
        changed = metadata()
        changed['selection_split'] = 'test'
        with self.assertRaises(ValueError):
            collector.validation_info(changed)
        missing = metadata()
        del missing['validation']['chat']['reference_loss']
        with self.assertRaises(ValueError):
            collector.validation_info(missing)

    def test_protocol_mismatch_is_not_silently_ranked(self):
        a = collector.validation_info(metadata())
        changed = metadata(score=1.)
        changed['validation']['chat']['rows'][0]['turns'][0]['content'] = 'Different held-out task'
        b = collector.validation_info(changed)
        a.update(id='a')
        b.update(id='b')
        with self.assertRaises(ValueError):
            collector.choose_candidate([a, b])
        changed = metadata(cases=25)
        with self.assertRaises(ValueError):
            collector.validation_info(changed)

    def test_cpu_checkpoint_validation_and_no_worse_local_publication(self):
        torch.set_num_threads(1)
        model = ByteGRULM(NeuralConfig(embedding_dim=4, hidden_size=8, num_layers=1, dropout=0.))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'output'
            output.mkdir()
            high = root / 'high.pt'
            low = root / 'low.pt'
            save_neural_checkpoint(high, model, metadata(score=.8, reference_loss=1.))
            save_neural_checkpoint(low, model, metadata(score=.5, reference_loss=.1))
            high_info = collector.validate_checkpoint(high)
            high_info.update(id='high', local_path=str(high), host='specht32', run_dir='runs/chat-sweep/high')
            first = collector._preserve_or_publish_local(high_info, output, 26)
            self.assertTrue(first['published'])
            low_info = collector.validate_checkpoint(low)
            low_info.update(id='low', local_path=str(low), host='specht32', run_dir='runs/chat-sweep/low')
            second = collector._preserve_or_publish_local(low_info, output, 26)
            self.assertFalse(second['published'])
            self.assertEqual(collector.sha256(output / 'champion.pt'), high_info['sha256'])
            self.assertEqual(collector.validate_checkpoint(output / 'champion.pt')['score'], .8)

    def test_nonfinite_checkpoint_rejected(self):
        model = ByteGRULM(NeuralConfig(embedding_dim=4, hidden_size=8, num_layers=1, dropout=0.))
        with torch.no_grad():
            model.output.bias[0] = float('nan')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.pt'
            save_neural_checkpoint(path, model, metadata())
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                collector.validate_checkpoint(path)

    def test_missing_reference_loss_is_computed_without_changing_source_bytes(self):
        model = ByteGRULM(NeuralConfig(embedding_dim=4, hidden_size=8, num_layers=1, dropout=0.))
        source_metadata = metadata()
        source_metadata['validation']['chat']['reference_loss'] = None
        for row in source_metadata['validation']['chat']['rows']:
            row.pop('reference_tokens')
            row.pop('reference_truncated')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.pt'
            save_neural_checkpoint(path, model, source_metadata)
            before = path.read_bytes()
            result = collector.validate_checkpoint(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(result['sha256'], hashlib.sha256(before).hexdigest())
            self.assertEqual(result['reference_loss_source'], 'collector_cpu_fixed_case_recomputation')
            self.assertGreater(result['reference_loss'], 0.)
            supplied = metadata(reference_loss=result['reference_loss'])
            self.assertEqual(result['protocol_sha256'], collector.validation_info(supplied)['protocol_sha256'])

    def test_actual_chat_training_checkpoint_with_reference_scoring_disabled_collects(self):
        from english_world.chat_training import train_chat
        from english_world.chat_eval import evaluate_chat
        data = {'texts': {}, 'conversations': {}, 'chat_cases': {}}
        for split in ('train', 'validation', 'test'):
            turns = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]
            data['texts'][split] = ['user: hi\nassistant: hello']
            data['conversations'][split] = [{'intent': 'greeting', 'turns': turns}]
            data['chat_cases'][split] = [{'id': split + '-' + str(i), 'intent': 'greeting',
                'turns': turns[:1], 'reference': 'hello',
                'rubric': {'required_any': ['hello'], 'max_response_bytes': 8}} for i in range(26)]
        config = {'strategies': ['ordinary'], 'rounds': 1, 'max_total_updates': 4,
                  'batch_size': 1, 'sequence_length': 32, 'threads': 1, 'device': 'cpu',
                  'max_seconds': 0, 'validation_batches': 1, 'validation_cases': 26,
                  'test_cases': 26, 'max_new_tokens': 8,
                  'model': {'embedding_dim': 4, 'hidden_size': 8, 'num_layers': 1, 'dropout': 0.}}

        def no_reference_scoring(model, dataset, **kwargs):
            kwargs['include_reference_nll'] = False
            return evaluate_chat(model, dataset, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch('english_world.chat_eval.evaluate_chat', side_effect=no_reference_scoring):
            report = train_chat(config, data, tmp)
            self.assertEqual(report['status'], 'complete')
            path = Path(tmp) / 'best.pt'
            saved = torch.load(path, weights_only=True)
            self.assertIsNone(saved['metadata']['validation']['chat']['reference_loss'])
            before = collector.sha256(path)
            result = collector.validate_checkpoint(path)
            self.assertEqual(result['sha256'], before)
            self.assertEqual(collector.sha256(path), before)
            self.assertEqual(result['reference_loss_source'], 'collector_cpu_fixed_case_recomputation')

    def test_ssh_script_is_stdin_not_shell_interpolated_manifest(self):
        with patch.object(collector, '_run', return_value='[]') as run:
            collector.inventory_host('specht32', collector.validate_manifest(
                {'schema': 'golem-chat-cluster-sweep-v1', 'jobs': [job()]}))
        args, kwargs = run.call_args
        self.assertEqual(args[0][-2:], ['specht32', 'python3 -'])
        self.assertNotIn('-n', args[0])
        self.assertIn('payload=', kwargs['stdin'])
        compile(kwargs['stdin'], '<remote-inventory>', 'exec')

    def test_publish_mode_keeps_raw_leader_separate_and_publishes_only_gated_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'schema': 'golem-chat-cluster-sweep-v1',
                                            'jobs': [job('specht32-cpu-0'), job('specht32-cpu-1')]}))
            raw = {'id': 'specht32-cpu-0', 'score': .9, 'reference_loss': 1., 'protocol_sha256': 'same',
                   'local_path': str(root / 'raw.pt'), 'sha256': 'a' * 64}
            good = {'id': 'specht32-cpu-1', 'score': .8, 'reference_loss': 2., 'protocol_sha256': 'same',
                    'local_path': str(root / 'good.pt'), 'sha256': 'b' * 64,
                    'release_gate': {'schema': 'golem-chat-release-gate-v1', 'passed': True,
                                     'pass_count': 7, 'required_checks': 7, 'checkpoint_sha256': 'b' * 64}}
            inventory = [{'id': row['id'], 'available': True, 'sha256': row['sha256']} for row in (raw, good)]
            with patch.object(collector, 'inventory_host', return_value=inventory), \
                 patch.object(collector, '_download', side_effect=[raw, good]), \
                 patch.object(collector, '_preserve_or_publish_local', return_value={'published': True}), \
                 patch('ops.chat_release_gate.run_release_gate', return_value={'chosen': good}) as gate, \
                 patch.object(collector, 'publish_candidate', return_value={'published': True}) as publish:
                report = collector.collect(manifest, root / 'collection', publish_host='specht32')
            self.assertEqual(report['selected']['id'], raw['id'])
            self.assertEqual(report['publication_selected']['id'], good['id'])
            self.assertEqual(publish.call_args.args[0]['id'], good['id'])
            gate.assert_called_once()

    def test_failed_release_gate_never_calls_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'schema': 'golem-chat-cluster-sweep-v1', 'jobs': [job()]}))
            raw = {'id': job()['id'], 'score': 1., 'reference_loss': .1, 'protocol_sha256': 'same',
                   'local_path': str(root / 'raw.pt'), 'sha256': 'a' * 64}
            with patch.object(collector, 'inventory_host', return_value=[{'id': raw['id'], 'available': True, 'sha256': raw['sha256']}]), \
                 patch.object(collector, '_download', return_value=raw), \
                 patch.object(collector, '_preserve_or_publish_local', return_value={'published': True}), \
                 patch('ops.chat_release_gate.run_release_gate', return_value={'chosen': None}), \
                 patch.object(collector, 'publish_candidate') as publish:
                report = collector.collect(manifest, root / 'collection', publish_host='specht32')
            publish.assert_not_called()
            self.assertIsNone(report['publication_selected'])
            self.assertFalse(report['remote_publication']['published'])
            self.assertEqual(report['selected']['score'], 1.)

    def test_direct_publish_requires_a_sha_bound_gate_before_any_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'candidate.pt'
            path.write_bytes(b'fixture; model loading mocked')
            candidate = {'local_path': str(path), 'sha256': collector.sha256(path)}
            with patch('english_world.neural_model.load_neural_checkpoint', return_value=(object(), {})), \
                 patch('ops.chat_release_gate.evaluate_release', return_value={'passed': False, 'pass_count': 6, 'required_checks': 7}), \
                 patch.object(collector, '_ssh_json') as remote:
                with self.assertRaisesRegex(ValueError, 'failed the known-task release gate'):
                    collector.publish_candidate(candidate)
            remote.assert_not_called()
            candidate['release_gate'] = {'schema': 'golem-chat-release-gate-v1', 'passed': True,
                                          'pass_count': 7, 'required_checks': 7, 'checkpoint_sha256': candidate['sha256']}
            checked = collector.require_release_gate(candidate)
            self.assertEqual(checked['release_gate']['checkpoint_sha256'], collector.sha256(path))


if __name__ == '__main__':
    unittest.main()
