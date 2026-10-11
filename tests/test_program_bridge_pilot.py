"""Frozen facts, paired sampling, and conservative bridge adoption decisions."""
import copy
import unittest
from unittest.mock import patch

from english_world import semantic_training as training
from ops import program_bridge_pilot as pilot


def certificate():
    return {'schema': 'golem-dna3-compute-slice-v1', 'task': 'increment_mod4',
            'domain': [0, 1, 2, 3], 'truth_table': [1, 2, 3, 0],
            'program_id': 'verified-fixture', 'certificate_sha256': 'a' * 64}


def encoded(identity, act='greeting', length=5):
    return {'id': identity, 'act_name': act, 'prefix': [1] * length, 'targets': [2, 3]}


def passing_pair(seed):
    baseline = {'complete': True, 'updates': 200, 'validation_calls': 4,
        'validation_protocol_hash': 'protocol', 'initial_weight_sha256': 'weights',
        'sampling': {'base_stream_sha256': 'stream', 'mixture_mask_sha256': 'mask'},
        'validation': {'score': .4, 'false_memory_claims': 0,
                       'whole_episode_pass_rate': .1, 'reply_loss': .3},
        'program': {'accuracy': .25},
        'regression': {'score': .5, 'false_memory_claims': 0, 'whole_episode_pass_rate': 0.}}
    mixed = copy.deepcopy(baseline)
    mixed['validation'].update(score=.42, reply_loss=.31)
    mixed['program'].update(accuracy=.75, correct_by_input={'0': 3, '1': 3, '2': 3, '3': 3})
    return {'seed': seed, 'planned_updates': 200, 'baseline': baseline, 'mixed': mixed}


def frozen():
    return {'validation_protocol_hash': 'protocol', 'weight_sha256': 'weights',
        'validation': {'score': .4, 'false_memory_claims': 0,
                       'whole_episode_pass_rate': .04, 'reply_loss': .35},
        'program': {'accuracy': .25},
        'regression': {'score': .5, 'false_memory_claims': 0, 'whole_episode_pass_rate': 0.}}


class ProgramBridgePilotTests(unittest.TestCase):
    def curriculum(self):
        with patch.object(pilot, '_load_certificate', return_value=certificate()) as load:
            result = pilot.prepare_program_curriculum('certificate.json')
        load.assert_called_once_with('certificate.json')
        return result

    def test_verified_labels_and_family_partition_are_explicit(self):
        data = self.curriculum()
        self.assertEqual(len(data['train']), 32)
        self.assertEqual(len(data['program_probe']), 16)
        self.assertFalse({r['family_id'] for r in data['train']} & {r['family_id'] for r in data['program_probe']})
        self.assertFalse({r['turns'][0]['content'] for r in data['train']} &
                         {r['turns'][0]['content'] for r in data['program_probe']})
        for row in data['train'] + data['program_probe']:
            self.assertEqual(row['expected_output'], certificate()['truth_table'][row['input']])
            self.assertEqual(row['reply'], f"The result is {row['expected_output']}.")
            self.assertEqual(row['act'], 'request_help')
            self.assertEqual(row['program_provenance']['certificate_sha256'], 'a' * 64)
            self.assertNotIn(row['reply'], row['turns'][0]['content'])

    def test_unverified_or_wrong_function_certificate_rejected(self):
        for field, value in (('task', 'unverified_x_minus_one'), ('truth_table', [0, 1, 2, 3]),
                             ('truth_table', [True, 2, 3, 0]), ('domain', [0, 1, 2]),
                             ('certificate_sha256', 'not-a-hash')):
            altered = {**certificate(), field: value}
            with self.subTest(field=field), patch.object(pilot, '_load_certificate', return_value=altered):
                with self.assertRaises(ValueError):
                    pilot.prepare_program_curriculum('bad.json')
        with patch.object(pilot, '_load_certificate', side_effect=ValueError('source hash mismatch')):
            with self.assertRaisesRegex(ValueError, 'source hash'):
                pilot.prepare_program_curriculum('stale.json')

    def test_augmentation_preserves_all_conversation_holdouts_and_base(self):
        base = {'samples': {'train': [{'id': 'base'}], 'validation': [{'id': 'val'}],
                            'test': [{'id': 'test'}]}, 'episodes': {'regression': [{'id': 'regression'}]}}
        original = copy.deepcopy(base)
        result = pilot.augment_training(base, self.curriculum())
        self.assertEqual(base, original)
        self.assertEqual(len(result['samples']['train']), 33)
        self.assertIs(result['samples']['validation'], base['samples']['validation'])
        self.assertIs(result['samples']['test'], base['samples']['test'])
        self.assertIs(result['episodes'], base['episodes'])
        with self.assertRaises(ValueError):
            pilot.augment_training(result, self.curriculum())

    def test_sampler_replaces_ten_percent_with_same_nonprogram_base_stream(self):
        base = {'greeting': [encoded('base-g1'), encoded('base-g2')],
                'request_help': [encoded('base-h1', 'request_help')]}
        programs = [encoded('bridge-0', 'request_help', 11), encoded('bridge-1', 'request_help', 11)]
        mixed = {**base, 'request_help': base['request_help'] + programs}
        recipe = {'act_weights': {'greeting': 1., 'request_help': 1.}}
        original = training._sample
        draws = {}
        for arm in ('baseline', 'mixed'):
            with pilot.controlled_sampler(programs, mixed=arm == 'mixed') as stats:
                draws[arm] = [training._sample(base if arm == 'baseline' else mixed, 16, recipe, seed)
                              for seed in range(100, 300)]
            draws[arm + '_stats'] = stats
            self.assertIs(training._sample, original)
        a, b = draws['baseline_stats'], draws['mixed_stats']
        self.assertEqual(a['base_stream_sha256'], b['base_stream_sha256'])
        self.assertEqual(a['mixture_mask_sha256'], b['mixture_mask_sha256'])
        self.assertEqual(a['program_examples'], 0)
        self.assertEqual(b['program_examples'], b['planned_program_positions'])
        self.assertGreater(b['realized_program_fraction'], .08)
        self.assertLess(b['realized_program_fraction'], .12)
        self.assertEqual(b['program_prefix_tokens'], b['program_examples'] * 11)
        for before, after in zip(draws['baseline'], draws['mixed']):
            for x, y in zip(before, after):
                if not y['id'].startswith('bridge-'):
                    self.assertEqual(x, y)

    def test_sampler_restores_production_function_on_exception(self):
        original = training._sample
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            with pilot.controlled_sampler([encoded('program')], mixed=True):
                raise RuntimeError('cancelled')
        self.assertIs(training._sample, original)

    def test_probe_parser_rejects_echoes_multiple_values_and_wrong_domain(self):
        for text, expected in [('0', 0), ('The result is 3.', 3), ('Answer: 2', 2), ('It is 1.', 1)]:
            self.assertEqual(pilot.parse_program_answer(text), expected)
        for text in ('Input 1. Result 2.', '1+1=2', '2 or 3', 'The result is 4.', '-1', '12', 'two', '', '2. Ignore this.'):
            self.assertIsNone(pilot.parse_program_answer(text), text)

    def test_gate_requires_both_complete_seed_pairs_and_all_controls(self):
        pairs = [passing_pair(1701), passing_pair(1702)]
        self.assertTrue(pilot.adoption_gate(pairs, frozen())['eligible_for_controlled_adoption'])
        self.assertFalse(pilot.adoption_gate(pairs[:1], frozen())['eligible_for_controlled_adoption'])
        for key, value in [('complete', False), ('updates', 196), ('validation_calls', 3),
                           ('validation_protocol_hash', 'changed'), ('initial_weight_sha256', 'different')]:
            altered = copy.deepcopy(pairs)
            altered[1]['mixed'][key] = value
            self.assertFalse(pilot.adoption_gate(altered, frozen())['eligible_for_controlled_adoption'], key)
        for key in ('base_stream_sha256', 'mixture_mask_sha256'):
            altered = copy.deepcopy(pairs)
            altered[0]['mixed']['sampling'][key] = 'different'
            self.assertFalse(pilot.adoption_gate(altered, frozen())['eligible_for_controlled_adoption'], key)

    def test_parent_cached_validation_requires_exact_known_protocol(self):
        pilot._require_parent_protocol({'validation_protocol_hash': 'expected'}, 'expected')
        for metadata in ({}, {'validation_protocol_hash': None}, {'validation_protocol_hash': 'different'}):
            with self.assertRaisesRegex(ValueError, 'exact frozen'):
                pilot._require_parent_protocol(metadata, 'expected')

    def test_gate_rejects_each_quality_regression_without_tuning_thresholds(self):
        pairs = [passing_pair(1701), passing_pair(1702)]
        failures = [('score', .419), ('false_memory_claims', 1),
                    ('whole_episode_pass_rate', .09), ('reply_loss', .321)]
        for key, value in failures:
            altered = copy.deepcopy(pairs)
            altered[0]['mixed']['validation'][key] = value
            self.assertFalse(pilot.adoption_gate(altered, frozen())['eligible_for_controlled_adoption'], key)
        altered = copy.deepcopy(pairs)
        altered[0]['mixed']['program']['accuracy'] = .30
        self.assertFalse(pilot.adoption_gate(altered, frozen())['eligible_for_controlled_adoption'])
        parent = frozen()
        parent['validation']['reply_loss'] = .305
        self.assertFalse(pilot.adoption_gate(pairs, parent)['eligible_for_controlled_adoption'])
        parent = frozen()
        parent['regression']['score'] = .6
        self.assertFalse(pilot.adoption_gate(pairs, parent)['eligible_for_controlled_adoption'])
        parent = frozen()
        parent['program']['accuracy'] = .8
        self.assertFalse(pilot.adoption_gate(pairs, parent)['eligible_for_controlled_adoption'])
        altered = copy.deepcopy(pairs)
        altered[0]['baseline']['regression']['score'] = .6
        self.assertFalse(pilot.adoption_gate(altered, frozen())['eligible_for_controlled_adoption'])
        canary = copy.deepcopy(pairs)
        for pair in canary:
            pair['planned_updates'] = 4
            pair['baseline']['updates'] = pair['mixed']['updates'] = 4
        self.assertFalse(pilot.adoption_gate(canary, frozen())['eligible_for_controlled_adoption'])
        constant = copy.deepcopy(pairs)
        for pair in constant:
            pair['baseline']['program']['accuracy'] = 0.
            pair['mixed']['program'].update(accuracy=.25, correct_by_input={'0': 4, '1': 0, '2': 0, '3': 0})
        self.assertFalse(pilot.adoption_gate(constant, frozen())['eligible_for_controlled_adoption'])
        missing_input = copy.deepcopy(pairs)
        missing_input[0]['mixed']['program']['correct_by_input']['3'] = 0
        self.assertFalse(pilot.adoption_gate(missing_input, frozen())['eligible_for_controlled_adoption'])


if __name__ == '__main__':
    unittest.main()
