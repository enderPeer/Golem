"""Name recall uses generated USER-grounded assertions, with paired controls."""
import copy
import unittest

from english_world import semantic_training as training
from english_world.semantic_model import ACTS
from ops import name_copy_pilot as pilot


def recall_report(text='Your name is Asha.', expected='Asha'):
    frame = {'act': 'name_recall_known', 'meaning_id': ACTS.index('name_recall_known'),
        'user_name': expected, 'source_user_turn': 0, 'turn_index': 3,
        'rubric': {'required_any': [expected], 'required_all': [], 'forbidden': [], 'max_response_bytes': 120}}
    row = {'act': 'name_recall_known', 'turn_index': 3, 'case_id': 'val-known', 'frame': frame,
           'expected_user_name': expected, 'generated': text,
           'input_history': [{'role': 'user', 'content': f'My name is {expected}.'},
                             {'role': 'assistant', 'content': 'Hello.'},
                             {'role': 'user', 'content': 'What is my name?'}]}
    return {'split': 'validation', 'rows': [{'id': 'val-episode', 'rows': [row]}]}


def names(accuracy):
    return {'accuracy': accuracy, 'denominator': 4, 'case_signature_sha256': 'cases', 'false_memory_claims': 0}


def frozen():
    return {'weight_sha256': 'weights', 'validation_protocol_hash': 'protocol',
        'validation': {'score': .4, 'false_memory_claims': 0, 'whole_episode_pass_rate': .05, 'reply_loss': .35},
        'name_recall': names(.25),
        'regression': {'score': .5, 'false_memory_claims': 0, 'whole_episode_pass_rate': 0.}}


def pair(seed):
    baseline = {'complete': True, 'updates': 200, 'validation_calls': 4,
        'initial_weight_sha256': 'weights', 'validation_protocol_hash': 'protocol',
        'data_fingerprint': 'same-data', 'verified_name_weight': 1.,
        'sampling': {'sample_stream_sha256': 'stream', 'examples': 1600, 'reply_tokens': 60000,
                     'prefix_tokens': 120000, 'verified_name_tokens': 500},
        'validation': {'score': .4, 'false_memory_claims': 0, 'whole_episode_pass_rate': .1, 'reply_loss': .3},
        'name_recall': names(.25),
        'regression': {'score': .5, 'false_memory_claims': 0, 'whole_episode_pass_rate': 0.}}
    targeted = copy.deepcopy(baseline)
    targeted['validation'].update(score=.42, reply_loss=.31)
    targeted['verified_name_weight'] = 4.
    targeted['name_recall'] = names(.5)
    return {'seed': seed, 'planned_updates': 200, 'baseline': baseline, 'targeted': targeted}


class NameCopyPilotTests(unittest.TestCase):
    def test_complete_correct_assertions_pass_and_case_signature_is_stable(self):
        signatures = set()
        for text in ('Your name is Asha.', 'Your name in this chat is Asha.',
                     'You told me your name is Asha.', 'You asked me to call you Asha.', 'Asha.', 'asha'):
            result = pilot.strict_name_recall(recall_report(text))
            self.assertEqual(result['accuracy'], 1., text)
            self.assertEqual(result['denominator'], 1)
            self.assertEqual(result['expected_names'], ['Asha'])
            signatures.add(result['case_signature_sha256'])
        self.assertEqual(len(signatures), 1)
        self.assertEqual(pilot.strict_name_recall(recall_report('Your name is May.', 'May'))['accuracy'], 1.)

    def test_substrings_extra_claims_guesses_denials_and_questions_fail(self):
        for text in ('Your name is Ashanti.', 'Your name is Asha. I will call you Ivo.',
                     'Maybe your name is Asha.', 'I think your name is Asha.',
                     'Your name might be Asha.', 'Your name is not Asha.',
                     "I don't know your name, Asha.", 'Your name is Asha?',
                     'Asha is a name.', "I'm Golem."):
            result = pilot.strict_name_recall(recall_report(text))
            self.assertEqual(result['accuracy'], 0., text)
        result = pilot.strict_name_recall(recall_report('Your name is Asha. I will call you Ivo.'))
        self.assertGreater(result['false_memory_claims'], 0)

    def test_assistant_claim_cannot_supply_expected_name_or_gate_partition(self):
        report = recall_report()
        report['rows'][0]['rows'][0]['input_history'][0] = {'role': 'assistant', 'content': 'Your name is Asha.'}
        with self.assertRaisesRegex(ValueError, 'USER evidence'):
            pilot.strict_name_recall(report)
        for split in ('train', 'test', 'regression'):
            with self.assertRaisesRegex(ValueError, 'VALIDATION'):
                pilot.strict_name_recall({**recall_report(), 'split': split})
        with self.assertRaisesRegex(ValueError, 'no known-name'):
            pilot.strict_name_recall({'split': 'validation', 'rows': []})

    def test_only_known_recall_rows_count_and_duplicates_rejected(self):
        report = recall_report()
        unrelated = copy.deepcopy(report['rows'][0]['rows'][0])
        unrelated.update(act='identity', generated='Your name is Ivo.', turn_index=5)
        report['rows'][0]['rows'].append(unrelated)
        self.assertEqual(pilot.strict_name_recall(report)['denominator'], 1)
        report = recall_report()
        report['rows'].append(copy.deepcopy(report['rows'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            pilot.strict_name_recall(report)

    def test_observer_preserves_rows_and_accounts_name_and_all_tokens(self):
        rows = [{'id': 'name', 'targets': [1, 2, 3], 'prefix': [4, 5], 'verified_name_mask': [True, True, False]},
                {'id': 'other', 'targets': [1, 2], 'prefix': [5], 'verified_name_mask': [False, False]}]
        buckets = {'name_set': rows}
        recipe = {'act_weights': {'name_set': 1.}}
        original = training._sample
        expected = [original(buckets, 8, recipe, seed) for seed in (1, 2)]
        with pilot.observe_sampler() as stats:
            actual = [training._sample(buckets, 8, recipe, seed) for seed in (1, 2)]
        self.assertEqual(actual, expected)
        self.assertIs(training._sample, original)
        flat = [row for batch in expected for row in batch]
        self.assertEqual(stats['verified_name_tokens'], sum(sum(row['verified_name_mask']) for row in flat))
        self.assertEqual(stats['reply_tokens'], sum(len(row['targets']) for row in flat))
        self.assertEqual(stats['prefix_tokens'], sum(len(row['prefix']) for row in flat))
        self.assertEqual(stats['examples'], 16)
        self.assertEqual(stats['calls'], 2)
        with self.assertRaises(RuntimeError):
            with pilot.observe_sampler():
                raise RuntimeError('cancelled')
        self.assertIs(training._sample, original)

    def test_all_predeclared_gates_and_two_seeds_required(self):
        pairs = [pair(3401), pair(3402)]
        result = pilot.adoption_gate(pairs, frozen())
        self.assertTrue(result['eligible_for_controlled_followup'])
        self.assertFalse(result['live_release_authorized'])
        self.assertFalse(pilot.adoption_gate(pairs[:1], frozen())['eligible_for_controlled_followup'])
        for key, value in [('complete', False), ('updates', 199), ('validation_calls', 3),
                           ('initial_weight_sha256', 'other'), ('validation_protocol_hash', 'other'),
                           ('data_fingerprint', 'other'), ('verified_name_weight', 1.)]:
            changed = copy.deepcopy(pairs)
            changed[1]['targeted'][key] = value
            self.assertFalse(pilot.adoption_gate(changed, frozen())['eligible_for_controlled_followup'], key)
        for key, value in [('sample_stream_sha256', 'other'), ('reply_tokens', 59999), ('verified_name_tokens', 499)]:
            changed = copy.deepcopy(pairs)
            changed[0]['targeted']['sampling'][key] = value
            self.assertFalse(pilot.adoption_gate(changed, frozen())['eligible_for_controlled_followup'], key)
        canary = copy.deepcopy(pairs)
        for value in canary:
            value['planned_updates'] = value['baseline']['updates'] = value['targeted']['updates'] = 4
        self.assertFalse(pilot.adoption_gate(canary, frozen())['eligible_for_controlled_followup'])

    def test_name_metric_gain_and_fixed_cases_are_required_not_classifier(self):
        for key, value in [('accuracy', .34), ('denominator', 3), ('case_signature_sha256', 'changed'), ('false_memory_claims', 1)]:
            pairs = [pair(3401), pair(3402)]
            pairs[0]['targeted']['name_recall'][key] = value
            self.assertFalse(pilot.adoption_gate(pairs, frozen())['eligible_for_controlled_followup'], key)
        parent = frozen()
        parent['name_recall']['accuracy'] = .5
        self.assertFalse(pilot.adoption_gate([pair(3401), pair(3402)], parent)['eligible_for_controlled_followup'])

    def test_conversation_parent_and_regression_degradation_block_adoption(self):
        for key, value in [('score', .419), ('false_memory_claims', 1), ('reply_loss', .321), ('whole_episode_pass_rate', .09)]:
            pairs = [pair(3401), pair(3402)]
            pairs[0]['targeted']['validation'][key] = value
            self.assertFalse(pilot.adoption_gate(pairs, frozen())['eligible_for_controlled_followup'], key)
        pairs = [pair(3401), pair(3402)]
        pairs[0]['targeted']['regression']['score'] = .49
        self.assertFalse(pilot.adoption_gate(pairs, frozen())['eligible_for_controlled_followup'])
        parent = frozen()
        parent['validation']['score'] = .43
        self.assertFalse(pilot.adoption_gate(pairs, parent)['eligible_for_controlled_followup'])


if __name__ == '__main__':
    unittest.main()
