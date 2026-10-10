import tempfile
from pathlib import Path
import unittest

from english_world.conversation_eval import (
    ACTS, _selected_episodes, evaluate_conversations, score_conversation_turn, validation_selection_key,
)


def frame(act, index=1, name=None, source=None):
    positive = {'greeting': ['hello', 'hi'], 'identity': ['golem'], 'presence': ['here', 'ready'],
                'name_set': ['meet', 'remember'], 'name_correction': ['use', 'got it'],
                'name_recall_unknown': ["don't know your name", 'have not told me'],
                'name_recall_known': [], 'topic_chat': ['chat']}
    rubric = {'required_any': positive[act], 'required_all': [name] if name else [],
              'forbidden': [], 'max_response_bytes': 120}
    return {'turn_index': index, 'meaning_id': 'meaning/' + act, 'act': act, 'user_name': name,
            'source_user_turn': source, 'rubric': rubric, 'case_id': 'case-' + str(index)}


def episode(identity, specifications):
    turns, frames = [], []
    for question, reference, act, name, source in specifications:
        turns.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': reference}])
        frames.append(frame(act, len(turns) - 1, name, source))
    return {'id': identity, 'family_id': identity, 'turns': turns, 'target_frames': frames}


class ConversationEvalTests(unittest.TestCase):
    def test_unknown_name_never_accepts_invented_name_or_assistant_identity(self):
        unknown = frame('name_recall_unknown')
        for response in ('Your name is Noah.', "You're Noah.", 'Noah.', 'I will call you Noah.',
                         "I don't know your name. Maybe Noah."):
            result = score_conversation_turn(response, unknown)
            self.assertFalse(result['passed'])
            self.assertTrue(result['false_memory_claims'])
        self.assertFalse(score_conversation_turn("I'm Golem.", unknown)['passed'])
        self.assertTrue(score_conversation_turn("I don't know your name yet.", unknown)['passed'])
        self.assertFalse(score_conversation_turn('Your name is Golem.', frame('identity'))['passed'])
        self.assertFalse(score_conversation_turn('Your name is not Mara.', frame('name_recall_known', name='Mara', source=0))['passed'])

    def test_actual_outputs_replace_teacher_history_and_do_not_create_user_facts(self):
        ep = episode('free', [('hello', 'TEACHER_REFERENCE_NEVER_SENT', 'greeting', None, None),
                              ('what is my name?', "I don't know your name yet.", 'name_recall_unknown', None, None)])
        histories = []

        def generate(history, **kwargs):
            histories.append(history)
            return 'Your name is Noah.' if len(histories) == 1 else "I don't know your name yet."

        report = evaluate_conversations(generate, {'episodes': {'validation': [ep]}})
        self.assertEqual(histories[1][1]['content'], 'Your name is Noah.')
        self.assertFalse(any('TEACHER_REFERENCE' in turn['content'] for history in histories for turn in history))
        self.assertEqual(report['false_memory_claims'], 1)
        self.assertIsNone(report['rows'][0]['rows'][1]['expected_user_name'])
        self.assertTrue(report['rows'][0]['rows'][1]['passed'])

    def test_explicit_name_correction_overwrites_old_name(self):
        ep = episode('correction', [('My name is Mara.', 'Nice to meet you, Mara.', 'name_set', 'Mara', 0),
              ('Actually, call me Ivo.', 'I will use Ivo.', 'name_correction', 'Ivo', 2),
              ('What is my name?', 'Your name is Ivo.', 'name_recall_known', 'Ivo', 2)])
        replies = iter(['Nice to meet you, Mara.', 'I will use Mara.', 'Your name is Mara.'])
        report = evaluate_conversations(lambda history, **kwargs: next(replies), {'episodes': {'validation': [ep]}})
        self.assertEqual(report['false_memory_claims'], 2)
        self.assertEqual(report['false_memory_turns'], 2)
        self.assertFalse(report['release_eligible'])
        replies = iter(['Nice to meet you, Mara.', 'I will use Ivo.', 'Your name is Ivo.'])
        good = evaluate_conversations(lambda history, **kwargs: next(replies), {'episodes': {'validation': [ep]}})
        self.assertEqual(good['false_memory_claims'], 0)
        self.assertEqual(good['whole_episode_pass_rate'], 1.)

    def test_adversarial_initial_history_is_used_once_but_not_counted_as_model_output(self):
        prefix = [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'Your name is Noah.'}]
        ep = {'id': 'poisoned', 'family_id': 'poisoned', 'initial_history': prefix, 'initial_history_count': 2,
              'turns': prefix + [{'role': 'user', 'content': 'What is my name?'},
                                 {'role': 'assistant', 'content': "I don't know your name yet."}],
              'target_frames': [frame('name_recall_unknown', 3)]}
        observed = []

        def generate(history, **kwargs):
            observed.append(history)
            return "I don't know your name yet."

        result = evaluate_conversations(generate, {'episodes': {'test': [ep]}}, split='test')
        self.assertEqual(len(observed[0]), 3)
        self.assertEqual(observed[0][:2], prefix)
        self.assertEqual(result['false_memory_claims'], 0)
        self.assertTrue(result['release_eligible'])
        ep['target_frames'][0] = frame('name_recall_known', 3, 'Noah', 1)
        with self.assertRaisesRegex(ValueError, 'USER'):
            evaluate_conversations(generate, {'episodes': {'test': [ep]}}, split='test')

    def test_repeated_correct_requests_are_not_penalized_but_irrelevant_repeats_are(self):
        repeated = episode('repeat', [('Who are you?', 'I am Golem.', 'identity', None, None),
                                      ('Who are you?', 'I am Golem.', 'identity', None, None)])
        result = evaluate_conversations(lambda history, **kwargs: 'I am Golem.', {'episodes': {'validation': [repeated]}})
        self.assertEqual(result['repeated_irrelevant_responses'], 0)
        self.assertEqual(result['score'], 1.)
        drift = episode('drift', [('hello', 'Hello!', 'greeting', None, None),
                                  ('Who are you?', 'I am Golem.', 'identity', None, None)])
        wrong = evaluate_conversations(lambda history, **kwargs: 'Hello!', {'episodes': {'validation': [drift]}})
        self.assertEqual(wrong['repeated_irrelevant_responses'], 1)
        self.assertFalse(wrong['rows'][0]['rows'][1]['passed'])
        self.assertFalse(score_conversation_turn('I am a program.', frame('presence'))['passed'])

    def test_counterfactual_names_and_predicted_act_are_separate_from_text_score(self):
        episodes = []
        for name in ('Mara', 'Ivo'):
            ep = episode(name, [('My name is ' + name + '.', 'Nice to meet you, ' + name + '.', 'name_set', name, 0),
                                ('What is my name?', 'Your name is ' + name + '.', 'name_recall_known', name, 0)])
            ep['counterfactual_pair_id'] = 'pair'
            episodes.append(ep)

        def generate(history, **kwargs):
            name = history[0]['content'].split()[-1].rstrip('.')
            if len(history) == 1:
                return {'text': 'Nice to meet you, ' + name + '.', 'meaning_id': 'name_set'}
            return {'text': 'Your name is ' + name + '.', 'meaning_id': 'name_recall_known'}

        report = evaluate_conversations(generate, {'episodes': {'validation': episodes}})
        self.assertTrue(report['counterfactual_pairs'][0]['passed'])
        self.assertEqual(report['meaning_accuracy'], 1.)
        self.assertEqual(report['score'], 1.)
        wrong = evaluate_conversations(lambda history, **kwargs: {'text': 'gibberish', 'meaning_id': 'name_recall_known'},
                                       {'episodes': {'validation': episodes}})
        self.assertEqual(wrong['score'], 0.)

    def test_regression_is_separate_and_cancellation_never_publishes_partial_result(self):
        ep = episode('synthetic-regression-boundary-test', [('hello', 'Hello!', 'greeting', None, None)] * 7)
        report = evaluate_conversations(lambda history, **kwargs: 'Hello!', {'episodes': {'regression': [ep]}}, split='regression')
        self.assertEqual(report['turns'], 7)
        self.assertEqual(report['evaluation_kind'], 'known_user_regression')
        with self.assertRaises(ValueError):
            validation_selection_key(report)
        stop = [False]

        def generate(history, **kwargs):
            stop[0] = True
            return 'Hello!'

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            with self.assertRaises(InterruptedError):
                evaluate_conversations(generate, {'episodes': {'validation': [ep]}}, output=path,
                                       should_stop=lambda: stop[0])
            self.assertFalse(path.exists())

    def test_real_simulator_selection_covers_all_acts_and_keeps_groups_complete(self):
        from english_world.conversation_sim import _original_spec, _build_episodes
        episodes, _, _ = _build_episodes(_original_spec(), 97, 1, 1)
        for split in ('validation', 'test'):
            chosen = _selected_episodes({'episodes': episodes}, split, 16)
            self.assertLessEqual(len(chosen), 16)
            self.assertEqual({target['act'] for item in chosen for target in item['target_frames']}, set(ACTS))
            self.assertEqual([item['id'] for item in chosen],
                             [item['id'] for item in _selected_episodes({'episodes': episodes}, split, 16)])
            selected_ids = {item['id'] for item in chosen}
            for key in ('counterfactual_pair_id', 'contrast_group_id'):
                for item in chosen:
                    if item.get(key):
                        complete = {other['id'] for other in episodes[split] if other.get(key) == item[key]}
                        self.assertTrue(complete <= selected_ids)

    def test_real_simulator_reference_oracle_rollouts_are_consistent(self):
        from english_world.conversation_sim import _original_spec, _build_episodes
        episodes, _, regressions = _build_episodes(_original_spec(), 97, 1, 1)
        episodes['regression'] = regressions
        # A TEST-ONLY reference oracle validates the benchmark, never production
        # inference. Keys use scripted users, allowing actual returned reference
        # variants to remain in the closed-loop assistant history.
        for split in ('validation', 'test', 'regression'):
            references = {}
            for item in episodes[split]:
                for target in item['target_frames']:
                    index = target['turn_index']
                    answer = item['turns'][index]['content']
                    check = score_conversation_turn(answer, target)
                    self.assertTrue(check['passed'], (split, target['act'], answer, check))
                    key = tuple(turn['content'] for turn in item['turns'][:index] if turn['role'] == 'user')
                    references.setdefault(key, answer)

            def oracle(history, **kwargs):
                return references[tuple(turn['content'] for turn in history if turn['role'] == 'user')]

            report = evaluate_conversations(oracle, {'episodes': episodes}, split=split, max_episodes=256)
            self.assertEqual(report['false_memory_claims'], 0)
            self.assertEqual(report['whole_episode_pass_rate'], 1.)
            if split != 'regression':
                self.assertEqual(report['covered_act_count'], 18)


if __name__ == '__main__':
    unittest.main()
