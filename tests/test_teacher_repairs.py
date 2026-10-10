"""Teacher selection cannot turn learner output or held-out labels into targets."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from english_world.conversation_eval import evaluate_conversations
from english_world.semantic_model import ACTS
from english_world import world_replay as replay
from ops.build_teacher_repairs import build_teacher_repairs


def fixture():
    turns = [{'role': 'user', 'content': 'My name is Maya'},
             {'role': 'assistant', 'content': 'Hello Maya.'},
             {'role': 'user', 'content': 'Who are you?'},
             {'role': 'assistant', 'content': 'I am Golem, a program.'},
             {'role': 'user', 'content': 'What is my name?'},
             {'role': 'assistant', 'content': 'Your name is Maya.'}]
    frames = [{'turn_index': index, 'act': act, 'meaning_id': ACTS.index(act),
               'user_name': 'Maya', 'source_user_turn': 0,
               'rubric': {'required_any': [word], 'required_all': [], 'forbidden': [],
                          'max_response_bytes': 120}}
              for index, act, word in ((1, 'name_set', 'Maya'), (3, 'identity', 'Golem'),
                                      (5, 'name_recall_known', 'Maya'))]
    episode = {'id': 'train-one', 'split': 'train', 'family_id': 'train-family',
               'turns': turns, 'target_frames': frames, 'initial_history': [],
               'initial_history_count': 0, 'known_regression': False}
    samples = [{'id': 'base-' + str(frame['turn_index']), 'episode_id': episode['id'],
                'turns': turns[:frame['turn_index']], 'reply': turns[frame['turn_index']]['content'],
                'act': frame['act'], 'meaning_id': frame['meaning_id'], 'frame': frame} for frame in frames]
    return {'schema': 'golem-conversation-sim-v1', 'meaning_vocab': list(ACTS),
            'episodes': {'train': [episode], 'validation': [{'id': 'heldout-val'}],
                         'test': [{'id': 'heldout-test'}], 'regression': []},
            'samples': {'train': samples, 'validation': [], 'test': []},
            'manifest': {'content_sha256': 'b' * 64}}


class TeacherRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = fixture()
        self.checkpoint = self.root / 'frozen.pt'
        self.checkpoint.write_bytes(b'Frozen checkpoint identity; model loading is not needed by selection.')
        self.report_path = self.root / 'train-report.json'
        self.output = self.root / 'teacher.json'
        self.report = self.make_report()
        self.save_report()

    def make_report(self, replies=None):
        frames = self.data['episodes']['train'][0]['target_frames']
        predictions = iter(zip(frames, replies or ['Your name is Noah.'] * 3))
        def generate(*args, **kwargs):
            frame, text = next(predictions)
            return {'text': text, 'meaning_id': frame['act']}
        report = evaluate_conversations(generate, {'episodes': {'regression': self.data['episodes']['train']}},
                                        split='regression', max_episodes=1)
        report.update(split='train', source_partition='train', evaluation_kind='training_diagnostic',
                      checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                      corpus_content_sha256=self.data['manifest']['content_sha256'])
        return report

    def save_report(self):
        self.report_path.write_text(json.dumps(self.report), encoding='utf-8')

    def build(self, **kwargs):
        return build_teacher_repairs(self.report_path, self.data, self.checkpoint, self.output, **kwargs)

    def test_only_verified_targets_added_and_heldout_objects_unchanged(self):
        original = copy.deepcopy(self.data)
        result = self.build()
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['skipped_unchanged_history'], 1)
        augmented = replay.apply_repair_overlay(self.data, self.output)
        added = augmented['samples']['train'][3:]
        self.assertEqual({row['reply'] for row in added}, {'I am Golem, a program.', 'Your name is Maya.'})
        self.assertTrue(all(row['frame']['user_name'] == 'Maya' for row in added))
        self.assertTrue(all('Noah' in row['turns'][1]['content'] for row in added))
        self.assertIs(augmented['samples']['validation'], self.data['samples']['validation'])
        self.assertIs(augmented['samples']['test'], self.data['samples']['test'])
        self.assertIs(augmented['episodes'], self.data['episodes'])
        self.assertEqual(self.data, original)
        artifact = json.loads(self.output.read_text())
        self.assertEqual(artifact['teacher_selection']['multiplicity_per_unique_context'], 1)
        self.assertEqual(len(artifact['teacher_selection']['evidence']), 2)
        self.assertEqual(len({row['id'] for row in added}), 2)

    def test_reply_failure_recomputed_not_trusted_from_report(self):
        for row in self.report['rows'][0]['rows']:
            row['passed'] = True
            row['predicted_meaning_correct'] = False
        self.save_report()
        self.assertEqual(self.build()['samples'], 2)

    def test_correct_responses_and_wrong_meanings_are_not_selected(self):
        self.report = self.make_report(['Your name is Noah.', 'I am Golem, a program.', 'Your name is Noah.'])
        self.report['rows'][0]['rows'][-1]['predicted_meaning_id'] = 'identity'
        self.save_report()
        result = self.build()
        self.assertEqual(result['samples'], 0)
        self.assertEqual(result['skipped_reply_passed'], 1)
        self.assertEqual(result['skipped_incorrect_meaning'], 1)

    def test_partition_checkpoint_and_corpus_identity_rejected(self):
        original = copy.deepcopy(self.report)
        variants = [('split', 'validation'), ('source_partition', 'test'),
                    ('checkpoint_sha256', 'a' * 64), ('corpus_content_sha256', 'c' * 64)]
        for key, value in variants:
            with self.subTest(key=key):
                self.report = copy.deepcopy(original)
                self.report[key] = value
                self.save_report()
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_forged_user_fact_target_frame_or_continuity_rejected(self):
        original = copy.deepcopy(self.report)
        for kind in ('user', 'reference', 'frame', 'prior_assistant', 'actual_history', 'duplicate', 'outside_train'):
            with self.subTest(kind=kind):
                self.report = copy.deepcopy(original)
                trace = self.report['rows'][0]
                row = trace['rows'][1]
                if kind == 'user':
                    row['input_history'][0]['content'] = 'My name is Noah'
                elif kind == 'reference':
                    row['reference'] = 'Your name is Noah.'
                elif kind == 'frame':
                    row['frame']['user_name'] = 'Noah'
                elif kind == 'prior_assistant':
                    row['input_history'][1]['content'] = 'An unrecorded reply.'
                elif kind == 'actual_history':
                    trace['actual_history'][1]['content'] = 'An unrecorded reply.'
                elif kind == 'duplicate':
                    self.report['rows'].append(copy.deepcopy(trace))
                else:
                    trace['id'] = 'heldout-test'
                self.save_report()
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_regressions_are_excluded_even_if_they_are_in_train(self):
        self.data['episodes']['train'][0]['known_regression'] = True
        with self.assertRaisesRegex(ValueError, 'regressions'):
            self.build()
        self.assertFalse(self.output.exists())

    def test_limits_focus_order_and_existing_output(self):
        for limit in (0, 128, True):
            with self.assertRaises(ValueError):
                self.build(max_samples=limit)
        for acts in ((), ('unknown',), ('identity', 'identity')):
            with self.assertRaises(ValueError):
                self.build(focus_acts=acts)
        result = self.build(max_samples=1, focus_acts=('identity', 'name_recall_known'))
        self.assertEqual(result['per_act'], {'identity': 1})
        original = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.build()
        self.assertEqual(self.output.read_bytes(), original)

    def test_cancelled_selection_publishes_nothing(self):
        calls = [0]
        def cancelled():
            calls[0] += 1
            return calls[0] >= 4
        with self.assertRaises(InterruptedError):
            self.build(should_stop=cancelled)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.*.tmp')))


if __name__ == '__main__':
    unittest.main()
