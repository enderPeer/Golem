"""Learner mistakes may become context, but cannot become facts or target labels."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from english_world import world_replay as replay
from english_world.neural_model import NeuralConfig
from english_world.semantic_model import ACTS, SemanticGRU, save_semantic_checkpoint


def base_data():
    turns = [{'role': 'user', 'content': 'My name is Maya'},
             {'role': 'assistant', 'content': 'Hello Maya.'},
             {'role': 'user', 'content': 'Who are you?'},
             {'role': 'assistant', 'content': 'I am Golem.'},
             {'role': 'user', 'content': 'What is my name?'},
             {'role': 'assistant', 'content': 'Your name is Maya.'}]
    frames = [{'turn_index': index, 'act': act, 'meaning_id': ACTS.index(act),
               'user_name': 'Maya', 'source_user_turn': 0} for index, act in
              ((1, 'name_set'), (3, 'identity'), (5, 'name_recall_known'))]
    episode = {'id': 'train-episode', 'split': 'train', 'family_id': 'train-family',
               'turns': turns, 'target_frames': frames, 'initial_history': [], 'initial_history_count': 0,
               'known_regression': False}
    samples = [{'id': 'base-' + str(frame['turn_index']), 'episode_id': episode['id'],
                'turns': turns[:frame['turn_index']], 'reply': turns[frame['turn_index']]['content'],
                'act': frame['act'], 'meaning_id': frame['meaning_id'], 'frame': frame} for frame in frames]
    return {'schema': 'golem-conversation-sim-v1', 'meaning_vocab': list(ACTS),
            'episodes': {'train': [episode], 'validation': [{'id': 'validation-forbidden'}],
                         'test': [{'id': 'test-forbidden'}], 'regression': [{'id': 'regression-forbidden'}]},
            'samples': {'train': samples, 'validation': [], 'test': [], 'regression': []}, 'manifest': {}}


class WorldReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source.pt'
        model = SemanticGRU(NeuralConfig(embedding_dim=8, hidden_size=12, num_layers=1, dropout=0.))
        save_semantic_checkpoint(self.source, model)
        self.data = base_data()

    def build(self, **kwargs):
        path = self.root / 'repair.json'
        with patch.object(replay, 'generate_semantic', return_value={'text': 'Your name is Noah.',
                                                                     'meaning_id': 'name_recall_known'}):
            report = replay.build_repair_overlay(self.source, self.data, path, seed=3, max_episodes=1, **kwargs)
        return path, report

    def rewrite(self, path, artifact):
        artifact['content_sha256'] = replay._digest({k: v for k, v in artifact.items() if k != 'content_sha256'})
        path.write_text(json.dumps(artifact), encoding='utf-8')

    def test_wrong_learner_name_is_context_only_and_base_heldout_objects_unchanged(self):
        original = copy.deepcopy(self.data)
        source_bytes = self.source.read_bytes()
        path, report = self.build()
        self.assertEqual(report['samples'], 2)
        augmented = replay.apply_repair_overlay(self.data, path)
        repairs = augmented['samples']['train'][len(self.data['samples']['train']):]
        self.assertEqual([row['reply'] for row in repairs], ['I am Golem.', 'Your name is Maya.'])
        self.assertTrue(all(row['frame']['user_name'] == 'Maya' for row in repairs))
        self.assertTrue(any('Noah' in turn['content'] for turn in repairs[-1]['turns']))
        self.assertIs(augmented['samples']['validation'], self.data['samples']['validation'])
        self.assertIs(augmented['samples']['test'], self.data['samples']['test'])
        self.assertIs(augmented['episodes'], self.data['episodes'])
        self.assertEqual(self.data, original)
        self.assertEqual(self.source.read_bytes(), source_bytes)
        with self.assertRaisesRegex(ValueError, 'pristine'):
            replay.apply_repair_overlay(augmented, path)

    def test_forged_teacher_target_or_user_fact_is_rejected_even_with_updated_checksum(self):
        path, _ = self.build()
        original = json.loads(path.read_text())
        for kind in ('reply', 'user', 'partition'):
            artifact = copy.deepcopy(original)
            if kind == 'reply':
                artifact['samples'][0]['reply'] = 'Your name is Noah.'
            elif kind == 'user':
                artifact['samples'][0]['turns'][0]['content'] = 'My name is Noah'
            else:
                artifact['samples'][0]['episode_id'] = 'validation-forbidden'
            self.rewrite(path, artifact)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                replay.apply_repair_overlay(self.data, path)

    def test_context_must_match_recorded_actual_rollout(self):
        path, _ = self.build()
        artifact = json.loads(path.read_text())
        history = artifact['samples'][0]['turns']
        history[1]['content'] = 'Another unrecorded reply.'
        artifact['samples'][0] = replay._entry(self.data['episodes']['train'][0], 3, history,
                                              artifact['source_checkpoint_sha256'])
        self.rewrite(path, artifact)
        with self.assertRaisesRegex(ValueError, 'recorded learner rollout'):
            replay.apply_repair_overlay(self.data, path)

    def test_overlay_limits_and_changed_base_are_rejected(self):
        path, report = self.build(max_samples=1)
        self.assertEqual(report['samples'], 1)
        changed = copy.deepcopy(self.data)
        changed['samples']['train'][0]['reply'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'another base'):
            replay.apply_repair_overlay(changed, path)
        with self.assertRaises(ValueError):
            replay.build_repair_overlay(self.source, self.data, self.root / 'too-large.json', 1, max_samples=513)

    def test_explicit_regression_episodes_are_not_replayed(self):
        self.data['episodes']['train'][0]['known_regression'] = True
        path, report = self.build()
        self.assertEqual(report['samples'], 0)
        self.assertEqual(report['episodes'], 0)
        self.assertEqual(replay.apply_repair_overlay(self.data, path)['samples']['train'], self.data['samples']['train'])

    def test_cancellation_never_publishes_partial_overlay(self):
        responses = []
        def generated(*args, **kwargs):
            responses.append('called')
            return {'text': 'untrusted learner reply'}
        path = self.root / 'cancelled.json'
        with patch.object(replay, 'generate_semantic', side_effect=generated):
            with self.assertRaises(InterruptedError):
                replay.build_repair_overlay(self.source, self.data, path, 1,
                    should_stop=lambda: len(responses) >= 1)
        self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
