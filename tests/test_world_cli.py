import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from english_world.cli import load_neural_input, parser


class WorldCliTests(unittest.TestCase):
    def test_world_defaults_select_separate_artifacts(self):
        args = parser().parse_args(['neural', 'world-train'])
        self.assertEqual(args.data, 'data/conversation-world/dataset.json')
        self.assertEqual(args.run_dir, 'runs/conversation-world')

    def test_world_loader_uses_verified_simulation_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.json'
            path.write_text(json.dumps({'schema': 'golem-conversation-sim-v1'}))
            with patch('english_world.conversation_sim.load_conversation_sim', return_value={'verified': True}) as load:
                self.assertEqual(load_neural_input(directory), {'verified': True})
                load.assert_called_once_with(path)

    def test_free_rollout_rejects_training_partition(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['neural', 'world-eval', '--checkpoint', 'x.pt', '--split', 'train'])

    def test_world_train_console_is_compact_but_full_rollouts_remain_on_disk(self):
        from tests.test_semantic_training import config, dataset, rollout
        from english_world import semantic_training
        cfg = {**config(), 'strategies': ['ordinary'], 'max_total_updates': 4,
               'rounds': 1, 'evaluate_test_on_complete': False}
        marker = 'FULL_ROLLOUT_DETAIL_' * 1000
        def detailed(*args):
            return {**rollout(*args), 'rows': [{'generated': marker}]}
        with tempfile.TemporaryDirectory() as directory, \
             patch('english_world.cli.load_config', return_value=cfg), \
             patch('english_world.cli.load_neural_input', return_value=dataset()), \
             patch.object(semantic_training, '_rollout', side_effect=detailed):
            args = parser().parse_args(['neural', 'world-train', '--run-dir', directory])
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                self.assertEqual(args.func(args), 0)
            summary = json.loads(console.getvalue())
            full = json.loads((Path(directory) / 'report.json').read_text())
            self.assertLess(len(console.getvalue()), 2048)
            self.assertNotIn('FULL_ROLLOUT_DETAIL', console.getvalue())
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['strategies']['ordinary']['updates'], 4)
            self.assertEqual(summary['report_path'], str(Path(directory).resolve() / 'report.json'))
            self.assertEqual(full['strategies']['ordinary']['best_validation']['rollout']['rows'][0]['generated'], marker)


if __name__ == '__main__':
    unittest.main()
