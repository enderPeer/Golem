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


if __name__ == '__main__':
    unittest.main()
