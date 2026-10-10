import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from english_world.cli import parser, load_neural_input


class ChatCliTests(unittest.TestCase):
    def test_greedy_terminal_command_is_accepted(self):
        args = parser().parse_args(['neural', 'chat', '--checkpoint', 'runs/chat-live/champion.pt',
                                    '--temperature', '0', '--max-bytes', '120'])
        self.assertEqual(args.temperature, 0)

    def test_negative_or_nonfinite_chat_temperatures_are_rejected(self):
        for value in ('-1', 'nan', 'inf'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(['neural', 'chat', '--checkpoint', 'x.pt', '--temperature', value])

    def test_original_chat_loader_verifies_the_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.json'
            path.write_text(json.dumps({'schema': 'golem-chat-corpus-v1'}))
            with patch('english_world.chat_corpus.load_chat_corpus', return_value={'verified': True}) as load:
                self.assertEqual(load_neural_input(directory), {'verified': True})
                load.assert_called_once_with(path)


if __name__ == '__main__':
    unittest.main()
