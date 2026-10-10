import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from english_world.cli import load_neural_input, main, parser


class GraphCliTests(unittest.TestCase):
    def test_relation_corpus_uses_its_verified_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.json'
            path.write_text(json.dumps({'schema': 'golem-relation-corpus-v1'}))
            with patch('english_world.relation_corpus.load_relation_corpus', return_value={'verified': True}) as loader:
                self.assertEqual(load_neural_input(directory), {'verified': True})
                loader.assert_called_once_with(path)

    def test_pair_command_labels_actual_path_nodes(self):
        graph = {'nodes': {'a': {'lemmas': ['cat']}, 'b': {'lemmas': ['animal']}}}
        result = {'found': True, 'path': [{'source': 'a', 'target': 'b', 'relation': 'hypernym'}]}
        output = io.StringIO()
        with patch('english_world.lexical_graph.load_graph', return_value=graph), \
             patch('english_world.lexical_graph.relation_path', return_value=result), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(['graph', 'pair', 'cat', 'animal']), 0)
        self.assertEqual(json.loads(output.getvalue())['readable_path'],
                         [{'from': ['cat'], 'relation': 'hypernym', 'to': ['animal']}])

    def test_corpus_limit_matches_preparer(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['graph', 'corpus', '--max-concepts', '20001'])


if __name__ == '__main__':
    unittest.main()
