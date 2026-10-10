import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from english_world.lexical_graph import (
    _word_index, build_graph, load_graph, normalize_word, pagerank,
    parse_data_line, parse_wordnet, rank_related, relation_path,
)


ADJECTIVES = (
    '00000001 00 a 02 hot 0 warm 0 001 ! 00000002 a 0102 | high temperature\n'
    '00000002 00 s 02 cold 0 chilly 0 001 ! 00000001 a 0201 | low temperature\n'
)
NOUNS = (
    '00000001 00 n 02 bank 0 depository 0 001 @ 00000003 n 0000 | a financial institution\n'
    '00000002 00 n 01 bank 1 000 | the side of a river\n'
    '00000003 00 n 01 institution 0 001 ~ 00000001 n 0000 | an organization\n'
)


def fixture_archive(path):
    with zipfile.ZipFile(path, 'w') as archive:
        for suffix, text, index in (
            ('noun', NOUNS, 'bank n 2 1 @ 2 0 00000002 00000001\ndepository n 1 0 1 0 00000001\ninstitution n 1 0 1 0 00000003\n'),
            ('adj', ADJECTIVES, 'hot a 1 0 1 0 00000001\nwarm a 1 0 1 0 00000001\ncold a 1 0 1 0 00000002\nchilly a 1 0 1 0 00000002\n'),
            ('verb', '00000010 00 v 01 run 0 000 01 + 08 00 | move fast\n', 'run v 1 0 1 0 00000010\n'),
            ('adv', '', ''),
        ):
            archive.writestr('wordnet/data.' + suffix, text)
            archive.writestr('wordnet/index.' + suffix, index)
        archive.writestr('wordnet/LICENSE', 'WordNet test-fixture license\n')


class LexicalGraphTests(unittest.TestCase):
    def test_parse_preserves_sense_specific_lemmas_and_lexical_pointer_indexes(self):
        identity, node, edges = parse_data_line(ADJECTIVES.splitlines()[0])
        self.assertEqual(identity, 'a:00000001')
        self.assertEqual(node['lemmas'], ['hot', 'warm'])
        self.assertEqual(edges[0]['relation'], 'antonym')
        self.assertEqual(edges[0]['source_lemma'], 'hot')
        self.assertEqual(edges[0]['target_index'], 2)
        self.assertNotIn('target_lemma', edges[0])  # Resolved only after all synsets exist.

    def test_adjective_markers_and_verb_frames(self):
        identity, node, _ = parse_data_line('00000021 00 s 01 able(p) a 000 | capable\n')
        self.assertEqual(identity, 'a:00000021')
        self.assertEqual(node['pos'], 's')
        self.assertEqual(node['lemmas'], ['able'])
        self.assertEqual(node['lex_ids'], [10])
        self.assertEqual(node['syntactic_markers'], ['p'])
        _, verb, _ = parse_data_line('00000010 00 v 01 run 0 000 01 + 08 00 | move fast\n')
        self.assertEqual(verb['verb_frames'], [{'frame': 8, 'word_index': 0}])

    def test_lexical_target_and_official_index_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'wordnet.zip'
            fixture_archive(archive)
            nodes, edges, _, _, order = parse_wordnet(archive)
            antonym = next(edge for edge in edges if edge['source'] == 'a:00000001')
            self.assertEqual(antonym['target_lemma'], 'chilly')
            self.assertEqual(antonym['target'], 'a:00000002')
            self.assertEqual(_word_index(nodes, order)['bank'], ['n:00000002', 'n:00000001'])

    def test_sparse_pagerank_subset_and_probability_mass(self):
        edges = [{'source': 'a', 'target': 'b', 'relation': 'hypernym'},
                 {'source': 'a', 'target': 'b', 'relation': 'also_see'},
                 {'source': 'b', 'target': 'c', 'relation': 'hypernym'},
                 {'source': 'c', 'target': 'outside', 'relation': 'hypernym'}]
        ranks = pagerank({'a': {}, 'b': {}, 'c': {}}, edges)
        self.assertAlmostEqual(sum(ranks.values()), 1.)
        self.assertGreater(ranks['c'], ranks['b'])
        self.assertGreater(ranks['b'], ranks['a'])
        self.assertNotIn('outside', ranks)
        self.assertEqual(ranks, pagerank(['c', 'b', 'a'], list(reversed(edges))))
        self.assertEqual(pagerank([], []), {})

    def test_queries_do_not_turn_antonyms_into_synonyms_or_cross_senses(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'wordnet.zip'
            fixture_archive(archive)
            nodes, edges, _, _, order = parse_wordnet(archive)
            graph = {'nodes': nodes, 'edges': edges, 'word_index': _word_index(nodes, order),
                     'rank': {'synsets': pagerank(nodes, edges)}}
            hot = rank_related(graph, 'hot')
            self.assertEqual(next(row for row in hot if row['relation'] == 'same_sense_synonym')['lemmas'], ['warm'])
            self.assertEqual(next(row for row in hot if row['relation'] == 'antonym')['lemmas'], ['chilly'])
            warm = rank_related(graph, 'warm')
            self.assertFalse(any(row['relation'] == 'antonym' for row in warm))
            river = rank_related(graph, 'bank', sense='n:00000002')
            self.assertEqual(river, [])
            with self.assertRaises(ValueError):
                rank_related(graph, 'bank', sense='a:00000001')

    def test_build_reuse_license_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'wordnet.zip'
            fixture_archive(archive)
            meta = {'url': 'https://test.invalid/wordnet.zip', 'bytes': archive.stat().st_size,
                    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
            with patch('english_world.lexical_graph.fetch_file', return_value=(archive, meta)) as fetch:
                path = build_graph(root / 'graph', iterations=10)
                graph = load_graph(path)
                self.assertEqual(graph['counts']['synsets'], 6)
                self.assertEqual(graph['counts']['lexical_edges'], 2)
                self.assertAlmostEqual(sum(graph['rank']['words'].values()), 1.)
                self.assertEqual(build_graph(root / 'graph', iterations=10), path)
                self.assertEqual(fetch.call_count, 1)
                with self.assertRaises(ValueError):
                    build_graph(root / 'graph', iterations=11)
            with path.open('a', encoding='utf-8') as handle:
                handle.write(' ')
            with self.assertRaises(ValueError):
                load_graph(path)

    def test_normalization_preserves_display_separately(self):
        self.assertEqual(normalize_word(' Domestic_Cat '), 'domestic cat')
        self.assertIsNone(parse_data_line('  1 Copyright notice\n'))

    def test_pair_path_preserves_polysemy_and_lexical_opposition(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'wordnet.zip'
            fixture_archive(archive)
            nodes, edges, _, _, order = parse_wordnet(archive)
            graph = {'nodes': nodes, 'edges': edges, 'word_index': _word_index(nodes, order)}
            opposite = relation_path(graph, 'hot', 'chilly')
            self.assertTrue(opposite['found'])
            self.assertTrue(opposite['contains_antonym'])
            self.assertEqual(opposite['interpretation'], 'opposition (antonym)')
            self.assertEqual(opposite['path'][0]['target_lemma'], 'chilly')
            self.assertFalse(relation_path(graph, 'hot', 'cold', max_hops=1)['found'])
            same = relation_path(graph, 'hot', 'warm')
            self.assertEqual(same['distance'], 0)
            self.assertEqual(same['interpretation'], 'same-sense synonyms')
            self.assertFalse(relation_path(graph, 'bank', 'institution')['found'])
            finance = relation_path(graph, 'bank', 'institution', source_sense='n:00000001')
            self.assertTrue(finance['found'])
            self.assertEqual(finance['distance'], 1)


if __name__ == '__main__':
    unittest.main()
