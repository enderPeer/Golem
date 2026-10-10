import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from english_world.lexical_export import (
    export_graphml, export_neighborhood, export_rank_table, ranked_words,
)


LICENSE = 'WordNet 3.0 Copyright 2006 Princeton University.\nPermission & conditions <retained> "in full".\n'


def graph_fixture():
    nodes = {
        'a:00000001': {'lemmas': ['hot', 'warm'], 'pos': 'a', 'definition': 'Heat <high> & "bright"'},
        'a:00000002': {'lemmas': ['cold', 'chilly'], 'pos': 'a', 'definition': 'Low <temperature> & ice'},
        'n:00000003': {'lemmas': ['hot'], 'pos': 'n', 'definition': 'An unrelated noun meaning'},
    }
    return {'nodes': nodes,
            'edges': [{'source': 'a:00000001', 'target': 'a:00000002', 'relation': 'antonym',
                       'source_lemma': 'hot', 'target_lemma': 'chilly'}],
            'word_index': {'hot': ['a:00000001', 'n:00000003'], 'warm': ['a:00000001'],
                           'cold': ['a:00000002'], 'chilly': ['a:00000002']},
            'rank': {'words': {'hot': .4, 'warm': .1, 'cold': .25, 'chilly': .25},
                     'synsets': {'a:00000001': .4, 'a:00000002': .5, 'n:00000003': .1}}}


class LexicalExportTests(unittest.TestCase):
    def test_rank_order_ties_meaning_counts_and_tsv_round_trip(self):
        graph = graph_fixture()
        rows = ranked_words(graph)
        self.assertEqual([row['word'] for row in rows], ['hot', 'chilly', 'cold', 'warm'])
        self.assertEqual([row['rank'] for row in rows], [1, 2, 3, 4])
        self.assertEqual(rows[0]['meanings'], 2)
        self.assertEqual(ranked_words(graph, 2), rows[:2])
        self.assertEqual(ranked_words(graph, 0), [])
        with self.assertRaises(ValueError):
            ranked_words(graph, -1)
        with tempfile.TemporaryDirectory() as tmp:
            path = export_rank_table(graph, Path(tmp) / 'ranks.tsv', license_text=LICENSE)
            with path.open(encoding='utf-8', newline='') as stream:
                saved = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual([row['word'] for row in saved], [row['word'] for row in rows])
            self.assertEqual(float(saved[0]['connectivity']), .4)
            self.assertEqual((Path(tmp) / 'WORDNET-LICENSE.txt').read_text(encoding='utf-8'), LICENSE)

    def test_graphml_is_valid_xml_and_preserves_lexical_relation_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_graphml(graph_fixture(), Path(tmp) / 'graph.graphml', license_text=LICENSE)
            document = ET.parse(path)
            ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
            nodes = document.findall('.//g:node', ns)
            edges = document.findall('.//g:edge', ns)
            self.assertEqual(len(nodes), 3)
            self.assertEqual(len(edges), 1)
            values = {datum.attrib['key']: datum.text for datum in nodes[0].findall('g:data', ns)}
            self.assertEqual(values['definition'], 'Heat <high> & "bright"')
            edge = edges[0]
            self.assertEqual(edge.attrib['source'], 'a:00000001')
            self.assertEqual(edge.attrib['target'], 'a:00000002')
            values = {datum.attrib['key']: datum.text for datum in edge.findall('g:data', ns)}
            self.assertEqual(values, {'relation': 'antonym', 'source_lemma': 'hot', 'target_lemma': 'chilly'})
            self.assertEqual(document.find('.//g:graph/g:data[@key="license"]', ns).text, LICENSE)
            self.assertEqual((Path(tmp) / 'WORDNET-LICENSE.txt').read_text(encoding='utf-8'), LICENSE)

    def test_svg_uses_selected_sense_and_correct_antonym_lemma(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_neighborhood(graph_fixture(), 'hot', Path(tmp) / 'hot.svg',
                                       sense='a:00000001', license_text=LICENSE)
            document = ET.parse(path)
            ns = {'s': 'http://www.w3.org/2000/svg'}
            texts = [node.text or '' for node in document.findall('.//s:text', ns)]
            self.assertIn('chilly', texts)
            self.assertNotIn('cold, chilly', texts)
            self.assertTrue(any(text.startswith('antonym') for text in texts))
            self.assertTrue(any(text.startswith('same sense synonym') for text in texts))
            self.assertFalse(any('unrelated noun' in text for text in texts))
            self.assertEqual(document.find('s:metadata', ns).text, LICENSE)
            self.assertTrue(any('Heat <high> & "bright"' in text for text in texts))

    def test_svg_escaping_and_missing_sense_rejection(self):
        graph = graph_fixture()
        graph['word_index']['x<&'] = ['a:00000001']
        with tempfile.TemporaryDirectory() as tmp:
            path = export_neighborhood(graph, 'x<&', Path(tmp) / 'escaped.svg', license_text=LICENSE)
            root = ET.parse(path).getroot()
            self.assertIn('x<&', root.find('{http://www.w3.org/2000/svg}title').text)
            with self.assertRaises(ValueError):
                export_neighborhood(graph, 'hot', Path(tmp) / 'wrong.svg', sense='n:unknown')

    @unittest.skipUnless(importlib.util.find_spec('matplotlib') and importlib.util.find_spec('PIL'),
                         'PNG export requires optional Matplotlib and Pillow')
    def test_png_has_full_license_metadata_and_sidecar(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = export_neighborhood(graph_fixture(), 'hot', Path(tmp) / 'hot.png', license_text=LICENSE)
            with Image.open(path) as image:
                self.assertEqual(image.info['Copyright'], LICENSE)
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, 500)
            self.assertEqual((Path(tmp) / 'WORDNET-LICENSE.txt').read_text(encoding='utf-8'), LICENSE)


if __name__ == '__main__':
    unittest.main()
