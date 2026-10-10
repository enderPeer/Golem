import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from english_world.neural_corpus import (
    SOURCE_FILE, _build_conversations, _render, load_neural_corpus, prepare_neural_corpus,
)


def message(tree, suffix, role, text, parent=None, rank=None, quality=.9, **overrides):
    result = {'message_id': tree + suffix, 'message_tree_id': tree, 'role': role, 'text': text,
              'parent_id': parent, 'rank': rank, 'lang': 'en', 'tree_state': 'ready_for_export',
              'review_result': True, 'deleted': False, 'synthetic': False,
              'labels': {'quality': {'value': quality, 'count': 3}}}
    result.update(overrides)
    return result


def fake_messages(count=12):
    records = []
    for i in range(count):
        tree = f'tree-{i:02d}'
        records += [message(tree, '-root', 'prompter', f'Question Number {i}?'),
                    message(tree, '-best', 'assistant', f'Correct Answer {i}.', parent=tree + '-root', rank=0),
                    message(tree, '-worse', 'assistant', f'Worse Answer {i}.', parent=tree + '-root', rank=1)]
    return records


class NeuralCorpusTests(unittest.TestCase):
    def test_tree_splits_before_branch_selection_and_best_rank(self):
        corpus, selected, stats = _build_conversations(fake_messages(20), 23, 20)
        self.assertEqual([len(corpus[x]) for x in ('train', 'validation', 'test')], [16, 2, 2])
        ids = [set(row['tree_id'] for row in corpus[split]) for split in corpus]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        for rows in corpus.values():
            for row in rows:
                self.assertTrue(row['message_ids'][1].endswith('-best'))
                self.assertEqual([turn['role'] for turn in row['turns']], ['user', 'assistant'])
        self.assertEqual(stats['selected_messages'], 40)
        self.assertEqual(len(selected), 40)

    def test_duplicate_root_group_does_not_cross_splits(self):
        records = fake_messages()
        records += [message('duplicate', '-root', 'prompter', ' question   number 0? '),
                    message('duplicate', '-reply', 'assistant', 'A Different Branch.', parent='duplicate-root', rank=0)]
        corpus, _, stats = _build_conversations(records, 23, 100)
        self.assertEqual(stats['eligible_root_groups'], 12)
        roots = [row['root_group_sha256'] for rows in corpus.values() for row in rows]
        self.assertEqual(len(roots), len(set(roots)))

    def test_rejected_nonenglish_and_low_quality_nodes_excluded(self):
        records = fake_messages()
        records += [message('extra', '-root', 'prompter', 'Otra pregunta', lang='es'),
                    message('extra', '-reply', 'assistant', 'A reply', parent='extra-root', rank=0)]
        records += [message('tree-00', '-bad', 'assistant', 'Rejected text', parent='tree-00-root', rank=0,
                            review_result=False),
                    message('tree-01', '-bad', 'assistant', 'Low quality text', parent='tree-01-root', rank=0, quality=.2)]
        _, selected, _ = _build_conversations(records, 23, 100)
        texts = {record['text'] for record in selected}
        self.assertNotIn('Rejected text', texts)
        self.assertNotIn('Low quality text', texts)
        self.assertNotIn('Otra pregunta', texts)

    def test_render_preserves_unicode_case_and_escapes_internal_role_lines(self):
        text = _render([{'role': 'user', 'content': 'Tell Me About café'},
                        {'role': 'assistant', 'content': 'Example:\nuser: hello\nassistant: Hi!\n    Code'}])
        self.assertIn('user: Tell Me About café', text)
        self.assertIn('\n user: hello', text)
        self.assertIn('\n assistant: Hi!', text)
        self.assertIn('\n    Code', text)

    def test_prepare_reuse_and_tamper_detection_with_fake_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / SOURCE_FILE
            with gzip.open(archive, 'wt', encoding='utf-8') as handle:
                for record in fake_messages():
                    handle.write(json.dumps(record) + '\n')
            license_file = root / 'LICENSE'
            license_file.write_text('Apache License Version 2.0\n', encoding='utf-8')
            card = root / 'README.md'
            card.write_text('Official fake test fixture only.\n', encoding='utf-8')

            def fake_fetch(url, cache_dir, **kwargs):
                path = archive if url.endswith(SOURCE_FILE) else license_file if url.endswith('/LICENSE') else card
                return path, {'url': url, 'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

            with patch('english_world.neural_corpus.fetch_file', side_effect=fake_fetch) as fetch:
                destination = prepare_neural_corpus(root / 'prepared', seed=23, max_conversations=12)
                self.assertEqual(fetch.call_count, 3)
                dataset = load_neural_corpus(destination)
                self.assertEqual(dataset['manifest']['license'], 'Apache-2.0')
                self.assertEqual(sum(len(v) for v in dataset['conversations'].values()), 12)
                again = prepare_neural_corpus(root / 'prepared', seed=23, max_conversations=12)
                self.assertEqual(destination, again)
                self.assertEqual(fetch.call_count, 3)
                with self.assertRaises(ValueError):
                    prepare_neural_corpus(root / 'prepared', seed=99, max_conversations=12)
            snapshot = root / 'prepared/source_snapshot/selected_messages.jsonl'
            with snapshot.open('a', encoding='utf-8') as handle:
                handle.write('{}\n')
            with self.assertRaises(ValueError):
                load_neural_corpus(destination)

    def test_deterministic_selection(self):
        a = _build_conversations(fake_messages(40), 3, 20)
        b = _build_conversations(list(reversed(fake_messages(40))), 3, 20)
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
