"""Origin, full-reference trace, bounded ABI and deterministic export checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from english_world import program_bridge as bridge


class ProgramBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proof = bridge.verify_increment_bridge()

    def test_frozen_observed_candidate_and_reference_hashes_are_exact(self):
        sources = bridge._sources()
        self.assertEqual(set(sources), set(bridge.PINNED))
        for name, content in sources.items():
            self.assertEqual(hashlib.sha256(content).hexdigest(), bridge.PINNED[name])
        origin = self.proof['origin']
        self.assertEqual(origin['codons'], 13)
        self.assertEqual(origin['operation_sequence_carriers'], 42577)
        self.assertEqual(origin['spelling_carriers'], 9846)
        self.assertEqual(origin['spelling'], bridge.SPELLING)
        self.assertEqual(origin['source_pcs_zero_based'], [11, 12, 6, 10])

    def test_all_four_inputs_match_reset_reference_in_every_declared_context(self):
        self.assertEqual(self.proof['truth_table'], [1, 2, 3, 0])
        cases = self.proof['verification']['reset_cases']
        self.assertEqual(len(cases), 12)
        for context in bridge.CONTEXTS:
            selected = [case for case in cases if case['context'] == context]
            self.assertEqual([case['answers'][0]['input'] for case in selected], [0, 1, 2, 3])
            self.assertEqual([case['answers'][0]['output'] for case in selected], [1, 2, 3, 0])
            self.assertTrue(all(case['instructions'] < 2048 for case in selected))

    def test_actual_wrapping_paths_cover_division_and_both_sense_branches(self):
        cases = self.proof['verification']['repeated_cases']
        paths = []
        for case in cases:
            self.assertEqual(len(case['answers']), 64)
            self.assertGreaterEqual(case['successful_parent_divisions'], 1)
            self.assertTrue(case['parent_gene_unchanged'])
            self.assertTrue(case['discarded_operations_preserve_A_H_io_xin_open'])
            for answer in case['answers']:
                self.assertEqual(answer['output'], (answer['input'] + 1) % 4)
                path = answer['path']
                self.assertEqual([pc for pc, op in path if op in bridge.SLICE_OPS], [11, 12, 6, 10])
                self.assertIn([3, 'COPY'], path)
                self.assertIn([4, 'TERM'], path)
                self.assertIn([7, 'SENSE'], path)
                paths.append(path)
        self.assertTrue(any([5, 'DIVIDE'] in path for path in paths))
        self.assertTrue(any([9, 'HARVEST'] in path for path in paths))
        self.assertTrue(any([9, 'HARVEST'] not in path for path in paths))
        self.assertEqual(self.proof['verification']['verified_answers'], 204)

    def test_changed_slice_or_ecological_dependency_is_rejected(self):
        with self.assertRaises(ValueError):
            bridge.execute_slice(0, ('IN', 'LISTEN', 'LDA', 'OUT'))
        sources = bridge._sources()
        reference = bridge._reference(sources)
        genome, _ = bridge._observed_candidate(sources, reference)
        # Replace the retained increment with LISTEN. The original ecological
        # VM now depends on a neighbor, and must fail the extraction proof.
        genome[12] = reference.codon_value('CGG')
        with self.assertRaisesRegex(ValueError, 'unapproved dependency|proposed noncontiguous slice|differs'):
            bridge._full_reference_fixture(reference, genome, [0], bridge.CONTEXTS[0])

    def test_export_is_deterministic_executable_and_resets_each_call(self):
        with tempfile.TemporaryDirectory() as root:
            first = bridge.build_increment_bridge(Path(root) / 'first')
            second = bridge.build_increment_bridge(Path(root) / 'second')
            for key in ('certificate_path', 'module_path'):
                self.assertEqual(Path(first[key]).read_bytes(), Path(second[key]).read_bytes())
            verified = bridge.load_verified_bridge(first['certificate_path'])
            self.assertEqual(verified['certificate_sha256'], first['certificate_sha256'])
            namespace = {}
            exec(compile(Path(first['module_path']).read_text(), first['module_path'], 'exec'), namespace)
            function = namespace['increment_mod4']
            self.assertEqual([function(x) for x in range(4)], verified['truth_table'])
            self.assertEqual([function(x) for x in [3, 3, 0, 2, 1] * 20], [0, 0, 1, 3, 2] * 20)
            for value in (-1, 4, True, False, 1.0, '1', None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    function(value)
            self.assertTrue((Path(root) / 'first' / 'LICENSE').is_file())
            with self.assertRaises(ValueError):
                bridge.build_increment_bridge(Path(root) / 'first')

    def test_declared_table_cannot_replace_proof_even_with_recomputed_certificate_hash(self):
        with tempfile.TemporaryDirectory() as root:
            result = bridge.build_increment_bridge(root)
            path = Path(result['certificate_path'])
            forged = json.loads(path.read_text())
            forged['truth_table'] = [0, 0, 0, 0]
            forged['certificate_sha256'] = bridge._digest({k: v for k, v in forged.items() if k != 'certificate_sha256'})
            path.write_text(json.dumps(forged), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'recomputed source proof'):
                bridge.load_verified_bridge(path)

    def test_tampered_export_is_never_executed_by_certificate_loader(self):
        with tempfile.TemporaryDirectory() as root:
            result = bridge.build_increment_bridge(root)
            marker = Path(root) / 'must-not-exist'
            Path(result['module_path']).write_text('raise RuntimeError("untrusted export")\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'export differs'):
                bridge.load_verified_bridge(root)
            self.assertFalse(marker.exists())

    def test_changed_frozen_reference_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for name, content in bridge._sources().items():
                (directory / name).write_bytes(content)
            (directory / 'dna_ref.py').write_text('raise RuntimeError("modified reference")\n', encoding='utf-8')
            with patch.object(bridge, 'FROZEN', directory), self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                bridge.verify_increment_bridge()


if __name__ == '__main__':
    unittest.main()
