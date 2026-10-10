"""Code-domain compatibility and objective boundaries, without host-code edits."""
import hashlib
import json
import math
import random
import unittest

from english_world.model import (
    ALPHABET, CODE_ALPHABET, CODE_FORMAT, FORMAT, Genome, Predictor,
    alphabet_for, baseline_metrics, evaluate, generate, mutate, normalize_text,
    reproduce, self_prediction_metrics,
)


class LegacyIdentityTests(unittest.TestCase):
    def test_default_serialization_and_hash_are_exactly_v1(self):
        expected = (b'{"format":"english-world-genome-v1","max_contexts":512,"max_order":3,'
                    b'"reproduction":["START","ALLOC","COPY","TERM","DIVIDE"],"rules":{},"smoothing":16}')
        genome = Genome()
        self.assertEqual(genome.serialize(), expected)
        self.assertEqual(genome.digest(), '71de55aa00f8042352eae34dd53f9197acd3e872ed2fcb7de7a8b56485c68596')
        self.assertEqual(Genome.from_dict(json.loads(expected)).serialize(), expected)
        self.assertNotIn('domain', genome.to_dict())

    def test_nonempty_legacy_genome_and_seeded_mutation_unchanged(self):
        genome = Genome(max_order=5, rules={'abc': {'z': 60}, '': {' ': 255}}, smoothing=23, max_contexts=100)
        self.assertEqual(genome.digest(), 'f0fb69bc1267896885ac2e7f7f58dc97e55fd4840c78d9968144cacd6767ab44')
        child, _ = mutate(Genome(), random.Random(7), [{'context': 'hello ', 'target': 'w'}],
                          guided_probability=.7, events=12)
        self.assertEqual(child.digest(), '5b435440580a9b42ab66fd2ca32b1a80415c7e38bb9c1c4310ad4ce213155db7')
        self.assertEqual(generate(Genome(), 'hello ', 25, seed=13), 'm77:jlhl9g0kov:3anh;!!? )')

    def test_english_behavior_and_metric_schema_stay_v1(self):
        self.assertEqual(alphabet_for(), ALPHABET)
        self.assertEqual(normalize_text('DEF\tFoo_1(X):\n    Return X'), 'def foo 1(x):\n return x')
        metrics = evaluate(Genome(), [{'context': 'hello', 'target': ' '}], generated_chars=0)
        self.assertNotIn('domain', metrics)
        self.assertNotIn('spelling_applicable', metrics)
        self.assertAlmostEqual(metrics['loss'], math.log(len(ALPHABET)))
        self.assertFalse(self_prediction_metrics(Genome())['applicable'])


class CodeModelTests(unittest.TestCase):
    def test_code_normalization_preserves_case_indentation_and_syntax(self):
        source = 'def Foo(Value=1):\r\n\tif Value > 0:\r\n        return {"A": [Value]}\r'
        expected = 'def Foo(Value=1):\n\tif Value > 0:\n        return {"A": [Value]}\n'
        self.assertEqual(normalize_text(source, 'code'), expected)
        self.assertEqual(normalize_text('caf\u00e9', 'code'), 'caf' + chr(92) + 'xe9')
        self.assertEqual(normalize_text(chr(0) + chr(127), 'code'), chr(92) + 'x00' + chr(92) + 'x7f')
        self.assertEqual(len(CODE_ALPHABET), 97)
        self.assertEqual(set(normalize_text(source, 'code')) - set(CODE_ALPHABET), set())

    def test_code_domain_version_and_round_trip_are_explicit(self):
        genome = Genome(domain='code', max_order=16, rules={'def Foo(': {')': 220}, '\t': {'R': 3}})
        self.assertEqual(genome.to_dict()['format'], CODE_FORMAT)
        self.assertEqual(genome.to_dict()['domain'], 'code')
        self.assertEqual(Genome.from_dict(genome.to_dict()).serialize(), genome.serialize())
        with self.assertRaises(ValueError):
            Genome(max_order=6)
        with self.assertRaises(ValueError):
            Genome(domain='code', max_order=17)
        with self.assertRaises(ValueError):
            Genome.from_dict({'format': FORMAT, 'domain': 'code'})
        with self.assertRaises(ValueError):
            Genome.from_dict({'format': CODE_FORMAT})

    def test_case_sensitive_prediction_and_context_beyond_five_characters(self):
        genome = Genome(domain='code', max_order=16,
                        rules={'ALPHA_suffix': {'A': 255}, 'BETA_suffix': {'B': 255},
                               'Foo': {'(': 220}, 'foo': {'[': 220}})
        self.assertEqual(generate(genome, 'ALPHA_suffix', 1, temperature=0), 'A')
        self.assertEqual(generate(genome, 'BETA_suffix', 1, temperature=0), 'B')
        self.assertEqual(generate(genome, 'Foo', 1, temperature=0), '(')
        self.assertEqual(generate(genome, 'foo', 1, temperature=0), '[')
        dist = Predictor(genome).distribution('Foo')
        self.assertEqual(set(dist), set(CODE_ALPHABET))
        self.assertAlmostEqual(sum(dist.values()), 1.)

    def test_code_fitness_ignores_dictionary_and_generated_spelling(self):
        genome = Genome(domain='code', rules={'': {'A': 80}, 'def ': {'F': 90}})
        examples = [{'context': 'def ', 'target': 'Foo():'}]
        short = evaluate(genome, examples, {'foo'}, generated_chars=0)
        long = evaluate(genome, examples, CODE_ALPHABET, generated_chars=100, generation_seed=99)
        self.assertEqual(short['fitness'], long['fitness'])
        self.assertEqual(short['loss'], long['loss'])
        self.assertEqual(long['spelling_score'], 0.)
        self.assertEqual(long['repetition_penalty'], 0.)
        self.assertFalse(long['spelling_applicable'])
        self.assertEqual(long['domain'], 'code')
        self.assertAlmostEqual(long['fitness'], math.log(97) - long['loss'] - .000001 * len(genome.serialize()))

    def test_guided_code_mutation_preserves_case_and_parent(self):
        parent = Genome(domain='code', max_order=16)
        before = parent.digest()
        child, report = mutate(parent, random.Random(9), [{'context': 'def Foo():\n\t', 'target': 'R'}],
                               guided_probability=1., events=4)
        self.assertEqual(parent.digest(), before)
        self.assertNotEqual(child.digest(), before)
        self.assertEqual(child.domain, 'code')
        self.assertEqual(report['guided_events'], 4)
        self.assertTrue(all(set(row) == {'R'} for row in child.rules.values()))
        self.assertLessEqual(len(child.rules), 4)

    def test_full_composite_code_genome_is_faithfully_copied(self):
        parent = Genome(domain='code', max_order=16,
                        rules={'def Foo(': {')': 120}, '\n    ': {'#': 10}, '': {'_': 5}})
        child, report = reproduce(parent)
        self.assertEqual(child.serialize(), parent.serialize())
        self.assertEqual(child.digest(), parent.digest())
        self.assertEqual(report['births'], 2)
        self.assertTrue(report['faithful_before_mutation'])
        child.rules['def Foo('][')'] += 1
        self.assertNotEqual(child.digest(), parent.digest())

    def test_self_prediction_is_bounded_deterministic_diagnostic_only(self):
        genome = Genome(domain='code', rules={'': {'"': 100, ':': 40}})
        before = genome.serialize()
        a = self_prediction_metrics(genome, max_examples=37)
        b = self_prediction_metrics(genome, max_examples=37)
        self.assertEqual(a, b)
        self.assertEqual(a['characters'], 37)
        self.assertEqual(a['source_bytes'], len(before))
        self.assertEqual(a['genome_hash'], hashlib.sha256(before).hexdigest())
        self.assertFalse(a['used_for_fitness'])
        self.assertNotIn('fitness', a)
        self.assertEqual(genome.serialize(), before)
        other = Genome.from_dict(genome.to_dict())
        other.rules['']['"'] += 1
        self.assertNotEqual(self_prediction_metrics(other)['genome_hash'], a['genome_hash'])
        with self.assertRaises(ValueError):
            self_prediction_metrics(genome, max_examples=100000)

    def test_code_controls_use_code_alphabet_and_preserve_case(self):
        examples = [{'context': 'def Foo():\n    return ', 'target': 'Value_1'}] * 3
        controls = baseline_metrics(examples, examples, domain='code')
        self.assertAlmostEqual(controls['uniform']['loss'], math.log(97))
        self.assertLess(controls['ngram_count_control']['loss'], controls['uniform']['loss'])
        self.assertEqual(controls['unigram_count_control']['domain'], 'code')


if __name__ == '__main__':
    unittest.main()
