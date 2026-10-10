import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from english_world.model import ALPHABET, Genome, Predictor, evaluate, generate, mutate, reproduce, spelling_metrics
from english_world.evolution import Trainer, train


def dataset():
    examples = [{'context': 'the ' + c, 'target': 'e'} for c in 'abcdef'] * 12
    return {'train': examples, 'validation': examples[:18], 'test': examples[18:36],
            'dictionary': ['the', 'bee', 'be', 'he', 'see']}


class ModelTests(unittest.TestCase):
    def test_uniform_and_prompt_sensitive_distribution(self):
        uniform = Predictor(Genome())
        self.assertAlmostEqual(sum(uniform.distribution('hello').values()), 1.)
        self.assertAlmostEqual(uniform.probability('hello', 'a'), 1 / len(ALPHABET))
        genome = Genome(rules={'cat': {'s': 100}, 'dog': {'!': 100}})
        self.assertGreater(Predictor(genome).probability('cat', 's'), Predictor(genome).probability('dog', 's'))
        self.assertNotEqual(generate(genome, 'cat', 1, temperature=0), generate(genome, 'dog', 1, temperature=0))

    def test_full_composite_copy_and_repeated_fecundity(self):
        genome = Genome(rules={'abc': {'z': 60}, '': {' ': 255}})
        child, report = reproduce(genome)
        self.assertEqual(child.serialize(), genome.serialize())
        self.assertEqual(child.digest(), genome.digest())
        self.assertEqual(report['births'], 2)
        self.assertEqual(report['bytes_copied'], 2 * (len(genome.serialize()) + 1))
        self.assertGreater(report['stalled_copies'], 0)
        self.assertEqual(report['template_pointer_after_divide'], 0)
        altered = Genome.from_dict(child.to_dict())
        altered.rules['abc']['z'] = 61
        self.assertNotEqual(altered.digest(), child.digest())

    def test_infertile_programs_are_bounded(self):
        for op in ('ALLOC', 'COPY', 'TERM', 'DIVIDE'):
            genome = Genome(reproduction=tuple(x for x in ('START', 'ALLOC', 'COPY', 'TERM', 'DIVIDE') if x != op))
            with self.assertRaises(ValueError):
                reproduce(genome, max_steps=300)

    def test_mutation_does_not_change_parent_or_fit_entire_corpus(self):
        parent = Genome()
        old = parent.digest()
        child, changes = mutate(parent, random.Random(3), dataset()['train'], guided_probability=1, events=2)
        self.assertEqual(parent.digest(), old)
        self.assertLessEqual(len(child.rules), 2)
        self.assertEqual(changes['guided_events'], 2)
        self.assertTrue(any('e' in row for row in child.rules.values()))

    def test_spelling_auxiliary_reward_bounded(self):
        metrics = evaluate(Genome(), dataset()['train'], dataset()['dictionary'])
        self.assertLessEqual(metrics['spelling_score'], 1)
        self.assertAlmostEqual(metrics['loss'], math.log(len(ALPHABET)))
        self.assertLessEqual(metrics['fitness'] - metrics['prediction_gain'], .1)

    def test_spelling_rejects_fragments_and_dictionary_spam(self):
        dictionary = {'z', 'o', 'f', 'd', 'h', 'k', 'the', 'hello', 'world', 'a', 'i'}
        garbage = 'z!-o.86fbs myp:)t\nd!;h81)k0 z o f d h k '
        self.assertEqual(spelling_metrics(garbage, dictionary)[0], 0.)
        self.assertEqual(spelling_metrics('hello, world! ', dictionary)[0], 1.)
        score, repetition = spelling_metrics('the the the the ', dictionary)
        self.assertEqual(score, .25)
        self.assertEqual(repetition, .75)
        self.assertEqual(spelling_metrics('hello123 the ', dictionary)[0], .5)

    def test_malformed_genome_mapping_rejected(self):
        for value in ([], None, 'not a genome'):
            with self.assertRaises(ValueError):
                Genome.from_dict(value)


class EvolutionTests(unittest.TestCase):
    def config(self, generations=5):
        return {'population_size': 8, 'generations': generations, 'seed': 4, 'eval_examples': 24,
                'validation_examples': 18, 'generated_chars': 16, 'max_seconds': 0,
                'guided_probability': 1., 'mutation_events': 6, 'checkpoint_seconds': 600.}

    def test_evolution_improves_predictive_loss_and_preserves_ancestry(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = train(self.config(), dataset(), tmp)
            self.assertEqual(result['status'], 'complete')
            self.assertLess(result['champion_metrics']['loss'], result['baseline_metrics']['loss'])
            state = json.loads((Path(tmp) / 'checkpoint.json').read_text())
            self.assertEqual(len(state['population']), 8)
            self.assertEqual(state['total_births'], 5 * 6)
            self.assertEqual(len(state['ancestry']), state['total_births'])
            self.assertGreater(sum(r['selection_differential'] for r in state['history']), 0)

    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as split:
            train(self.config(5), dataset(), whole)
            train(self.config(2), dataset(), split)
            train(self.config(5), dataset(), split, resume=True)
            a = json.loads((Path(whole) / 'checkpoint.json').read_text())
            b = json.loads((Path(split) / 'checkpoint.json').read_text())
            for key in ('population', 'rng_state', 'champion', 'history', 'ancestry', 'total_births'):
                self.assertEqual(a[key], b[key], key)

    def test_mid_generation_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as split:
            train(self.config(3), dataset(), whole)
            trainer = Trainer(self.config(3), dataset(), split)
            for _ in range(15):
                trainer._step()
            self.assertEqual(trainer.state['stage'], 'reproduce')
            trainer.checkpoint()
            train(self.config(3), dataset(), split, resume=True)
            a = json.loads((Path(whole) / 'checkpoint.json').read_text())
            b = json.loads((Path(split) / 'checkpoint.json').read_text())
            for key in ('population', 'rng_state', 'champion', 'history', 'ancestry'):
                self.assertEqual(a[key], b[key], key)

    def test_resume_rejects_changed_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            train(self.config(1), dataset(), tmp)
            changed = dataset()
            changed['train'] = changed['train'] + [{'context': 'x', 'target': 'q'}]
            with self.assertRaises(ValueError):
                train(self.config(2), changed, tmp, resume=True)

    def test_nonfinite_budget_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config()
            cfg['max_seconds'] = float('nan')
            with self.assertRaises(ValueError):
                train(cfg, dataset(), tmp)


if __name__ == '__main__':
    unittest.main()
