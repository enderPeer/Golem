"""Regression checks for the byte-for-byte vendored DNA-2 reference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "upstream" / "dna2"


class ReplicationBaselineTests(unittest.TestCase):
    def test_vendored_reference_is_unchanged(self):
        provenance = json.loads((VENDOR / "PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256((VENDOR / "dna_ref.py").read_bytes()).hexdigest(),
            provenance["sha256"],
        )
        self.assertEqual(
            hashlib.sha256((VENDOR / "LICENSE").read_bytes()).hexdigest(),
            provenance["license_sha256"],
        )

    def test_dna2_512_tick_replication_baseline(self):
        spec = importlib.util.spec_from_file_location("frozen_dna_ref", VENDOR / "dna_ref.py")
        dna = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dna)
        world = dna.World(
            N=64 * 256, G=256, I=16, T=64, mu=0, D=0, F=0,
            blank=True, seed=1, n_seed=1, dna2=True,
            genome=dna.parse_genome("ATG GCT CTT TTA GGT"),
        )
        for _ in range(512):
            world.tick()
        self.assertEqual(world.births, 471)
        self.assertEqual(world.faithful, 471)
        self.assertEqual(len(world.live), 54)
        self.assertEqual(world.state_hash(), 0xFF66E97FDAF7F31D)


if __name__ == "__main__":
    unittest.main()
