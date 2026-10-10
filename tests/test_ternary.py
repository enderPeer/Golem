"""Exact low-bit format and arithmetic checks, without optional ML packages."""

import itertools
import random
import unittest

from english_world.ternary import (
    INT32_MAX, INT32_MIN, dot_int8, mac_int8, matvec_int8,
    pack_ternary, packed_weight_bytes, self_test, storage_accounting, unpack_ternary,
)


class TernaryTests(unittest.TestCase):
    def test_known_byte_layout_and_nonbyte_aligned_rows(self):
        self.assertEqual(pack_ternary([-1, 0, 1, -1]), b"\x24")
        self.assertEqual(pack_ternary([1, 1, 1, 1]), b"\xaa")
        weights = [-1, 0, 1, 1, 0, -1]
        self.assertEqual(matvec_int8(pack_ternary(weights), 2, 3, [-128, 2, 127]), [255, -255])
        self.assertEqual(unpack_ternary(pack_ternary(weights), 6), weights)

    def test_roundtrips_exhaustive_small_vectors(self):
        for length in range(6):
            for weights in itertools.product((-1, 0, 1), repeat=length):
                packed = pack_ternary(iter(weights))
                self.assertEqual(len(packed), packed_weight_bytes(length))
                self.assertEqual(unpack_ternary(packed, length), list(weights))

    def test_all_int8_inputs_and_ternary_weights(self):
        for weight in (-1, 0, 1):
            for activation in range(-128, 128):
                self.assertEqual(mac_int8(17, weight, activation), 17 + weight * activation)
        self.assertEqual(mac_int8(0, -1, -128), 128)

    def test_wide_accumulation_does_not_wrap_int8(self):
        for width in (2560, 6912):
            self.assertEqual(dot_int8([-1] * width, [-128] * width), 128 * width)
            self.assertEqual(dot_int8([1] * width, [-128] * width), -128 * width)

    def test_overflow_is_checked_even_when_final_result_would_fit(self):
        self.assertEqual(mac_int8(INT32_MAX, 0, -128), INT32_MAX)
        self.assertEqual(mac_int8(INT32_MIN, 0, 127), INT32_MIN)
        with self.assertRaises(OverflowError):
            mac_int8(INT32_MAX, 1, 1)
        with self.assertRaises(OverflowError):
            mac_int8(INT32_MIN, 1, -1)
        with self.assertRaises(OverflowError):
            dot_int8([1, -1], [1, 1], accumulator=INT32_MAX)

    def test_reserved_codes_padding_and_dimensions_rejected(self):
        for position in range(4):
            with self.assertRaisesRegex(ValueError, "Reserved"):
                unpack_ternary(bytes([3 << (2 * position)]), 4)
        with self.assertRaisesRegex(ValueError, "padding"):
            unpack_ternary(b"\x04", 1)
        with self.assertRaises(ValueError):
            unpack_ternary(b"\x00", 5)
        with self.assertRaises(ValueError):
            unpack_ternary(b"\x00\x00", 1)
        with self.assertRaises(ValueError):
            dot_int8([1, 0], [1])
        with self.assertRaises(ValueError):
            matvec_int8(b"\x00", 0, 1, [1])
        with self.assertRaises(ValueError):
            matvec_int8(b"\x00", 1, 2, [1])
        with self.assertRaises(ValueError):
            matvec_int8(b"\x00", 1, 1, [128])

    def test_bad_numeric_types_and_ranges_rejected(self):
        for value in (True, 1.0, "1"):
            with self.assertRaises(TypeError):
                pack_ternary([value])
        for value in (-2, 2):
            with self.assertRaises(ValueError):
                pack_ternary([value])
        for value in (-129, 128):
            with self.assertRaises(ValueError):
                mac_int8(0, 1, value)
        with self.assertRaises(ValueError):
            packed_weight_bytes(-1)
        with self.assertRaises(TypeError):
            unpack_ternary([0], 4)

    def test_seeded_matrices_match_independent_integer_reference(self):
        rng = random.Random(74)
        for _ in range(100):
            rows, cols = rng.randrange(1, 6), rng.randrange(1, 33)
            weights = [rng.choice((-1, 0, 1)) for _ in range(rows * cols)]
            inputs = [rng.randrange(-128, 128) for _ in range(cols)]
            expected = [sum(w * x for w, x in zip(weights[row * cols:(row + 1) * cols], inputs))
                        for row in range(rows)]
            self.assertEqual(matvec_int8(pack_ternary(weights), rows, cols, inputs), expected)

    def test_storage_report_is_payload_not_runtime_memory(self):
        counts = storage_accounting(3, 3)
        self.assertEqual(counts["packed_weight_bytes"], 3)
        self.assertEqual(counts["int8_weight_bytes"], 9)
        self.assertEqual(counts["bf16_weight_bytes"], 18)
        self.assertEqual(counts["minimal_array_payload_bytes"], 3 + 3 + 12)
        self.assertFalse(counts["runtime_memory_measured"])
        self.assertGreater(len(counts["excluded"]), 0)

    def test_public_self_test_is_deterministic(self):
        report = self_test()
        self.assertEqual(report, self_test())
        self.assertTrue(report["passed"])
        self.assertEqual(report["packing_roundtrips"], 1093)
        self.assertEqual(report["single_mac_cases"], 768)
        self.assertEqual(report["dot_product_cases"], 3618)
        self.assertEqual(report["invalid_or_overflow_cases_rejected"], 5)
        self.assertFalse(report["hardware_benchmark"])


if __name__ == "__main__":
    unittest.main()
