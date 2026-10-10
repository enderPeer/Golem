"""Exact ternary storage and signed-int8 arithmetic reference, without NumPy.

Weights use two bits each: -1 -> 00, 0 -> 01, +1 -> 10. Code 11 is invalid.
Flat weights follow row-major order. The first weight occupies the least
significant pair of the first byte. Unused high pairs must be zero. This is a
project interchange format, not an adapter for a particular BitNet checkpoint.

The Python implementation establishes arithmetic and layout correctness only.
It is not an optimized CPU kernel, FPGA implementation, or energy benchmark.
"""

from __future__ import annotations

import itertools
import random
from typing import Iterable

INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
ENCODING = {-1: 0, 0: 1, 1: 2}
DECODING = (-1, 0, 1)


def _integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python integer (not bool or float)")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _dimension(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def packed_weight_bytes(weight_count: int) -> int:
    """Payload size only; excludes shape metadata, scales and runtime buffers."""
    if type(weight_count) is not int:
        raise TypeError("weight_count must be a Python integer")
    if weight_count < 0:
        raise ValueError("weight_count must be nonnegative")
    return (weight_count + 3) // 4


def pack_ternary(values: Iterable[int]) -> bytes:
    """Pack a flat sequence in order, four weights per byte."""
    packed = bytearray()
    for index, value in enumerate(values):
        _integer(value, f"weight[{index}]", -1, 1)
        if index % 4 == 0:
            packed.append(0)
        packed[-1] |= ENCODING[value] << (2 * (index % 4))
    return bytes(packed)


def _validate_packed(packed: bytes | bytearray, count: int) -> None:
    expected = packed_weight_bytes(count)
    if not isinstance(packed, (bytes, bytearray)):
        raise TypeError("packed weights must be bytes or bytearray")
    if len(packed) != expected:
        raise ValueError(f"Packed length {len(packed)} differs from expected {expected}")
    if count % 4 and packed[-1] >> (2 * (count % 4)):
        raise ValueError("Unused high bit pairs must be zero padding")
    for index in range(count):
        if ((packed[index // 4] >> (2 * (index % 4))) & 3) == 3:
            raise ValueError(f"Reserved ternary code 11 at weight {index}")


def unpack_ternary(packed: bytes | bytearray, count: int) -> list[int]:
    """Decode exactly count weights; reject malformed, extra or missing bytes."""
    _validate_packed(packed, count)
    return [DECODING[(packed[index // 4] >> (2 * (index % 4))) & 3] for index in range(count)]


def mac_int8(accumulator: int, weight: int, activation: int) -> int:
    """One exact ternary/int8 MAC with a checked signed-int32 accumulator.

    Multiplication takes place in Python's wide integer type. In particular,
    -1 * -128 is +128, not an overflowing signed-int8 intermediate. Overflow
    raises rather than silently wrapping or saturating.
    """
    _integer(accumulator, "accumulator", INT32_MIN, INT32_MAX)
    _integer(weight, "weight", -1, 1)
    _integer(activation, "activation", INT8_MIN, INT8_MAX)
    result = accumulator + weight * activation
    if not INT32_MIN <= result <= INT32_MAX:
        raise OverflowError("Ternary MAC exceeds signed-int32 accumulator range")
    return result


def dot_int8(weights: Iterable[int], activations: Iterable[int], *, accumulator: int = 0) -> int:
    """Dot product with equal-length vectors and overflow checked at each step."""
    _integer(accumulator, "accumulator", INT32_MIN, INT32_MAX)
    missing = object()
    for weight, activation in itertools.zip_longest(weights, activations, fillvalue=missing):
        if weight is missing or activation is missing:
            raise ValueError("Weight and activation vector dimensions differ")
        accumulator = mac_int8(accumulator, weight, activation)
    return accumulator


def matvec_int8(packed: bytes | bytearray, rows: int, cols: int,
                activations: Iterable[int]) -> list[int]:
    """Multiply a packed row-major (rows, cols) matrix by one int8 vector.

    Rows need not align to bytes. Shape is explicit and is not inferred from
    padded storage. No bias, scaling, normalization or activation quantization
    is applied by this arithmetic primitive.
    """
    _dimension(rows, "rows")
    _dimension(cols, "cols")
    vector = list(activations)
    if len(vector) != cols:
        raise ValueError(f"Activation vector length {len(vector)} differs from cols={cols}")
    for index, value in enumerate(vector):
        _integer(value, f"activation[{index}]", INT8_MIN, INT8_MAX)
    _validate_packed(packed, rows * cols)
    result = []
    for row in range(rows):
        accumulator = 0
        for column, activation in enumerate(vector):
            index = row * cols + column
            weight = DECODING[(packed[index // 4] >> (2 * (index % 4))) & 3]
            accumulator = mac_int8(accumulator, weight, activation)
        result.append(accumulator)
    return result


def storage_accounting(rows: int, cols: int) -> dict:
    """Analytical array payload counts, explicitly not measured process memory."""
    _dimension(rows, "rows")
    _dimension(cols, "cols")
    weights = rows * cols
    packed_bytes = packed_weight_bytes(weights)
    return {
        "rows": rows, "cols": cols, "weight_count": weights,
        "packed_weight_bytes": packed_bytes,
        "int8_weight_bytes": weights, "bf16_weight_bytes": 2 * weights,
        "fp32_weight_bytes": 4 * weights,
        "int8_input_payload_bytes": cols,
        "int32_output_payload_bytes": 4 * rows,
        "minimal_array_payload_bytes": packed_bytes + cols + 4 * rows,
        "runtime_memory_measured": False,
        "excluded": ["Python objects and temporary allocations", "shape metadata",
                     "quantization scales", "activation/KV caches", "other model parameters",
                     "training master weights, gradients and optimizer state", "kernel workspaces"],
    }


def self_test() -> dict:
    """Run deterministic exact arithmetic checks; return JSON-serializable results.

    This uses independent Python sum(w*x) expectations, not floating point.
    Timing and energy are deliberately absent from the report.
    """
    roundtrips = 0
    for length in range(7):
        for weights in itertools.product((-1, 0, 1), repeat=length):
            if unpack_ternary(pack_ternary(weights), length) != list(weights):
                raise AssertionError("Exhaustive ternary packing mismatch")
            roundtrips += 1
    single_macs = 0
    for weight in (-1, 0, 1):
        for activation in range(INT8_MIN, INT8_MAX + 1):
            if mac_int8(0, weight, activation) != weight * activation:
                raise AssertionError("Signed-int8 reference MAC mismatch")
            single_macs += 1
    dots = 0
    for length in range(4):
        for weights in itertools.product((-1, 0, 1), repeat=length):
            for activations in itertools.product((-128, -1, 0, 1, 127), repeat=length):
                if dot_int8(weights, activations) != sum(w * x for w, x in zip(weights, activations)):
                    raise AssertionError("Exhaustive small dot product mismatch")
                dots += 1
    rng = random.Random(20261010)
    matrices = 0
    for _ in range(64):
        rows, cols = rng.randrange(1, 9), rng.randrange(1, 66)
        weights = [rng.choice((-1, 0, 1)) for _ in range(rows * cols)]
        activations = [rng.randrange(-128, 128) for _ in range(cols)]
        expected = [sum(weights[row * cols + column] * activations[column] for column in range(cols))
                    for row in range(rows)]
        if matvec_int8(pack_ternary(weights), rows, cols, activations) != expected:
            raise AssertionError("Seeded matrix-vector mismatch")
        matrices += 1
    for width in (2560, 6912):
        if dot_int8([-1] * width, [-128] * width) != 128 * width:
            raise AssertionError("Wide accumulator mismatch")
        dots += 1
    expected_rejections = [
        lambda: unpack_ternary(b"\x03", 1),
        lambda: unpack_ternary(b"\x04", 1),
        lambda: mac_int8(INT32_MAX, 1, 1),
        lambda: mac_int8(INT32_MIN, 1, -1),
        lambda: dot_int8([1, -1], [1, 1], accumulator=INT32_MAX),
    ]
    rejected = 0
    for operation in expected_rejections:
        try:
            operation()
        except (ValueError, OverflowError):
            rejected += 1
        else:
            raise AssertionError("Invalid encoding or overflowing arithmetic was accepted")
    return {
        "passed": True, "reference": "exact Python integer arithmetic",
        "encoding": {"-1": "00", "0": "01", "+1": "10", "reserved": "11"},
        "layout": "row-major, consecutive weights in low-to-high bit pairs; zero unused high pairs",
        "packing_roundtrips": roundtrips, "single_mac_cases": single_macs,
        "dot_product_cases": dots, "seeded_matrix_cases": matrices,
        "invalid_or_overflow_cases_rejected": rejected,
        "storage_example": storage_accounting(64, 128),
        "hardware_benchmark": False,
        "claims": "Arithmetic and packed payload sizes only; no measured latency, power or energy advantage.",
    }
