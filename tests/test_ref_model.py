"""
Unit tests for the Astrixcore AX-1 INT8 Neural Accelerator Reference Model.

These tests verify the correctness of INT8 arithmetic, MAC accumulation,
quantization, and saturation behavior that should match the RTL implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compiler'))

import numpy as np
from ref_model import (
    int8_mult, int8_saturate_accumulator, int8_round_shift,
    mac_single, mac_array, quantize_fp_to_int8, dequantize_int8_to_fp,
    run_all_tests
)
import pytest


class TestInt8Mult:
    """Tests for INT8 multiplication function."""

    def test_basic_multiplication(self):
        """Test basic INT8 multiplication cases."""
        # Positive * positive
        assert int8_mult(1, 1) == 1
        assert int8_mult(5, 3) == 15

        # Negative * negative = positive
        assert int8_mult(-5, -3) == 15
        assert int8_mult(-128, -128) > 0  # wraps to positive

        # Negative * positive = negative
        assert int8_mult(-5, 3) == -15
        assert int8_mult(5, -3) == -15

        # Zero cases
        assert int8_mult(0, 100) == 0
        assert int8_mult(50, 0) == 0
        assert int8_mult(0, 0) == 0

        # Boundary values
        assert int8_mult(127, 127) > 0
        assert int8_mult(-128, 127) < 0
        assert int8_mult(127, -128) < 0

    def test_boundary_values(self):
        """Test multiplication with boundary INT8 values."""
        # Max values
        result = int8_mult(127, 127)
        assert result > 0

        # Min values (most negative)
        result = int8_mult(-128, -128)
        assert result > 0  # negative * negative = positive

        # Min * max
        result = int8_mult(-128, 127)
        assert result < 0

        # Max * min
        result = int8_mult(127, -128)
        assert result < 0


class TestSaturation:
    """Tests for accumulator saturation behavior."""

    def test_saturate_positive_overflow(self):
        """Saturate large positive values to 127."""
        assert int8_saturate_accumulator(200) == 127
        assert int8_saturate_accumulator(300) == 127
        assert int8_saturate_accumulator(127) == 127

    def test_saturate_negative_overflow(self):
        """Saturate large negative values to -128."""
        assert int8_saturate_accumulator(-200) == -128
        assert int8_saturate_accumulator(-300) == -128
        assert int8_saturate_accumulator(-128) == -128

    def test_saturate_within_range(self):
        """Values within INT8 range pass through unchanged."""
        assert int8_saturate_accumulator(0) == 0
        assert int8_saturate_accumulator(127) == 127
        assert int8_saturate_accumulator(-128) == -128
        assert int8_saturate_accumulator(-64) == -64


class TestMacSingle:
    """Tests for single MAC operation."""

    def test_basic_mac(self):
        """Test basic MAC without shift."""
        # 2 * 3 = 6, acc starts at 0
        result = mac_single(2, 3, acc=0, shift=0)
        assert result == 6, f"Expected 6, got {result}"

        # 1 + 2*3 = 7
        result = mac_single(2, 3, acc=1, shift=0)
        assert result == 7, f"Expected 7, got {result}"

    def test_mac_saturation(self):
        """Test MAC overflow saturation."""
        # 100 * 100 = 10000, saturated to 127
        result = mac_single(100, 100, acc=0, shift=0)
        assert result == 127, f"Expected 127, got {result}"

        # -128 * 127 = -16256, saturated to -128
        result = mac_single(-128, 127, acc=0, shift=0)
        assert result == -128, f"Expected -128, got {result}"

    def test_mac_negative(self):
        """Test MAC with negative values."""
        # -5 * 10 = -50
        result = mac_single(-5, 10, acc=0, shift=0)
        assert result == -50, f"Expected -50, got {result}"

        # Negative accum with positive product
        result = mac_single(5, 10, acc=-30, shift=0)
        # -30 + 50 = 20
        assert result == 20, f"Expected 20, got {result}"


class TestMacWithShift:
    """Tests for MAC with right-shift quantization."""

    def test_shift_by_4(self):
        """Test MAC with shift=4 (common in INT8 networks)."""
        # 2 * 3 = 6, shift right 4 = round(6/16) = 0
        result = mac_single(2, 3, acc=0, shift=4)
        assert result == 0, f"Expected 0, got {result}"

        # 15 * 15 = 225, shift right 4 = round(225/16) = 14
        result = mac_single(15, 15, acc=0, shift=4)
        assert result == 14, f"Expected 14, got {result}"

    def test_shift_rounding(self):
        """Test rounding behavior in shift."""
        # 7 * 1 = 7, shift right 3 = round(7/8) = 1
        result = mac_single(7, 1, acc=0, shift=3)
        # round(7/8) = 1 (since 0.875 rounds to 1)
        assert result == 1, f"Expected 1, got {result}"

        # 1 * 1 = 1, shift right 3 = round(1/8) = 0
        result = mac_single(1, 1, acc=0, shift=3)
        assert result == 0, f"Expected 0, got {result}"


class TestQuantizeDequantize:
    """Tests for quantization/dequantization pipeline."""

    def test_quantize_zero(self):
        """Quantize zero value."""
        scale = 0.01
        q = quantize_fp_to_int8(0.0, scale, 0)
        assert q == 0, f"Expected 0, got {q}"

    def test_quantize_positive(self):
        """Quantize positive floating-point value."""
        scale = 0.1  # 0.1 scale means values are divided by 0.1
        # 0.5 / 0.1 = 5, round = 5, + 0 zero_point = 5
        q = quantize_fp_to_int8(0.5, scale, 0)
        assert q == 5, f"Expected 5, got {q}"

        # 0.99 / 0.1 = 9.9, round = 10
        q = quantize_fp_to_int8(0.99, scale, 0)
        assert q == 10, f"Expected 10, got {q}"

    def test_dequantize_roundtrip(self):
        """Test quantize then dequantize round-trip."""
        scale = 0.00390625  # 1/256
        zero_point = 0

        test_values = [0.0, 0.5, 1.0, -0.5, -1.0]
        for fp_val in test_values:
            q = quantize_fp_to_int8(fp_val, scale, zero_point)
            deq = dequantize_int8_to_fp(q, scale, zero_point)
            # Should be within reasonable error
            assert abs(deq - fp_val) < 0.1, \
                f"Roundtrip error too large: {fp_val} -> {q} -> {deq}"


class TestMacArray:
    """Tests for MAC array operation."""

    def test_mac_array_basic(self):
        """Test basic MAC array with multiple weights."""
        weights = np.array([[2, 3, 4]], dtype=np.int8)
        inputs = np.array([5, 6, 7], dtype=np.int8)

        # 2*5 + 3*6 + 4*7 = 10 + 18 + 28 = 56
        result = mac_array(weights, inputs, init_acc=0, shift=0)
        assert result == 56, f"Expected 56, got {result}"

    def test_mac_array_with_shift(self):
        """Test MAC array with right-shift."""
        weights = np.array([[2, 3, 4]], dtype=np.int8)
        inputs = np.array([5, 6, 7], dtype=np.int8)

        # 56 >> 1 with rounding = 28
        result = mac_array(weights, inputs, init_acc=0, shift=1)
        assert result == 28, f"Expected 28, got {result}"


class TestRunAll:
    """Tests that all reference model functions run without errors."""

    def test_run_all_functions(self):
        """Run the main test function to verify all code paths work."""
        # This should not raise any exceptions
        run_all_tests()


if __name__ == "__main__":
    # Run tests when invoked directly
    pytest.main([__file__, "-v"])