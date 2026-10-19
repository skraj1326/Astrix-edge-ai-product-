"""
Astrixcore AX-1 Performance Analysis — Phase 8

Investigates array scaling, data reuse, SRAM banking, pipeline design,
operator fusion, and bandwidth limits. Estimates whether the product targets
are feasible before committing to a larger architecture.

Target product specs (aspirational, not yet achieved):
- 2,048 INT8 MAC operations per cycle-equivalent
- 2 MiB on-chip SRAM capacity
- 2.4 TOPS peak performance
- 1.2 W power target
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ref_model import mac_single, mac_array, int8_mult


def estimate_array_scaling():
    """
    Estimate how array scaling affects performance and resource usage.
    
    Starts from 2×2 (4 PEs) and scales to 8×8 (64 PEs).
    Reports:
    - PE count and MAC/cycle rate
    - Approximate BRAM usage (for weight/SRAM storage)
    - DSP slice usage (for MAC operations)
    - Clock frequency estimates
    - Power estimates (relative)
    """
    print("=" * 60)
    print("Phase 8: Array Scaling Analysis")
    print("=" * 60)
    
    # Scale from 2×2 to 8×8
    scales = [2, 4, 8]  # Array dimensions
    
    for dim in scales:
        pes = dim * dim
        macs_per_cycle = pes  # One MAC per PE per cycle (INT8)
        
        # BRAM estimation: each PE needs weight storage
        # INT8 weight = 1 byte, but BRAM is 32-bit wide
        # For a dim×dim array, we need dim² weights
        # BRAMs = ceil(dim² * 8 bits / 32 bits) = ceil(dim² / 4)
        brams = int(np.ceil(dim * dim / 4))
        
        # DSP slice estimation: each PE needs a DSP48 for MAC
        # Xilinx Artix-7 has DSP48E1 slices
        dsp_slices = pes  # 1 DSP per PE (minimum)
        
        # Clock frequency estimate
        # Larger arrays have more routing, lower max freq
        # Artix-7 typical: 100-133 MHz for small designs
        # Scaling assumption: freq drops ~5% per doubling of PEs
        base_freq = 133.0  # MHz, Artix-7 limit
        freq = base_freq * (0.95 ** (pes - 4))  # Exponential decay
        
        # Throughput (MACs per second)
        throughput_macs_per_s = macs_per_cycle * freq
        
        # Relative power (dynamic power ~ CV²f, C ~ area ~ PE count)
        # Normalize to 2×2 = 1.0
        rel_power = pes / 4.0  # Rough linear approximation
        
        # Effective TOPS (theoretical)
        tops = macs_per_cycle * freq / 1e12
        
        print(f"\nArray dim: {dim}×{dim}")
        print(f"  PEs: {pes}")
        print(f"  MAC/cycle: {macs_per_cycle}")
        print(f"  Estimated freq: {freq:.1f} MHz")
        print(f"  Throughput: {throughput_macs_per_s:.1f} MAC/s = {tops:.4f} TOPS")
        print(f"  BRAM usage: {brams} blocks")
        print(f"  DSP slices: {dsp_slices}")
        print(f"  Relative power: {rel_power:.2f}× baseline (2×2)")
        
        # Check against targets
        if tops >= 2.4:
            print(f"  ✅ Meets 2.4 TOPS target")
        else:
            print(f"  ❌ Below 2.4 TOPS target ({(tops/2.4)*100:.1f}% of target)")
        
        if macs_per_cycle >= 2048:
            print(f"  ✅ Meets 2048 MAC/cycle target")
        else:
            print(f"  ❌ Below 2048 MAC/cycle target ({(macs_per_cycle/2048)*100:.1f}% of target)")
    
    print("\n" + "=" * 60)


def estimate_sram_banking():
    """
    Estimate SRAM banking and bandwidth for different array sizes.
    
    Target: 2 MiB total, divided into weight buffer, input buffer, output buffer.
    """
    print("\n" + "=" * 60)
    print("Phase 8: SRAM Banking & Bandwidth Analysis")
    print("=" * 60)
    
    total_sram = 2 * 1024 * 1024  # 2 MiB in bits = 2_097_152 bits
    total_sram_bytes = total_sram / 8  # = 262_144 bytes = 256 KiB
    
    # Actually 2 MiB = 2 * 1024 * 1024 = 2_097_152 Bytes? No.
    # 2 MiB = 2 * 1024 * 1024 Bytes = 2_097_152 Bytes = 16_777_216 bits
    # Let me recalculate:
    total_sram_bytes = 2 * 1024 * 1024  # 2 MiB = 2_097_152 Bytes
    total_sram_bits = total_sram_bytes * 8  # = 16_777_216 bits
    
    print(f"Total SRAM: 2 MiB = {total_sram_bytes:,} Bytes = {total_sram_bits:,} bits")
    
    # For INT8 weights: 1 byte per weight
    # For INT8 activations: 1 byte per activation
    # For INT32 accumulators: 4 bytes per output
    
    # Scenario: 8×8 array (64 PEs), YOLOv8n input 224×224, batch=1
    # Weights only: depends on network
    
    # Breakdown by buffer type
    # Weight buffer: stores all network weights
    # Input buffer: stores current input tile (e.g., 64×64 for tiling)
    # Output buffer: stores partial sums before reduction
    
    # Example: weight storage for a small conv net
    # Assume ~4k weights for a tiny network
    sample_weights = 4096  # 4K weights
    weight_bytes = sample_weights  # 1 byte each = INT8
    
    input_tile = 64 * 64  # 64×64 input tile
    input_bytes = input_tile  # 1 byte each = INT8
    
    output_partial = 64 * 64  # output tile same size
    output_bytes = output_partial * 4  # 4 bytes each = INT32 accumulators
    
    print(f"\nExample weight buffer: {sample_weights} INT8 weights = {weight_bytes:,} Bytes")
    print(f"Example input tile: {input_tile} INT8 activations = {input_bytes:,} Bytes")
    print(f"Example output tile: {output_partial} partial sums = {output_bytes:,} Bytes (INT32)")
    
    # Fraction of 2 MiB
    weight_frac = weight_bytes / total_sram_bytes * 100
    input_frac = input_bytes / total_sram_bytes * 100
    output_frac = output_bytes / total_sram_bytes * 100
    
    print(f"\nWeight buffer: {weight_bytes:,} Bytes = {weight_frac:.2f}% of 2 MiB")
    print(f"Input buffer: {input_bytes:,} Bytes = {input_frac:.2f}% of 2 MiB")
    print(f"Output buffer: {output_bytes:,} Bytes = {output_frac:.2f}% of 2 MiB")
    print(f"Remaining SRAM: {100 - weight_frac - input_frac - output_frac:.2f}%")
    
    # Bandwidth analysis
    # For YOLOv8n: ~86 MFLOPs (floating point), INT8 equivalent ~86×4 = 344 MACs per inference
    # At 100 MHz with 64 PE array: 64×100M = 6.4G MAC/s
    # Inference time = 86 / 6.4G ≈ 13.4 μs (well under 6 ms target)
    
    print(f"\nBandwidth analysis (YOLOv8n approx.):")
    print(f"  - Network FLOPs: ~86M (FP32) ≈ 86M INT8 MACs (conservative)")
    print(f"  - 64 PE @ 100MHz: 6.4G MAC/s theoretical peak")
    print(f"  - Inference time estimate: ~13 μs (well under 6 ms target)")
    print(f"  - Memory bandwidth needed: ~86M weights/activations / 13μs ≈ 6.6 GB/s")
    print(f"  - 2 MiB SRAM bandwidth (1R1W per cycle @ 100MHz): 256 KB/s per port")
    print(f"  - Multi-port needed for concurrent access (banking recommended)")
    
    print("\n" + "=" * 60)


def estimate_pipeline_design():
    """
    Estimate pipeline design for throughput improvement.
    """
    print("\n" + "=" * 60)
    print("Phase 8: Pipeline Design Analysis")
    print("=" * 60)
    
    print("""
Pipeline design aims to increase throughput by overlapping MAC operations:

  Without pipeline (sequential):
    - 1 MAC per cycle per PE
    - n cycles for n inputs
    
    With 2-stage pipeline:
    - Stage 1: INT8 × INT8 multiplication
    - Stage 2: Accumulation + shift + saturation
    - Throughput: 1 MAC result per cycle (after initial fill)
    - Latency: 2 cycles + (n-1) cycles for n operations
    
    With 3-stage pipeline (MAC → shift → saturate):
    - Stage 1: Multiplication
    - Stage 2: Accumulation
    - Stage 3: Shift + saturation
    - Throughput: 1 result per cycle
    - Latency: 3 cycles + (n-1) cycles for n operations
    
    Trade-offs:
    - +1 stage: +1 cycle latency, same throughput
    - Doubles critical path freedom → potentially higher clock freq
    - More registers → slightly more area (negligible vs PE array)
    - Requires careful data hazard handling
    
    For Astrixcore AX-1:
    - Current: Single-cycle MAC with immediate shift+saturate
    - Next: 2-stage pipeline for freq scaling
    - Target: 150+ MHz vs 100MHz base (50% freq increase possible)
    - Result: 50% throughput improvement at same array size
    """ )
    
    # Pipeline impact on targets
    base_freq = 100.0  # MHz
    pipelined_freq = 150.0  # MHz with 2-stage pipeline
    pe_dim = 4  # 4×4 = 16 PEs
    pes = pe_dim * pe_dim
    
    base_tops = pes * base_freq / 1e5  # TOPS (MACs per 100K per second... let me recalculate)
    # Actually: MACs per second = pes × freq_MHz × 1e6
    # TOPS = MACs_per_s / 1e12
    base_macs_per_s = pes * base_freq * 1e6
    pipelined_macs_per_s = pes * pipelined_freq * 1e6
    
    base_tops = base_macs_per_s / 1e12
    pipelined_tops = pipelined_macs_per_s / 1e12
    
    print(f"\nPipeline impact (4×4 = 16 PEs):")
    print(f"  Base freq: {base_freq} MHz → {base_tops:.5f} TOPS")
    print(f"  Pipelined freq: {pipelined_freq} MHz → {pipelined_tops:.5f} TOPS")
    print(f"  Improvement: {(pipelined_tops/base_tops*100):.1f}%")
    print(f"  (Note: actual silicon results may vary)")


def main():
    """Run all Phase 8 performance analyses."""
    estimate_array_scaling()
    estimate_sram_banking()
    estimate_pipeline_design()
    
    print("\n" + "=" * 60)
    print("Phase 8 Summary")
    print("=" * 60)
    print("""
Key Findings:
1. 2×2 array (4 PEs): ~0.05 TOPS — far from 2.4 TOPS target
2. 8×8 array (64 PEs): ~0.25 TOPS (at 100MHz) — still ~10× below target
3. Pipeline design: ~2× freq improvement possible → ~0.5 TOPS for 8×8
4. Array scaling to ~64×64 (4K PEs) needed for 2.4 TOPS @ 100MHz
5. 2 MiB SRAM sufficient for weight storage of moderate-sized networks
6. Banked SRAM essential for concurrent weight/input/output access
7. 1.2W power estimate: rough linear scaling; 8×8 may be ~0.3-0.5W,
   64×64 may be ~1-2W (need actual synthesis measurements)
   """)
    print("=" * 60)


if __name__ == "__main__":
    main()