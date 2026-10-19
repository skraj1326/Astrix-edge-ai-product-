# Astrixcore AX-1 Microarchitecture

## Accelerator Datapath

### MAC Array

- Parameterizable dimensions (N×M processing elements)
- Each PE performs: `accumulator += weight * input` where all values are INT8
- Partial sum accumulation in local accumulator registers
- Saturation logic on read-back to prevent overflow

### Accumulator

- 32-bit accumulated sum per output channel
- Write-back to SRAM after MAC operation completes
- Saturation behavior: clamped to INT32 range [−2^31, 2^31−1]

### Dataflow

Weights stationary: weights are loaded once and reused across input activations.
Input stationary alternative for future exploration.

## Register Interface (CSRs)

Accelerator control resides in a memory-mapped register block:

| Offset | Name | Width | Reset Value | Description |
|---|---|---|---|---|
| 0x00 | cmd_reg | 32 | 0 | Accelerator command (start, stop, reset) |
| 0x04 | status_reg | 32 | 0 | Accelerator status (idle, busy, done, error) |
| 0x08 | dim_reg | 32 | 0 | Array dimensions (N, M packed) |
| 0x0C | addr_reg | 32 | 0 | SRAM read/write address |
| 0x10 | data_reg | 32 | 0 | Data input/output register |
| 0x14 | cfg_reg | 32 | 0 | Configuration (activation mode, precision) |
| 0x18–0x3F | reserved | — | — | — |

## Memory Organization

### On-Chip SRAM

- Parameterizable size (target: 2 MiB)
- Dual-port for concurrent read/write
- Organized as weight buffer + input buffer + output buffer
- Address mapping depends on configuration

### Weight Storage

- INT8 weights stored in SRAM
- Row-major ordering for convolution kernels
- Padding for kernel alignment (future)

### Input/Output Buffering

- Double-buffering for ping-pong I/O
- Direct host access via memory-mapped addresses

## Precision and Quantization

### INT8 Format

- Signed 8-bit values: [−128, 127]
- Multiplication: `result = weight * input` (full precision intermediate)
- Accumulation: 32-bit accumulator with saturation
- Output quantization: right-shift with rounding before read-back

### Quantization Pipeline

1. MAC operation produces INT32 partial sum
2. Optional right-shift (arithmetic shift right) for scaling
3. Saturation to INT8 range if output must be INT8
4. Round-to-nearest-even for shift operations

## Reset Behavior

- All registers reset to zero
- Accelerator FSM enters idle state
- SRAM retains contents (SRAM is power-retentive)
- Clock gating disabled until stable clock

## Clocking

- Single clock domain, no asynchronous reset crossing in core
- Reset pin must be pulsed high during power-on
- Clock enable pin for power-saving (future extension)