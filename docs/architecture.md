# Astrixcore AX-1 Architecture

## High-Level Block Diagram

The Astrixcore AX-1 consists of the following major components:

1. **32-bit RISC-V Control Processor** — Standard RISC-V ISA (RV32IMA with extensions), responsible for command dispatch, configuration, and host interaction.

2. **Neural Accelerator Array** — Parameterized INT8 MAC array with configurable dimensions. Starts small (e.g., 4×4 or 8×8) and scales based on resource justification.

3. **On-Chip SRAM** — 2 MiB target capacity, organized for weight and input buffering. DRAM-less architecture is a design objective.

4. **Memory Interface** — Simple bus interface for host DMA and external memory (if/when DRAM is added).

5. **Register File Interface** — Memory-mapped control and status registers (CSRs) for the accelerator.

6. **Clock and Reset Generation** — Synchronous design with reset polarity documented.

## Instruction Set Extensions

The accelerator supports a custom instruction set via the RISC-V extension mechanism:

- `ACC` — Execute MAC operation on accelerator array
- `LOAD` — Load weights/inputs from SRAM
- `STORE` — Store results to SRAM/host memory

## Memory Map

| Address Range | Description |
|---|---|
| 0x0000_0000 – 0x0000_3FFF | Accelerator CSR register space |
| 0x0000_4000 – 0x0000_7FFF | On-chip SRAM (parameterizable) |
| 0x0000_8000 – 0xFFFF_FFFF | Reserved / host memory |

## Clocking Assumptions

- Single clock domain (synchronous design)
- Reset pin with active-high polaritity documented in reset behavior
- Clock enable for power gating (future)

## Power Estimation

- Target: 1.2 W at typical INT8 workload
- DRAM-less design reduces memory energy significantly
- Accelerator array power scales with active MAC count