# Astrixcore AX-1 Hardware-Software Interface

## Programming Model

### Accelerator Command Format

The accelerator is commanded via the control register at offset 0x00:

```
bit [31:8]  reserved, must be zero
bit [7]     start         — 1 initiates MAC operation, self-clears on completion
bit [6:4]   array_size    — selects accelerator array dimension (0=1×1, 1=2×2, 2=4×4, 3=8×8)
bit [3:0]   operation     — operation type (0=MAC, 1=LOAD, 2=STORE, 3=RESERVED)
```

### Status Register

The status register at offset 0x04 reports:

```
bit [31:4]  reserved, must be zero
bit [3]     done          — 1 when last operation completed
bit [2]     busy          — 1 when accelerator is processing
bit [1]     error         — 1 if invalid command or saturation occurred
bit [0]     idle          — 1 when accelerator is idle and ready
```

### Configuration Register

Offset 0x14 configures the accelerator:

```
bit [31:8]  reserved
bit [7]     weight_precision — 0=INT8, 1=INT4 (future)
bit [6:4]   array_dimension  — matches cmd_reg array_size
bit [3:0]   mac_count        — number of active PEs (0=disabled, 1–15=enabled)
```

### SRAM Access

Host reads/writes via memory-mapped addresses:

- Offset 0x0C (addr_reg): Sets SRAM address for read/write
- Offset 0x10 (data_reg): Writes data to current SRAM address, reads data from current SRAM address
- Auto-increment mode available (cfg_reg bit 0)

### ISA Extensions (RISC-V Custom)

The RISC-V control processor can issue custom accelerator instructions:

```
acc <size>, <op>, <addr>   — Issue accelerator command
ld.w <sram_addr>, <reg>    — Load word from SRAM
st.w <reg>, <sram_addr>    — Store word to SRAM
```

## Software Workflow

1. **Initialize**: Write array dimensions and configuration to CSRs
2. **Load Weights**: DMA weights from host memory to SRAM via `st.w` instructions
3. **Load Inputs**: DMA input data to SRAM
4. **Execute**: Write `start=1` to cmd_reg to begin MAC computation
5. **Poll Status**: Read status_reg until `done=1` and `busy=0`
6. **Read Result**: Read accumulator from SRAM or data_reg
7. **Repeat** or halt

## Error Handling

- Invalid command → error bit set in status_reg, accelerator halts
- Overflow/saturation → error bit set, result clamped
- Timeout (busy stuck) → host software should reset accelerator via cmd_reg bit 2

## ABI and Calling Convention

- All CSR accesses use 32-bit word operations
- Little-endian byte order (RISC-V default)
- Alignment: 4-byte aligned accesses only
- Exception handler: illegal instruction exception for unsupported custom ops