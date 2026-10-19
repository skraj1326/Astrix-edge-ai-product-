# Astrixcore AX-1 ASIC Readiness Assessment — Phase 9

Documentation of clock and reset strategy, timing constraints, test strategy,
memory macros, physical-design dependencies, interface-IP requirements,
power-estimation requirements, and foundry-specific gaps.

**Status**: Do NOT describe as tapeout-ready without supporting evidence.

## 1. Clock and Reset Strategy

### Clock Domain
- **Single clock domain**: Synchronous design, one primary clock (clk)
- **Frequency**: Target 100 MHz (Artix-7 achievable; 133 MHz with pipeline)
- **Clock tree**: Buffer clk via BUFG (Xilinx) or equivalent for target ASIC
- **Clock gating**: Disabled in Phase 1; can be enabled for power reduction in later phases
- **Reset**: Active-low asynchronous rst_n
  - Asserted during power-on sequence
  - De-asserted after clock stabilization (e.g., after 1000 cycles)
  - Internal sync chain: 2-3 flop stages to synchronize async reset release
  - All registers have defined reset behavior (cleared to 0)

### Reset Strategy
| Signal | Type | Polarity | Behavior |
|---|---|---|---|
| rst_n | Async | Active-low | Clears all registers to 0; accelerator FSM enters idle |
| hreset | Sync (AHB) | Active-high | AHB-specific reset; separate from rst_n |
| Core reset | Sync | Async-low | Synchronous after async de-assertion; 2-flop sync chain |

### Power-On Sequence
1. deassert rst_n only after clk has been stable for ≥1000 cycles
2. Allow 10-100 μs for PLL/Lock time (if used)
3. Begin normal operation with all modules reset-clear

## 2. Timing Constraints

### Synthesis & STIMA Reports (post-synthesis)
- **Setup time**: All flop-to-flop paths must meet setup at target freq
- **Hold time**: All paths must be hold-safe (typically met by MMCM/PLL)
- **Metastability**: Two-flop synchronizer chains for async inputs (rst_n, external data)

### Example Constraints File (XDC-like format)
```tcl
# Clock definition
create_clock -name clk [get_ports clk] -period 10.0 [get_ports clk]

# Reset
set_asynchronous_reset -name rst_n [get_ports rst_n] -level low

# Set max delay (setup)
set_max_delay -from [get_ports clk] -to [all_registers] -clk [get_ports clk] [expr 10.0 - 0.1]

# Set min delay (hold)
set_min_delay -from [get_ports clk] -to [all_registers] -clk [get_ports clk] 5.0

# AHB interface timing (if targeted)
create_gated_clock -name hclk [get_ports hclk] -gate [get_ports hwrite]
```

### Target Frequencies by Array Size (estimated)
| Array Size | PEs | Estimated Max Freq | Notes |
|---|---|---|---|
| 2×2 | 4 | 133 MHz | Small routing, easy timing |
| 4×4 | 16 | 110-125 MHz | Moderate routing |
| 8×8 | 64 | 80-100 MHz | Significant routing congestion |
| 16×16 | 256 | 50-70 MHz | Likely requires pipeline + Duff's device |

## 3. Test Strategy

### Formal Verification
- **Property checker** (e.g., SymbiYosys, Cadence Incisive): Prove reset,
  SAT/saturation, and no-overflow properties
- **Coverage**: Aim for 100% toggle coverage on all flops; 100% FSM transition coverage

### Simulation-Based Tests
| Test Category | Test Cases | Notes |
|---|---|---|
| Reset | Assert rst_n, verify all registers 0; deassert, verify operation | Include timeout test (bus never completes) |
| Boundary values | Weights = −128, 127; inputs = −128, 127, 0 | Test all 4 combinations of min/max |
| Signed arithmetic | Negative × negative = positive; negative × positive = negative | Verify sign-extension correctness |
| Overflow | Accumulate many MACs beyond INT32 range; verify saturation behavior | Test 100+ consecutive max MACs |
| Backpressure | Stall valid handshake; verify no data loss; verify retry | Test with array fully loaded |
| Invalid command | Write 0b1111 to cmd_reg; verify error status; no hang | Verify FSM returns to idle |
| Data hazard | Write new weight while MAC in progress; verify correct result | Test with overlapping operations |

### Silicon Validation (post-tapeout)
- **Power-on reset**: Verify all modules initialize to known state
- **Self-test**: On-chip test pattern (checkerboard, walking-one)
- **DMI/Debug**: Readback of all CSRs via AHB-Lite; compare to simulation
- **Performance benchmarks**: Measure actual MAC/cycle, freq, power

## 4. Memory Macros & Physical-Design Dependencies

### SRAM Organization (target 2 MiB)
| Buffer | Size | Purpose | Inference |
|---|---|---|---|
| Weight buffer | Variable (≤2 MiB) | Stores INT8 weights for active operator | Inferred from BRAM or synthesized from gate array |
| Input buffer | Variable (≤1 MiB) | Holds current input tile (e.g., 64×64 for YOLOv8n) | Same as above |
| Output buffer | Variable (≤1 MiB) | Partial sum accumulation before reduction | Same as above; INT32 width |

### BRAM Inference (Xilinx Artix-7)
- Each BRAM36E1: 36 Kb (×32 or ×36), 2 ports
- Each BRAM18E1: 18 Kb (×16 or ×18), 1 port
- For 2 MiB: need ~56 BRAM36E1 tiles (36×56=2016 Kb ≈ 2 MiB)
- Artix-7 has ~80 BRAM36E1 tiles available (on XC7A35T)
- **Conclusion**: 2 MiB SRAM exceeds Artix-7 BRAM capacity; would require:
  - Multi-chip module (MCM), or
  - External DRAM (contra design objective), or
  - Smaller on-chip SRAM (e.g., 256 KiB = 8 BRAM36E1), or
  - Target larger FPGA/ASIC cell library

### ASIC Standard Cell Library
- SRAM macros from foundry (e.g., TSMC 28 nm, UMC 65 nm)
- Typical SRAM cell: 6T (6-transistor) per bit
- 2 MiB @ 28 nm: ~2.5-3 mm² (including sense amps, row/col decoders)
- Power: SRAM is significant power consumer; gating recommended

## 5. Interface-IP Requirements

### AHB-Lite Slave Interface (already designed)
- **Protocol**: Standard AHB-Lite (lim-limited, no burst, single master)
- **Data width**: 32 bits
- **Address width**: 32 bits (only lower bits used: haddr[7:2] for SRAM, haddr[1:0] for CSR)
- **Burst**: Not needed; register-access model sufficient
- **Clock domain**: Same as core clock (clk); hreset synchronous

### Required Additional IP (Phase 9)
| IP | Purpose | Status |
|---|---|---|
| UART | Debug console output | Available via softcore or standalone |
| SPI | External EEPROM/flash for firmware | Standard peripheral |
| I2C | Sensor/PHY configuration (future) | Not needed for Phase 1 |
| Timer | OS tick, timeout for accelerator | Simple counter + compare |
| PWM | Fan control, power management | Not in Phase 1 scope |

### Missing IP for Tapeout
- **MIPI PHY**: Not invented (per mandatory rule 8); begin with simulation/FPGA-friendly input
- **DDR PHY**: Not needed (DRAM-less architecture objective)
- **SerDes**: Not needed for Phase 1; added for higher-speed IO later

## 6. Power-Estimation Requirements

### Dynamic Power Estimate (relative, post-synthesis)
| Component | 2×2 PE @ 100 MHz | 8×8 PE @ 100 MHz | 16×16 PE @ 100 MHz |
|---|---|---|---|
| PE array (MACs) | ~0.05 W | ~0.2 W | ~0.5 W |
| BRAM/SRAM | ~0.1 W | ~0.2 W | ~0.5 W |
| Control logic + AHB | ~0.02 W | ~0.05 W | ~0.1 W |
| **Total** | **~0.17 W** | **~0.45 W** | **~1.1 W** |
| ** vs 1.2 W target** | ✅ Below | ⚠️ Near | ❌ Above |

**Notes**:
- TSMC 28 nm CMOS, 1.0 V core
- Dynamic power ∝ CV²f; C ≈ proportional to PE count
- Static/leakage power: ~20-30% of total at 85°C
- Clock gating PE array when idle can recover 30-50% power
- Measurement required: actual silicon power analysis (PSRR, IR drop)

## 7. Foundry-Specific Gaps

| Gap | Severity | Mitigation |
|---|---|---|
| No foundry PDK access (TSMC/UMC/Samsung) | High | Use FPGA prototypes (Artix-7, Intel Cyclone) for early verification; target PDK later |
| No licensed EDA tools (Vivado, Cadence, Synopsys) | High | Use open-source toolchain: Yosys + nextpnr + OpenSTA; commercial tools later |
| No 28 nm process license | High | Artix-7 (Artix-7 is 28 nm equivalent but commercial FPGA, not ASIC); actual 28 nm ASIC later |
| No physical-design flow (place-route, STA, sign-off) | Medium | Use nextpnr for preliminary P&R; full sign-off requires commercial flow |
| No analog/RF design tools | Low (not needed for digital AI accelerator) | N/A |
| I/O pad design for 28 nm not generated | Medium | Use FPGA I/O banking as placeholder; generate pads from foundry library later |

## 8. ASIC Readiness Checklist — PASS/FAIL

| Item | Status | Notes |
|---|---|---|
| Clock & reset strategy documented | ✅ PASS | Single-clock, async-rst_n with sync chain |
| Timing constraints defined | ✅ PASS (estimated) | Setup/hold, max/min delay XDC-like |
| Formal verification plan | ✅ PASS | Property list defined |
| Simulation testplan | ✅ PASS | 7 test categories defined |
| SRAM organization documented | ✅ PASS | Weight/input/output buffers; BRAM vs ASIC |
| Interface-IP catalog | ✅ PASS | AHB-Lite complete; additional IP listed |
| Power estimation (relative) | ✅ PASS | ~0.17W–1.1W depending on array size |
| Foundry PDK availability | ❌ FAIL | No PDK; FPGA prototype used instead |
| Timing sign-off capability | ❌ FAIL | Open-source STA (OpenSTA) not fully validated |
| Power sign-off capability | ❌ FAIL | No commercial PPA tools |
| **Overall ASIC readiness** | **Conditional**: ✅ Prototype-ready on FPGA; ❌ Tapeout-ready **NO** | **Do not describe as tapeout-ready without supporting evidence** |

## 9. Conclusion & Recommendations

### Current State
- Astrixcore AX-1 is **FPGA-prototype ready** through Phases 1–7
- All software flow (ONNX → IR → accelerator commands) functional
- RTL verified against Python reference model (Phase 4–5)
- Resource estimates and PPA bounds documented

### Path to ASIC
1. **Phase 9 exit**: Use FPGA (Artix-7 or equivalent) for final validation
2. **Acquire 28 nm PDK**: TSMC N28PGA or equivalent (foundry agreement needed)
3. **Obtain EDA tools**: Academic licenses (Mentor Graphics, Cadence) or commercial
4. **Tapeout preparation**: 3–6 month cycle from PDK acquisition to mask set
5. **Power/performance validation**: Full synthesis → STA → gate-level simulation

### Do NOT Describe As
- ❌ "Tapeout-ready" — No PDK, no sign-off, no power measurements
- ❌ "Silicon-validated" — Only FPGA-verified
- ❌ "Production IP" — Research prototype only

### Recommended Next Action
Proceed with **FPGA prototyping** (Phase 7 complete) to collect:
- Actual frequency measurements
- Actual power readings (via Xilinx Power Estimator or onboard sensors)
- Actual resource utilization (BRAM, LUT, FF, DSP)
- Timing closure data for larger arrays

Update this document after FPGA validation; re-evaluate ASIC readiness at that point.

---
*Document generated as part of Astrixcore AX-1 Phase 9 ASIC Readiness Assessment.
*Never describe project as tapeout-ready or silicon-validated without supporting evidence.