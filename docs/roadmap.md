# Astrixcore AX-1 Roadmap

## Phase 1 — Project Foundation ✅

- [x] Repository creation and directory structure
- [x] README.md, LICENSE, .gitignore
- [x] Documentation: architecture, microarchitecture, HW/S interface, verification plan, roadmap
- [x] requirements.txt with dependencies
- [x] GitHub Actions CI workflow skeleton

## Phase 2 — Executable Reference Model

- [ ] Python reference model for INT8 arithmetic, MAC accumulation, quantization, saturation
- [ ] First supported neural-network operators (conv1d,relu)
- [ ] Unit tests for reference model
- [ ] Integration tests with pytest

## Phase 3 — Minimal RTL Accelerator

- [ ] Small parameterized MAC datapath (e.g., 2×2 PE array)
- [ ] Accumulator with saturation logic
- [ ] Control FSM (command, status, config registers)
- [ ] Simple SRAM interface (read/write)
- [ ] Register-level testbench with SystemVerilog
- [ ] Basic cocotb tests for MAC correctness

## Phase 4 — Verification

- [ ] Self-checking SystemVerilog testbenches
- [ ] Cocotb regression tests
- [ ] Reset tests, boundary values, signed arithmetic, overflow
- [ ] Backpressure and invalid-command tests
- [ ] Code coverage > 80%

## Phase 5 — Compiler MVP

- [ ] ONNX import pipeline for documented operator subset
- [ ] Graph validation and supported-operator checks
- [ ] INT8 quantization pipeline
- [ ] Intermediate representation (IR) for accelerator commands
- [ ] Generate accelerator commands only for supported operations

## Phase 6 — End-to-End Execution

- [ ] Small neural-network workload (e.g., conv3x3 + relu) through compiler + RTL
- [ ] Compare outputs with Python reference model
- [ ] Report: correctness, latency (simulated cycles), memory usage, limitations

## Phase 7 — FPGA Preparation

- [ ] Synthesis scripts for supported FPGA toolchain (Yosys + nextpnr or Vivado)
- [ ] XDC constraints (clock, reset, SRAM initialization)
- [ ] Report inferred resource usage and timing
- [ ] Clearly distinguish simulated results from actual FPGA measurements

## Phase 8 — Performance Expansion

- [ ] Array scaling analysis (4×4 → 8×8 → 16×16)
- [ ] Data reuse and SRAM banking investigation
- [ ] Pipeline design for throughput improvement
- [ ] Operator fusion opportunities
- [ ] Bandwidth limit characterization
- [ ] Feasibility estimation against product targets

## Phase 9 — ASIC Readiness Assessment

- [ ] Clock and reset strategy documentation
- [ ] Timing constraints and test pattern specification
- [ ] Memory macros and physical-design dependencies
- [ ] Interface-IP requirements (UART, SPI, etc.)
- [ ] Power-estimation methodology
- [ ] Foundry-specific gaps analysis
- [ ] Do NOT describe as tapeout-ready without supporting evidence

## Success Criteria for Each Phase

Each phase must produce verifiable artifacts before advancing:

1. **Working implementation** — not just simulation scripts, actual runnable code
2. **Test coverage** — self-checking tests with documented pass/fail criteria
3. **Documentation** — updated specs reflecting actual implementation
4. **Known limitations** — explicitly listed unsupported features or unverified behaviors