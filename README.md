# Astrixcore AX-1

A low-power RISC-V-based neural accelerator for industrial automated optical inspection (AOI).

## Overview

The Astrixcore AX-1 is a research and engineering prototype designed to deliver efficient INT8 inference for AOI applications. The project implements a credible RISC-V-controlled neural accelerator from scratch using open-source tools.

## Product Targets (Aspirational)

- **2,048** INT8 MAC operations per cycle-equivalent accelerator array
- **2 MiB** on-chip SRAM capacity
- **2.4** TOPS peak performance
- **1.2** W power target
- Under **6 ms** YOLOv8n inference
- **32-bit** RISC-V control processor
- DRAM-less architecture (design objective)

## Development Phases

- **Phase 1** — Project foundation (current)
- **Phase 2** — Executable reference model
- **Phase 3** — Minimal RTL accelerator
- **Phase 4** — Verification
- **Phase 5** — Compiler MVP
- **Phase 6** — End-to-end execution
- **Phase 7** — FPGA preparation
- **Phase 8** — Performance expansion
- **Phase 9** — ASIC readiness assessment

## Status

This is a research prototype. No hardware claims (synthesizability, verification, FPGA testing, or silicon readiness) are made until corresponding evidence exists.

## License

See LICENSE for licensing details.

## Project Structure

```
Astrixcore_AX-1/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── docs/
│   ├── architecture.md
│   ├── microarchitecture.md
│   ├── hardware_software_interface.md
│   ├── verification_plan.md
│   └── roadmap.md
├── rtl/
│   └── ... (Register Transfer Level sources)
├── compiler/
│   └── ... (ONNX import pipeline, graph validation)
├── verification/
│   └── ... (SystemVerilog testbenches, cocotb tests)
├── tests/
│   └── ... (Unit tests, integration tests)
├── fpga/
│   └── ... (Synthesis scripts, constraints)
└── scripts/
    └── ... (Build, test, simulation scripts)
```

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd Astrixcore_AX-1

# Install dependencies
pip install -r requirements.txt

# Run basic tests
python -m pytest tests/ -v
```