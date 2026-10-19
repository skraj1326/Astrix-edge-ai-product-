# Astrixcore AX-1 Verification Plan

## Verification Strategy

The verification approach follows a tiered strategy from simple unit tests to full end-to-end execution:

### Tier 1 — Unit Tests (Python Reference Model)

- Test individual MAC operations with known input/output pairs
- Test quantization, saturation, and shift operations
- Test edge cases: zero, min/max values, signed wrapping
- Test FSM transitions in the RTL model

### Tier 2 — Microbenchmarks (SystemVerilog + Cocotb)

- Basic MAC operation: verify accumulator = sum(weight * input) for all INT8 combinations
- Reset behavior: verify all registers clear to zero on asserted reset
- Boundary values: min (−128) and max (127) inputs and weights
- Signed arithmetic: negative * negative = positive, negative * positive = negative
- Overflow: accumulation beyond INT32 range
- Backpressure: verify stall behavior when output not consumed
- Invalid command: verify error status on unrecognized command opcode

### Tier 3 — Functional Tests (Full Accelerator)

- End-to-end convolution-like MAC operation with known kernel
- Multi-output accumulation (macro-unrolling)
- Command sequence: load weights → load inputs → execute → read result
- Command termination: stop command clears busy flag

### Tier 4 — Compiler Integration

- ONNX import pipeline validates operator support
- Quantization correctness: INT8 weights dequantize to near-floating-point accuracy
- Generated commands only for operations supported by current RTL

## Testbench Architecture

### SystemVerilog Testbench Structure

```systemverilog
module tb_accelerator;
  // Clock and reset
  logic clk;
  logic rst_n;
  
  // DUT interfaces
  logic [31:0] csr_cmd;
  logic [31:0] csr_status;
  // ... other interfaces
  
  // Test sequences
  initial begin
    // Initialize
    rst_n = 0;
    #100 rst_n = 1;
    
    // Test 1: Reset all registers
    // ...
    
    // Test 2: Basic MAC operation
    // ...
    
    // Test 3: Signed arithmetic
    // ...
    
    // Test 4: Overflow behavior
    // ...
    
    #100 $finish;
  end
endmodule
```

### Cocotb Tests (Python-embedded)

- Cocotb test vectors for MAC operations
- Parameterized test cases across input/weight combinations
- Coverage collection on operation coverage bins

## Coverage Metrics

- **Functional coverage**: bins for all command opcodes, array sizes, saturation events
- **Code coverage**: line/block coverage on RTL testbench
- **Functional hit rate**: percentage of INT8 input combinations exercised

## Regression Testing

- Nightly run of full testsuite via GitHub Actions
- Automatic failure on: simulation timeout, coverage drop, new X-propagation
- Artefacts: VCD files, coverage reports, test logs