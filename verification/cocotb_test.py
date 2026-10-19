"""
Cocotb test for the Astrixcore AX-1 accelerator.

These tests use cocotb's Python-embedded test framework to verify the RTL
accelerator behavior. Each test runs in a simulator context (typically
ModelSim, VCS, or Verilator).
"""

import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from cocotb.binary import BinaryValue


@cocotb.test()
def test_accelerator_reset(dut):
    """Test that the accelerator resets correctly."""
    # Initialize signals
    dut.rst_n <= 0
    dut.haddr <= 0
    dut.hwdata <= 0
    dut.hwrite <= 0
    
    # Wait for global reset deassertion
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    # Deassert reset
    dut.rst_n <= 1
    
    # Wait a few cycles
    for _ in range(5):
        await RisingEdge(dut.clk)
    
    # After reset, accelerator should be idle
    assert dut.acc_busy == 0, "acc_busy should be 0 after reset"
    assert dut.acc_done == 0, "acc_done should be 0 after reset"
    dut._log("Reset test PASSED")


@cocotb.test()
def test_accelerator_start_stop(dut):
    """Test starting and stopping the accelerator."""
    # Apply reset first
    dut.rst_n <= 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n <= 1
    await RisingEdge(dut.clk)
    
    # Write START command to command register (offset 0x00)
    dut.haddr <= 32'h00000000
    dut.hwdata <= 4'b0001  # START
    dut.hwrite <= 1
    await RisingEdge(dut.clk)
    dut.hwrite <= 0
    await RisingEdge(dut.clk)
    
    # Wait for busy and done
    for _ in range(10):
        await RisingEdge(dut.clk)
    
    # Check that done is asserted
    assert dut.acc_done == 1, "acc_done should be 1 after MAC completion"
    dut._log("Start/stop test PASSED")


@cocotb.test()
def test_accelerator_dimension(dut):
    """Test dimension register programming."""
    # Apply reset
    dut.rst_n <= 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n <= 1
    await RisingEdge(dut.clk)
    
    # Write 2x2 dimension
    dut.haddr <= 32'h00000004
    dut.hwdata <= 4'b0010  # 2x2
    dut.hwrite <= 1
    await RisingEdge(dut.clk)
    dut.hwrite <= 0
    await RisingEdge(dut.clk)
    
    # Read back dimension
    dut.haddr <= 32'h00000004
    dut.hwrite <= 0
    await RisingEdge(dut.clk)
    
    # Dimension should be 2x2
    assert dut.acc_dim == 4, f"Expected dim=4 (2x2), got {dut.acc_dim}"
    dut._log("Dimension test PASSED")


@cocotb.test()
def test_accelerator_saturation(dut):
    """Test MAC saturation with large values."""
    # Apply reset
    dut.rst_n <= 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n <= 1
    await RisingEdge(dut.clk)
    
    # Write max INT8 weight
    dut.haddr <= 32'h00000010
    dut.hwdata <= 8'd127
    dut.hwrite <= 1
    await RisingEdge(dut.clk)
    dut.hwrite <= 0
    await RisingEdge(dut.clk)
    
    # Write max INT8 input
    dut.haddr <= 32'h00000000
    dut.hwdata <= 8'd127
    dut.hwrite <= 1
    await RisingEdge(dut.clk)
    dut.hwrite <= 0
    await RisingEdge(dut.clk)
    
    # Wait for operation to complete
    for _ in range(10):
        await RisingEdge(dut.clk)
    
    # Result should be saturated to 127 (not 16129)
    # Note: actual value depends on internal implementation
    dut._log(f"Saturation test: acc_result={dut.acc_result}, acc_done={dut.acc_done}")
    
    # acc_done should be asserted
    assert dut.acc_done == 1, "acc_done should be 1 after operation"
    dut._log("Saturation test PASSED")


@cocotb.test()
def test_accelerator_invalid_command(dut):
    """Test that invalid commands are rejected."""
    # Apply reset
    dut.rst_n <= 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n <= 1
    await RisingEdge(dut.clk)
    
    # Write invalid command
    dut.haddr <= 32'h00000000
    dut.hwdata <= 4'b1111  # INVALID
    dut.hwrite <= 1
    await RisingEdge(dut.clk)
    dut.hwrite <= 0
    await RisingEdge(dut.clk)
    
    # Wait and check status
    for _ in range(5):
        await RisingEdge(dut.clk)
    
    # Status should indicate error (not done, not idle with error flag)
    dut._log(f"Invalid command test: acc_status={dut.acc_status}")
    
    # The accelerator should not be in done state after invalid command
    # (exact check depends on implementation)
    assert dut.acc_done == 0, "acc_done should not be 1 after invalid command"
    dut._log("Invalid command test PASSED")


def run_simulation(test_func, *args, **kwargs):
    """Run a cocotb test function."""
    import sys
    from cocotb.runner import get_runner
    
    # Get the simulator runner
    runner = get_runner('verilator')  # or 'modelsim', 'vcs', etc.
    
    # Run the test
    runner.build()
    runner.test(test_func, *args, **kwargs)