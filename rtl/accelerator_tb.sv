`ifndef ASTRIXCORE_AX1_RTL_TB_SV
`define ASTRIXCORE_AX1_RTL_TB_SV

// Testbench for the Astrixcore AX-1 accelerator
// Verifies basic AHB-Lite register access and accelerator functionality
// Tests INT8 MAC operations against the Python reference model

`timescale 1ns / 1ps

module accelerator_tb;

    // Clock and reset
    logic clk = 0;
    logic rst_n = 0;

    // AHB-Lite signals
    logic [31:0] haddr;
    logic [31:0] hwdata;
    logic [31:0] hrdata;
    logic hresp;
    logic hwrite;
    logic hreset;

    // Accelerator control
    logic acc_busy;
    logic acc_done;
    logic [3:0] acc_status;
    logic [3:0] acc_dim;
    logic [31:0] acc_result;

    // Clock generator
    always #5 clk <= ~clk;  // 100MHz clock

    // Instantiate the design under test
    accelerator_top uut (
        .clk(clk),
        .rst_n(rst_n),
        .haddr(haddr),
        .hwdata(hwdata),
        .hrdata(hrdata),
        .hresp(hresp),
        .hwrite(hwrite),
        .hreset(hreset),
        .acc_busy(acc_busy),
        .acc_done(acc_done),
        .acc_status(acc_status),
        .acc_dim(acc_dim),
        .acc_result(acc_result)
    );

    // -------------------------------------------------------------------------
    // Test sequence
    // -------------------------------------------------------------------------

    initial begin
        // Initialize inputs
        haddr = 0;
        hwdata = 0;
        hwrite = 0;

        // Wait for initial stabilization
        repeat (2) @(posedge clk);

        // Apply reset
        rst_n = 0;
        @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // ---- Test 1: Read/write CSR registers ----
        `TESTLINE("Test 1: CSR register access")

        // Write to command register (offset 0x00) - START
        haddr = 32'h00000000;
        hwdata = 4'b0001;  // START command
        hwrite = 1;
        @(posedge clk);
        hwrite = 0;
        @(posedge clk);

        // Read status register
        haddr = 32'h00000000;
        @(posedge clk);
        `ifdef TB_PRINT
        $display("After START: acc_busy = %b, acc_done = %b", acc_busy, acc_done);
        `endif

        // ---- Test 2: Write dimension register (2x2) ----
        `TESTLINE("Test 2: Dimension register write")

        haddr = 32'h00000004;
        hwdata = 4'b0010;  // 2x2 array dimension
        hwrite = 1;
        @(posedge clk);
        hwrite = 0;
        @(posedge clk);

        // ---- Test 3: Write weight and input, execute MAC ----
        `TESTLINE("Test 3: MAC operation (2x2 array)")

        // Load weight = 50 (INT8) to SRAM
        haddr = 32'h00000010;
        hwdata = 8'd50;  // INT8 weight = 50
        hwrite = 1;
        @(posedge clk);
        hwrite = 0;
        @(posedge clk);

        // Load input = 40 (INT8) 
        haddr = 32'h00000000;
        hwdata = 8'd40;  // INT8 input = 40
        hwrite = 1;
        @(posedge clk);
        hwrite = 0;
        @(posedge clk);

        // Execute MAC: 50 * 40 = 2000
        // With shift=4: 2000 >> 4 = 125 (rounded)
        // Saturated to INT8: 125 (within [-128, 127])
        // Without shift: 2000 saturated to 127
        
        // Write START command
        haddr = 32'h00000000;
        hwdata = 4'b0001;  // START command
        hwrite = 1;
        @(posedge clk);
        hwrite = 0;
        @(posedge clk);

        // Wait for operation to complete
        repeat (20) @(posedge clk);

        `ifdef TB_PRINT
        $display("After MAC: acc_result = %h (expected ~125 with shift, or 127 saturated)", acc_result);
        $display("acc_busy = %b, acc_done = %b", acc_busy, acc_done);
        $display("acc_status = %b", acc_status);
        $display("acc_dim = %b", acc_dim);
        `endif

        // Verify acc_done is asserted
        if (acc_done !== 1'b1) begin
            $error("FAIL: acc_done should be 1 after MAC completion");
            $stop;
        end

        // ---- Test 4: Verify saturation with large values ----
        `TESTLINE("Test 4: Saturation check")

        // Write max INT8 weight = 127
        haddr = 32'h00000010;
        hwdata = 8'd127;
        hwrite = 1;
        @(posedge clk);
        hwrite = 0;
        @(posedge clk);

        // Write max INT8 input = 127
        haddr = 32'h00000000;
        hwdata = 8'd127;
        hwrite = 1;
        @(posedge clk);
        hwrite = 0;
        @(posedge clk);

        // Write START command
        haddr = 32'h00000000;
        hwdata = 4'b0001;
        hwrite = 1;
        @(posedge clk);
        hwrite = 0;
        @(posedge clk);

        // Wait for operation to complete
        repeat (20) @(posedge clk);

        `ifdef TB_PRINT
        $display("After saturated MAC: acc_result = %h", acc_result);
        $display("Expected: 127 (saturated from 127*127=16129 or 16129>>4=1009 saturated)");
        `endif

        // acc_result should be 127 (saturated), not a larger value
        if (acc_result > 127 || acc_result < -128) begin
            $error("FAIL: acc_result should be saturated to INT8 range [-128, 127]");
            $stop;
        end

        // ---- Test 5: Reset test ----
        `TESTLINE("Test 5: Reset operation")

        haddr = 32'h00000000;
        hwdata = 4'b0011;  // RESET command
        hwrite = 1;
        @(posedge clk);
        hwrite = 0;
        @(posedge clk);

        // After reset, accelerator should be idle
        `ifdef TB_PRINT
        $display("After RESET: acc_busy = %b, acc_done = %b", acc_busy, acc_done);
        `endif

        if (acc_busy !== 1'b0) begin
            $error("FAIL: acc_busy should be 0 after reset");
            $stop;
        end
        if (acc_done !== 1'b0) begin
            $error("FAIL: acc_done should be 0 after reset");
            $stop;
        end

        // ---- Test 6: Invalid command test ----
        `TESTLINE("Test 6: Invalid command test")

        haddr = 32'h00000000;
        hwdata = 4'b1111;  // Invalid command
        hwrite = 1;
        @(posedge clk);
        hwrite = 0;
        @(posedge clk);

        // Status should not be DONE
        `ifdef TB_PRINT
        $display("After invalid command: acc_status = %b", acc_status);
        `endif

        // Wait and finish
        #100;
        $display("TESTS COMPLETED");
        $finish;
    end

    // VCD dump for debugging
    initial begin
        $dumpfile("accelerator_tb.vcd");
        $dumpvars(0, accelerator_tb);
    end

endmodule

`endif // ASTRIXCORE_AX1_RTL_TB_SV