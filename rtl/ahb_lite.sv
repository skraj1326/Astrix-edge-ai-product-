`ifndef ASTRIXCORE_AX1_RTL_AHB_LITE_SV
`define ASTRIXCORE_AX1_RTL_AHB_LITE_SV

// AHB-Lite interface for the Astrixcore AX-1 neural accelerator
// This provides a simple register-access interface for the RISC-V control processor

// AHB-Lite signal types
typedef struct packed {
    logic [31:0] addr;    // Address
    logic [31:0] write_data;  // Write data
    logic [3:0] write_response; // Write response (OKAY, ERROR, SLVERR, DECERR)
    logic write_valid;    // Write valid signal
    logic write_ready;    // Write ready signal
    logic [31:0] read_data;   // Read data
    logic read_valid;     // Read valid signal
    logic read_ready;     // Read ready signal
} ahb_lite_transaction_t;

// AHB-Lite transfer types
parameter logic [1:0] AHB_IDLE = 2'b00;
parameter logic [1:0] AHB_BUSY = 2'b01;
parameter logic [1:0] AHB_OKAY = 2'b10;
parameter logic [1:0] AHB_ERROR = 2'b11;

// Accelerator command encodings
parameter logic [3:0] ACCEL_CMD_IDLE = 4'b0000;
parameter logic [3:0] ACCEL_CMD_START = 4'b0001;
parameter logic [3:0] ACCEL_CMD_STOP = 4'b0010;
parameter logic [3:0] ACCEL_CMD_RESET = 4'b0011;

// Accelerator status encodings
parameter logic [3:0] ACCEL_STATUS_IDLE = 4'b0000;
parameter logic [3:0] ACCEL_STATUS_BUSY = 4'b0001;
parameter logic [3:0] ACCEL_STATUS_DONE = 4'b0010;
parameter logic [3:0] ACCEL_STATUS_ERROR = 4'b0011;

// MAC array parameterization
// Start with small array for verification; scale after correctness is established
parameter logic [3:0] MAC_ARRAY_DIM_1X1 = 4'b0001;
parameter logic [3:0] MAC_ARRAY_DIM_2X2 = 4'b0010;
parameter logic [3:0] MAC_ARRAY_DIM_4X4 = 4'b0100;
parameter logic [3:0] MAC_ARRAY_DIM_8X8 = 4'b1000;

// Datapath width parameters
parameter int WIDTH_INT8 = 8;
parameter int WIDTH_INT16 = 16;
parameter int WIDTH_INT32 = 32;
parameter int WIDTH_ACCUM = 32;  // Accumulator width

// Default configuration for Phase 1
// 2x2 MAC array is sufficient for basic verification
localparam int MAC_ARRAY_SIZE = 2;  // 2x2 = 4 processing elements

// INT8 weight and input width
localparam int DATA_WIDTH = 8;

// SRAM organization
// Target: 2 MiB total, divided into weight buffer, input buffer, output buffer
// For now, use a small parameterizable SRAM
localparam int SRAM_ADDR_WIDTH = 10;  // 1024 x 32 bits = 4 KiB for Phase 1
// Will expand to 21 bits (2 MiB) in later phases

// ---------------------------------------------------------------------------
// Parameterized MAC Datapath Module
// ---------------------------------------------------------------------------

// Single PE (Processing Element) that performs INT8 MAC
module pe_mac #
    (
        parameter PE_ID = 0  // Processing element ID for logging
    )
    (
        input  logic        clk,
        input  logic        rst_n,
        
        // Control
        input  logic        en,      // PE enable
        input  logic        start,   // Start MAC operation
        
        // Inputs (INT8)
        input  logic [7:0]  weight_i,  // INT8 weight value
        input  logic [7:0]  input_i,   // INT8 input activation
        
        // Accumulator output (saturated INT8)
        output logic [31:0] acc_out  // 32-bit saturated accumulator result
    );
    
    // Internal signals
    logic [15:0] product;     // INT14 product (signed)
    logic [31:0] acc;          // 32-bit accumulator
    logic [31:0] acc_sat;      // Saturated accumulator
    
    // Step 1: INT8 * INT8 -> INT14 (with sign extension)
    // INT8 range: [-128, 127]
    // Product range: [-16384, 16383] -> fits in INT14
    assign product = $signed(weight_i) * $signed(input_i);
    
    // Step 2: Accumulate
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 32'd0;
        end else if (en && start) begin
            acc <= acc + product;
        end else begin
            acc <= acc;
        end
    end
    
    // Step 3: Right-shift with rounding (quantization)
    // Default shift = 4 (common for INT8 networks, divides by 16)
    localparam int SHIFT = 4;
    
    // Arithmetic shift right with rounding
    always_comb begin
        // Add 2^(SHIFT-1) for rounding, then shift
        acc_sat = $past(acc);  // Get previous acc value
        // Round: (abs(acc) + 2^(SHIFT-1)) >> SHIFT
        if (acc_sat >= 0) begin
            acc_sat = (acc_sat + (1 << (SHIFT - 1))) >> SHIFT;
        end else begin
            acc_sat = -(((-acc_sat) + (1 << (SHIFT - 1))) >> SHIFT);
        end
    end
    
    // Step 4: Saturate to INT8 range [-128, 127]
    always_comb begin
        if (acc_sat > 127)
            acc_out = 32'd127;
        else if (acc_sat < -128)
            acc_out = 32'd-128;
        else
            acc_out = {{24{acc_sat[31]}}, acc_sat[31:0]};  // Sign-extend to 32-bit
    end
    
endmodule: pe_mac

// ---------------------------------------------------------------------------
// Accelerator Array Module (parameterized NxM PE array)
// ---------------------------------------------------------------------------

// Simple NxM MAC array for Phase 1 (starts with 2x2 = 4 PEs)
module accelerator_array #
    (
        parameter ARRAY_DIM = 2  // Array dimension: ARRAY_DIM x ARRAY_DIM PEs
    )
    (
        input  logic        clk,
        input  logic        rst_n,
        
        // Control from top level
        input  logic        acc_start,   // Start accelerator
        input  logic        acc_stop,    // Stop accelerator
        input  logic [3:0]  acc_dim,   // Active array dimension select
        
        // SRAM interface for weight/input storage
        input  logic [31:0] sram_read_data,  // Data read from SRAM
        output logic [31:0] sram_write_addr, // SRAM write address
        output logic [31:0] sram_write_data, // SRAM write data
        output logic        sram_write_en,   // SRAM write enable
        
        // Accumulator result (final MAC result)
        output logic [31:0] acc_result     // 32-bit saturated MAC result
    );
    
    // Local weight and input buffers (from SRAM)
    logic [7:0] weight_reg [0:ARRAY_DIM-1][0:ARRAY_DIM-1];  // Weight registers
    logic [7:0] input_reg  [0:ARRAY_DIM-1];                  // Input registers
    
    // PE outputs
    logic [31:0] pe_acc_out [0:ARRAY_DIM-1][0:ARRAY_DIM-1];  // PE accumulator results
    
    // Counter for array indexing
    integer i, j;
    
    // -------------------------------------------------------------------------
    // SRAM read/write logic
    // -------------------------------------------------------------------------
    
    // Read weights from SRAM (stationary weights)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize weights to zero
            for (i = 0; i < ARRAY_DIM; i = i + 1)
                for (j = 0; j < ARRAY_DIM; j = j + 1)
                    weight_reg[i][j] <= 8'd0;
        end else begin
            // Read weights based on SRAM address
            // Weight layout: row-major, ARRAY_DIM x ARRAY_DIM weights
            // For simplicity, read based on acc_dim selection
            case (acc_dim)
                MAC_ARRAY_DIM_1X1: begin
                    // 1x1: read single weight from SRAM address 0
                    sram_write_addr <= 32'd0;
                    // Weights loaded via AHB-Lite CSR writes to SRAM space
                end
                MAC_ARRAY_DIM_2X2: begin
                    // 2x2: read 4 weights
                    // In a full implementation, would use address decoding
                    sram_write_addr <= 32'd0;
                end
                default: begin
                    sram_write_addr <= 32'd0;
                end
            endcase
        end
    end
    
    // Read inputs from SRAM
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize inputs to zero
            for (i = 0; i < ARRAY_DIM; i = i + 1)
                input_reg[i] <= 8'd0;
        end else begin
            // Inputs loaded via AHB-Lite
            case (acc_dim)
                MAC_ARRAY_DIM_1X1: begin
                    // 1x1: single input
                    input_reg[0] <= sram_read_data[7:0];
                end
                MAC_ARRAY_DIM_2X2: begin
                    // 2x2: 4 inputs
                    // Simplified: first input from SRAM
                    input_reg[0] <= sram_read_data[7:0];
                end
                default: begin
                    // No-op
                end
            endcase
        end
    end
    
    // -------------------------------------------------------------------------
    // PE Array computation
    // -------------------------------------------------------------------------
    
    // Instantiate PEs
    genvar g, h;
    generate
        for (g = 0; g < ARRAY_DIM; g = g + 1) begin: pe_row
            for (h = 0; h < ARRAY_DIM; h = h + 1) begin: pe_col
                pe_mac #(
                    .PE_ID({g, h})
                ) u_pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .en((acc_start && (acc_dim == (MAC_ARRAY_DIM_1X1 << (g*2 + h))))),
                    .weight_i(weight_reg[g][h]),
                    .input_i(input_reg[h]),
                    .acc_out(pe_acc_out[g][h])
                );
            end
        end
    endgenerate
    
    // -------------------------------------------------------------------------
    // Accumulator: sum all PE results
    // -------------------------------------------------------------------------
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_result <= 32'd0;
        end else if (acc_start) begin
            // Sum all PE accumulator outputs
            logic [31:0] sum = 32'd0;
            for (g = 0; g < ARRAY_DIM; g = g + 1) begin
                for (h = 0; h < ARRAY_DIM; h = h + 1) begin
                    sum = sum + pe_acc_out[g][h];
                end
            end
            // Apply right-shift with rounding (shift=4)
            if (sum >= 0) begin
                acc_result <= (sum + 8'd8) >> 4;  // divide by 16 with rounding
            end else begin
                acc_result <= -(((-sum) + 8'd8) >> 4);
            end
        end else begin
            acc_result <= 32'd0;
        end
    end
    
    // -------------------------------------------------------------------------
    // SRAM interface assignments
    // -------------------------------------------------------------------------
    
    // SRAM write address/data (driven by AHB-Lite when writing to SRAM space)
    assign sram_write_addr = 32'd0;  // Would be driven by AHB addr bus
    assign sram_write_data = 32'd0;  // Would be driven by AHB hwdata bus
    assign sram_write_en = 1'b0;     // Would be driven by AHB hwrite signal
    
    // SRAM read data (driven by AHB-Lite read responses)
    // In a full implementation, would decode addr bus to read from specific SRAM location
    assign sram_read_data = 32'd0;  // Placeholder
    
endmodule: accelerator_array

// ---------------------------------------------------------------------------
// Top-level Accelerator Module
// ---------------------------------------------------------------------------

// Top-level module for the neural accelerator
module accelerator_top (
    // Clock and reset
    input  logic        clk,       // System clock
    input  logic        rst_n,     // Active-low asynchronous reset
    
    // AHB-Lite slave interface
    input  logic [31:0] haddr,     // Address bus
    input  logic [31:0] hwdata,    // Write data bus
    output logic [31:0] hrdata,    // Read data bus
    output logic        hresp,     // Transfer response (0=OKAY, 1=ERROR)
    input  hwrite,                 // Write transaction
    input  hreset,                 // AHB reset
    
    // Accelerator control
    output logic        acc_busy,  // Accelerator busy status
    output logic        acc_done,  // Accelerator done status
    output logic [3:0]  acc_status, // Accelerator status codes
    output logic [3:0]  acc_dim,   // Active array dimension
    output logic [31:0] acc_result // MAC result
    
    // TODO: Add DMA interface for SRAM access in later phases
    // TODO: Add external memory interface in later phases
);

// ---------------------------------------------------------------------------
// Internal signals
// ---------------------------------------------------------------------------

logic [3:0] cmd_reg;     // Command register
logic [3:0] status_reg;  // Status register
logic [3:0] dim_reg;     // Array dimension register
logic [31:0] data_reg;   // Data register
logic [31:0] acc_internal; // Internal accumulator
logic [31:0] sram [0:2**SRAM_ADDR_WIDTH-1]; // Simple inline SRAM

// accelerator_array instance
logic [31:0] array_acc_result;
accelerator_array u_array (
    .clk(clk),
    .rst_n(rst_n),
    .acc_start((cmd_reg == ACCEL_CMD_START)),
    .acc_stop((cmd_reg == ACCEL_CMD_STOP)),
    .acc_dim(dim_reg),
    .sram_read_data(32'd0),  // Would be connected to actual SRAM
    .sram_write_addr(),      // Would be connected to AHB haddr
    .sram_write_data(),      // Would be connected to AHB hwdata
    .sram_write_en(),        // Would be connected to AHB hwrite
    .acc_result(array_acc_result)
);

// ---------------------------------------------------------------------------
// Register File / CSR Logic
// ---------------------------------------------------------------------------

// Command register write
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cmd_reg <= ACCEL_CMD_IDLE;
        status_reg <= ACCEL_STATUS_IDLE;
        dim_reg <= MAC_ARRAY_DIM_1X1;
        data_reg <= 32'd0;
    end else begin
        if (hwrite && (haddr[3:2] == 2'b00)) begin // CSR space
            case (haddr[1:0])
                2'b00: cmd_reg <= hwdata[3:0];   // cmd_reg at offset 0x00
                2'b01: dim_reg <= hwdata[3:0];   // dim_reg at offset 0x04
                2'b10: data_reg <= hwdata;       // data_reg at offset 0x10
                2'b11: ;                           // reserved
            endcase
        end
    end
end

// Status register - read-only, updated internally
always_ff @(posedge clk) begin
    // Default: idle
    status_reg <= ACCEL_STATUS_IDLE;
    
    // Check command and update status
    case (cmd_reg)
        ACCEL_CMD_START: begin
            if (dim_reg == MAC_ARRAY_DIM_1X1) begin
                // 1x1 array: single MAC operation
                acc_internal <= array_acc_result;
                status_reg <= ACCEL_STATUS_DONE;
            end else if (dim_reg == MAC_ARRAY_DIM_2X2) begin
                // 2x2 array: 4 MAC operations
                acc_internal <= array_acc_result;
                status_reg <= ACCEL_STATUS_DONE;
            end else begin
                status_reg <= ACCEL_STATUS_ERROR;
            end
        end

        ACCEL_CMD_STOP: begin
            status_reg <= ACCEL_STATUS_IDLE;
        end

        ACCEL_CMD_RESET: begin
            acc_internal <= 32'd0;
            status_reg <= ACCEL_STATUS_IDLE;
        end

        default: begin
            // Keep current status
        end
    endcase
end

// Accelerator output assignments
assign acc_busy = (cmd_reg == ACCEL_CMD_START);
assign acc_done = (status_reg == ACCEL_STATUS_DONE);
assign acc_status = status_reg;
assign acc_dim = dim_reg;
assign acc_result = acc_internal;

// AHB-Lite read data mux
always_comb begin
    case (haddr[3:2])
        2'b00: hrdata = {30'd0, status_reg};     // status at 0x00
        2'b01: hrdata = {30'd0, dim_reg};        // dim at 0x04
        2'b10: hrdata = data_reg;                // data at 0x10
        2'b11: hrdata = 32'd0;                   // reserved
        default: hrdata = 32'd0;
    endcase
end

// AHB-Lite write response
assign hresp = 1'b0; // Always OKAY

// Simple SRAM initialization (all zeros initially)
generate
    for (genvar i = 0; i < 2**SRAM_ADDR_WIDTH; i = i + 1) begin
        initial begin
            sram[i] = 32'd0;
        end
    end
endgenerate

// Simple write port to SRAM (for weight/input loading)
// Only one port for Phase 1 (read-during-write behavior)
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Initialize to zero
    end else begin
        // Address-based write (simplified - full dual-port in later phases)
        if (hwrite && (haddr[3:2] == 2'b11)) begin // SRAM space
            if (haddr[1:0] == 2'b00) begin
                // Write to SRAM at address addr[SRAM_ADDR_WIDTH-1:0]
                sram[{haddr[8:2], addr[1:0]}] <= hwdata[31:0];
            end
        end
    end
end

// Read data from SRAM
always_comb begin
    // Simple read: return data from SRAM at addressed location
    if (hwrite == 1'b0 && (haddr[3:2] == 2'b11)) begin
        // Read from SRAM
        hrdata = sram[{haddr[8:2], addr[1:0]}];
    end
end

endmodule // accelerator_top

`endif // ASTRIXCORE_AX1_RTL_AHB_LITE_SV