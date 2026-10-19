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
// Accelerator Control Module
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
                // acc_internal = data_reg * sram[0]  (simplified)
                acc_internal <= data_reg * sram[0];
                status_reg <= ACCEL_STATUS_DONE;
            end else if (dim_reg == MAC_ARRAY_DIM_2X2) begin
                // 2x2 array: 4 MAC operations
                // For now, just accumulate first element
                acc_internal <= data_reg * sram[0];
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
                // For now, just acknowledge
            end
        end
    end
end

// Read data from SRAM
always_comb begin
    // Simple read: return data from SRAM at addressed location
    if (hwrite == 1'b0 && (haddr[3:2] == 2'b11)) begin
        // Read from SRAM
        // In a full implementation, would use addr bus bits
        hrdata = sram[0]; // Simplified: always return sram[0]
    end
end

endmodule

`endif // ASTRIXCORE_AX1_RTL_AHB_LITE_SV