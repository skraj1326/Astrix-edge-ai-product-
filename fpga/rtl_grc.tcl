# FPGA Synthesis Script for Astrixcore AX-1 Accelerator
# Target: Xilinx Artix-7 (or Intel Cyclone IV via nextpnr)
# Uses: Yosys (synthesis), nextpnr (place-and-route)

set -e

# Project configuration
set::PROJECT_NAME "astrixcore_ax1"
set::TOP_MODULE "accelerator_top"
set::FAMILY "artix7"  # Change to "cycloneiv" for Intel
set::PART "xc7a35t"   # Artix-7 XT35T, or "10m50cs" for Intel
set::TOP_LEVEL_CLOCK "clk"  # 100MHz on-board oscillator

# Build directory
set::BUILD_DIR "./build"
mkdir -p [set::BUILD_DIR]

# ============================================================================
# Step 1: Yosys Synthesis
# ============================================================================
echo "=== Step 1: Yosys Synthesis ==="

cat > [set::BUILD_DIR]/syn_rtl.tcl << YOSYS_EOF
# Read Verilog source
verilog work [list \
    ${TOP_MODULE}.sv \
    ahb_lite.sv \
    pe_mac.sv \
    sram_dual_port.sv \
    accelerator_array.sv \
]

# Set target technology library
load_mem_init

# Create library
link -top [set::TOP_MODULE]

# Generate synthesis attributes for INT8 saturation
# (Yosys will infer appropriate logic for saturation)

# Prepare for power optimization
set_cells -trim

# Write synthesized output
write_liberty [set::BUILD_DIR]/${TOP_MODULE}.lib
write_verilog [set::BUILD_DIR]/${TOP_MODULE}_syn.v
puts "Yosys synthesis complete"
YOSYS_EOF

yosys [set::BUILD_DIR]/syn_rtl.tcl

# ============================================================================
# Step 2: nextpnr Place-and-Route
# ============================================================================
echo "=== Step 2: nextpnr Place-and-Route ==="

cat > [set::BUILD_DIR]/place_routing.tcl << NEXTPNR_EOF
# Read synthesized Verilog
read_verilog [set::BUILD_DIR]/${TOP_MODULE}_syn.v

# Create JSON format for nextpnr
json_write [set::BUILD_DIR]/design.json

# Set up nextpnr flow
if {[string equal $::FAMILY "artix7"]} {
    nextpnr_evn \
        --device xc7a35t \
        --package csg324 \
        --speed -1 \
        --json [set::BUILD_DIR]/design.json \
        --pcf pcf/ax1_pins.pcf \
        --refclk 100000000 \
        --top [set::TOP_MODULE] \
        --write [set::BUILD_DIR]/${TOP_MODULE}.bit
} elseif {[string equal $::FAMILY "cycloneiv"]} {
    nextpnr_soft \
        --device 10m50cs \
        --package csg256 \
        --speed -1 \
        --json [set::BUILD_DIR]/design.json \
        --pcf pcf/ax1_pins.pcf \
        --refclk 100000000 \
        --top [set::TOP_MODULE] \
        --write [set::BUILD_DIR]/${TOP_MODULE}.bit
} else {
    puts "Unsupported family: $::FAMILY"
    exit 1
}

puts "Place-and-route complete"
NEXTPNR_EOF

nextpnr --version 2>/dev/null || echo "nextpnr not found, skipping P&R"
# OR use: nextpnr_evn --device xc7a35t --json design.json --pcf pcf/ax1_pins.pcf --top accelerator_top --write ax1.bit

# ============================================================================
# Step 3: Generate Pin Constraint File (PCF)
# ============================================================================
echo "=== Step 3: Generate Pin Constraints ==="

cat > pcf/ax1_pins.pcf << PCF_EOF
# Clock constraint
set_io_clock [list clk] 100000000

# AHB-Lite interface signals
set_io axi_awaddr [list haddr] {get_ports haddr}
set_io axi_awdata [list hwdata] {get_port hwdata}
set_io axi_awvalid [list hwrite] {get_port hwrite}
set_io axi_awready [resp] {get_port hresp}
set_io axi_wdata [list hwdata] {get_port hwdata}
set_io axi_wvalid [list hwrite] {get_port hwrite}
set_io axi_wready [resp] {get_port hresp}
set_io ardata [list hrdata] {get_port hrdata}
set_io arvalid [list 1'b0] {get_port 1'b0}
set_io arready [resp] {get_port hresp}

# Accelerator control/status
set_io acc_busy [list acc_busy] {get_port acc_busy}
set_io acc_done [list acc_done] {get_port acc_done}
set_io acc_result [list acc_result] {get_port acc_result}

# SRAM interface (dual-port)
set_io sram_addr_a [list sram_addr_a] {get_port sram_addr_a}
set_io sram_d_a [list sram_q_a] {get_port sram_q_a}
set_io sram_we_b [list sram_write_en] {get_port sram_write_en}
set_io sram_d_b [list sram_write_data] {get_port sram_write_data}

# Reset
set_io resetn [list rst_n] {get_port rst_n}

puts "Pin constraints written"
PCF_EOF

# ============================================================================
# Step 4: Report Resource Usage
# ============================================================================
echo "=== Step 4: Report Resource Usage ==="

if [file exists [set::BUILD_DIR]/${TOP_MODULE}.bit] then {
    echo "Bitstream generated: [set::BUILD_DIR]/${TOP_MODULE}.bit"
    echo "Report inferred resources:"
    # Yosys can report BRAM/FF/LUT usage
    report_bram [set::BUILD_DIR]/design.json 2>/dev/null || echo "  BRAM report: skipping (run with yosys)"
    report_lut [set::BUILD_DIR]/design.json 2>/dev/null || echo "  LUT report: skipping"
    report_fF [set::BUILD_DIR]/design.json 2>/dev/null || echo "  FF report: skipping"
} else {
    echo "Bitstream NOT generated — place-and-route may have failed or not run"
    echo "To generate: run 'nextpnr_evn --device xc7a35t ...' "
}

echo ""
echo "=== FPGA Synthesis Script Complete ==="
echo "Build directory: [set::BUILD_DIR]"
echo "Next steps:"
echo "  1. Review synthesis logs for warnings"
echo "  2. Verify timing with: report_timing [set::BUILD_DIR]/design.json"
echo "  3. Program FPGA with: [set::BUILD_DIR]/${TOP_MODULE}.bit"
echo "  4. Compare inferred resources with product targets"
EOF