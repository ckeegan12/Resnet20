# ============================================================================
# Vitis HLS TCL Script for AdderNet20 on Kria KV260
# ============================================================================
# Usage: vitis_hls -f run_hls.tcl
# This script performs C Simulation, Synthesis, and optionally Co-Simulation

# Project settings
set PROJECT_NAME "addernet20_hls"
set TOP_FUNCTION "ADDERNET20_2_0"
set SOLUTION_NAME "solution1"

# Kria KV260 part number (Zynq UltraScale+ MPSoC)
set PART_NAME "xck26-sfvc784-2LV-c"

# Clock period (100 MHz = 10ns)
set CLOCK_PERIOD 10

# ============================================================================
# Create project
# ============================================================================
open_project -reset $PROJECT_NAME

# Add source files
add_files Addernet20.cpp
add_files Addernet20.h
add_files parameters.h

# Add testbench for C Simulation
add_files -tb addernet20_tb.cpp

# Set top function
set_top $TOP_FUNCTION

# ============================================================================
# Create solution
# ============================================================================
open_solution -reset $SOLUTION_NAME

# Set target device
set_part $PART_NAME

# Set clock constraint
create_clock -period $CLOCK_PERIOD -name default

# Set clock uncertainty (10% of period)
set_clock_uncertainty 1

# ============================================================================
# C Simulation
# ============================================================================
puts "=== Running C Simulation ==="
csim_design

# ============================================================================
# Synthesis
# ============================================================================
puts "=== Running HLS Synthesis ==="
csynth_design

# ============================================================================
# Co-Simulation (Optional - commented out as it can take a long time)
# ============================================================================
# puts "=== Running Co-Simulation ==="
# cosim_design -rtl verilog

# ============================================================================
# Export RTL (IP for Vivado)
# ============================================================================
puts "=== Exporting IP ==="
export_design -format ip_catalog -description "AdderNet20 Int5 Accelerator" -vendor "custom" -library "hls" -version "2.0" -display_name "AdderNet20_Accelerator"

puts "=== HLS Flow Complete ==="
puts "Check reports in: $PROJECT_NAME/$SOLUTION_NAME/syn/report/"
exit
