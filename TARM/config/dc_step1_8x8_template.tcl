set search_path         ". /home/lao/IH28HK12T_V1p0/IH28HK_VHS_V1p0_basic/IH28HK_VHS_RVT_V1p0/synopsys"
set target_library      "ih28hk_vhs_rvt_tt_0p90_25c_basic.db"
set link_library        "* $target_library"

set top "{UNIT_NAME}"

analyze -format sverilog -vcs [glob -nocomplain -directory ../../syn_2x2/mapped/ *_synthesized.v]
analyze -format sverilog -vcs [glob -nocomplain -directory ../../syn_tree/mapped/ *_synthesized.v]
analyze -format sverilog -vcs [glob -nocomplain -directory ../../../config/adder_tree_8bit/ *.v]
analyze -format sverilog -vcs ../rtl/$top.v

elaborate $top
current_design $top
check_design

link

set CLK_PERIOD  1
create_clock -period $CLK_PERIOD -name vclk
set_input_delay 0.0 -clock vclk [all_inputs]
set_output_delay 0.0 -clock vclk [all_outputs]

set_max_area 0 -ignore_tns

compile_ultra

write -f verilog -output ../mapped/{UNIT_NAME}_synthesized.v

exit
