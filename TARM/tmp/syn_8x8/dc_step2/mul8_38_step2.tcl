set search_path         ". /home/lao/IH28HK12T_V1p0/IH28HK_VHS_V1p0_basic/IH28HK_VHS_RVT_V1p0/synopsys"
set target_library      "ih28hk_vhs_rvt_tt_0p90_25c_basic.db"
set link_library        "* $target_library"

read_verilog ../mapped/mul8_38_synthesized.v

set CLK_PERIOD  1
create_clock -period $CLK_PERIOD -name vclk
set_input_delay 0.0 -clock vclk [all_inputs]
set_output_delay 0.0 -clock vclk [all_outputs]

read_saif -input ../vcs/saif/mul8_38.saif -instance top_sim/top

report_saif > ../out/mul8_38.saif.rpts
report_area -hierarchy > ../out/mul8_38.area.rpts
report_power -analysis_effort high > ../out/mul8_38.power.rpts
report_timing -path full -delay max -max_paths 1 -nworst 1 -significant_digits 5 > ../out/mul8_38.timing.rpts

exit