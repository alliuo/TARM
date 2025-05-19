module final_tree_acc(
    input  wire [7:0]  pp_hh,
    input  wire [7:0]  pp_hl,
    input  wire [7:0]  pp_lh,
    input  wire [7:0]  pp_ll,
    output wire [15:0] result
);

    wire [8:0] tmp_sum;
    rca_8bit ADD_0 (.a(pp_hl), .b(pp_lh), .cout(tmp_sum[8]), .sum(tmp_sum[7:0]));

    wire       cla_cin;
    wire [2:0] carry;

    assign result[3:0] = pp_ll[3:0];
    half_adder RCA_HA0 (.a(tmp_sum[0]), .b(pp_ll[4]), .cout(cla_cin), .sum(result[4]));
    cla_8bit ADD_1 (.a(tmp_sum[8:1]), .b({{pp_hh[4:0]}, {pp_ll[7:5]}}), .cin(cla_cin), .cout(carry[0]), .sum(result[12:5]));
    half_adder RCA_HA1 (.a(pp_hh[5]), .b(carry[0]), .cout(carry[1]), .sum(result[13]));
    half_adder RCA_HA2 (.a(pp_hh[6]), .b(carry[1]), .cout(carry[2]), .sum(result[14]));
    assign result[15] = pp_hh[7] | carry[2];

endmodule