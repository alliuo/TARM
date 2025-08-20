
module mul8_15 (
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    output wire [15:0] product
);

wire [3:0] pp_hh_hh;
wire [3:0] pp_hh_hl;
wire [3:0] pp_hh_lh;
wire [3:0] pp_hh_ll;
wire [7:0] pp_hh;

wire [3:0] pp_hl_hh;
wire [3:0] pp_hl_hl;
wire [3:0] pp_hl_lh;
wire [3:0] pp_hl_ll;
wire [7:0] pp_hl;

wire [3:0] pp_lh_hh;
wire [3:0] pp_lh_hl;
wire [3:0] pp_lh_lh;
wire [3:0] pp_lh_ll;
wire [7:0] pp_lh;

wire [3:0] pp_ll_hh;
wire [3:0] pp_ll_hl;
wire [3:0] pp_ll_lh;
wire [3:0] pp_ll_ll;
wire [7:0] pp_ll;

mul2_acc mul_hh_hh (.a(a[7:6]), .b(b[7:6]), .out(pp_hh_hh));
mul2_acc mul_hh_hl (.a(a[7:6]), .b(b[5:4]), .out(pp_hh_hl));
mul2_acc mul_hh_lh (.a(a[5:4]), .b(b[7:6]), .out(pp_hh_lh));
mul2_acc mul_hh_ll (.a(a[5:4]), .b(b[5:4]), .out(pp_hh_ll));
tree_1 tree_hh (.pp_hh(pp_hh_hh), .pp_hl(pp_hh_hl), .pp_lh(pp_hh_lh), .pp_ll(pp_hh_ll), .result(pp_hh));

mul2_2 mul_hl_hh (.a(a[7:6]), .b(b[3:2]), .out(pp_hl_hh));
mul2_3 mul_hl_hl (.a(a[7:6]), .b(b[1:0]), .out(pp_hl_hl));
mul2_acc mul_hl_lh (.a(a[5:4]), .b(b[3:2]), .out(pp_hl_lh));
mul2_acc mul_hl_ll (.a(a[5:4]), .b(b[1:0]), .out(pp_hl_ll));
tree_2 tree_hl (.pp_hh(pp_hl_hh), .pp_hl(pp_hl_hl), .pp_lh(pp_hl_lh), .pp_ll(pp_hl_ll), .result(pp_hl));

mul2_2 mul_lh_hh (.a(a[3:2]), .b(b[7:6]), .out(pp_lh_hh));
mul2_acc mul_lh_hl (.a(a[3:2]), .b(b[5:4]), .out(pp_lh_hl));
mul2_2 mul_lh_lh (.a(a[1:0]), .b(b[7:6]), .out(pp_lh_lh));
mul2_acc mul_lh_ll (.a(a[1:0]), .b(b[5:4]), .out(pp_lh_ll));
tree_1 tree_lh (.pp_hh(pp_lh_hh), .pp_hl(pp_lh_hl), .pp_lh(pp_lh_lh), .pp_ll(pp_lh_ll), .result(pp_lh));

mul2_acc mul_ll_hh (.a(a[3:2]), .b(b[3:2]), .out(pp_ll_hh));
mul2_2 mul_ll_hl (.a(a[3:2]), .b(b[1:0]), .out(pp_ll_hl));
mul2_acc mul_ll_lh (.a(a[1:0]), .b(b[3:2]), .out(pp_ll_lh));
mul2_2 mul_ll_ll (.a(a[1:0]), .b(b[1:0]), .out(pp_ll_ll));
tree_10 tree_ll (.pp_hh(pp_ll_hh), .pp_hl(pp_ll_hl), .pp_lh(pp_ll_lh), .pp_ll(pp_ll_ll), .result(pp_ll));

final_tree_acc tree (.pp_hh(pp_hh), .pp_hl(pp_hl), .pp_lh(pp_lh), .pp_ll(pp_ll), .result(product));

endmodule
