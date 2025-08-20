
module tree_acc ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_AD1_1 U13 ( .A(pp_ll[2]), .B(pp_lh[0]), .CI(pp_hl[0]), .CO(n22), .S(
        result[2]) );
  VHSR_AD1_1 U14 ( .A(n16), .B(n20), .CI(n15), .CO(n17), .S(result[4]) );
  VHSR_AD1_1 U15 ( .A(n14), .B(pp_hh[2]), .CI(n13), .CO(n12), .S(result[6]) );
  VHSR_AD1_1 U16 ( .A(pp_lh[3]), .B(pp_hl[3]), .CI(pp_hh[1]), .CO(n14), .S(n18) );
  VHSR_AD1_1 U17 ( .A(pp_lh[2]), .B(pp_hl[2]), .CI(pp_hh[0]), .CO(n19), .S(n16) );
  VHSR_AND2_2 U18 ( .A1(n22), .A2(n21), .Z(n20) );
  VHSR_AD1_1 U19 ( .A(pp_lh[1]), .B(pp_ll[3]), .CI(pp_hl[1]), .CO(n15), .S(n21) );
  VHSR_OR2_2 U20 ( .A1(pp_hh[3]), .A2(n12), .Z(result[7]) );
  VHSR_AD1_1 U21 ( .A(n19), .B(n18), .CI(n17), .CO(n13), .S(result[5]) );
  VHSR_IAO21_2 U22 ( .A1(n22), .A2(n21), .B(n20), .ZN(result[3]) );
endmodule

