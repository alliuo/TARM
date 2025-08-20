
module tree_7 ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_AD1_1 U15 ( .A(n18), .B(pp_hl[1]), .CI(n16), .CO(n14), .S(result[3]) );
  VHSR_AD1_1 U16 ( .A(n15), .B(pp_hh[0]), .CI(n14), .CO(n12), .S(result[4]) );
  VHSR_AD1_1 U17 ( .A(n13), .B(pp_hh[1]), .CI(n12), .CO(n10), .S(result[5]) );
  VHSR_OR2_2 U18 ( .A1(pp_lh[3]), .A2(pp_hl[3]), .Z(n13) );
  VHSR_OR2_2 U19 ( .A1(pp_lh[2]), .A2(pp_hl[2]), .Z(n15) );
  VHSR_IN_2 U20 ( .I(pp_lh[0]), .ZN(n19) );
  VHSR_NOR2_1 U21 ( .A1(pp_ll[2]), .A2(pp_hl[0]), .ZN(n20) );
  VHSR_NOR2_1 U22 ( .A1(n19), .A2(n20), .ZN(n18) );
  VHSR_OR2_2 U23 ( .A1(pp_ll[3]), .A2(pp_lh[1]), .Z(n16) );
  VHSR_CLKNAND2_2 U24 ( .A1(n10), .A2(pp_hh[2]), .ZN(n17) );
  VHSR_OAI21_2 U25 ( .A1(n10), .A2(pp_hh[2]), .B(n17), .ZN(n11) );
  VHSR_IN_2 U26 ( .I(n11), .ZN(result[6]) );
  VHSR_INAND2_2 U27 ( .A1(pp_hh[3]), .B1(n17), .ZN(result[7]) );
  VHSR_AOI21_2 U28 ( .A1(n20), .A2(n19), .B(n18), .ZN(result[2]) );
endmodule

