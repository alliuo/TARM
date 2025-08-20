
module tree_9 ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n9, n10, n11, n12, n13, n14, n15, n16, n17, n18;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_AD1_1 U14 ( .A(n16), .B(pp_hl[1]), .CI(n14), .CO(n12), .S(result[3]) );
  VHSR_AD1_1 U15 ( .A(n13), .B(pp_hh[0]), .CI(n12), .CO(n11), .S(result[4]) );
  VHSR_AD1_1 U16 ( .A(pp_lh[3]), .B(pp_hh[1]), .CI(n11), .CO(n9), .S(result[5]) );
  VHSR_OR2_2 U17 ( .A1(pp_lh[2]), .A2(pp_hl[2]), .Z(n13) );
  VHSR_IN_2 U18 ( .I(pp_lh[0]), .ZN(n17) );
  VHSR_NOR2_1 U19 ( .A1(pp_ll[2]), .A2(pp_hl[0]), .ZN(n18) );
  VHSR_NOR2_1 U20 ( .A1(n17), .A2(n18), .ZN(n16) );
  VHSR_OR2_2 U21 ( .A1(pp_ll[3]), .A2(pp_lh[1]), .Z(n14) );
  VHSR_CLKNAND2_2 U22 ( .A1(n9), .A2(pp_hh[2]), .ZN(n15) );
  VHSR_OAI21_2 U23 ( .A1(n9), .A2(pp_hh[2]), .B(n15), .ZN(n10) );
  VHSR_IN_2 U24 ( .I(n10), .ZN(result[6]) );
  VHSR_INAND2_2 U25 ( .A1(pp_hh[3]), .B1(n15), .ZN(result[7]) );
  VHSR_AOI21_2 U26 ( .A1(n18), .A2(n17), .B(n16), .ZN(result[2]) );
endmodule

