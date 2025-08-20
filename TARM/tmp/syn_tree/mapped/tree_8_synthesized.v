
module tree_8 ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n17, n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_IOA21_2 U23 ( .A1(n25), .A2(n26), .B(n24), .ZN(n20) );
  VHSR_INAND2_2 U24 ( .A1(pp_hh[3]), .B1(n22), .ZN(result[7]) );
  VHSR_AD1_1 U25 ( .A(pp_hl[1]), .B(n27), .CI(n21), .CO(n25), .S(result[3]) );
  VHSR_AD1_1 U26 ( .A(pp_hh[1]), .B(n20), .CI(n19), .CO(n17), .S(result[5]) );
  VHSR_IN_2 U27 ( .I(pp_lh[0]), .ZN(n28) );
  VHSR_NOR2_1 U28 ( .A1(pp_ll[2]), .A2(pp_hl[0]), .ZN(n29) );
  VHSR_NOR2_1 U29 ( .A1(n28), .A2(n29), .ZN(n27) );
  VHSR_OR2_2 U30 ( .A1(pp_ll[3]), .A2(pp_lh[1]), .Z(n21) );
  VHSR_OR3_2 U31 ( .A1(pp_lh[2]), .A2(pp_hh[0]), .A3(pp_hl[2]), .Z(n26) );
  VHSR_MAOI222_2 U32 ( .A(pp_lh[2]), .B(pp_hh[0]), .C(pp_hl[2]), .ZN(n24) );
  VHSR_OR2_2 U33 ( .A1(pp_lh[3]), .A2(pp_hl[3]), .Z(n19) );
  VHSR_CLKNAND2_2 U34 ( .A1(n17), .A2(pp_hh[2]), .ZN(n22) );
  VHSR_OAI21_2 U35 ( .A1(n17), .A2(pp_hh[2]), .B(n22), .ZN(n18) );
  VHSR_IN_2 U36 ( .I(n18), .ZN(result[6]) );
  VHSR_AOI21_2 U37 ( .A1(n26), .A2(n24), .B(n25), .ZN(n23) );
  VHSR_AOI31_2 U38 ( .A1(n26), .A2(n25), .A3(n24), .B(n23), .ZN(result[4]) );
  VHSR_AOI21_2 U39 ( .A1(n29), .A2(n28), .B(n27), .ZN(result[2]) );
endmodule

