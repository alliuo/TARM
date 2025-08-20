
module tree_2 ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34,
         n35, n36, n37;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_NOR2_1 U28 ( .A1(n37), .A2(n36), .ZN(n35) );
  VHSR_IN_2 U29 ( .I(n21), .ZN(result[2]) );
  VHSR_CLKN_1 U30 ( .I(n27), .ZN(result[6]) );
  VHSR_INAND2_1 U31 ( .A1(pp_hh[3]), .B1(n30), .ZN(result[7]) );
  VHSR_INAND2_1 U32 ( .A1(n31), .B1(n25), .ZN(n29) );
  VHSR_INOR2_1 U33 ( .A1(n24), .B1(n32), .ZN(n31) );
  VHSR_AD1_1 U34 ( .A(pp_hh[1]), .B(n29), .CI(n28), .CO(n26), .S(result[5]) );
  VHSR_MAOI222_2 U35 ( .A(pp_ll[2]), .B(pp_hl[0]), .C(pp_lh[0]), .ZN(n37) );
  VHSR_OAI31_2 U36 ( .A1(pp_ll[2]), .A2(pp_hl[0]), .A3(pp_lh[0]), .B(n37), 
        .ZN(n21) );
  VHSR_OR2_2 U37 ( .A1(pp_lh[1]), .A2(pp_hl[1]), .Z(n22) );
  VHSR_AOI22_2 U38 ( .A1(pp_ll[3]), .A2(n22), .B1(pp_lh[1]), .B2(pp_hl[1]), 
        .ZN(n34) );
  VHSR_OAI31_2 U39 ( .A1(pp_ll[3]), .A2(pp_lh[1]), .A3(pp_hl[1]), .B(n34), 
        .ZN(n36) );
  VHSR_IN_2 U40 ( .I(n35), .ZN(n33) );
  VHSR_CLKNAND2_2 U41 ( .A1(n34), .A2(n33), .ZN(n24) );
  VHSR_OR2_2 U42 ( .A1(pp_hl[2]), .A2(pp_hh[0]), .Z(n23) );
  VHSR_AOI22_2 U43 ( .A1(pp_lh[2]), .A2(n23), .B1(pp_hl[2]), .B2(pp_hh[0]), 
        .ZN(n25) );
  VHSR_OAI31_2 U44 ( .A1(pp_lh[2]), .A2(pp_hl[2]), .A3(pp_hh[0]), .B(n25), 
        .ZN(n32) );
  VHSR_OR2_2 U45 ( .A1(pp_lh[3]), .A2(pp_hl[3]), .Z(n28) );
  VHSR_CLKNAND2_2 U46 ( .A1(n26), .A2(pp_hh[2]), .ZN(n30) );
  VHSR_OAI21_2 U47 ( .A1(n26), .A2(pp_hh[2]), .B(n30), .ZN(n27) );
  VHSR_AOI31_2 U48 ( .A1(n34), .A2(n33), .A3(n32), .B(n31), .ZN(result[4]) );
  VHSR_AOI21_2 U49 ( .A1(n37), .A2(n36), .B(n35), .ZN(result[3]) );
endmodule

