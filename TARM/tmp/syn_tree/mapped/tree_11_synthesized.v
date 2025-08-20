
module tree_11 ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32,
         n33, n34, n35, n36, n37, n38;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_INAND2_2 U26 ( .A1(n29), .B1(n25), .ZN(n26) );
  VHSR_NOR2_1 U27 ( .A1(n35), .A2(n34), .ZN(n33) );
  VHSR_IN_2 U28 ( .I(n19), .ZN(result[2]) );
  VHSR_CLKN_1 U29 ( .I(n27), .ZN(result[6]) );
  VHSR_INAND2_1 U30 ( .A1(pp_hh[3]), .B1(n28), .ZN(result[7]) );
  VHSR_INOR2_1 U31 ( .A1(n24), .B1(n30), .ZN(n29) );
  VHSR_MAOI222_2 U32 ( .A(pp_ll[2]), .B(pp_hl[0]), .C(pp_lh[0]), .ZN(n38) );
  VHSR_OAI31_2 U33 ( .A1(pp_ll[2]), .A2(pp_hl[0]), .A3(pp_lh[0]), .B(n38), 
        .ZN(n19) );
  VHSR_OR2_2 U34 ( .A1(pp_hl[2]), .A2(pp_hh[0]), .Z(n20) );
  VHSR_AOI22_2 U35 ( .A1(pp_lh[2]), .A2(n20), .B1(pp_hl[2]), .B2(pp_hh[0]), 
        .ZN(n32) );
  VHSR_IN_2 U36 ( .I(n21), .ZN(n37) );
  VHSR_NOR2_1 U37 ( .A1(n37), .A2(n38), .ZN(n36) );
  VHSR_AD1_1 U38 ( .A(pp_ll[3]), .B(pp_lh[1]), .CI(pp_hl[1]), .CO(n22), .S(n21) );
  VHSR_NOR2_1 U39 ( .A1(n36), .A2(n22), .ZN(n35) );
  VHSR_OAI31_2 U40 ( .A1(pp_lh[2]), .A2(pp_hl[2]), .A3(pp_hh[0]), .B(n32), 
        .ZN(n34) );
  VHSR_IN_2 U41 ( .I(n33), .ZN(n31) );
  VHSR_CLKNAND2_2 U42 ( .A1(n32), .A2(n31), .ZN(n24) );
  VHSR_OR2_2 U43 ( .A1(pp_hl[3]), .A2(pp_hh[1]), .Z(n23) );
  VHSR_AOI22_2 U44 ( .A1(pp_lh[3]), .A2(n23), .B1(pp_hl[3]), .B2(pp_hh[1]), 
        .ZN(n25) );
  VHSR_OAI31_2 U45 ( .A1(pp_lh[3]), .A2(pp_hl[3]), .A3(pp_hh[1]), .B(n25), 
        .ZN(n30) );
  VHSR_CLKNAND2_2 U46 ( .A1(pp_hh[2]), .A2(n26), .ZN(n28) );
  VHSR_OAI21_2 U47 ( .A1(pp_hh[2]), .A2(n26), .B(n28), .ZN(n27) );
  VHSR_AOI31_2 U48 ( .A1(n32), .A2(n31), .A3(n30), .B(n29), .ZN(result[5]) );
  VHSR_AOI21_2 U49 ( .A1(n35), .A2(n34), .B(n33), .ZN(result[4]) );
  VHSR_AOI21_2 U50 ( .A1(n38), .A2(n37), .B(n36), .ZN(result[3]) );
endmodule

