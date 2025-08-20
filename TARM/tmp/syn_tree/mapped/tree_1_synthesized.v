
module tree_1 ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33,
         n34, n35, n36, n37, n38, n39;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_INAND2_2 U27 ( .A1(n29), .B1(n25), .ZN(n26) );
  VHSR_NOR2_1 U28 ( .A1(n38), .A2(n39), .ZN(n37) );
  VHSR_IN_2 U29 ( .I(n20), .ZN(result[2]) );
  VHSR_CLKN_1 U30 ( .I(n27), .ZN(result[6]) );
  VHSR_INAND2_1 U31 ( .A1(pp_hh[3]), .B1(n28), .ZN(result[7]) );
  VHSR_INOR2_1 U32 ( .A1(n24), .B1(n30), .ZN(n29) );
  VHSR_MAOI222_2 U33 ( .A(pp_ll[2]), .B(pp_hl[0]), .C(pp_lh[0]), .ZN(n39) );
  VHSR_OAI31_2 U34 ( .A1(pp_ll[2]), .A2(pp_hl[0]), .A3(pp_lh[0]), .B(n39), 
        .ZN(n20) );
  VHSR_OR2_2 U35 ( .A1(pp_hl[2]), .A2(pp_hh[0]), .Z(n21) );
  VHSR_AOI22_2 U36 ( .A1(pp_lh[2]), .A2(n21), .B1(pp_hl[2]), .B2(pp_hh[0]), 
        .ZN(n31) );
  VHSR_IN_2 U37 ( .I(n22), .ZN(n38) );
  VHSR_AD1_1 U38 ( .A(pp_ll[3]), .B(pp_lh[1]), .CI(pp_hl[1]), .CO(n36), .S(n22) );
  VHSR_NOR2_1 U39 ( .A1(n37), .A2(n36), .ZN(n34) );
  VHSR_OAI31_2 U40 ( .A1(pp_lh[2]), .A2(pp_hl[2]), .A3(pp_hh[0]), .B(n31), 
        .ZN(n33) );
  VHSR_OR2_2 U41 ( .A1(n34), .A2(n33), .Z(n32) );
  VHSR_CLKNAND2_2 U42 ( .A1(n31), .A2(n32), .ZN(n24) );
  VHSR_OR2_2 U43 ( .A1(pp_hl[3]), .A2(pp_hh[1]), .Z(n23) );
  VHSR_AOI22_2 U44 ( .A1(pp_lh[3]), .A2(n23), .B1(pp_hl[3]), .B2(pp_hh[1]), 
        .ZN(n25) );
  VHSR_OAI31_2 U45 ( .A1(pp_lh[3]), .A2(pp_hl[3]), .A3(pp_hh[1]), .B(n25), 
        .ZN(n30) );
  VHSR_CLKNAND2_2 U46 ( .A1(pp_hh[2]), .A2(n26), .ZN(n28) );
  VHSR_OAI21_2 U47 ( .A1(pp_hh[2]), .A2(n26), .B(n28), .ZN(n27) );
  VHSR_AOI31_2 U48 ( .A1(n31), .A2(n32), .A3(n30), .B(n29), .ZN(result[5]) );
  VHSR_IOA21_2 U49 ( .A1(n34), .A2(n33), .B(n32), .ZN(n35) );
  VHSR_IOA21_2 U50 ( .A1(n36), .A2(n37), .B(n35), .ZN(result[4]) );
  VHSR_AOI21_2 U51 ( .A1(n39), .A2(n38), .B(n37), .ZN(result[3]) );
endmodule

