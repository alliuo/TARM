
module tree_3 ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33,
         n34, n35, n36, n37, n38, n39, n40;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_NOR2_1 U28 ( .A1(n40), .A2(n39), .ZN(n38) );
  VHSR_IN_2 U29 ( .I(n20), .ZN(result[2]) );
  VHSR_CLKN_1 U30 ( .I(n28), .ZN(result[6]) );
  VHSR_INAND2_1 U31 ( .A1(pp_hh[3]), .B1(n29), .ZN(result[7]) );
  VHSR_INOR2_1 U32 ( .A1(n25), .B1(n31), .ZN(n30) );
  VHSR_CLKN_1 U33 ( .I(n34), .ZN(n32) );
  VHSR_INOR2_1 U34 ( .A1(n23), .B1(n35), .ZN(n34) );
  VHSR_MAOI222_2 U35 ( .A(pp_ll[2]), .B(pp_hl[0]), .C(pp_lh[0]), .ZN(n40) );
  VHSR_OAI31_2 U36 ( .A1(pp_ll[2]), .A2(pp_hl[0]), .A3(pp_lh[0]), .B(n40), 
        .ZN(n20) );
  VHSR_OR2_2 U37 ( .A1(pp_hl[2]), .A2(pp_hh[0]), .Z(n21) );
  VHSR_AOI22_2 U38 ( .A1(pp_lh[2]), .A2(n21), .B1(pp_hl[2]), .B2(pp_hh[0]), 
        .ZN(n33) );
  VHSR_OR2_2 U39 ( .A1(pp_lh[1]), .A2(pp_hl[1]), .Z(n22) );
  VHSR_AOI22_2 U40 ( .A1(pp_ll[3]), .A2(n22), .B1(pp_lh[1]), .B2(pp_hl[1]), 
        .ZN(n37) );
  VHSR_OAI31_2 U41 ( .A1(pp_ll[3]), .A2(pp_lh[1]), .A3(pp_hl[1]), .B(n37), 
        .ZN(n39) );
  VHSR_IN_2 U42 ( .I(n38), .ZN(n36) );
  VHSR_CLKNAND2_2 U43 ( .A1(n37), .A2(n36), .ZN(n23) );
  VHSR_OAI31_2 U44 ( .A1(pp_lh[2]), .A2(pp_hl[2]), .A3(pp_hh[0]), .B(n33), 
        .ZN(n35) );
  VHSR_CLKNAND2_2 U45 ( .A1(n33), .A2(n32), .ZN(n25) );
  VHSR_OR2_2 U46 ( .A1(pp_hl[3]), .A2(pp_hh[1]), .Z(n24) );
  VHSR_AOI22_2 U47 ( .A1(pp_lh[3]), .A2(n24), .B1(pp_hl[3]), .B2(pp_hh[1]), 
        .ZN(n26) );
  VHSR_OAI31_2 U48 ( .A1(pp_lh[3]), .A2(pp_hl[3]), .A3(pp_hh[1]), .B(n26), 
        .ZN(n31) );
  VHSR_INAND2_2 U49 ( .A1(n30), .B1(n26), .ZN(n27) );
  VHSR_CLKNAND2_2 U50 ( .A1(pp_hh[2]), .A2(n27), .ZN(n29) );
  VHSR_OAI21_2 U51 ( .A1(pp_hh[2]), .A2(n27), .B(n29), .ZN(n28) );
  VHSR_AOI31_2 U52 ( .A1(n33), .A2(n32), .A3(n31), .B(n30), .ZN(result[5]) );
  VHSR_AOI31_2 U53 ( .A1(n37), .A2(n36), .A3(n35), .B(n34), .ZN(result[4]) );
  VHSR_AOI21_2 U54 ( .A1(n40), .A2(n39), .B(n38), .ZN(result[3]) );
endmodule

