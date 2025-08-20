
module tree_0 ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34,
         n35, n36, n37, n38, n39, n40, n41;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_INAND3_2 U27 ( .A1(n41), .B1(n25), .B2(n37), .ZN(n26) );
  VHSR_INAND2_2 U28 ( .A1(pp_hh[3]), .B1(n28), .ZN(result[7]) );
  VHSR_MAOI222_2 U29 ( .A(pp_ll[2]), .B(pp_hl[0]), .C(pp_lh[0]), .ZN(n35) );
  VHSR_OAI31_2 U30 ( .A1(pp_ll[2]), .A2(pp_hl[0]), .A3(pp_lh[0]), .B(n35), 
        .ZN(n21) );
  VHSR_IN_2 U31 ( .I(n21), .ZN(result[2]) );
  VHSR_IN_2 U32 ( .I(n22), .ZN(n33) );
  VHSR_IN_2 U33 ( .I(n23), .ZN(n36) );
  VHSR_NOR2_1 U34 ( .A1(n36), .A2(n35), .ZN(n34) );
  VHSR_AD1_1 U35 ( .A(pp_ll[3]), .B(pp_lh[1]), .CI(pp_hl[1]), .CO(n29), .S(n23) );
  VHSR_NOR2_1 U36 ( .A1(n34), .A2(n29), .ZN(n32) );
  VHSR_NOR2_1 U37 ( .A1(n33), .A2(n32), .ZN(n30) );
  VHSR_AD1_1 U38 ( .A(pp_lh[2]), .B(pp_hl[2]), .CI(pp_hh[0]), .CO(n24), .S(n22) );
  VHSR_NOR2_1 U39 ( .A1(n30), .A2(n24), .ZN(n40) );
  VHSR_MAOI222_2 U40 ( .A(pp_hl[3]), .B(pp_hh[1]), .C(pp_lh[3]), .ZN(n25) );
  VHSR_OAI31_2 U41 ( .A1(pp_hl[3]), .A2(pp_hh[1]), .A3(pp_lh[3]), .B(n25), 
        .ZN(n38) );
  VHSR_NOR2_1 U42 ( .A1(n40), .A2(n38), .ZN(n41) );
  VHSR_CLKNAND2_2 U43 ( .A1(n30), .A2(n24), .ZN(n37) );
  VHSR_CLKNAND2_2 U44 ( .A1(pp_hh[2]), .A2(n26), .ZN(n28) );
  VHSR_OAI21_2 U45 ( .A1(pp_hh[2]), .A2(n26), .B(n28), .ZN(n27) );
  VHSR_IN_2 U46 ( .I(n27), .ZN(result[6]) );
  VHSR_CLKNAND2_2 U47 ( .A1(n34), .A2(n29), .ZN(n31) );
  VHSR_AOI22_2 U48 ( .A1(n33), .A2(n32), .B1(n31), .B2(n30), .ZN(result[4]) );
  VHSR_AOI21_2 U49 ( .A1(n36), .A2(n35), .B(n34), .ZN(result[3]) );
  VHSR_IN_2 U50 ( .I(n37), .ZN(n39) );
  VHSR_OAI32_2 U51 ( .A1(n41), .A2(n40), .A3(n39), .B1(n38), .B2(n41), .ZN(
        result[5]) );
endmodule

