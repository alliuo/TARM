
module tree_5 ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34, n35, n36, n37,
         n38, n39, n40, n41, n42, n43, n44;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_INAND3_2 U30 ( .A1(n44), .B1(n28), .B2(n40), .ZN(n29) );
  VHSR_INAND2_2 U31 ( .A1(pp_hh[3]), .B1(n31), .ZN(result[7]) );
  VHSR_MAOI222_2 U32 ( .A(pp_ll[2]), .B(pp_hl[0]), .C(pp_lh[0]), .ZN(n38) );
  VHSR_OAI31_2 U33 ( .A1(pp_ll[2]), .A2(pp_hl[0]), .A3(pp_lh[0]), .B(n38), 
        .ZN(n24) );
  VHSR_IN_2 U34 ( .I(n24), .ZN(result[2]) );
  VHSR_IN_2 U35 ( .I(n25), .ZN(n36) );
  VHSR_IN_2 U36 ( .I(n26), .ZN(n39) );
  VHSR_NOR2_1 U37 ( .A1(n39), .A2(n38), .ZN(n37) );
  VHSR_AD1_1 U38 ( .A(pp_ll[3]), .B(pp_lh[1]), .CI(pp_hl[1]), .CO(n32), .S(n26) );
  VHSR_NOR2_1 U39 ( .A1(n37), .A2(n32), .ZN(n35) );
  VHSR_NOR2_1 U40 ( .A1(n36), .A2(n35), .ZN(n33) );
  VHSR_AD1_1 U41 ( .A(pp_lh[2]), .B(pp_hl[2]), .CI(pp_hh[0]), .CO(n27), .S(n25) );
  VHSR_NOR2_1 U42 ( .A1(n33), .A2(n27), .ZN(n43) );
  VHSR_OAI21_2 U43 ( .A1(pp_lh[3]), .A2(pp_hl[3]), .B(pp_hh[1]), .ZN(n28) );
  VHSR_OAI31_2 U44 ( .A1(pp_lh[3]), .A2(pp_hl[3]), .A3(pp_hh[1]), .B(n28), 
        .ZN(n41) );
  VHSR_NOR2_1 U45 ( .A1(n43), .A2(n41), .ZN(n44) );
  VHSR_CLKNAND2_2 U46 ( .A1(n33), .A2(n27), .ZN(n40) );
  VHSR_CLKNAND2_2 U47 ( .A1(pp_hh[2]), .A2(n29), .ZN(n31) );
  VHSR_OAI21_2 U48 ( .A1(pp_hh[2]), .A2(n29), .B(n31), .ZN(n30) );
  VHSR_IN_2 U49 ( .I(n30), .ZN(result[6]) );
  VHSR_CLKNAND2_2 U50 ( .A1(n37), .A2(n32), .ZN(n34) );
  VHSR_AOI22_2 U51 ( .A1(n36), .A2(n35), .B1(n34), .B2(n33), .ZN(result[4]) );
  VHSR_AOI21_2 U52 ( .A1(n39), .A2(n38), .B(n37), .ZN(result[3]) );
  VHSR_IN_2 U53 ( .I(n40), .ZN(n42) );
  VHSR_OAI32_2 U54 ( .A1(n44), .A2(n43), .A3(n42), .B1(n41), .B2(n44), .ZN(
        result[5]) );
endmodule

