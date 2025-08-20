
module tree_4 ( pp_hh, pp_hl, pp_lh, pp_ll, result );
  input [3:0] pp_hh;
  input [3:0] pp_hl;
  input [3:0] pp_lh;
  input [3:0] pp_ll;
  output [7:0] result;
  wire   n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34, n35,
         n36, n37, n38, n39;
  assign result[1] = pp_ll[1];
  assign result[0] = pp_ll[0];

  VHSR_IOA21_2 U28 ( .A1(pp_lh[0]), .A2(pp_hl[0]), .B(n22), .ZN(n38) );
  VHSR_INAND2_2 U29 ( .A1(pp_hh[3]), .B1(n30), .ZN(result[7]) );
  VHSR_AD1_1 U30 ( .A(pp_hh[1]), .B(n29), .CI(n28), .CO(n26), .S(result[5]) );
  VHSR_OAI21_2 U31 ( .A1(pp_lh[0]), .A2(pp_hl[0]), .B(pp_ll[2]), .ZN(n22) );
  VHSR_CLKNAND2_2 U32 ( .A1(n37), .A2(n38), .ZN(n35) );
  VHSR_IN_2 U33 ( .I(n35), .ZN(n36) );
  VHSR_AD1_1 U34 ( .A(pp_ll[3]), .B(pp_lh[1]), .CI(pp_hl[1]), .CO(n32), .S(n37) );
  VHSR_NOR2_1 U35 ( .A1(n36), .A2(n32), .ZN(n25) );
  VHSR_OR2_2 U36 ( .A1(pp_hl[2]), .A2(pp_hh[0]), .Z(n23) );
  VHSR_AOI22_2 U37 ( .A1(pp_lh[2]), .A2(n23), .B1(pp_hl[2]), .B2(pp_hh[0]), 
        .ZN(n24) );
  VHSR_OAI31_2 U38 ( .A1(pp_lh[2]), .A2(pp_hl[2]), .A3(pp_hh[0]), .B(n24), 
        .ZN(n31) );
  VHSR_OAI21_2 U39 ( .A1(n25), .A2(n31), .B(n24), .ZN(n29) );
  VHSR_OR2_2 U40 ( .A1(pp_lh[3]), .A2(pp_hl[3]), .Z(n28) );
  VHSR_CLKNAND2_2 U41 ( .A1(n26), .A2(pp_hh[2]), .ZN(n30) );
  VHSR_OAI21_2 U42 ( .A1(n26), .A2(pp_hh[2]), .B(n30), .ZN(n27) );
  VHSR_IN_2 U43 ( .I(n27), .ZN(result[6]) );
  VHSR_NOR2_1 U44 ( .A1(n32), .A2(n31), .ZN(n34) );
  VHSR_AOI22_2 U45 ( .A1(n32), .A2(n31), .B1(n35), .B2(n34), .ZN(n33) );
  VHSR_OAI21_2 U46 ( .A1(n35), .A2(n34), .B(n33), .ZN(result[4]) );
  VHSR_IAO21_2 U47 ( .A1(n37), .A2(n38), .B(n36), .ZN(result[3]) );
  VHSR_OR2_2 U48 ( .A1(pp_ll[2]), .A2(pp_hl[0]), .Z(n39) );
  VHSR_IAO21_2 U49 ( .A1(pp_lh[0]), .A2(n39), .B(n38), .ZN(result[2]) );
endmodule

