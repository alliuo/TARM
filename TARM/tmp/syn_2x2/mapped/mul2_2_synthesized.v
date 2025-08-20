
module mul2_2 ( a, b, out );
  input [1:0] a;
  input [1:0] b;
  output [3:0] out;
  wire   \*Logic0* , n3;
  assign out[3] = \*Logic0* ;

  VHSR_PULL0_0 U8 ( .Z(\*Logic0* ) );
  VHSR_AND2_2 U9 ( .A1(b[1]), .A2(a[1]), .Z(out[2]) );
  VHSR_AND2_2 U10 ( .A1(a[0]), .A2(b[0]), .Z(out[0]) );
  VHSR_AOI22_2 U11 ( .A1(a[0]), .A2(b[1]), .B1(b[0]), .B2(a[1]), .ZN(n3) );
  VHSR_IN_2 U12 ( .I(n3), .ZN(out[1]) );
endmodule

