
module mul2_acc ( a, b, out );
  input [1:0] a;
  input [1:0] b;
  output [3:0] out;
  wire   n5, n6, n7;

  VHSR_AND2_2 U11 ( .A1(a[0]), .A2(b[0]), .Z(out[0]) );
  VHSR_CLKNAND2_2 U12 ( .A1(b[1]), .A2(a[1]), .ZN(n5) );
  VHSR_NOR2_1 U13 ( .A1(out[0]), .A2(n5), .ZN(out[2]) );
  VHSR_AND3_2 U14 ( .A1(b[1]), .A2(a[1]), .A3(out[0]), .Z(out[3]) );
  VHSR_CLKNAND2_2 U15 ( .A1(b[0]), .A2(a[1]), .ZN(n7) );
  VHSR_CLKNAND2_2 U16 ( .A1(a[0]), .A2(b[1]), .ZN(n6) );
  VHSR_CLKXOR2_2 U17 ( .A1(n7), .A2(n6), .Z(out[1]) );
endmodule

