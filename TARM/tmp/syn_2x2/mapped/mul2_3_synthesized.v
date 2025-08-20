
module mul2_3 ( a, b, out );
  input [1:0] a;
  input [1:0] b;
  output [3:0] out;
  wire   \*Logic0* ;
  assign out[3] = \*Logic0* ;

  VHSR_PULL0_0 U7 ( .Z(\*Logic0* ) );
  VHSR_AND2_2 U8 ( .A1(a[0]), .A2(b[0]), .Z(out[0]) );
  VHSR_AND2_2 U9 ( .A1(a[0]), .A2(b[1]), .Z(out[1]) );
  VHSR_AND2_2 U10 ( .A1(out[1]), .A2(a[1]), .Z(out[2]) );
endmodule

