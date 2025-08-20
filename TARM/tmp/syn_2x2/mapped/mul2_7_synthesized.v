
module mul2_7 ( a, b, out );
  input [1:0] a;
  input [1:0] b;
  output [3:0] out;
  wire   \*Logic0* ;
  assign out[0] = \*Logic0* ;
  assign out[3] = \*Logic0* ;

  VHSR_PULL0_0 U6 ( .Z(\*Logic0* ) );
  VHSR_AND2_2 U7 ( .A1(b[1]), .A2(a[1]), .Z(out[2]) );
  VHSR_AND2_2 U8 ( .A1(b[1]), .A2(a[0]), .Z(out[1]) );
endmodule

