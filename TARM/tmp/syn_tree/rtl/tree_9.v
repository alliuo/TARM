
module tree_9 (
    input  wire [3:0] pp_hh,
    input  wire [3:0] pp_hl,
    input  wire [3:0] pp_lh,
    input  wire [3:0] pp_ll,
    output wire [7:0] result
);

wire [1:0] tmp_sum3;
wire [1:0] tmp_sum2;
wire [1:0] tmp_sum1;
wire [1:0] tmp_sum0;

// Full Adder 3
assign tmp_sum3 = ({pp_hh[1], pp_hl[3], pp_lh[3]} == 3'b000) ? 2'b00 : (
                  ({pp_hh[1], pp_hl[3], pp_lh[3]} == 3'b001) ? 2'b01 : (
                  ({pp_hh[1], pp_hl[3], pp_lh[3]} == 3'b010) ? 2'b00 : (
                  ({pp_hh[1], pp_hl[3], pp_lh[3]} == 3'b011) ? 2'b01 : (
                  ({pp_hh[1], pp_hl[3], pp_lh[3]} == 3'b100) ? 2'b01 : (
                  ({pp_hh[1], pp_hl[3], pp_lh[3]} == 3'b101) ? 2'b10 : (
                  ({pp_hh[1], pp_hl[3], pp_lh[3]} == 3'b110) ? 2'b01 : (
                  ({pp_hh[1], pp_hl[3], pp_lh[3]} == 3'b111) ? 2'b10 : 2'b00)))))));

// Full Adder 2
assign tmp_sum2 = ({pp_hh[0], pp_hl[2], pp_lh[2]} == 3'b000) ? 2'b00 : (
                  ({pp_hh[0], pp_hl[2], pp_lh[2]} == 3'b001) ? 2'b01 : (
                  ({pp_hh[0], pp_hl[2], pp_lh[2]} == 3'b010) ? 2'b01 : (
                  ({pp_hh[0], pp_hl[2], pp_lh[2]} == 3'b011) ? 2'b01 : (
                  ({pp_hh[0], pp_hl[2], pp_lh[2]} == 3'b100) ? 2'b01 : (
                  ({pp_hh[0], pp_hl[2], pp_lh[2]} == 3'b101) ? 2'b10 : (
                  ({pp_hh[0], pp_hl[2], pp_lh[2]} == 3'b110) ? 2'b10 : (
                  ({pp_hh[0], pp_hl[2], pp_lh[2]} == 3'b111) ? 2'b10 : 2'b00)))))));

// Full Adder 1
assign tmp_sum1 = ({pp_hl[1], pp_lh[1], pp_ll[3]} == 3'b000) ? 2'b00 : (
                  ({pp_hl[1], pp_lh[1], pp_ll[3]} == 3'b001) ? 2'b01 : (
                  ({pp_hl[1], pp_lh[1], pp_ll[3]} == 3'b010) ? 2'b01 : (
                  ({pp_hl[1], pp_lh[1], pp_ll[3]} == 3'b011) ? 2'b01 : (
                  ({pp_hl[1], pp_lh[1], pp_ll[3]} == 3'b100) ? 2'b01 : (
                  ({pp_hl[1], pp_lh[1], pp_ll[3]} == 3'b101) ? 2'b10 : (
                  ({pp_hl[1], pp_lh[1], pp_ll[3]} == 3'b110) ? 2'b10 : (
                  ({pp_hl[1], pp_lh[1], pp_ll[3]} == 3'b111) ? 2'b10 : 2'b00)))))));

// Full Adder 0
assign tmp_sum0 = ({pp_hl[0], pp_lh[0], pp_ll[2]} == 3'b000) ? 2'b00 : (
                  ({pp_hl[0], pp_lh[0], pp_ll[2]} == 3'b001) ? 2'b01 : (
                  ({pp_hl[0], pp_lh[0], pp_ll[2]} == 3'b010) ? 2'b01 : (
                  ({pp_hl[0], pp_lh[0], pp_ll[2]} == 3'b011) ? 2'b10 : (
                  ({pp_hl[0], pp_lh[0], pp_ll[2]} == 3'b100) ? 2'b01 : (
                  ({pp_hl[0], pp_lh[0], pp_ll[2]} == 3'b101) ? 2'b01 : (
                  ({pp_hl[0], pp_lh[0], pp_ll[2]} == 3'b110) ? 2'b10 : (
                  ({pp_hl[0], pp_lh[0], pp_ll[2]} == 3'b111) ? 2'b10 : 2'b00)))))));           

wire carry0, carry1, carry2, carry3;

// RCA
assign result[2:0] = {tmp_sum0[0], pp_ll[1:0]};

assign {carry0, result[3]} = ({tmp_sum1[0], tmp_sum0[1]} == 2'b00) ? 2'b00 : (
                               ({tmp_sum1[0], tmp_sum0[1]} == 2'b01) ? 2'b01 : (
                               ({tmp_sum1[0], tmp_sum0[1]} == 2'b10) ? 2'b01 : (
                               ({tmp_sum1[0], tmp_sum0[1]} == 2'b11) ? 2'b10 : 2'b00)));  
                
assign {carry1, result[4]} = ({tmp_sum2[0], tmp_sum1[1], carry0} == 3'b000) ? 2'b00 : (
                               ({tmp_sum2[0], tmp_sum1[1], carry0} == 3'b001) ? 2'b01 : (
                               ({tmp_sum2[0], tmp_sum1[1], carry0} == 3'b010) ? 2'b01 : (
                               ({tmp_sum2[0], tmp_sum1[1], carry0} == 3'b011) ? 2'b01 : (
                               ({tmp_sum2[0], tmp_sum1[1], carry0} == 3'b100) ? 2'b01 : (
                               ({tmp_sum2[0], tmp_sum1[1], carry0} == 3'b101) ? 2'b10 : (
                               ({tmp_sum2[0], tmp_sum1[1], carry0} == 3'b110) ? 2'b10 : (
                               ({tmp_sum2[0], tmp_sum1[1], carry0} == 3'b111) ? 2'b10 : 2'b00)))))));

assign {carry2, result[5]} = ({tmp_sum3[0], tmp_sum2[1], carry1} == 3'b000) ? 2'b00 : (
                               ({tmp_sum3[0], tmp_sum2[1], carry1} == 3'b001) ? 2'b01 : (
                               ({tmp_sum3[0], tmp_sum2[1], carry1} == 3'b010) ? 2'b01 : (
                               ({tmp_sum3[0], tmp_sum2[1], carry1} == 3'b011) ? 2'b01 : (
                               ({tmp_sum3[0], tmp_sum2[1], carry1} == 3'b100) ? 2'b01 : (
                               ({tmp_sum3[0], tmp_sum2[1], carry1} == 3'b101) ? 2'b10 : (
                               ({tmp_sum3[0], tmp_sum2[1], carry1} == 3'b110) ? 2'b10 : (
                               ({tmp_sum3[0], tmp_sum2[1], carry1} == 3'b111) ? 2'b10 : 2'b00)))))));

assign {carry3, result[6]} = ({pp_hh[2], tmp_sum3[1], carry2} == 3'b000) ? 2'b00 : (
                               ({pp_hh[2], tmp_sum3[1], carry2} == 3'b001) ? 2'b01 : (
                               ({pp_hh[2], tmp_sum3[1], carry2} == 3'b010) ? 2'b01 : (
                               ({pp_hh[2], tmp_sum3[1], carry2} == 3'b011) ? 2'b01 : (
                               ({pp_hh[2], tmp_sum3[1], carry2} == 3'b100) ? 2'b01 : (
                               ({pp_hh[2], tmp_sum3[1], carry2} == 3'b101) ? 2'b10 : (
                               ({pp_hh[2], tmp_sum3[1], carry2} == 3'b110) ? 2'b10 : (
                               ({pp_hh[2], tmp_sum3[1], carry2} == 3'b111) ? 2'b10 : 2'b00)))))));

assign result[7] = pp_hh[3] | carry3;

endmodule
