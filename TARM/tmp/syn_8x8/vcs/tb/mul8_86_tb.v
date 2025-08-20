
`timescale 1ns / 1ps

module top_sim();
    wire [15:0] product;
    reg  [7:0]  a;
    reg  [7:0]  b;

    mul8_86 top (.a(a), .b(b), .product(product));

    initial
    begin
        `ifdef DUMP_VPD
                $vcdpluson();
        `endif
        a = {$random}%256;
        b = {$random}%256;
        #1000000
        `ifdef DUMP_VPD
                $vcdplusoff();
        `endif
        $finish;
    end

    always
    begin
        forever #1   begin b = {$random}%256;end
    end
    always
    begin
        forever #1   begin a = {$random}%256;end
    end

endmodule
