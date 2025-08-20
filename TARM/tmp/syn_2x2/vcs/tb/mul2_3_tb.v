
`timescale 1ns / 1ps

module top_sim();
    wire [3:0] out;
    reg  [1:0] a;
    reg  [1:0] b;

    mul2_3 top (.a(a), .b(b), .out(out));

    initial
    begin
        `ifdef DUMP_VPD
                $vcdpluson();
        `endif
        a = {$random}%4;
        b = {$random}%4;
        #1000000
        `ifdef DUMP_VPD
                $vcdplusoff();
        `endif
        $finish;
    end

    always
    begin
        forever #1   begin b = {$random}%4;end
    end
    always
    begin
        forever #1   begin a = {$random}%4;end
    end

endmodule
