
`timescale 1ns / 1ps

module top_sim();
    wire [7:0] result;
    reg  [3:0] pp_hh;
    reg  [3:0] pp_hl;
    reg  [3:0] pp_lh;
    reg  [3:0] pp_ll;

    tree_0 top (.pp_hh(pp_hh), .pp_hl(pp_hl), .pp_lh(pp_lh), .pp_ll(pp_ll), .result(result));

    initial
    begin
        `ifdef DUMP_VPD
                $vcdpluson();
        `endif
        pp_hh = {$random}%16;
        pp_hl = {$random}%16;
        pp_lh = {$random}%16;
        pp_ll = {$random}%16;
        #1000000
        `ifdef DUMP_VPD
                $vcdplusoff();
        `endif
        $finish;
    end

    always
    begin
        forever #1   begin pp_hh = {$random}%16;end
    end
    always
    begin
        forever #1   begin pp_hl = {$random}%16;end
    end
    always
    begin
        forever #1   begin pp_lh = {$random}%16;end
    end
    always
    begin
        forever #1   begin pp_ll = {$random}%16;end
    end

endmodule
