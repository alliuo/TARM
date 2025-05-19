module rca_8bit (
    input  wire [7:0] a,
    input  wire [7:0] b,
    output wire       cout,
    output wire [7:0] sum 
);
    wire [6:0] rca_c;

    half_adder     HA0 (.a(a[0]), .b(b[0]),                 .cout(rca_c[0]), .sum(sum[0]));
    full_adder_acc FA0 (.a(a[1]), .b(b[1]), .cin(rca_c[0]), .cout(rca_c[1]), .sum(sum[1]));
    full_adder_acc FA1 (.a(a[2]), .b(b[2]), .cin(rca_c[1]), .cout(rca_c[2]), .sum(sum[2]));
    full_adder_acc FA2 (.a(a[3]), .b(b[3]), .cin(rca_c[2]), .cout(rca_c[3]), .sum(sum[3]));
    full_adder_acc FA3 (.a(a[4]), .b(b[4]), .cin(rca_c[3]), .cout(rca_c[4]), .sum(sum[4]));
    full_adder_acc FA4 (.a(a[5]), .b(b[5]), .cin(rca_c[4]), .cout(rca_c[5]), .sum(sum[5]));
    full_adder_acc FA5 (.a(a[6]), .b(b[6]), .cin(rca_c[5]), .cout(rca_c[6]), .sum(sum[6]));
    full_adder_acc FA6 (.a(a[7]), .b(b[7]), .cin(rca_c[6]), .cout(cout),     .sum(sum[7]));
    
endmodule