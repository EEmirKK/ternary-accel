`timescale 1ns/1ps

// Output deskew. Column c leaves the array c cycles after column 0, so it is delayed
// by N-1-c registers, and all N column sums then appear in the same cycle.

module deskew #(
    parameter N = 4,
    parameter W = 18
) (
    input  wire         clk,
    input  wire [N*W-1:0] in_cols,
    output wire [N*W-1:0] out_cols
);
    genvar c;
    generate
        for (c = 0; c < N; c = c + 1) begin : g_col
            if (c == N - 1) begin : g_pass
                assign out_cols[c*W +: W] = in_cols[c*W +: W];
            end else begin : g_dly
                reg [W-1:0] sr [0:N-2-c];
                integer i;
                always @(posedge clk) begin
                    sr[0] <= in_cols[c*W +: W];
                    for (i = 1; i <= N-2-c; i = i + 1) sr[i] <= sr[i-1];
                end
                assign out_cols[c*W +: W] = sr[N-2-c];
            end
        end
    endgenerate
endmodule
