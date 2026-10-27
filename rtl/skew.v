`timescale 1ns/1ps

// Input skew. Row r is delayed by r+1 registers, so the activation for row r reaches
// PE(r,c) in the same cycle as the partial sum computed above it.

module skew #(
    parameter N = 4,
    parameter W = 8                   // 8 for activations, 1 for bank tags
) (
    input  wire           clk,
    input  wire [N*W-1:0] in_lanes,   // zeros when idle, so the array doesn't toggle
    output wire [N*W-1:0] out_lanes
);
    genvar r;
    generate
        for (r = 0; r < N; r = r + 1) begin : g_row
            reg [W-1:0] sr [0:r];
            integer i;
            always @(posedge clk) begin
                sr[0] <= in_lanes[r*W +: W];
                for (i = 1; i <= r; i = i + 1) sr[i] <= sr[i-1];
            end
            assign out_lanes[r*W +: W] = sr[r];
        end
    endgenerate
endmodule
