`timescale 1ns/1ps

// N x N weight-stationary array. PE(r,c) holds weight W[k0+r][m0+c] of the current
// tile. Activations enter each row from the left, already skewed, and move right.
// Partial sums start at zero at the top and move down, so the bottom of column c
// gives the sum over r of W[r][c] * x[r] for each activation vector.

module systolic_array #(
    parameter N       = 4,
    parameter TERNARY = 0,
    parameter WW      = TERNARY ? 2 : 8,    // weight width
    parameter PSUM_W  = 18,
    parameter DOUBLE_BUF = 0
) (
    input  wire                  clk,
    input  wire [N-1:0]          w_we,     // one write enable per row
    input  wire [N*N*WW-1:0]     w_data,   // weight for PE(r,c) at [(r*N+c)*WW +: WW]
    input  wire                  w_bank,
    input  wire [N*8-1:0]        a_left,   // row r at [r*8 +: 8]
    input  wire [N-1:0]          b_left,   // bank tag of each row's activation
    output wire [N*PSUM_W-1:0]   psum_bot  // column c at [c*PSUM_W +: PSUM_W]
);
    // a_h[r][c] and p_v[r][c] are the activation and partial sum entering PE(r,c).
    // The extra index is the unused right edge for a_h and the bottom edge for p_v.
    wire [7:0]        a_h [0:N-1][0:N];
    wire              b_h [0:N-1][0:N];
    wire [PSUM_W-1:0] p_v [0:N][0:N-1];

    genvar r, c;
    generate
        for (r = 0; r < N; r = r + 1) begin : g_row
            assign a_h[r][0] = a_left[r*8 +: 8];
            assign b_h[r][0] = b_left[r];
            for (c = 0; c < N; c = c + 1) begin : g_col
                if (r == 0) begin : g_top
                    assign p_v[0][c] = {PSUM_W{1'b0}};
                end
                if (TERNARY) begin : g_pe
                    pe_ternary #(.PSUM_W(PSUM_W), .DOUBLE_BUF(DOUBLE_BUF)) u_pe (
                        .clk(clk), .w_we(w_we[r]), .w_in(w_data[(r*N+c)*WW +: WW]), .w_bank(w_bank),
                        .a_in(a_h[r][c]), .b_in(b_h[r][c]), .psum_in(p_v[r][c]),
                        .a_out(a_h[r][c+1]), .b_out(b_h[r][c+1]), .psum_out(p_v[r+1][c]));
                end else begin : g_pe
                    pe_int8 #(.PSUM_W(PSUM_W), .DOUBLE_BUF(DOUBLE_BUF)) u_pe (
                        .clk(clk), .w_we(w_we[r]), .w_in(w_data[(r*N+c)*WW +: WW]), .w_bank(w_bank),
                        .a_in(a_h[r][c]), .b_in(b_h[r][c]), .psum_in(p_v[r][c]),
                        .a_out(a_h[r][c+1]), .b_out(b_h[r][c+1]), .psum_out(p_v[r+1][c]));
                end
            end
        end
        for (c = 0; c < N; c = c + 1) begin : g_bot
            assign psum_bot[c*PSUM_W +: PSUM_W] = p_v[N][c];
        end
    endgenerate
endmodule
