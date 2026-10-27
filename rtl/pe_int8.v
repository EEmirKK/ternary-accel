`timescale 1ns/1ps

// int8 processing element. It holds one signed 8-bit weight, passes each activation
// on to the right and adds weight x activation into the partial sum moving down its
// column. Both outputs are registered.
//
// With DOUBLE_BUF it keeps a second weight. Each activation carries a bank tag that
// selects which weight it is multiplied by, so the next tile can load while the
// current one is still in use.

module pe_int8 #(
    parameter PSUM_W     = 18,
    parameter DOUBLE_BUF = 0
) (
    input  wire                     clk,
    input  wire                     w_we,
    input  wire        [7:0]        w_in,
    input  wire                     w_bank,    // bank that w_we writes (DOUBLE_BUF only)
    input  wire signed [7:0]        a_in,
    input  wire                     b_in,      // bank tag of a_in (DOUBLE_BUF only)
    input  wire signed [PSUM_W-1:0] psum_in,
    output reg  signed [7:0]        a_out,
    output wire                     b_out,
    output reg  signed [PSUM_W-1:0] psum_out
);
    wire signed [7:0] w;        // the weight this cycle's activation uses

    always @(posedge clk) a_out <= a_in;

    generate
        if (DOUBLE_BUF) begin : g_db
            reg signed [7:0] w0, w1;
            reg              b_q;
            always @(posedge clk) begin
                if (w_we & ~w_bank) w0 <= w_in;
                if (w_we &  w_bank) w1 <= w_in;
                b_q <= b_in;
            end
            assign w     = b_in ? w1 : w0;
            assign b_out = b_q;
        end else begin : g_sb
            reg signed [7:0] w_r;
            always @(posedge clk) if (w_we) w_r <= w_in;
            assign w     = w_r;
            assign b_out = 1'b0;
        end
    endgenerate

    wire signed [15:0] prod = w * a_in;
    always @(posedge clk) psum_out <= psum_in + prod;
endmodule
