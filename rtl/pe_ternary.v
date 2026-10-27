`timescale 1ns/1ps

// Ternary processing element. The 2-bit weight is two's complement, 01 = +1 and
// 11 = -1, with 00 and the unused 10 both meaning 0. Instead of multiplying, it adds
// the activation, subtracts it, or adds nothing, so there is no multiplier. The
// product needs 9 bits, because -(-128) = +128. Otherwise it matches pe_int8.v.

module pe_ternary #(
    parameter PSUM_W     = 11,
    parameter DOUBLE_BUF = 0
) (
    input  wire                     clk,
    input  wire                     w_we,
    input  wire        [1:0]        w_in,
    input  wire                     w_bank,
    input  wire signed [7:0]        a_in,
    input  wire                     b_in,
    input  wire signed [PSUM_W-1:0] psum_in,
    output reg  signed [7:0]        a_out,
    output wire                     b_out,
    output reg  signed [PSUM_W-1:0] psum_out
);
    wire [1:0] w;               // the weight this cycle's activation uses
    wire w_pos = (w == 2'b01);
    wire w_neg = (w == 2'b11);

    always @(posedge clk) a_out <= a_in;

    generate
        if (DOUBLE_BUF) begin : g_db
            reg [1:0] w0, w1;
            reg       b_q;
            always @(posedge clk) begin
                if (w_we & ~w_bank) w0 <= w_in;
                if (w_we &  w_bank) w1 <= w_in;
                b_q <= b_in;
            end
            assign w     = b_in ? w1 : w0;
            assign b_out = b_q;
        end else begin : g_sb
            reg [1:0] w_r;
            always @(posedge clk) if (w_we) w_r <= w_in;
            assign w     = w_r;
            assign b_out = 1'b0;
        end
    endgenerate

    wire [PSUM_W-1:0] a_ext   = {{(PSUM_W-8){a_in[7]}}, a_in};
    wire [PSUM_W-1:0] operand = w_neg ? ~a_ext : (w_pos ? a_ext : {PSUM_W{1'b0}});
    // For -1, w_neg is the carry-in that turns ~a into -a, so no separate negator is needed.
    always @(posedge clk)
        psum_out <= psum_in + operand + {{(PSUM_W-1){1'b0}}, w_neg};
endmodule
