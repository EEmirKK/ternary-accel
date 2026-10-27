`timescale 1ns/1ps

// Requantiser: turns a wide accumulator value back into an int8 output,
// y = clamp((acc * M0 + 2^(s-1)) >> s), rounding half up. The clamp range is
// [0, 127] with ReLU and [-128, 127] without. All N outputs share this one unit
// in turn, so it is pipelined over three stages. That latency costs almost
// nothing, because the unit runs once per output, and keeps it off the critical path.

module requant #(
    parameter ACC_W = 26,
    parameter TAG_W = 2
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_v,
    input  wire [TAG_W-1:0]  in_tag,
    input  wire [ACC_W-1:0]  acc,
    input  wire [15:0]       m0,
    input  wire [5:0]        s,
    input  wire              relu,
    output wire              out_v,
    output wire [TAG_W-1:0]  out_tag,
    output wire [7:0]        y
);
    // |acc*M0| < 2^(ACC_W+15), plus one bit of headroom for the rounding constant.
    localparam PW = ACC_W + 18;
    localparam HW = ACC_W + 9;          // acc times one zero-extended byte of M0

    reg [2:0]              v;           // valid bit of each pipeline stage
    reg [TAG_W-1:0]        t0, t1, t2;
    reg signed [ACC_W-1:0] a0;
    reg signed [HW-1:0]    p_lo, p_hi;
    reg signed [PW-1:0]    sum;

    // No rounding term when s = 0. The shift assumes s is at most ACC_W+16.
    wire signed [PW-1:0] one = {{(PW-1){1'b0}}, 1'b1};
    wire signed [PW-1:0] rnd = (s == 6'd0) ? {PW{1'b0}} : (one <<< (s - 6'd1));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v <= 3'b000;
        else        v <= {v[1:0], in_v};
    end

    always @(posedge clk) begin
        // Stage 1 registers the accumulator read, so the bank's read mux gets a pipeline stage of its own.
        if (in_v) begin
            a0 <= acc;
            t0 <= in_tag;
        end
        if (v[0]) begin
            p_lo <= a0 * $signed({1'b0, m0[7:0]});
            p_hi <= a0 * $signed({1'b0, m0[15:8]});
            t1   <= t0;
        end
        if (v[1]) begin
            sum <= p_lo + (p_hi <<< 8) + rnd;
            t2  <= t1;
        end
    end

    assign out_v   = v[2];
    assign out_tag = t2;

    // The shift and clamp are combinational, straight into the output buffer.
    wire signed [PW-1:0] t  = sum >>> s;
    wire signed [PW-1:0] hi = 127;
    wire signed [PW-1:0] lo = relu ? 0 : -128;
    assign y = (t > hi) ? 8'sd127 : (t < lo) ? lo[7:0] : t[7:0];
endmodule
