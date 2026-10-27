`timescale 1ns/1ps

// Accumulator bank: one ACC_W-bit register per (image, output column),
// B_MAX x N in total. At this size, registers are simpler than SRAM.
//
// Bias beats preload every image's row with its bias. After that, each
// column sum coming out of the array is added to the row of the image
// it belongs to.
//
// Only one image's row is updated per cycle, so the bank shares N adders
// across all rows instead of having one adder per register. The same row
// mux also feeds the requantiser's read port. The two uses never overlap,
// because requantisation only starts after every activation has been
// accumulated.

module acc_bank #(
    parameter N      = 4,
    parameter B_MAX  = 8,
    parameter ACC_W  = 26,
    parameter PSUM_W = 18,
    parameter BW     = 3,                 // width of a batch index
    parameter CW     = 2,                 // width of a column index
    parameter NBB    = (ACC_W + 7) / 8,   // bias beats
    parameter BIW    = 2                  // width of a bias beat index
) (
    input  wire                clk,
    input  wire                bias_we,
    input  wire [BIW-1:0]      bias_idx,
    input  wire [N*8-1:0]      bias_lanes,
    input  wire                acc_en,
    input  wire [BW-1:0]       acc_b,
    input  wire [N*PSUM_W-1:0] col_in,
    input  wire [BW-1:0]       rd_b,
    input  wire [CW-1:0]       rd_c,
    output wire [ACC_W-1:0]    rd_data
);
    wire [B_MAX*N*ACC_W-1:0] flat;      // all registers, [(b*N+c)*ACC_W +: ACC_W]
    wire [BW-1:0]            row_b = acc_en ? acc_b : rd_b;
    wire [N*ACC_W-1:0]       row   = flat[row_b*N*ACC_W +: N*ACC_W];
    wire [N*ACC_W-1:0]       sum;       // row + column sums (N shared adders)
    wire [N*ACC_W-1:0]       bias_v;    // row 0 with bias byte bias_idx replaced

    genvar b, c;
    generate
        for (c = 0; c < N; c = c + 1) begin : g_col
            // Sign-extend the column sum to accumulator width, then add it to the row.
            wire [PSUM_W-1:0] col = col_in[c*PSUM_W +: PSUM_W];
            assign sum[c*ACC_W +: ACC_W] =
                row[c*ACC_W +: ACC_W] + {{(ACC_W-PSUM_W){col[PSUM_W-1]}}, col};

            // The bias arrives one byte per beat. Every row receives the same bytes,
            // so all rows always hold the same value, and row 0 can be used as the
            // base when merging in the next byte. This relies on the host always
            // sending all NBB bias beats.
            /* verilator lint_off UNUSEDSIGNAL */
            reg [NBB*8-1:0] padded;     // top bits beyond ACC_W are dropped
            /* verilator lint_on UNUSEDSIGNAL */
            always @(*) begin
                padded = {(NBB*8){1'b0}};
                padded[ACC_W-1:0] = flat[c*ACC_W +: ACC_W];
                padded[bias_idx*8 +: 8] = bias_lanes[c*8 +: 8];
            end
            assign bias_v[c*ACC_W +: ACC_W] = padded[ACC_W-1:0];
        end

        for (b = 0; b < B_MAX; b = b + 1) begin : g_b
            for (c = 0; c < N; c = c + 1) begin : g_c
                reg [ACC_W-1:0] acc;
                always @(posedge clk) begin
                    if (bias_we)
                        acc <= bias_v[c*ACC_W +: ACC_W];
                    else if (acc_en && acc_b == b)
                        acc <= sum[c*ACC_W +: ACC_W];
                end
                assign flat[(b*N+c)*ACC_W +: ACC_W] = acc;
            end
        end
    endgenerate

    assign rd_data = row[rd_c*ACC_W +: ACC_W];
endmodule
