`timescale 1ns/1ps

// Top level of the accelerator. The host drives the tile loop over one valid/ready
// input stream, and this block loads weights, skews activations into the array,
// accumulates across input tiles and requantises the results.
//
// Input beats by in_type: 00 CONFIG holds M0 [15:0], s [21:16], ReLU [22] and the
// batch size B [31:24]. 01 BIAS takes ceil(ACC_W/8) beats, beat j carrying byte j of
// each output's bias. 10 WEIGHT is one beat per row for int8 (lane c = W[r][c]), or one
// beat per four rows for ternary (row q at bits 2*(q*N+c), 01 = +1, 11 = -1). 11 ACT is
// one activation vector, with in_last on the last one of the last input tile, which
// starts requantisation. Each output beat holds one image's N int8 outputs.
//
// With PERF_COUNTERS, a CONFIG beat with B = 0 is a STATUS request instead. It returns
// two beats, the total cycles and the accepted ACT beats since the last request, and
// clears both. With DOUBLE_BUF, each weight tile loads into the bank the in-flight ACTs
// are not using, so a WEIGHT beat only waits for its own bank to drain.

module accel_top #(
    parameter N       = 4,
    parameter TERNARY = 0,
    parameter B_MAX   = 8,
    parameter K_MAX   = 784,
    parameter DOUBLE_BUF = 0,
    parameter PERF_COUNTERS = 0
) (
    input  wire           clk,
    input  wire           rst_n,

    input  wire [8*N-1:0] in_data,
    input  wire [1:0]     in_type,
    input  wire           in_last,
    input  wire           in_valid,
    output wire           in_ready,

    output wire [8*N-1:0] out_data,
    output reg            out_valid,
    input  wire           out_ready
);
    localparam WW       = TERNARY ? 2 : 8;
    localparam PROD_W   = TERNARY ? 9 : 16;
    localparam PSUM_W   = PROD_W + $clog2(N);
    localparam ACC_W    = $clog2(K_MAX) + (TERNARY ? 7 : 14) + 2;
    localparam WBEATS   = TERNARY ? (N + 3) / 4 : N;
    localparam BBEATS   = (ACC_W + 7) / 8;
    localparam LAT      = 2 * N;        // cycles from accepting an ACT to its deskewed column sums
    // Cycles after the last ACT before a WEIGHT beat may overwrite the weights that ACT uses.
    // A ternary beat rewrites four rows at once, so it must wait for PE(N-1,N-1), 2N-2
    // cycles. An int8 beat rewrites one row, starting with row 0, so only row 0 bounds it.
    localparam DRAIN    = TERNARY ? 2 * N - 2 : N - 1;
    localparam BW       = (B_MAX > 1) ? $clog2(B_MAX) : 1;
    localparam CW       = (N > 1) ? $clog2(N) : 1;
    localparam WBW      = (WBEATS > 1) ? $clog2(WBEATS) : 1;
    localparam BIW      = (BBEATS > 1) ? $clog2(BBEATS) : 1;
    localparam DW       = $clog2(DRAIN + 1);

    localparam [1:0] T_CONFIG = 2'b00, T_BIAS = 2'b01, T_WEIGHT = 2'b10, T_ACT = 2'b11;

    reg [15:0]  cfg_m0;
    reg [5:0]   cfg_s;
    reg         cfg_relu;
    reg [BW:0]  cfg_b;          // 1 .. B_MAX

    reg [WBW-1:0] w_beat;       // next weight beat within the tile
    reg [BIW-1:0] bias_beat;
    reg [BW-1:0]  act_b;        // image index of the next ACT beat
    reg [DW-1:0]  drain_cnt;    // nonzero while an ACT may still read bank 0's weights
    reg [DW-1:0]  drain_cnt1;   // the same for bank 1
    reg           act_bank;     // bank the next ACTs use
    reg           load_bank;    // bank the current weight tile is loading into
    reg [31:0]    cnt_cycles;
    reg [31:0]    cnt_acts;
    reg [31:0]    st_word;      // STATUS beat being sent
    reg [1:0]     st_left;      // STATUS beats still to send
    reg           busy;         // from in_last accepted until the last output beat leaves

    reg [LAT-1:0] tag_v;        // tags run alongside the datapath, so the bank knows each sum's image
    reg [LAT-1:0] tag_last;
    reg [BW-1:0]  tag_b [0:LAT-1];

    wire pipe_empty = ~|tag_v;

    // A new weight tile goes to the bank the in-flight ACTs are not using.
    wire w_tgt   = DOUBLE_BUF ? ((w_beat == {WBW{1'b0}}) ? ~act_bank : load_bank) : 1'b0;
    wire w_clear = (DOUBLE_BUF && w_tgt) ? (drain_cnt1 == {DW{1'b0}}) : (drain_cnt == {DW{1'b0}});

    assign in_ready = ~busy & (
        (in_type == T_ACT)    ? 1'b1 :
        (in_type == T_WEIGHT) ? w_clear :
                                pipe_empty);            // CONFIG and BIAS wait for the array to empty

    wire       acc_xfer  = in_valid & in_ready;
    wire       is_config = acc_xfer & (in_type == T_CONFIG);
    wire       is_bias   = acc_xfer & (in_type == T_BIAS);
    wire       is_weight = acc_xfer & (in_type == T_WEIGHT);
    wire       is_act    = acc_xfer & (in_type == T_ACT);
    wire       is_status = PERF_COUNTERS && is_config && (in_data[31:24] == 8'd0);

    reg           rq_issue;     // still issuing elements of the current beat
    reg [BW-1:0]  rq_b;
    reg [CW-1:0]  rq_c;
    wire          rq_done_v;
    wire [CW-1:0] rq_done_c;
    wire [7:0]    rq_y;
    reg  [7:0]    out_buf [0:N-1];
    reg  [31:0]   st_first;     // first STATUS beat (total cycles)

    wire          last_arrives = tag_v[LAT-1] & tag_last[LAT-1];
    wire          out_xfer     = out_valid & out_ready;
    wire          rq_xfer;      // out_xfer of a requant output beat (not a STATUS beat)

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_m0    <= 16'd0;
            cfg_s     <= 6'd0;
            cfg_relu  <= 1'b0;
            cfg_b     <= 1;
            w_beat    <= {WBW{1'b0}};
            bias_beat <= {BIW{1'b0}};
            act_b     <= {BW{1'b0}};
            drain_cnt <= {DW{1'b0}};
            drain_cnt1 <= {DW{1'b0}};
            act_bank  <= 1'b0;
            load_bank <= 1'b0;
            cnt_cycles <= 32'd0;
            cnt_acts  <= 32'd0;
            st_word   <= 32'd0;
            st_left   <= 2'd0;
            busy      <= 1'b0;
            tag_v     <= {LAT{1'b0}};
            tag_last  <= {LAT{1'b0}};
            rq_issue  <= 1'b0;
            rq_b      <= {BW{1'b0}};
            rq_c      <= {CW{1'b0}};
            out_valid <= 1'b0;
        end else begin
            if (is_config && !is_status) begin
                cfg_m0   <= in_data[15:0];
                cfg_s    <= in_data[21:16];
                cfg_relu <= in_data[22];
                cfg_b    <= in_data[24 +: BW+1];
            end

            // Each beat counter restarts whenever a beat of another type arrives.
            if (is_weight) w_beat <= (w_beat == WBEATS - 1) ? {WBW{1'b0}} : w_beat + 1'b1;
            else if (acc_xfer) w_beat <= {WBW{1'b0}};

            if (is_bias) bias_beat <= (bias_beat == BBEATS - 1) ? {BIW{1'b0}} : bias_beat + 1'b1;
            else if (acc_xfer) bias_beat <= {BIW{1'b0}};

            if (is_act) act_b <= (act_b == cfg_b - 1) ? {BW{1'b0}} : act_b + 1'b1;
            else if (acc_xfer) act_b <= {BW{1'b0}};

            // Hold off WEIGHT beats until in-flight ACTs have used the old weights,
            // per bank when double-buffered.
            if (is_act && !(DOUBLE_BUF && act_bank)) drain_cnt <= DRAIN;
            else if (drain_cnt != 0) drain_cnt <= drain_cnt - 1'b1;
            if (DOUBLE_BUF) begin
                if (is_act && act_bank) drain_cnt1 <= DRAIN;
                else if (drain_cnt1 != 0) drain_cnt1 <= drain_cnt1 - 1'b1;
                if (is_weight) load_bank <= w_tgt;
                if (is_weight && w_beat == WBEATS - 1) act_bank <= w_tgt;   // tile complete, the next ACTs use it
            end

            if (PERF_COUNTERS) begin
                if (is_status) begin
                    cnt_cycles <= 32'd1;        // this cycle starts the next interval
                    cnt_acts   <= 32'd0;
                    st_word    <= cnt_acts;     // sent second
                    st_left    <= 2'd2;
                    busy       <= 1'b1;
                end else begin
                    cnt_cycles <= cnt_cycles + 1'b1;
                    if (is_act) cnt_acts <= cnt_acts + 1'b1;
                end
                if (st_left != 2'd0 && out_xfer) begin
                    st_left <= st_left - 1'b1;
                    if (st_left == 2'd1) busy <= 1'b0;
                end
            end

            if (is_act & in_last) busy <= 1'b1;

            tag_v    <= {tag_v[LAT-2:0], is_act};
            tag_last <= {tag_last[LAT-2:0], is_act & in_last};

            // Requantise and send B beats once the last ACT has been accumulated.
            if (last_arrives) begin
                rq_issue  <= 1'b1;
                rq_b      <= {BW{1'b0}};
                rq_c      <= {CW{1'b0}};
            end else if (rq_issue) begin
                rq_c <= rq_c + 1'b1;
                if (rq_c == N - 1) rq_issue <= 1'b0;
            end else if (rq_xfer) begin
                if (rq_b == cfg_b - 1) begin
                    busy      <= 1'b0;
                    rq_b      <= {BW{1'b0}};    // clean idle state
                end else begin
                    rq_b     <= rq_b + 1'b1;
                    rq_c     <= {CW{1'b0}};
                    rq_issue <= 1'b1;
                end
            end

            if (rq_done_v && rq_done_c == N - 1) out_valid <= 1'b1;
            else if (is_status)                  out_valid <= 1'b1;
            else if (out_xfer && st_left == 2'd2) out_valid <= 1'b1;   // second STATUS beat
            else if (out_xfer)                   out_valid <= 1'b0;
        end
    end

    always @(posedge clk) begin
        tag_b[0] <= act_b;
        for (i = 1; i < LAT; i = i + 1) tag_b[i] <= tag_b[i-1];
        if (rq_done_v) out_buf[rq_done_c] <= rq_y;
        if (is_status) st_first <= cnt_cycles;
    end

    genvar g;
    wire [8*N-1:0] rq_data;
    generate
        for (g = 0; g < N; g = g + 1) begin : g_out
            assign rq_data[g*8 +: 8] = out_buf[g];
        end
        if (PERF_COUNTERS) begin : g_st
            assign rq_xfer = out_xfer & (st_left == 2'd0);
            wire [31:0] st_beat = (st_left == 2'd2) ? st_first : st_word;
            assign out_data = (st_left != 2'd0) ? {{(8*N-32){1'b0}}, st_beat} : rq_data;
        end else begin : g_nost
            assign rq_xfer  = out_xfer;
            assign out_data = rq_data;
        end
    endgenerate

    // An int8 beat r writes row r from all lanes. A ternary beat j writes rows 4j to
    // 4j+3, row 4j+q taking bits [2N*q +: 2N].
    wire [N-1:0]      w_we;
    wire [N*N*WW-1:0] w_data;
    generate
        for (g = 0; g < N; g = g + 1) begin : g_wrow
            if (TERNARY) begin : g_t
                assign w_we[g] = is_weight & (w_beat == g / 4);
                assign w_data[g*N*WW +: N*WW] = in_data[(g % 4)*N*WW +: N*WW];
            end else begin : g_i
                assign w_we[g] = is_weight & (w_beat == g);
                assign w_data[g*N*WW +: N*WW] = in_data;
            end
        end
    endgenerate

    wire [8*N-1:0]      a_skewed;
    wire [N-1:0]        b_skewed;
    wire [N*PSUM_W-1:0] psum_bot, col_sum;
    wire [ACC_W-1:0]    acc_rd;

    // Idle cycles feed zeros, so they don't toggle the array.
    skew #(.N(N)) u_skew (
        .clk(clk), .in_lanes(is_act ? in_data : {8*N{1'b0}}), .out_lanes(a_skewed));

    // Each activation's bank tag is skewed exactly like the activation (always 0 when single-buffered).
    generate
        if (DOUBLE_BUF) begin : g_bskew
            skew #(.N(N), .W(1)) u_bskew (
                .clk(clk), .in_lanes({N{is_act & act_bank}}), .out_lanes(b_skewed));
        end else begin : g_nobskew
            assign b_skewed = {N{1'b0}};
        end
    endgenerate

    systolic_array #(.N(N), .TERNARY(TERNARY), .WW(WW), .PSUM_W(PSUM_W),
                     .DOUBLE_BUF(DOUBLE_BUF)) u_array (
        .clk(clk), .w_we(w_we), .w_data(w_data), .w_bank(w_tgt),
        .a_left(a_skewed), .b_left(b_skewed), .psum_bot(psum_bot));

    deskew #(.N(N), .W(PSUM_W)) u_deskew (
        .clk(clk), .in_cols(psum_bot), .out_cols(col_sum));

    acc_bank #(.N(N), .B_MAX(B_MAX), .ACC_W(ACC_W), .PSUM_W(PSUM_W),
               .BW(BW), .CW(CW), .NBB(BBEATS), .BIW(BIW)) u_acc (
        .clk(clk),
        .bias_we(is_bias), .bias_idx(bias_beat), .bias_lanes(in_data),
        .acc_en(tag_v[LAT-1]), .acc_b(tag_b[LAT-1]), .col_in(col_sum),
        .rd_b(rq_b), .rd_c(rq_c), .rd_data(acc_rd));

    requant #(.ACC_W(ACC_W), .TAG_W(CW)) u_rq (
        .clk(clk), .rst_n(rst_n),
        .in_v(rq_issue & ~last_arrives), .in_tag(rq_c), .acc(acc_rd),
        .m0(cfg_m0), .s(cfg_s), .relu(cfg_relu),
        .out_v(rq_done_v), .out_tag(rq_done_c), .y(rq_y));
endmodule
