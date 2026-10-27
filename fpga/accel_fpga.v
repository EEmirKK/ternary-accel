`timescale 1ns/1ps

// FPGA wrapper around the accelerator, independent of the board. It receives the
// program and the images over a UART, keeps them in block RAM and replays the host's
// tile loop from on-chip memory, so the accelerator runs at full speed. It supports
// only N = 4, where one stream beat is one 32-bit word.
//
// Host to board, little-endian. 'P', u32 n, then n program words: for each layer and
// output tile the BIAS beats, then the WEIGHT beats of each input tile. 'L', then two
// layer-table entries {u32 CONFIG word, u16 input tiles, u16 output tiles, u32 program
// start address}, with the CONFIG word's batch field ignored. 'R', u8 B, then B images
// of (input tiles x 4) int8 bytes, after which the board runs. The reply is 'D', then
// B x (layer-1 output tiles x 4) bytes of int8 logits, image-major, then u32 total
// cycles and u32 ACT beats from the on-chip counters.
//
// A run is a STATUS request (clearing the counters), layer 0 from the image RAM into
// the hidden RAM, layer 1 from the hidden RAM into the logits RAM, and a second STATUS
// request to read the counters.

module accel_fpga #(
    parameter TERNARY      = 0,
    parameter B_MAX        = 8,
    parameter CLKS_PER_BIT = 434,
    parameter PROG_DEPTH   = 13312,   // int8 MLP needs 12,812 words, ternary 3,241
    parameter KT0_MAX      = 196,     // layer-0 k-tiles (784 / 4)
    parameter MT0_MAX      = 16,      // layer-0 m-tiles = layer-1 k-tiles (64 / 4)
    parameter MT1_MAX      = 3        // layer-1 m-tiles (10 -> 12 / 4)
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [3:0] status_led     // [0] program loaded, [1] running, [2] reply, [3] heartbeat
);
    localparam N = 4;
    localparam ACC_W  = 10 + (TERNARY ? 7 : 14) + 2;          // must match accel_top with K_MAX = 784
    localparam BBEATS = (ACC_W + 7) / 8;
    localparam WBEATS = TERNARY ? 1 : N;
    localparam IMG_DEPTH = B_MAX * KT0_MAX;
    localparam HID_DEPTH = B_MAX * MT0_MAX;
    localparam LOG_DEPTH = B_MAX * MT1_MAX;
    localparam PAW = $clog2(PROG_DEPTH);
    localparam IAW = $clog2(IMG_DEPTH);
    localparam HAW = $clog2(HID_DEPTH);
    localparam LAW = $clog2(LOG_DEPTH);
    localparam [1:0] T_CONFIG = 2'b00, T_BIAS = 2'b01, T_WEIGHT = 2'b10, T_ACT = 2'b11;

    wire [7:0] rx_data;
    wire       rx_valid;
    reg  [7:0] tx_data;
    reg        tx_valid;
    wire       tx_ready;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (.clk(clk), .rst_n(rst_n), .rx(uart_rx),
                                                  .data(rx_data), .valid(rx_valid));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (.clk(clk), .rst_n(rst_n), .data(tx_data),
                                                  .valid(tx_valid), .ready(tx_ready), .tx(uart_tx));

    // Simple dual-port RAMs, one write port and one registered read port, so they map
    // to M9K blocks. max_depth = 256 builds the deep ones from 256 x 32 slices, because
    // otherwise Quartus picks deep narrow slices, rounds the depth up to a power of two
    // and runs out of M9K.
    (* max_depth = 256 *) reg [31:0] prog_mem [0:PROG_DEPTH-1];
    (* max_depth = 256 *) reg [31:0] img_mem  [0:IMG_DEPTH-1];
    reg [31:0] hid_mem  [0:HID_DEPTH-1];
    reg [31:0] log_mem  [0:LOG_DEPTH-1];

    reg            prog_we;  reg [PAW-1:0] prog_wa;  reg [31:0] prog_wd;
    reg            img_we;   reg [IAW-1:0] img_wa;   reg [31:0] img_wd;
    reg            hid_we;   reg [HAW-1:0] hid_wa;   reg [31:0] hid_wd;
    reg            log_we;   reg [LAW-1:0] log_wa;   reg [31:0] log_wd;
    // Read addresses come combinationally from the requester's state, so each registered
    // output holds the word asked for in the previous cycle.
    wire [PAW-1:0] prog_ra;  reg [31:0] prog_q;
    wire [IAW-1:0] img_ra;   reg [31:0] img_q;
    wire [HAW-1:0] hid_ra;   reg [31:0] hid_q;
    wire [LAW-1:0] log_ra;   reg [31:0] log_q;

    always @(posedge clk) begin
        if (prog_we) prog_mem[prog_wa] <= prog_wd;
        if (img_we)  img_mem[img_wa]   <= img_wd;
        if (hid_we)  hid_mem[hid_wa]   <= hid_wd;
        if (log_we)  log_mem[log_wa]   <= log_wd;
        prog_q <= prog_mem[prog_ra];
        img_q  <= img_mem[img_ra];
        hid_q  <= hid_mem[hid_ra];
        log_q  <= log_mem[log_ra];
    end

    reg [31:0] l_cfg  [0:1];
    reg [15:0] l_kt   [0:1];
    reg [15:0] l_mt   [0:1];
    reg [31:0] l_base [0:1];
    reg [7:0]  run_b;               // batch size of the current run (1..B_MAX)

    localparam P_CMD = 3'd0, P_PCNT = 3'd1, P_PWORD = 3'd2, P_LTAB = 3'd3, P_RB = 3'd4, P_RIMG = 3'd5,
               P_BUSY = 3'd6;
    reg [2:0]  pst;
    reg [1:0]  bytei;               // byte within a 32-bit word
    reg [31:0] word;                // word being assembled (bytes shift in from the top)
    reg [31:0] pcount, pidx;        // program words expected / received
    reg [4:0]  lbyte;               // layer-table byte index 0..23
    reg [7:0]  rb;                  // image being received
    reg [15:0] rkt;                 // word within that image
    reg        prog_loaded;
    reg        start_run;           // pulse: images received, start the sequencer
    reg        run_done;            // pulse from the reply FSM
    wire [31:0] word_next = {rx_data, word[31:8]};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pst <= P_CMD; bytei <= 0; word <= 0; pcount <= 0; pidx <= 0; lbyte <= 0; rb <= 0; rkt <= 0;
            prog_loaded <= 1'b0; start_run <= 1'b0; prog_we <= 1'b0; img_we <= 1'b0; run_b <= 8'd1;
        end else begin
            start_run <= 1'b0;
            prog_we   <= 1'b0;
            img_we    <= 1'b0;
            if (pst == P_BUSY) begin
                if (run_done) pst <= P_CMD;
            end else if (rx_valid) begin
                case (pst)
                    P_CMD: begin
                        bytei <= 0;
                        if (rx_data == "P") pst <= P_PCNT;
                        else if (rx_data == "L") begin pst <= P_LTAB; lbyte <= 0; end
                        else if (rx_data == "R") pst <= P_RB;
                    end
                    P_PCNT: begin
                        word <= word_next; bytei <= bytei + 1'b1;
                        if (bytei == 2'd3) begin
                            pcount <= word_next; pidx <= 0;
                            pst <= (word_next == 0) ? P_CMD : P_PWORD;
                        end
                    end
                    P_PWORD: begin
                        word <= word_next; bytei <= bytei + 1'b1;
                        if (bytei == 2'd3) begin
                            prog_we <= 1'b1; prog_wa <= pidx[PAW-1:0]; prog_wd <= word_next;
                            pidx <= pidx + 1'b1;
                            if (pidx + 1 == pcount) begin pst <= P_CMD; prog_loaded <= 1'b1; end
                        end
                    end
                    P_LTAB: begin
                        word <= word_next; bytei <= bytei + 1'b1;
                        lbyte <= lbyte + 1'b1;
                        // entry e = lbyte / 12: bytes 0-3 cfg, 4-5 k_tiles, 6-7 m_tiles, 8-11 base
                        case (lbyte % 12)
                            3:  l_cfg[lbyte / 12]  <= word_next;
                            7:  begin l_kt[lbyte / 12] <= word_next[15:0]; l_mt[lbyte / 12] <= word_next[31:16]; end
                            11: l_base[lbyte / 12] <= word_next;
                            default: ;
                        endcase
                        if (lbyte == 5'd23) pst <= P_CMD;
                    end
                    P_RB: begin
                        run_b <= (rx_data == 0 || rx_data > B_MAX) ? B_MAX[7:0] : rx_data;
                        rb <= 0; rkt <= 0; bytei <= 0; pst <= P_RIMG;
                    end
                    P_RIMG: begin
                        word <= word_next; bytei <= bytei + 1'b1;
                        if (bytei == 2'd3) begin
                            img_we <= 1'b1; img_wa <= rb * KT0_MAX + rkt; img_wd <= word_next;
                            if (rkt + 1 == l_kt[0]) begin
                                rkt <= 0;
                                if (rb + 1 == run_b) begin pst <= P_BUSY; start_run <= 1'b1; end
                                rb <= rb + 1'b1;
                            end else rkt <= rkt + 1'b1;
                        end
                    end
                    default: pst <= P_CMD;
                endcase
            end
        end
    end

    wire [31:0] acc_in_data;
    wire [1:0]  acc_in_type;
    wire        acc_in_last, acc_in_valid, acc_in_ready;
    wire [31:0] acc_out_data;
    wire        acc_out_valid;

    accel_top #(.N(N), .TERNARY(TERNARY), .B_MAX(B_MAX), .K_MAX(784),
                .DOUBLE_BUF(1), .PERF_COUNTERS(1)) u_acc (
        .clk(clk), .rst_n(rst_n),
        .in_data(acc_in_data), .in_type(acc_in_type), .in_last(acc_in_last),
        .in_valid(acc_in_valid), .in_ready(acc_in_ready),
        .out_data(acc_out_data), .out_valid(acc_out_valid), .out_ready(1'b1));

    // The sequencer's G_* states describe the next beat to issue. Each issue reads one
    // RAM word, or uses a constant, and the word arrives a cycle later in a 2-entry FIFO
    // in front of the accelerator. The credit count includes this cycle's pop, so the
    // accelerator can take a beat every cycle.
    localparam G_IDLE = 3'd0, G_STAT = 3'd1, G_CFG = 3'd2, G_BIAS = 3'd3, G_WT = 3'd4, G_ACT = 3'd5,
               G_DONE = 3'd6;
    reg [2:0]  gst;
    reg        glayer;
    reg        gstat_n;             // which STATUS: 0 = before, 1 = after
    reg [15:0] gmt, gkt;
    reg [2:0]  gj;                  // beat within a BIAS or WEIGHT group
    reg [7:0]  gb;                  // image within the ACT group
    reg [31:0] gaddr;               // program word address

    reg [34:0] fifo [0:1];          // {type[1:0], last, data[31:0]}
    reg        fifo_rd, fifo_wr_ptr;
    reg [1:0]  fifo_cnt;
    wire       fifo_pop = (fifo_cnt != 0) && acc_in_ready;
    assign acc_in_valid = (fifo_cnt != 0);
    assign {acc_in_type, acc_in_last, acc_in_data} = fifo[fifo_rd];

    reg        pend;                // a read was issued last cycle
    reg [1:0]  pend_src;            // 0 const, 1 prog, 2 img, 3 hid
    reg [1:0]  pend_type;
    reg        pend_last;
    reg [31:0] pend_const;
    wire [31:0] pend_data = (pend_src == 2'd1) ? prog_q : (pend_src == 2'd2) ? img_q :
                            (pend_src == 2'd3) ? hid_q : pend_const;

    wire g_active = (gst != G_IDLE) && (gst != G_DONE);
    wire issue    = g_active && ({1'b0, fifo_cnt} + pend - fifo_pop < 2);

    wire [15:0] kt_n = l_kt[glayer];
    wire [15:0] mt_n = l_mt[glayer];
    wire [31:0] cfg_w = {run_b, l_cfg[glayer][23:0]};
    assign prog_ra = gaddr[PAW-1:0];
    assign img_ra  = gb * KT0_MAX + gkt;
    assign hid_ra  = gb * MT0_MAX + gkt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gst <= G_IDLE; glayer <= 0; gstat_n <= 0; gmt <= 0; gkt <= 0; gj <= 0; gb <= 0; gaddr <= 0;
            fifo_rd <= 0; fifo_wr_ptr <= 0; fifo_cnt <= 0; pend <= 1'b0;
        end else begin
            if (pend) begin
                fifo[fifo_wr_ptr] <= {pend_type, pend_last, pend_data};
                fifo_wr_ptr <= ~fifo_wr_ptr;
            end
            if (fifo_pop) fifo_rd <= ~fifo_rd;
            fifo_cnt <= fifo_cnt + pend - fifo_pop;

            pend <= issue;
            if (start_run) begin
                gst <= G_STAT; gstat_n <= 1'b0; glayer <= 1'b0;
            end else if (issue) begin
                case (gst)
                    G_STAT: begin
                        pend_src <= 2'd0; pend_type <= T_CONFIG; pend_last <= 1'b0;
                        pend_const <= 32'd0;                    // a CONFIG beat with B = 0 is a STATUS request
                        if (gstat_n) gst <= G_DONE;
                        else begin gst <= G_CFG; glayer <= 1'b0; end
                    end
                    G_CFG: begin
                        pend_src <= 2'd0; pend_type <= T_CONFIG; pend_last <= 1'b0; pend_const <= cfg_w;
                        gst <= G_BIAS; gmt <= 0; gj <= 0; gaddr <= l_base[glayer];
                    end
                    G_BIAS: begin
                        pend_src <= 2'd1; pend_type <= T_BIAS; pend_last <= 1'b0;
                        gaddr <= gaddr + 1'b1;
                        if (gj == BBEATS - 1) begin gst <= G_WT; gj <= 0; gkt <= 0; end
                        else gj <= gj + 1'b1;
                    end
                    G_WT: begin
                        pend_src <= 2'd1; pend_type <= T_WEIGHT; pend_last <= 1'b0;
                        gaddr <= gaddr + 1'b1;
                        if (gj == WBEATS - 1) begin gst <= G_ACT; gj <= 0; gb <= 0; end
                        else gj <= gj + 1'b1;
                    end
                    G_ACT: begin
                        pend_type <= T_ACT;
                        pend_last <= (gkt + 1 == kt_n) && (gb + 1 == run_b);
                        pend_src <= (glayer == 1'b0) ? 2'd2 : 2'd3;     // image RAM or hidden RAM
                        if (gb + 1 == run_b) begin
                            gb <= 0;
                            if (gkt + 1 == kt_n) begin
                                gkt <= 0;
                                if (gmt + 1 == mt_n) begin
                                    gmt <= 0;
                                    if (glayer == 1'b0) begin glayer <= 1'b1; gst <= G_CFG; end
                                    else begin gst <= G_STAT; gstat_n <= 1'b1; end
                                end else begin gmt <= gmt + 1'b1; gst <= G_BIAS; gj <= 0; end
                            end else begin gkt <= gkt + 1'b1; gst <= G_WT; gj <= 0; end
                        end else gb <= gb + 1'b1;
                    end
                    default: ;
                endcase
            end else if (gst == G_DONE && run_done) begin
                gst <= G_IDLE;
            end
        end
    end

    // Output beats arrive in this order: 2 STATUS words, MT0*B for layer 0 (output-tile
    // major), MT1*B for layer 1, then 2 STATUS words.
    localparam O_ST0 = 2'd0, O_L0 = 2'd1, O_L1 = 2'd2, O_ST1 = 2'd3;
    reg [1:0]  ost;
    reg [15:0] omt;
    reg [7:0]  ob;
    reg        osw;                 // STATUS word index
    reg [31:0] st_cycles, st_acts;
    reg        out_done;            // all outputs of the run captured

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ost <= O_ST0; omt <= 0; ob <= 0; osw <= 0; out_done <= 1'b0; hid_we <= 1'b0; log_we <= 1'b0;
            st_cycles <= 0; st_acts <= 0;
        end else begin
            hid_we <= 1'b0;
            log_we <= 1'b0;
            if (start_run || run_done) begin
                ost <= O_ST0; omt <= 0; ob <= 0; osw <= 0; out_done <= 1'b0;
            end else if (acc_out_valid) begin
                case (ost)
                    O_ST0: begin osw <= ~osw; if (osw) ost <= O_L0; end
                    O_L0, O_L1: begin
                        if (ost == O_L0) begin hid_we <= 1'b1; hid_wa <= ob * MT0_MAX + omt; hid_wd <= acc_out_data; end
                        else             begin log_we <= 1'b1; log_wa <= ob * MT1_MAX + omt; log_wd <= acc_out_data; end
                        if (ob + 1 == run_b) begin
                            ob <= 0;
                            if (omt + 1 == l_mt[ost == O_L1]) begin omt <= 0; ost <= ost + 1'b1; end
                            else omt <= omt + 1'b1;
                        end else ob <= ob + 1'b1;
                    end
                    O_ST1: begin
                        if (!osw) st_cycles <= acc_out_data;
                        else begin st_acts <= acc_out_data; out_done <= 1'b1; end
                        osw <= ~osw;
                    end
                endcase
            end
        end
    end

    // The reply: 'D', the logits (B x MT1 words), then the cycle and ACT-beat counts.
    localparam R_IDLE = 3'd0, R_HDR = 3'd1, R_RD = 3'd2, R_WAITQ = 3'd3, R_BYTE = 3'd4, R_DONE = 3'd5;
    reg [2:0]  rst_s;
    reg [15:0] rw;                  // word index: 0 .. B*MT1-1 logits, then 2 counter words
    reg [1:0]  rbyte;
    reg [31:0] rword;
    wire [15:0] n_logit_words = run_b * l_mt[1];
    assign log_ra = rw[LAW-1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rst_s <= R_IDLE; tx_valid <= 1'b0; run_done <= 1'b0; rw <= 0; rbyte <= 0;
        end else begin
            tx_valid <= 1'b0;
            run_done <= 1'b0;
            case (rst_s)
                // The !run_done check matters in the cycle a reply finishes. out_done and the
                // parser's busy state only clear at the next edge, so without it the board would
                // reply twice.
                R_IDLE: if (out_done && pst == P_BUSY && !run_done) rst_s <= R_HDR;
                R_HDR: if (tx_ready && !tx_valid) begin
                    tx_data <= "D"; tx_valid <= 1'b1; rw <= 0; rst_s <= R_RD;
                end
                R_RD: rst_s <= R_WAITQ;   // log_q <= log_mem[rw] at this edge
                R_WAITQ: begin            // log_q valid now
                    rword <= (rw < n_logit_words) ? log_q : (rw == n_logit_words) ? st_cycles : st_acts;
                    rbyte <= 0; rst_s <= R_BYTE;
                end
                R_BYTE: if (tx_ready && !tx_valid) begin
                    tx_data <= rword[7:0]; tx_valid <= 1'b1; rword <= rword >> 8;
                    rbyte <= rbyte + 1'b1;
                    if (rbyte == 2'd3) begin
                        if (rw + 1 == n_logit_words + 2) rst_s <= R_DONE;
                        else begin rw <= rw + 1'b1; rst_s <= R_RD; end
                    end
                end
                R_DONE: if (tx_ready && !tx_valid) begin
                    run_done <= 1'b1; rst_s <= R_IDLE;
                end
                default: rst_s <= R_IDLE;
            endcase
        end
    end

    reg [25:0] hb = 26'd0;
    always @(posedge clk) hb <= hb + 1'b1;
    assign status_led = {hb[25], rst_s != R_IDLE, gst != G_IDLE, prog_loaded};
endmodule
