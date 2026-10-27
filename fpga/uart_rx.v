`timescale 1ns/1ps

// UART receiver, 8N1, LSB first. It samples each bit in its middle, and
// CLKS_PER_BIT is the clock frequency over the baud rate.

module uart_rx #(
    parameter CLKS_PER_BIT = 434          // 50 MHz / 115200
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,                 // idle high
    output reg  [7:0] data,
    output reg        valid               // one-cycle pulse per received byte
);
    localparam CW = $clog2(CLKS_PER_BIT + 1);
    reg [1:0]    sync;                    // rx is asynchronous
    reg [CW-1:0] cnt;
    reg [3:0]    bitn;                    // 0 = start bit, 1..8 = data, 9 = stop
    reg          busy;
    wire         rxs = sync[1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sync <= 2'b11; cnt <= 0; bitn <= 0; busy <= 1'b0; valid <= 1'b0; data <= 8'd0;
        end else begin
            sync  <= {sync[0], rx};
            valid <= 1'b0;
            if (!busy) begin
                if (!rxs) begin                       // falling edge: start bit
                    busy <= 1'b1; bitn <= 0; cnt <= CLKS_PER_BIT / 2;
                end
            end else if (cnt != 0) begin
                cnt <= cnt - 1'b1;
            end else begin
                cnt <= CLKS_PER_BIT - 1;
                if (bitn == 0) begin
                    if (rxs) busy <= 1'b0;            // glitch, not a real start bit
                end else if (bitn <= 8) begin
                    data <= {rxs, data[7:1]};
                end else begin
                    busy  <= 1'b0;
                    valid <= rxs;                     // drop the byte on a framing error
                end
                bitn <= bitn + 1'b1;
            end
        end
    end
endmodule
