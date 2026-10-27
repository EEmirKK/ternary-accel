`timescale 1ns/1ps

// Test top that echoes every byte from rx back on tx, to check uart_rx and uart_tx
// together. The 4-byte FIFO is needed because back-to-back bytes arrive as fast as
// tx can send them.

module uart_loop #(parameter CLKS_PER_BIT = 434) (
    input wire clk, input wire rst_n, input wire rx, output wire tx
);
    wire [7:0] d; wire v; wire rdy;
    reg  [7:0] fifo [0:3];
    reg  [1:0] wp, rp;
    reg  [2:0] cnt;
    wire       send = (cnt != 0) && rdy;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (.clk(clk), .rst_n(rst_n), .rx(rx), .data(d), .valid(v));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (.clk(clk), .rst_n(rst_n), .data(fifo[rp]), .valid(send),
                                                  .ready(rdy), .tx(tx));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin wp <= 0; rp <= 0; cnt <= 0; end
        else begin
            if (v) begin fifo[wp] <= d; wp <= wp + 1'b1; end
            if (send) rp <= rp + 1'b1;
            cnt <= cnt + v - send;
        end
    end
endmodule
