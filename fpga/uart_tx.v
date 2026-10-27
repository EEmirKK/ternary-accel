`timescale 1ns/1ps

// UART transmitter, 8N1, LSB first. It takes a byte whenever ready is high.

module uart_tx #(
    parameter CLKS_PER_BIT = 434
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data,
    input  wire       valid,
    output wire       ready,
    output reg        tx                  // idle high
);
    localparam CW = $clog2(CLKS_PER_BIT + 1);
    reg [CW-1:0] cnt;
    reg [3:0]    bitn;
    reg [9:0]    shreg;                   // {stop, data, start}
    reg          busy;
    assign ready = ~busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx <= 1'b1; cnt <= 0; bitn <= 0; shreg <= 10'h3ff; busy <= 1'b0;
        end else if (!busy) begin
            if (valid) begin
                shreg <= {1'b1, data, 1'b0};
                busy  <= 1'b1; bitn <= 0; cnt <= CLKS_PER_BIT - 1;
                tx    <= 1'b0;                        // start bit
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
        end else begin
            cnt <= CLKS_PER_BIT - 1;
            if (bitn == 9) begin
                busy <= 1'b0; tx <= 1'b1;
            end else begin
                tx    <= shreg[bitn + 1];
                bitn  <= bitn + 1'b1;
            end
        end
    end
endmodule
