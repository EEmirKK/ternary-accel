`timescale 1ns/1ps

// DE0-Nano top level: 50 MHz clock, KEY[0] reset, the UART on the GPIO-0 header
// through a 3.3 V adapter, and status LEDs. Pins are in fpga/quartus/de0nano.qsf.

module de0nano_top #(
    parameter TERNARY = 0
) (
    input  wire       CLOCK_50,
    input  wire [1:0] KEY,          // active low; KEY[0] = reset
    input  wire       UART_RXD,     // from the adapter's TX
    output wire       UART_TXD,     // to the adapter's RX
    output wire [7:0] LED
);
    // Synchronise the reset release.
    reg [1:0] rst_sync = 2'b00;
    always @(posedge CLOCK_50 or negedge KEY[0])
        if (!KEY[0]) rst_sync <= 2'b00;
        else         rst_sync <= {rst_sync[0], 1'b1};
    wire rst_n = rst_sync[1];

    wire [3:0] st;
    // Program RAM sized to the network (int8 12,812 words, ternary 3,241), rounded up
    // to whole 256-word M9K slices.
    accel_fpga #(.TERNARY(TERNARY), .CLKS_PER_BIT(434),
                 .PROG_DEPTH(TERNARY ? 3328 : 13312)) u_core (
        .clk(CLOCK_50), .rst_n(rst_n), .uart_rx(UART_RXD), .uart_tx(UART_TXD), .status_led(st));
    assign LED = {4'b0000, st};
endmodule
