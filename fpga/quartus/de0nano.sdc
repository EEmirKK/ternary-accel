create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]
derive_clock_uncertainty
# asynchronous inputs/outputs: UART (synchronised in uart_rx), reset key, LEDs
set_false_path -from [get_ports {UART_RXD KEY[*]}]
set_false_path -to   [get_ports {UART_TXD LED[*]}]
