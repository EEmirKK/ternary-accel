`timescale 1ns/1ps

// Extra top module for Icarus gate-level runs. It dumps a VCD for the power analysis
// to +vcd=<path>, or nothing without it.

module gl_dump;
    reg [1023:0] fname;
    initial begin
        if ($value$plusargs("vcd=%s", fname)) begin
            $dumpfile(fname);
            $dumpvars(0, accel_top);
        end
    end
endmodule
