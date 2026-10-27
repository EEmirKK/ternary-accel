# Setup slack of every post-route endpoint at one corner, to find which logic limits the
# clock. Inputs come from env ODB, SPEF, LIB and SDC, and OUT gets one "name slack" line per
# endpoint.
read_liberty $::env(LIB)
read_db $::env(ODB)
read_sdc $::env(SDC)
read_spef $::env(SPEF)
set_propagated_clock [all_clocks]
set f [open $::env(OUT) w]
foreach path [find_timing_paths -path_delay max -group_path_count 100000 -endpoint_path_count 1 -path_group clk] {
    set pin [get_property $path endpoint]
    set inst [get_property $pin full_name]
    puts $f "$inst [get_property $path slack]"
}
close $f
