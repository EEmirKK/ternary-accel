# Area of a finished LibreLane run, read from its final layout (env ODB) and written
# as JSON (env OUT). Fill, decap and tap cells hold no logic and are left out of the
# cell area, antenna diodes are counted separately, and everything else, including
# clock-tree and resizer buffers, is logic. Logic under u_array is also counted as
# PE-array area. Any instance that isn't a sky130 HD standard cell is an error.

read_db $::env(ODB)
set block [ord::get_db_block]
set dbu [$block getDbUnitsPerMicron]
set um2 [expr {double($dbu) * $dbu}]

foreach k {fill decap tap diode logic array_logic buffer array_buffer} {
    set area($k) 0.0
    set count($k) 0
}
set bad {}
foreach inst [$block getInsts] {
    set m [$inst getMaster]
    set mname [$m getName]
    set iname [$inst getName]
    if {![string match "sky130_fd_sc_hd__*" $mname]} { lappend bad "$iname:$mname"; continue }
    set a [expr {double([$m getWidth]) * [$m getHeight] / $um2}]
    if {[string match "*__fill_*" $mname]} { set cls fill
    } elseif {[string match "*__decap_*" $mname]} { set cls decap
    } elseif {[string match "*__tapvpwrvgnd_*" $mname]} { set cls tap
    } elseif {[string match "*__diode_*" $mname]} { set cls diode
    } else { set cls logic }
    set area($cls) [expr {$area($cls) + $a}]
    incr count($cls)
    set in_array [string match "u_array/*" $iname]
    set is_buf [regexp {__(buf|clkbuf|dlygate|dlymetal|clkdlybuf)} $mname]
    if {$cls eq "logic"} {
        if {$in_array} {
            set area(array_logic) [expr {$area(array_logic) + $a}]
            incr count(array_logic)
        }
        if {$is_buf} {
            set area(buffer) [expr {$area(buffer) + $a}]
            incr count(buffer)
            if {$in_array} {
                set area(array_buffer) [expr {$area(array_buffer) + $a}]
                incr count(array_buffer)
            }
        }
    }
}
if {[llength $bad] > 0} {
    puts "ERROR: non-standard-cell instances: [lrange $bad 0 9]"
    exit 1
}

set core [$block getCoreArea]
set die [$block getDieArea]
set core_a [expr {double([$core dx]) * [$core dy] / $um2}]
set die_a [expr {double([$die dx]) * [$die dy] / $um2}]

set f [open $::env(OUT) w]
puts $f "{"
foreach k {fill decap tap diode logic array_logic buffer array_buffer} {
    puts $f "  \"${k}_area_um2\": [format %.4f $area($k)], \"${k}_count\": $count($k),"
}
puts $f "  \"core_area_um2\": [format %.4f $core_a], \"die_area_um2\": [format %.4f $die_a],"
puts $f "  \"core_w_um\": [format %.3f [expr {double([$core dx])/$dbu}]], \"core_h_um\": [format %.3f [expr {double([$core dy])/$dbu}]],"
puts $f "  \"instances\": [llength [$block getInsts]]"
puts $f "}"
close $f
puts "area written to $::env(OUT)"
