# Gate-level power of a post-route netlist from simulated switching activity. Inputs come
# from env LIB (typical-corner liberty), ODB, SDC, SPEF (nominal), VCD and SCOPE (the DUT's
# path in the VCD). It writes the annotation reports, power.rpt with the design totals by
# group, and array_power.rpt with every cell under u_array/, which flow/parse_power.py sums.
# All of it goes to OUTDIR.
read_liberty $::env(LIB)
read_db $::env(ODB)
read_sdc $::env(SDC)
read_spef $::env(SPEF)
set_propagated_clock [all_clocks]
read_vcd -scope $::env(SCOPE) $::env(VCD)
report_activity_annotation > $::env(OUTDIR)/annotation.rpt
report_activity_annotation -report_unannotated > $::env(OUTDIR)/unannotated.rpt
report_power -digits 9 > $::env(OUTDIR)/power.rpt
report_power -instances [get_cells u_array/*] -digits 12 > $::env(OUTDIR)/array_power.rpt
