open_hw_manager
connect_hw_server -allow_non_jtag -url [lindex $argv 1]
set adapter ""
append adapter "*/" [lindex $argv 2]
set target [get_hw_targets -quiet $adapter]
if {[string equal $target ""]} {
  disconnect_hw_server
  close_hw_manager
  quit
}

open_hw_target $target
set dev [get_hw_devices -quiet {*xcu50*}]
if {[string equal $dev ""]} {
 close_hw_target $target
 disconnect_hw_server
 close_hw_manager
 quit
}

current_hw_device $dev
set_property PROGRAM.FILE [lindex $argv 0] [current_hw_device]
program_hw_devices [current_hw_device]

close_hw_target $target
disconnect_hw_server
close_hw_manager
quit
