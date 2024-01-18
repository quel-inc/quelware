proc graceful_quit {retval target msg} {
  if {[string length $target] > 0} {
    close_hw_target $target
  }
  disconnect_hw_server
  close_hw_manager
  puts stderr $msg
  quit $retval
}

if {$argc < 3} {
  puts "usage: $argv0 BITFILE SERVER_URI ADAPTER_ID"
  quit 1
}

# connecting to hw_server
open_hw_manager
connect_hw_server -allow_non_jtag -url [lindex $argv 1]

# finding a target with adapter id
set adapter ""
append adapter "*/" [lindex $argv 2]
set target [get_hw_targets -quiet $adapter]
if {[string equal $target ""]} {
  graceful_quit 1 $target "ERROR: cannot find the adapter: $adapter"
}

open_hw_target $target
set dev [get_hw_devices -quiet {*xcu50*}]
if {[string equal $dev ""]} {
 close_hw_target $target
 graceful_quit 1 $target "ERROR: the specified target looks invalid"
}

current_hw_device $dev
set_property PROGRAM.FILE [lindex $argv 0] [current_hw_device]
program_hw_devices [current_hw_device]

# closing
close_hw_target $target
graceful_quit 0 $target "XINFO: programming bit into $adapter is completed successfully"
