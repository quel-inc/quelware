proc graceful_quit {retval target msg} {
  if {[string length $target] > 0} {
    close_hw_target $target
  }
  disconnect_hw_server
  close_hw_manager
  puts stderr $msg
  quit $retval
}

if {$argc < 1} {
  puts "usage: $argv0 ADAPTER_ID"
  quit 1
}

# connecting to hw_server
open_hw_manager -quiet
connect_hw_server -allow_non_jtag -quiet -url [lindex $argv 0]

# finding a target with adapter id
set adapter "*/"
append adapter [lindex $argv 1]
set target [get_hw_targets -quiet $adapter]
if {[string equal $target ""]} {
  graceful_quit 1 $target "ERROR: cannot find the adapter: $adapter"
}

open_hw_target $target
set dev [current_hw_device [get_hw_devices -quiet {*}]]
if {[string equal $dev ""]} {
  graceful_quit 1 $target "ERROR: the specified target looks invalid"
}

# reboot!
boot_hw_device [lindex $dev 0]
graceful_quit 0 $target "INFO: Rebooted successfully"
