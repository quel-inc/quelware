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
  puts "usage: $argv0 MCSFILE SERVER_URI ADAPTER_ID"
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
set dev [current_hw_device [get_hw_devices -quiet {*xcu50*}]]
if {[string equal $dev ""]} {
  close_hw_target $target
  graceful_quit 1 $target "ERROR: the specified target looks invalid"
}

create_hw_cfgmem -hw_device $dev [lindex [get_cfgmem_parts {mt25qu01g-spi-x1_x2_x4}] 0]
set_property PROGRAM.ADDRESS_RANGE  {use_file} [ get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.FILES [lindex $argv 0] [ get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.PRM_FILE {} [ get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.UNUSED_PIN_TERMINATION {pull-none} [ get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.BLANK_CHECK 0 [ get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.ERASE       1 [ get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.CFG_PROGRAM 1 [ get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.VERIFY      1 [ get_property PROGRAM.HW_CFGMEM $dev]
set_property PROGRAM.CHECKSUM    0 [ get_property PROGRAM.HW_CFGMEM $dev]

create_hw_bitstream -hw_device $dev [get_property PROGRAM.HW_CFGMEM_BITFILE $dev]
program_hw_devices $dev
program_hw_cfgmem -hw_cfgmem [ get_property PROGRAM.HW_CFGMEM $dev]

close_hw_target $target
graceful_quit 0 $target "XINFO: programming mcs into $adapter is completed successfully"
