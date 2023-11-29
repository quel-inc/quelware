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
set dev [current_hw_device [get_hw_devices -quiet {*7a200t*}]]
if {[string equal $dev ""]} {
 close_hw_target $target
 disconnect_hw_server
 close_hw_manager
 quit
}

create_hw_cfgmem -hw_device $dev [lindex [get_cfgmem_parts {mt25ql128-spi-x1_x2_x4}] 0]
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
disconnect_hw_server
close_hw_manager
quit
