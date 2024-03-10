write_cfgmem -force -format mcs -interface SPIx4 -size 16 -loadbit [list up 0x0 [lindex $argv 0]] -loaddata [list up 0xFFFFFA [lindex $argv 1]] -file [lindex $argv 2]
