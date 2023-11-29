write_cfgmem -force -format mcs -size 16 -interface SPIx4 -loadbit [list up 0x00000000 [lindex $argv 0]] -file [lindex $argv 1]
