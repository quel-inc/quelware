write_cfgmem -force -format mcs -size 128 -interface SPIx4 -loadbit [list up 0x01002000 [lindex $argv 0]] -file [lindex $argv 1]
