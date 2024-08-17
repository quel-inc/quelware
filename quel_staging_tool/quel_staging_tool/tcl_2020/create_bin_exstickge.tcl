write_cfgmem -force -format bin -interface SPIx4 -size 16 -loadbit "up 0x0 [lindex $argv 0]" -file [lindex $argv 1]
