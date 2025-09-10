Some CLI commands utilize a system configuration file.
In this file, you can declare the configuration of your system, which consists of multiple QuEL devices.
The file format is YAML, and its structure is as follows:

```
version: 2

clockmaster:
  - ipaddr: 10.3.0.200

boxes:
  - name: box167
    ipaddr: 10.1.0.167
    boxtype: quel1se-riken8
  - name: box171
    ipaddr: 10.1.0.171
    boxtype: quel1se-riken8
  - name: box092
    ipaddr: 10.1.0.92
    boxtype: quel1se-riken8
  - ...
```

This file is searched for in `$XDG_CONFIG_HOME/quelware/sysconf.yaml`. The environment variable `$XDG_CONFIG_HOME` typically defaults to `~/.config`.
If you place this file in the default location, some commands will use it by default, even without a specified path.
