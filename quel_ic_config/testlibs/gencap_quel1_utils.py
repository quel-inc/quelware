from quel_ic_config.quel1_box_intrinsic import Quel1BoxIntrinsic


def config_lines(boxi: Quel1BoxIntrinsic):
    # AWG: 10.0 GHz
    boxi.config_line(
        0, 0, cnco_freq=1.5e9, fullscale_current=40527, lo_freq=8.5e9, sideband="U", vatt=0xA00, rfswitch="block"
    )
    boxi.config_channel(0, 0, 0, fnco_freq=0.0)

    # AWG: 9.5GHz, 9.7GHz, 9.3GHz
    boxi.config_line(
        0, 3, cnco_freq=2.0e9, fullscale_current=40527, lo_freq=11.5e9, sideband="L", vatt=0xA00, rfswitch="block"
    )
    boxi.config_channel(0, 3, 0, fnco_freq=0.0)
    boxi.config_channel(0, 3, 1, fnco_freq=-200.0e6)
    boxi.config_channel(0, 3, 2, fnco_freq=200.0e6)

    # AWG: 9.0GHz, 9.2GHz, 8.8GHz
    boxi.config_line(
        1, 3, cnco_freq=2.5e9, fullscale_current=40527, lo_freq=11.5e9, sideband="L", vatt=0xA00, rfswitch="block"
    )
    boxi.config_channel(1, 3, 0, fnco_freq=0.0)
    boxi.config_channel(1, 3, 1, fnco_freq=-200.0e6)
    boxi.config_channel(1, 3, 2, fnco_freq=200.0e6)


def config_rlines(boxi: Quel1BoxIntrinsic):
    # 10GHz
    boxi.config_rline(0, "r", cnco_freq=1.5e9, lo_freq=8.5e9, rfswitch="loop")

    # 9.5GHz
    boxi.config_rline(0, "m", cnco_freq=2.0e9, lo_freq=11.5e9, rfswitch="loop")

    # 9.0GHz
    boxi.config_rline(1, "m", cnco_freq=2.5e9, lo_freq=11.5e9, rfswitch="loop")
