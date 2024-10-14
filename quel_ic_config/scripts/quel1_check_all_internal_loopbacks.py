import argparse
import logging
from ipaddress import ip_address
from typing import Any, Dict, Final, Mapping, Set, Tuple, Union

import matplotlib

from quel_ic_config import QUEL1_BOXTYPE_ALIAS, CaptureReturnCode, Quel1BoxType
from quel_ic_config_utils.common_arguments import complete_ipaddrs
from testlibs.general_looptest_common_updated import (
    BoxPool,
    PulseCap,
    PulseGen,
    VportSettingType,
    find_chunks,
    plot_iqs,
)

logger = logging.getLogger()


DEFAULT_PULSE_DETECTION_THRESHOLD: Final[float] = 2000.0


CAP_VPORT_SETTINGS_QUEL1A: Dict[str, Mapping[str, VportSettingType]] = {
    "READ0": {
        "create": {
            "boxname": "BOX0",
            "port": 0,  # (0, "r")
            "runits": {0},
        },
        "config": {
            "lo_freq": 8.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [2048],
                "num_blank_samples": [4],
            },
        },
    },
    "READ1": {
        "create": {
            "boxname": "BOX0",
            "port": 7,  # (1, "r")
            "runits": {0},
        },
        "config": {
            "lo_freq": 8.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [2048],
                "num_blank_samples": [4],
            },
        },
    },
    "MON0": {
        "create": {
            "boxname": "BOX0",
            "port": 5,  # (0, "m")
            "runits": {0},
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [2048],
                "num_blank_samples": [4],
            },
        },
    },
    "MON1": {
        "create": {
            "boxname": "BOX0",
            "port": 12,  # (1, "m")
            "runits": {0},
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [2048],
                "num_blank_samples": [4],
            },
        },
    },
}

GEN_VPORT_SETTINGS_QUEL1A: Dict[str, Mapping[str, VportSettingType]] = {
    "GEN01": {
        "create": {
            "boxname": "BOX0",
            "port": 1,
            "channel": 0,
        },
        "config": {
            # "lo_freq": 8.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "U",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN02": {
        "create": {
            "boxname": "BOX0",
            "port": 2,  # ctrl
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN03": {
        "create": {
            "boxname": "BOX0",
            "port": 3,  # pump-#0
            "channel": 0,
        },
        "config": {
            # "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (512, 0),
        },
    },
    "GEN04": {
        "create": {
            "boxname": "BOX0",
            "port": 4,  # ctrl
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (1024, 0),
        },
    },
    "GEN08": {
        "create": {
            "boxname": "BOX0",
            "port": 8,  # readout
            "channel": 0,
        },
        "config": {
            # "lo_freq": 8.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "U",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN09": {
        "create": {
            "boxname": "BOX0",
            "port": 9,  # ctrl
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN10": {
        "create": {
            "boxname": "BOX0",
            "port": 10,  # pump
            "channel": 0,
        },
        "config": {
            # "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (512, 0),
        },
    },
    "GEN11": {
        "create": {
            "boxname": "BOX0",
            "port": 11,  # ctrl
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (1024, 0),
        },
    },
}

CAP_VPORT_SETTINGS_QUEL1B: Dict[str, Mapping[str, VportSettingType]] = {
    "MON0": {
        "create": {
            "boxname": "BOX0",
            "port": 5,  # (0, "m")
            "runits": {0},
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [3072],
                "num_blank_samples": [4],
            },
        },
    },
    "MON1": {
        "create": {
            "boxname": "BOX0",
            "port": 12,  # (1, "m")
            "runits": {0},
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [3072],
                "num_blank_samples": [4],
            },
        },
    },
}

GEN_VPORT_SETTINGS_QUEL1B: Dict[str, Mapping[str, VportSettingType]] = {
    "GEN01": {
        "create": {
            "boxname": "BOX0",
            "port": 1,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN02": {
        "create": {
            "boxname": "BOX0",
            "port": 2,
            "channel": 0,
        },
        "config": {
            # "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (512, 0),
        },
    },
    "GEN03": {
        "create": {
            "boxname": "BOX0",
            "port": 3,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (1024, 0),
        },
    },
    "GEN04": {
        "create": {
            "boxname": "BOX0",
            "port": 4,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (1536, 0),
        },
    },
    "GEN08": {
        "create": {
            "boxname": "BOX0",
            "port": 8,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN09": {
        "create": {
            "boxname": "BOX0",
            "port": 9,
            "channel": 0,
        },
        "config": {
            # "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (512, 0),
        },
    },
    "GEN10": {
        "create": {
            "boxname": "BOX0",
            "port": 10,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (1024, 0),
        },
    },
    "GEN11": {
        "create": {
            "boxname": "BOX0",
            "port": 11,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (1536, 0),
        },
    },
}

CAP_VPORT_SETTINGS_QUBERIKENA: Dict[str, Mapping[str, VportSettingType]] = {
    "READ0": {
        "create": {
            "boxname": "BOX0",
            "port": 1,  # (0, "r")
            "runits": {0},
        },
        "config": {
            "lo_freq": 8.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [2048],
                "num_blank_samples": [4],
            },
        },
    },
    "READ1": {
        "create": {
            "boxname": "BOX0",
            "port": 12,  # (1, "r")
            "runits": {0},
        },
        "config": {
            "lo_freq": 8.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [2048],
                "num_blank_samples": [4],
            },
        },
    },
    "MON0": {
        "create": {
            "boxname": "BOX0",
            "port": 4,  # (0, "m")
            "runits": {0},
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [2048],
                "num_blank_samples": [4],
            },
        },
    },
    "MON1": {
        "create": {
            "boxname": "BOX0",
            "port": 9,  # (1, "m")
            "runits": {0},
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [2048],
                "num_blank_samples": [4],
            },
        },
    },
}

GEN_VPORT_SETTINGS_QUBERIKENA: Dict[str, Mapping[str, VportSettingType]] = {
    "GEN00": {
        "create": {
            "boxname": "BOX0",
            "port": 0,  # readout
            "channel": 0,
        },
        "config": {
            # "lo_freq": 8.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "U",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN02": {
        "create": {
            "boxname": "BOX0",
            "port": 2,  # pump
            "channel": 0,
        },
        "config": {
            # "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN05": {
        "create": {
            "boxname": "BOX0",
            "port": 5,  # ctrl
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (512, 0),
        },
    },
    "GEN06": {
        "create": {
            "boxname": "BOX0",
            "port": 6,  # ctrl
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (1024, 0),
        },
    },
    "GEN13": {
        "create": {
            "boxname": "BOX0",
            "port": 13,  # readout
            "channel": 0,
        },
        "config": {
            # "lo_freq": 8.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "U",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN11": {
        "create": {
            "boxname": "BOX0",
            "port": 11,  # pump
            "channel": 0,
        },
        "config": {
            # "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN08": {
        "create": {
            "boxname": "BOX0",
            "port": 8,  # ctrl
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (512, 0),
        },
    },
    "GEN07": {
        "create": {
            "boxname": "BOX0",
            "port": 7,  # ctrl
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (1024, 0),
        },
    },
}

CAP_VPORT_SETTINGS_QUBERIKENB: Dict[str, Mapping[str, VportSettingType]] = {
    "MON0": {
        "create": {
            "boxname": "BOX0",
            "port": 4,  # (0, "m")
            "runits": {0},
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [3072],
                "num_blank_samples": [4],
            },
        },
    },
    "MON1": {
        "create": {
            "boxname": "BOX0",
            "port": 9,  # (1, "m")
            "runits": {0},
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "rfswitch": "loop",
        },
        "simple_parameters": {
            0: {
                "num_delay_sample": 0,
                "num_integration_section": 1,
                "num_capture_samples": [3072],
                "num_blank_samples": [4],
            },
        },
    },
}

GEN_VPORT_SETTINGS_QUBERIKENB: Dict[str, Mapping[str, VportSettingType]] = {
    "GEN00": {
        "create": {
            "boxname": "BOX0",
            "port": 0,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN02": {
        "create": {
            "boxname": "BOX0",
            "port": 2,
            "channel": 0,
        },
        "config": {
            # "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (512, 0),
        },
    },
    "GEN05": {
        "create": {
            "boxname": "BOX0",
            "port": 5,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (1024, 0),
        },
    },
    "GEN06": {
        "create": {
            "boxname": "BOX0",
            "port": 6,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (1536, 0),
        },
    },
    "GEN13": {
        "create": {
            "boxname": "BOX0",
            "port": 13,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (0, 0),
        },
    },
    "GEN11": {
        "create": {
            "boxname": "BOX0",
            "port": 11,
            "channel": 0,
        },
        "config": {
            # "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (512, 0),
        },
    },
    "GEN08": {
        "create": {
            "boxname": "BOX0",
            "port": 8,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 128,
            "num_repeats": (1, 1),
            "num_wait_samples": (1024, 0),
        },
    },
    "GEN07": {
        "create": {
            "boxname": "BOX0",
            "port": 7,
            "channel": 0,
        },
        "config": {
            "lo_freq": 11.5e9,
            "cnco_freq": 1.5e9,
            "fnco_freq": 0.0,
            "fullscale_current": 40000,
            "sideband": "L",
            "vatt": 0xC00,
        },
        "cw_parameter": {
            "amplitude": 32767.0,
            "num_wave_sample": 64,
            "num_repeats": (1, 1),
            "num_wait_samples": (1536, 0),
        },
    },
}


def parse_boxtype(boxtypename: str) -> Quel1BoxType:
    if boxtypename not in QUEL1_BOXTYPE_ALIAS:
        raise ValueError
    return Quel1BoxType.fromstr(boxtypename)


def single_schedule(cp: PulseCap, pg_trigger: PulseGen, pgs: Set[PulseGen], boxpool: BoxPool, power_thr: float):
    if pg_trigger not in pgs:
        raise ValueError("trigerring pulse generator is not included in activated pulse generators")
    thunk = cp.capture_at_single_trigger_of(pg=pg_trigger)
    boxpool.emit_at(cp=cp, pgs=pgs, min_time_offset=125_000_000, time_counts=(0,))

    s0, iqs = thunk.result()
    iq0 = iqs[0]
    assert s0 == CaptureReturnCode.SUCCESS
    chunks = find_chunks(iq0, power_thr=power_thr)
    return iq0, chunks


class LoopbackTest:
    def __init__(
        self,
        cap_vport_settings: Dict[str, Mapping[str, VportSettingType]],
        gen_vport_settings: Dict[str, Mapping[str, VportSettingType]],
        cap_gen_map: Dict[str, Tuple[int, ...]],
    ):
        self.cap_vport_settings: Dict[str, Mapping[str, VportSettingType]] = cap_vport_settings
        self.gen_vport_settings: Dict[str, Mapping[str, VportSettingType]] = gen_vport_settings
        self.cap_gen_map: Dict[str, Tuple[int, ...]] = cap_gen_map
        self.boxpool: Union[BoxPool, None] = None
        self.cps: Dict[str, PulseCap] = {}
        self.pgs: Dict[str, PulseGen] = {}

    def initialize(self, boxpool):
        if self.boxpool is not None:
            raise RuntimeError("device handles are already created")
        self.boxpool = boxpool
        self.cps = PulseCap.create(self.cap_vport_settings, self.boxpool)
        self.pgs = PulseGen.create(self.gen_vport_settings, self.boxpool)

    @property
    def pulsecap(self):
        return self.cps

    @property
    def pulsegen(self):
        return self.pgs

    def _do_background_check(self, cp: PulseCap, power_thr: float) -> bool:
        noise_max, noise_avg, _ = cp.measure_background_noise()
        return noise_max < power_thr * 0.75

    def do_test(self, power_thr: float):
        if self.boxpool is None:
            raise RuntimeError("not initialized yet")

        # Notes: it is fine to measure time diff with a single input port because all the input ports belong to the same
        #        box in this script.
        self.boxpool.measure_timediff(self.cps[list(self.cap_gen_map.keys())[0]])

        for capname, genidxs in self.cap_gen_map.items():
            cp = self.cps[capname]

            if not self._do_background_check(cp, power_thr):
                logger.warning(
                    f"the input port-#{cp.port:02d} is too noise for the given power threshold of pulse detection, "
                    "you may see sprious pulses in the results"
                )

            idx0 = genidxs[0]
            iqs, chunks = single_schedule(
                cp,
                self.pgs[f"GEN{idx0:02d}"],
                {self.pgs[f"GEN{idx:02d}"] for idx in genidxs},
                self.boxpool,
                power_thr=power_thr,
            )
            if len(chunks) != len(genidxs):
                logger.error(
                    f"the number of pulses captured by {capname} is expected to be {len(genidxs)} "
                    f"but is actually {len(chunks)}, something wrong"
                )

            ll = [f"{capname}: "]
            for idx in genidxs:
                pgname = f"GEN{idx:02d}"
                port = self.pgs[pgname].port
                ll.append(f"port-#{port:02d}")
            label = " ".join(ll)
            plot_iqs({label: iqs})


TEST_CONFIGS = {
    Quel1BoxType.QuEL1_TypeA: LoopbackTest(
        CAP_VPORT_SETTINGS_QUEL1A,
        GEN_VPORT_SETTINGS_QUEL1A,
        {
            "READ0": (1,),
            "MON0": (2, 3, 4),
            "READ1": (8,),
            "MON1": (9, 10, 11),
        },
    ),
    Quel1BoxType.QuEL1_TypeB: LoopbackTest(
        CAP_VPORT_SETTINGS_QUEL1B,
        GEN_VPORT_SETTINGS_QUEL1B,
        {
            "MON0": (1, 2, 3, 4),
            "MON1": (8, 9, 10, 11),
        },
    ),
    Quel1BoxType.QuBE_RIKEN_TypeA: LoopbackTest(
        CAP_VPORT_SETTINGS_QUBERIKENA,
        GEN_VPORT_SETTINGS_QUBERIKENA,
        {
            "READ0": (0,),
            "MON0": (2, 5, 6),
            "READ1": (13,),
            "MON1": (11, 8, 7),
        },
    ),
    Quel1BoxType.QuBE_RIKEN_TypeB: LoopbackTest(
        CAP_VPORT_SETTINGS_QUBERIKENB,
        GEN_VPORT_SETTINGS_QUBERIKENB,
        {
            "MON0": (0, 2, 5, 6),
            "MON1": (13, 11, 8, 7),
        },
    ),
}

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    for lgrname, lgr in logging.root.manager.loggerDict.items():
        if lgrname in {"root"}:
            pass
        elif lgrname.startswith("testlibs."):
            pass
        else:
            if isinstance(lgr, logging.Logger):
                lgr.setLevel(logging.WARNING)

    matplotlib.use("Gtk3agg")

    parser = argparse.ArgumentParser("observing signals from all the output port via internal loop-back paths")
    parser.add_argument(
        "--ipaddr_clk",
        type=ip_address,
        required=True,
        help="IP address of clock master",
    )
    parser.add_argument(
        "--ipaddr_wss",
        type=ip_address,
        required=True,
        help="IP address of the wave generation/capture subsystem of the target box",
    )
    parser.add_argument(
        "--ipaddr_sss",
        type=ip_address,
        default=0,
        help="IP address of the synchronization subsystem of the target box",
    )
    parser.add_argument(
        "--ipaddr_css",
        type=ip_address,
        default=0,
        help="IP address of the configuration subsystem of the target box",
    )
    parser.add_argument(
        "--boxtype",
        type=parse_boxtype,
        required=True,
        help=f"a type of the target box A: either of "
        f"{', '.join([t for t in QUEL1_BOXTYPE_ALIAS if not t.startswith('x_')])}",
    )
    parser.add_argument(
        "--pulse_detection_threshold",
        type=float,
        default=DEFAULT_PULSE_DETECTION_THRESHOLD,
        help="threshold of amplitude for detecting the pulses, "
        f"default value (= {DEFAULT_PULSE_DETECTION_THRESHOLD}) is fine for QuEL-1",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)

    if args.boxtype not in TEST_CONFIGS:
        logger.error(f"boxtype '{Quel1BoxType.tostr(args.boxtype)}' is not supported")
        sys.exit(1)

    DEVICE_SETTINGS: Dict[str, Mapping[str, Any]] = {
        "CLOCK_MASTER": {
            "ipaddr": str(args.ipaddr_clk),
        },
        "BOX0": {
            "ipaddr_wss": str(args.ipaddr_wss),
            "ipaddr_sss": str(args.ipaddr_sss),
            "ipaddr_css": str(args.ipaddr_css),
            "boxtype": args.boxtype,
            "ignore_crc_error_of_mxfe": {0, 1},
        },
    }

    boxpool0 = BoxPool(DEVICE_SETTINGS)
    boxpool0.init(resync=False)

    test_config = TEST_CONFIGS[args.boxtype]
    test_config.initialize(boxpool0)
    test_config.do_test(args.pulse_detection_threshold)
