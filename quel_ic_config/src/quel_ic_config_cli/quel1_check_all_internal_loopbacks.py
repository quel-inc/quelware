import argparse
import logging
import sys
from collections.abc import Collection, Mapping
from ipaddress import ip_address
from typing import Any, Final, Optional, cast

from e7awghal import CapIqDataReader

from quel_ic_config import QUEL1_BOXTYPE_ALIAS, Quel1BoxType, Quel1PortType
from quel_ic_config_utils import (
    BoxPool,
    BoxSettingType,
    VportSettingType,
    VportTypicalSettingType,
    complete_ipaddrs,
    find_chunks,
    plot_iqs,
)

logger = logging.getLogger()


DEFAULT_PULSE_DETECTION_THRESHOLD: Final[float] = 2000.0
DEFAULT_PULSE_DETECTION_THRESHOLD_FUJITSU11_MON: Final[float] = 1200.0


CAP_VPORT_SETTINGS_QUEL1SE_FUJITSU11_A: dict[str, dict[str, VportSettingType]] = {
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD_FUJITSU11_MON,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD_FUJITSU11_MON,
        },
    },
}

CAP_VPORT_SETTINGS_QUEL1SE_FUJITSU11_B: dict[str, dict[str, VportSettingType]] = {
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD_FUJITSU11_MON,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD_FUJITSU11_MON,
        },
    },
}

GEN_VPORT_SETTINGS_QUEL1SE_FUJITSU11_B: dict[str, dict[str, VportSettingType]] = {
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

CAP_VPORT_SETTINGS_QUEL1A: dict[str, dict[str, VportSettingType]] = {
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
        },
    },
}

GEN_VPORT_SETTINGS_QUEL1A: dict[str, dict[str, VportSettingType]] = {
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

CAP_VPORT_SETTINGS_QUEL1B: dict[str, dict[str, VportSettingType]] = {
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
        },
    },
}

GEN_VPORT_SETTINGS_QUEL1B: dict[str, dict[str, VportSettingType]] = {
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

CAP_VPORT_SETTINGS_QUBERIKENA: dict[str, dict[str, VportSettingType]] = {
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
        },
    },
}

GEN_VPORT_SETTINGS_QUBERIKENA: dict[str, dict[str, VportSettingType]] = {
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

CAP_VPORT_SETTINGS_QUBERIKENB: dict[str, dict[str, VportSettingType]] = {
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
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
        "check": {
            "pulse_threshold": DEFAULT_PULSE_DETECTION_THRESHOLD,
        },
    },
}

GEN_VPORT_SETTINGS_QUBERIKENB: dict[str, dict[str, VportSettingType]] = {
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


class LoopbackTest:
    def __init__(
        self,
        cm_setting: Mapping[str, Any],
        box_settings: Mapping[str, BoxSettingType],
        cap_vport_settings: Mapping[str, Mapping[str, VportSettingType]],
        gen_vport_settings: Mapping[str, Mapping[str, VportSettingType]],
        cap_gen_map: Mapping[str, tuple[int, ...]],
    ):
        self.cap_vport_settings: Mapping[str, Mapping[str, VportSettingType]] = cap_vport_settings
        self.gen_vport_settings: Mapping[str, Mapping[str, VportSettingType]] = gen_vport_settings
        self.cap_gen_map: Mapping[str, tuple[int, ...]] = cap_gen_map
        self.boxpool = BoxPool(cm_setting, box_settings, cap_vport_settings, gen_vport_settings)
        self.cps = set(self.cap_vport_settings.keys())
        self.pgs = set(self.gen_vport_settings.keys())

    def initialize(self):
        self.boxpool.initialize(allow_resync=False)

    def _do_background_check(self, pulse_threshold: Optional[float] = None) -> None:
        bgnoise = self.boxpool.check_background_noise(self.cps)
        for cpname, bgnoise_max in bgnoise.items():
            check_setting = cast(VportTypicalSettingType, self.cap_vport_settings[cpname]["check"])
            thr: float = pulse_threshold or check_setting["pulse_threshold"]
            if bgnoise_max > thr * 0.75:
                logger.warning(
                    f"the runit '{cpname}' is too noisy for the given power threshold of pulse detection (= {thr})"
                    "you may see spurious pulses in the results"
                )

    def _single_schedule(self, cp: str, pgs: Collection[str], power_thr: float):
        _, c_tasks, g_tasks = self.boxpool.start_at(runit_name=cp, channel_names=pgs, min_time_offset=125_000_000 // 10)
        for boxname, g_task in g_tasks.items():
            g_task.result()
            logger.info(f"wave generation of box {boxname} is completed")

        rdrs: dict[str, dict[tuple[Quel1PortType, int], CapIqDataReader]] = {}
        for boxname, c_task in c_tasks.items():
            rdrs[boxname] = c_task.result()
            logger.info(f"capture of box {boxname} is completed")

        if len(rdrs) != 1:
            raise AssertionError("too much reader objects...")

        cp_box, cp_port, cp_runits = self.boxpool._runits[cp]

        rdr = rdrs[cp_box]
        iq = rdr[cp_port, list(cp_runits)[0]].as_wave_dict()["s0"][0]
        chunks = find_chunks(iq, power_thr=power_thr)
        return iq, chunks

    def _test_loopback(self, cp: str, pgs: Collection[str], power_thr: float):
        # check_background_noise(cp, power_thr)
        iq, chunks = self._single_schedule(cp, pgs, power_thr)
        if len(chunks) != len(pgs):
            logger.error(
                f"the number of pulses captured by the runit '{cp}` is "
                f"expected to be {len(pgs)} but is actually {len(chunks)}, something wrong"
            )
        return iq, chunks

    def do_test(self, pulse_detection_threshold: Optional[float] = None):
        self._do_background_check()

        # Notes: it is fine to measure time diff with a single input port because all the input ports belong to the same
        #        box in this script.
        self.boxpool.measure_timediff(list(self.cps)[0])

        epochs = {}
        for capname, genidxs in self.cap_gen_map.items():
            gennames = [f"GEN{idx:02d}" for idx in genidxs]
            thr: float = cast(VportTypicalSettingType, self.cap_vport_settings[capname]["check"])["pulse_threshold"]
            iqs, chunks = self._test_loopback(
                capname,
                gennames,
                cast(float, pulse_detection_threshold or thr),
            )

            ll = [f"{capname}: "]
            ll.extend(gennames)
            label = " ".join(ll)
            epochs[label] = iqs

        plot_iqs(epochs, same_range=False)


TEST_CONFIGS = {
    Quel1BoxType.QuEL1SE_FUJITSU11_TypeA: lambda cms, bxs: LoopbackTest(
        cms,
        bxs,
        CAP_VPORT_SETTINGS_QUEL1SE_FUJITSU11_A,
        GEN_VPORT_SETTINGS_QUEL1A,
        {
            "READ0": (1,),
            "MON0": (2, 3, 4),
            "READ1": (8,),
            "MON1": (9, 10, 11),
        },
    ),
    Quel1BoxType.QuEL1SE_FUJITSU11_TypeB: lambda cms, bxs: LoopbackTest(
        cms,
        bxs,
        CAP_VPORT_SETTINGS_QUEL1SE_FUJITSU11_B,
        GEN_VPORT_SETTINGS_QUEL1SE_FUJITSU11_B,
        {
            "MON0": (1, 2, 3, 4),
            "MON1": (8, 9, 10, 11),
        },
    ),
    Quel1BoxType.QuEL1_TypeA: lambda cms, bxs: LoopbackTest(
        cms,
        bxs,
        CAP_VPORT_SETTINGS_QUEL1A,
        GEN_VPORT_SETTINGS_QUEL1A,
        {
            "READ0": (1,),
            "MON0": (2, 3, 4),
            "READ1": (8,),
            "MON1": (9, 10, 11),
        },
    ),
    Quel1BoxType.QuEL1_TypeB: lambda cms, bxs: LoopbackTest(
        cms,
        bxs,
        CAP_VPORT_SETTINGS_QUEL1B,
        GEN_VPORT_SETTINGS_QUEL1B,
        {
            "MON0": (1, 2, 3, 4),
            "MON1": (8, 9, 10, 11),
        },
    ),
    Quel1BoxType.QuBE_RIKEN_TypeA: lambda cms, bxs: LoopbackTest(
        cms,
        bxs,
        CAP_VPORT_SETTINGS_QUBERIKENA,
        GEN_VPORT_SETTINGS_QUBERIKENA,
        {
            "READ0": (0,),
            "MON0": (2, 5, 6),
            "READ1": (13,),
            "MON1": (11, 8, 7),
        },
    ),
    Quel1BoxType.QuBE_RIKEN_TypeB: lambda cms, bxs: LoopbackTest(
        cms,
        bxs,
        CAP_VPORT_SETTINGS_QUBERIKENB,
        GEN_VPORT_SETTINGS_QUBERIKENB,
        {
            "MON0": (0, 2, 5, 6),
            "MON1": (13, 11, 8, 7),
        },
    ),
}


def main():
    logging.basicConfig(level=logging.INFO, format="{asctime} [{levelname:.4}] {name}: {message}", style="{")
    for lgrname, lgr in logging.root.manager.loggerDict.items():
        if lgrname in {"root"}:
            pass
        elif lgrname.startswith("quel_ic_config_utils.simple_multibox_framework"):
            pass
        else:
            if isinstance(lgr, logging.Logger):
                lgr.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser("observing signals from all the output port via internal loop-back paths")
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
        default=None,
        help="threshold of amplitude for detecting the pulses",
    )

    args = parser.parse_args()
    complete_ipaddrs(args)

    if args.boxtype not in TEST_CONFIGS:
        logger.error(f"boxtype '{Quel1BoxType.tostr(args.boxtype)}' is not supported")
        sys.exit(1)

    CLOCKMASTER_SETTINGS: dict[str, Any] = {}

    BOX_SETTINGS: dict[str, BoxSettingType] = {
        "BOX0": {
            "ipaddr_wss": str(args.ipaddr_wss),
            "ipaddr_sss": str(args.ipaddr_sss),
            "ipaddr_css": str(args.ipaddr_css),
            "boxtype": args.boxtype,
            "ignore_crc_error_of_mxfe": {0, 1},
        },
    }

    test_config = TEST_CONFIGS[args.boxtype](CLOCKMASTER_SETTINGS, BOX_SETTINGS)
    test_config.initialize()
    test_config.do_test(pulse_detection_threshold=args.pulse_detection_threshold)

    del test_config


if __name__ == "__main__":
    main()
