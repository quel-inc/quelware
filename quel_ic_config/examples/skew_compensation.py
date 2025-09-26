#
# This script demonstrates skew adjustment using deskew_tools; it performs no meaningful task.
#
import json
import logging
from os.path import expanduser

import numpy as np

import quel_ic_config as qi
from quel_ic_config_utils import configuration, deskew_tools

TARGET_BOXES = ["staging-167", "staging-171"]

SOURCE_PORT = 3
SOURCE_CHANNEL = 0
DEST_PORT = 4
DEST_RUNIT = 0

WAVEDATA = (2**15 - 1) * np.ones((64,), dtype=np.complex64)

logging.basicConfig(format=logging.BASIC_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sysconf = configuration.load_default_configuration()
name_to_box = {
    box.name: box
    for box in configuration.get_boxes_in_parallel(filter(lambda b: b.name in TARGET_BOXES, sysconf.boxes))
}
configuration.reconnect_and_get_link_status_in_parallel(name_to_box.values())

for box_name in TARGET_BOXES:
    name_to_box[box_name].register_wavedata(SOURCE_PORT, 0, "testwave", WAVEDATA)
    name_to_box[box_name].register_wavedata(SOURCE_PORT, 0, "testwave", WAVEDATA)

with open(expanduser("~/") + ".config/quelware/deskew.json") as io:
    obj = json.load(io)
    deskew_config = deskew_tools.DeskewConfiguration.model_validate(obj)
# or
# deskew_config = deskew_tools.load_default_configuration()  # typically loads ~/.config/quelware/deskew.json

wait_amount_resolver = deskew_tools.WaitAmountResolver.from_deskew_configuration(deskew_config)
count_proposer = deskew_tools.StableCountProposer.from_deskew_configuration(deskew_config)
delay_compensator = deskew_tools.E7awgDelayCompensator()
awg_param = qi.AwgParam(num_wait_word=0)
awg_param.chunks.append(qi.WaveChunk(name_of_wavedata="testwave"))

for box_name in TARGET_BOXES:
    deskew_tools.register_blank_wavedata(name_to_box[box_name], SOURCE_PORT, SOURCE_CHANNEL)
    adjusted_awg_param_source = delay_compensator.adjust_awg_param(
        awg_param, wait_amount_resolver.get_word_to_wait(box_name, SOURCE_PORT)
    )
    name_to_box[box_name].config_channel(SOURCE_PORT, SOURCE_CHANNEL, awg_param=adjusted_awg_param_source)

cap_length_word = 200
cap_param = qi.CapParam(num_wait_word=0)
cap_param.sections.append(
    qi.CapSection(
        name="capsec",
        num_capture_word=cap_length_word,
        num_blank_word=1,
    )
)

for box_name in TARGET_BOXES:
    adjusted_cap_param = delay_compensator.adjust_cap_param(
        cap_param, wait_amount_resolver.get_word_to_wait(box_name, DEST_PORT)
    )
    name_to_box[box_name].config_runit(DEST_PORT, 0, capture_param=adjusted_cap_param)

trigger_counts = count_proposer.propose_trigger_counts(
    name_to_box[TARGET_BOXES[0]].get_current_timecounter(),
    TARGET_BOXES,
    delay_sec=0.2,
)

name_to_captask = {}
name_to_gentask = {}
for box_name in TARGET_BOXES:
    capture_task, wavegen_task = name_to_box[box_name].start_capture_by_awg_trigger(
        {(DEST_PORT, DEST_RUNIT)}, {(SOURCE_PORT, SOURCE_CHANNEL)}, trigger_counts[box_name]
    )
    name_to_captask[box_name] = capture_task
    name_to_gentask[box_name] = wavegen_task

for box_name, gentask in name_to_gentask.items():
    gentask.result()

for box_name, captask in name_to_captask.items():
    iq_readers = captask.result()

    wave_dict = deskew_tools.extract_wave_dict(iq_readers[DEST_PORT, DEST_RUNIT])
    # or
    wave_list = deskew_tools.extract_wave_list(iq_readers[DEST_PORT, DEST_RUNIT])
