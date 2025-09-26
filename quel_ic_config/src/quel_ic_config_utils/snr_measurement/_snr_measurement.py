import logging

import numpy as np

import quel_ic_config as qi

SAMPLING_RATE = 500_000_000
CLOCK_FREQ = 125_000_000
AMP = 2**15 - 1
CAPTURE_MARGIN_WORD = 256
SAMPLES_PER_WORD = 4
AWG_INIT_BLANK_BASE_WORD = 1024
SYSREF_PERIOD = 2000
SNR_THRESHOLD_DEFAULT = 10

logger = logging.getLogger(__name__)


def measure_snr(
    source_box: qi.Quel1Box,
    source_port: qi.Quel1PortType,
    dest_box: qi.Quel1Box,
    dest_port: qi.Quel1PortType,
) -> float:
    """Measure Signal-to-Noise Ratio (SNR).

    This function generates a random noise signal from a **source_port** and
    attempts to capture it on a **dest_port**. It performs the same process
    for a silence signal. By comparing the power of the captured noise signal
    to the power of the captured silence signal, it calculates the SNR.

    Note that 0-th channel/runit for each port will be used.

    :param source_box: The ``Quel1Box`` instance that generates the signal.
    :param source_port: The port on the ``source_box`` used to transmit the signal.
    :param dest_box: The ``Quel1Box`` instance that captures the signal.
    :param dest_port: The port on the ``capture_box`` used for the capture.
    :returns: Measured SNR in dB.
    """
    rng = np.random.default_rng()
    noise_wavedata = rng.choice((AMP + 1j * AMP, AMP - 1j * AMP, -AMP + 1j * AMP, -AMP - 1j * AMP), 2**16).astype(
        np.complex64
    )
    source_box.register_wavedata(source_port, 0, "noise", noise_wavedata)

    # measures background noise
    logger.debug(f"Measurement of background noise's power on port-{dest_port} of {dest_box.name} will be started.")
    cap_param = qi.CapParam()
    cap_param.sections.append(qi.CapSection(name="capsec", num_capture_word=8192 // SAMPLES_PER_WORD))
    dest_box.config_runit(dest_port, 0, capture_param=cap_param)
    capture_task = dest_box.start_capture_now({(dest_port, 0)})
    iq_readers = capture_task.result()
    iq = iq_readers[dest_port, 0].as_wave_dict()["capsec"][0]

    background_power = np.sum(np.abs(iq / AMP) ** 2)
    logger.debug(
        f"Measurement of background noise's power on port-{dest_port} of {dest_box.name} has been finished. The power is {background_power}."
    )

    # measures signal power
    awg_param_noise = qi.AwgParam(num_wait_word=0)
    awg_param_noise.chunks.append(qi.WaveChunk(name_of_wavedata="noise", num_repeat=0xFFFF_FFFF))
    source_box.config_channel(source_port, 0, awg_param=awg_param_noise)

    source_task = source_box.start_wavegen({(source_port, 0)})

    logger.debug(f"The test signal has started to emit from port-{source_port} of {source_box.name}.")
    logger.debug(f"Measurement of test signal's power on port-{dest_port} of {dest_box.name} will be started.")

    try:
        cap_param = qi.CapParam()
        cap_param.sections.append(qi.CapSection(name="capsec", num_capture_word=8192 // SAMPLES_PER_WORD))
        dest_box.config_runit(dest_port, 0, capture_param=cap_param)
        capture_task = dest_box.start_capture_now({(dest_port, 0)})
        iq_readers = capture_task.result()
        iq = iq_readers[dest_port, 0].as_wave_dict()["capsec"][0]

        signal_power = np.sum(np.abs(iq / AMP) ** 2)
        logger.debug(
            f"Measurement of test signal's power on port-{dest_port} of {dest_box.name} has been finished. The power is {signal_power}."
        )
    except Exception as e:
        raise e
    finally:
        source_task.cancel()
        logger.debug(f"The emission of the test signal from port-{source_port} of {source_box.name} has been stopped.")

    if signal_power - background_power > 0:
        snr = 10 * np.log10((signal_power - background_power) / background_power)
    else:
        snr = -float("inf")
    return snr


def test_continuity(
    source_box: qi.Quel1Box,
    source_port: qi.Quel1PortType,
    dest_box: qi.Quel1Box,
    dest_port: qi.Quel1PortType,
    snr_threshold: float = SNR_THRESHOLD_DEFAULT,
) -> bool:
    """Tests the continuity of a signal path by checking for signal presence.

    This function measures Signal-to-Noise Ratio (SNR) to determine if a signal
    was successfully transmitted and detected.

    :param source_box: The ``Quel1Box`` instance that generates the signal.
    :param source_port: The port on the ``source_box`` used to transmit the signal.
    :param dest_box: The ``Quel1Box`` instance that captures the signal.
    :param dest_port: The port on the ``dest_box`` used for the capture.
    :param snr_threshold: The minimum SNR value (in dB) required for the signal to be
        considered successfully detected.
    :returns: A boolean value: ``True`` if the calculated SNR is greater than or
        equal to the specified ``snr_threshold``, indicating a detected signal;
        ``False`` otherwise.
    """
    snr = measure_snr(
        source_box=source_box,
        source_port=source_port,
        dest_box=dest_box,
        dest_port=dest_port,
    )
    if snr >= snr_threshold:
        signal_detected = True
    else:
        signal_detected = False

    logger.info(
        f"{source_box.name}/port-{source_port} -> {dest_box.name}/port-{dest_port}, {signal_detected=} (SNR: {snr} dB, Threshold: {snr_threshold} dB)"
    )
    return signal_detected
