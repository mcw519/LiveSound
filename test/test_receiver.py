import sys

import pytest

sys.path.insert(0, "./")

from livesound.libs.receiver import LiveRecorder


@pytest.mark.receiver
def test_local_recoder():
    (device_idx2name, _) = LiveRecorder.get_all_devices()
    print(device_idx2name)

    recoder = LiveRecorder(
        sampling_rate=16000,
        record_channels=1,
        stream_chunk_size=800,
        device_index=0,
        timeout=10,
        show_recorded=False,
    )

    print("Please say 3 seconds speech to test your microphone gain")
    print("normally we would control the overall gain around -20 dB")
    print("Mic gain(dB)", round(recoder.test_microphone_gain(record_length_sec=3), 2))

    recoder.set_pre_gain(pre_gain_db=10)
    recoder.start_record()
