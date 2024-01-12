import sys
from glob import glob

import pytest

sys.path.insert(0, "./")

from livesound.libs.player import LivePlayer

TESTCASES = glob("./samples/*.wav")


@pytest.mark.player
def test_local_player():
    media_player = LivePlayer(stream_chunk_size=2048, device_index=0)
    all_devices = media_player.get_all_devices()
    print(all_devices)

    for fpath in TESTCASES:
        media_player.play_local_wavfile(file_path=fpath, loop_play=True, visiable=True)
