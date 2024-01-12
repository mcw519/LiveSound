"""
Created on Jan 12 2024

@author:
    Meng Wu, 2024
"""
import argparse
from glob import glob
from typing import Any

from livesound.libs.player import LivePlayer


def player(argv: Any = None):
    parser = argparse.ArgumentParser(
        description="This is a local audio player CLI program"
    )
    parser.add_argument(
        "-sd",
        "--select_device",
        action="store_true",
        help="choose your output speaker device index",
    )
    parser.add_argument(
        "-spg",
        "--set_post_gain",
        action="store_true",
        help="set output speaker post gain (dB)",
    )
    parser.add_argument(
        "-v",
        "--visiable",
        action="store_true",
        help="open visiable screen",
    )
    parser.add_argument(
        "-pf",
        "--play_folder",
        action="store_true",
        help="play all waveform (*.wav) inside the folder",
    )
    parser.add_argument(
        "-cs", "--chunk_size", type=int, default=2048, help="stream chunk buffer size"
    )

    args = parser.parse_args(argv)

    if args.select_device:
        (device_idx2name, _) = LivePlayer.get_all_devices()
        print(device_idx2name)
        device_id = int(input("Enter your device index: "))
        print("=====" * 10)
    else:
        device_id = 0

    runner = LivePlayer(
        stream_chunk_size=args.chunk_size,
        device_index=device_id,
    )

    if args.set_post_gain:
        boost_gain = float(input("Enter the boost speaker post gain value: "))
        print("=====" * 10)
        runner.set_post_gain(post_gain_db=boost_gain)

    if args.play_folder:
        f_path = input("Enter the audio folder path: ")
        all_wav = glob(f"{f_path}/*.wav")
        for file in all_wav:
            runner.play_local_wavfile(file_path=file, loop_play=True, visiable=False)
    else:
        f_path = input("Enter the audio path: ")
        runner.play_local_wavfile(file_path=f_path, visiable=args.visiable)
