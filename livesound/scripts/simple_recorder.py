"""
Created on Jan 05 2024

@author:
    Meng Wu, 2024
"""
import argparse
from typing import Any

from livesound.libs.gui import LiveSoundRecorder
from livesound.libs.receiver import LiveRecorder


def recorder(argv: Any = None):
    parser = argparse.ArgumentParser(
        description="This is a local audio recorder CLI program"
    )
    parser.add_argument(
        "-gui",
        "--use_gui",
        action="store_true",
        help="open recorder GUI",
    )
    parser.add_argument(
        "-sd",
        "--select_device",
        action="store_true",
        help="choose your microphone device index",
    )
    parser.add_argument(
        "-tg",
        "--test_gain",
        action="store_true",
        help="test microphone average gain (dB)",
    )
    parser.add_argument(
        "-spg",
        "--set_pre_gain",
        action="store_true",
        help="set microphone pre gain (dB)",
    )
    parser.add_argument(
        "-sr", "--sample_rate", type=int, default=16000, help="recording sample rate"
    )
    parser.add_argument(
        "-cs", "--chunk_size", type=int, default=1024, help="stream chunk buffer size"
    )

    args = parser.parse_args(argv)

    if args.use_gui:
        runner = LiveSoundRecorder(
            sampling_rate=args.sample_rate, stream_chunk_size=args.chunk_size
        )

    else:
        if args.select_device:
            (device_idx2name, _) = LiveRecorder.get_all_devices()
            print(device_idx2name)
            device_id = int(input("Enter your device index: "))
            print("=====" * 10)
        else:
            device_id = 0

        runner = LiveRecorder(
            sampling_rate=args.sample_rate,
            record_channels=1,
            stream_chunk_size=args.chunk_size,
            device_index=device_id,
        )

        if args.set_pre_gain:
            boost_gain = float(input("Enter the boost gain value: "))
            print("=====" * 10)
            runner.set_pre_gain(pre_gain_db=boost_gain)

        if args.test_gain:
            print("Please say 3 seconds speech to test your microphone gain")
            print("normally we would control the overall gain around -20 dB")
            input("Press Enter to start recording 3 seconds:")
            print(
                "Mic gain(dB)",
                round(runner.test_microphone_gain(record_length_sec=3), 2),
            )
            print("=====" * 10)

        save_to_where = input("Enter the save folder path: ")
        input("Press Enter to start recording:")
        runner.start_record(save_path=save_to_where)
