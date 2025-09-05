"""
CLI interface for LiveSound Recorder
Simple and clean command-line interface
"""

import argparse
import signal
import sys
import time
from pathlib import Path

from livesound.libs.gui import create_gui
from livesound.libs.i18n import DEFAULT_LANG, LANGS, t
from livesound.libs.recorder import AudioConfig, AudioRecorder

# Global recorder instance
recorder = None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n" + t("cli_stopping", signal_handler.lang))
    global recorder
    if recorder:
        recorder.stop_recording()
    sys.exit(0)


def cli_record(args):
    """CLI recording mode"""
    global recorder
    lang = args.lang or DEFAULT_LANG
    signal_handler.lang = lang

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create recorder
    config = AudioConfig(
        sample_rate=args.sample_rate,
        channels=args.channels,
        chunk_size=args.chunk_size,
        device_index=args.device_index,
        pre_gain_db=args.pre_gain,
    )

    recorder = AudioRecorder(config=config, output_dir=str(output_dir))

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    print(t("cli_will_save_to", lang, path=output_dir.absolute()))

    # List devices if requested
    if args.list_devices:
        recorder.list_devices()
        recorder.cleanup()
        return

    # Test gain if requested
    if args.test_gain:
        print(t("cli_test_gain_start", lang))
        if recorder._open_stream():
            levels = []
            start_time = time.time()

            while time.time() - start_time < 3.0:
                try:
                    data = recorder.stream.read(
                        recorder.config.chunk_size, exception_on_overflow=False
                    )
                    import numpy as np

                    audio_data = np.frombuffer(data, dtype=np.float32)
                    rms_db, _ = recorder._calculate_levels(audio_data)
                    levels.append(rms_db)
                except:
                    break

            recorder._close_stream()

            if levels:
                avg_level = sum(levels) / len(levels)
                print(t("cli_avg_level", lang, level=avg_level))

                if avg_level < -40:
                    print(t("cli_gain_up", lang))
                elif avg_level > -12:
                    print(t("cli_gain_down", lang))
                else:
                    print(t("cli_gain_ok", lang))

        recorder.cleanup()
        return

    # Start recording
    filename = args.filename or None

    print(t("cli_press_ctrl_c", lang))

    # Add level callback for CLI feedback
    def on_level_update(data):
        level_db = data["rms"]
        duration = data["duration"]
        # Simple text-based level meter
        level_normalized = max(0, min(1, (level_db + 60) / 60))
        meter_width = 20
        filled = int(level_normalized * meter_width)
        meter = "█" * filled + "░" * (meter_width - filled)
        print(
            f"\r{t('cli_recording', lang)} {duration:.1f}s [{meter}] {level_db:.1f}dB",
            end="",
            flush=True,
        )

    recorder.add_callback("on_level_update", on_level_update)

    # Start recording
    if recorder.start_recording(filename):
        try:
            while recorder.state.value != "idle":
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        print("\n" + t("cli_saving", lang))
        saved_path = recorder.stop_recording()

        if saved_path:
            print(t("cli_saved", lang, path=saved_path))
        else:
            print(t("cli_save_failed", lang))
    else:
        print(t("cli_cannot_start", lang))

    recorder.cleanup()


def main(argv=None):
    """Main function"""
    if argv is None:
        argv = sys.argv[1:]

    # temp parser for lang to localize help
    temp = argparse.ArgumentParser(add_help=False)
    temp.add_argument("--lang", choices=list(LANGS.keys()), default=DEFAULT_LANG)
    known, _ = temp.parse_known_args(argv)
    lang = known.lang

    parser = argparse.ArgumentParser(
        description=t("cli_desc", lang),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument("--gui", action="store_true", help=t("cli_help_gui", lang))

    # Output
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./recordings",
        help=t("cli_help_output_dir", lang),
    )
    parser.add_argument("-f", "--filename", help=t("cli_help_filename", lang))

    # Device
    parser.add_argument(
        "--list-devices", action="store_true", help=t("cli_help_list_devices", lang)
    )
    parser.add_argument(
        "-d", "--device-index", type=int, help=t("cli_help_device_index", lang)
    )
    parser.add_argument(
        "--test-gain", action="store_true", help=t("cli_help_test_gain", lang)
    )

    # Audio settings
    parser.add_argument(
        "-sr",
        "--sample-rate",
        type=int,
        default=44100,
        choices=[8000, 16000, 22050, 44100, 48000],
        help=t("cli_help_sample_rate", lang),
    )
    parser.add_argument(
        "-c",
        "--channels",
        type=int,
        default=1,
        choices=[1, 2],
        help=t("cli_help_channels", lang),
    )
    parser.add_argument(
        "-cs",
        "--chunk-size",
        type=int,
        default=1024,
        help=t("cli_help_chunk_size", lang),
    )
    parser.add_argument(
        "-g", "--pre-gain", type=float, default=0.0, help=t("cli_help_pre_gain", lang)
    )
    parser.add_argument(
        "--lang",
        choices=list(LANGS.keys()),
        default=lang,
        help=t("cli_help_lang", lang),
    )

    args = parser.parse_args(argv)

    # Validate gain
    if not -20 <= args.pre_gain <= 20:
        parser.error(t("cli_gain_range_error", lang))

    try:
        if args.gui:
            create_gui(args.output_dir, lang=args.lang)
        else:
            cli_record(args)
    except KeyboardInterrupt:
        print("\n" + t("cli_cancelled", args.lang))
    except Exception as e:
        print(t("cli_error", args.lang, msg=e))
        sys.exit(1)


# For entry point
recorder_main = main

if __name__ == "__main__":
    main()
