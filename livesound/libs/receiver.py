"""
Created on Jan 05 2024

@author:
    Meng Wu, 2024
"""

import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import soundfile


class LiveRecorder:
    def __init__(
        self,
        sampling_rate: int = 16000,
        record_channels: int = 1,
        stream_chunk_size: int = 1024,
        device_index: int = 0,
        timeout: Optional[int] = None,
        show_recorded: bool = True,
    ):
        self.sr = sampling_rate
        if self.sr <= 16000:
            self.nfft = 512
        else:
            self.nfft = 1024

        self.n_channels = record_channels
        self.stream_chunk_size = stream_chunk_size
        self.device_index = device_index
        self.timeout = timeout
        self.show_recorded = show_recorded

        # default setting
        self.mic_pre_gain = 1
        self.recorded_wav = []
        self.processed_wav = []

    @staticmethod
    def get_all_devices() -> Tuple:
        """
        Returns:
            device_idx2name_dict (dict): mapping device index to its name
            device_name2idx_dict (dict): mapping device nema to its index
        """
        p = pyaudio.PyAudio()
        total_devices_num = p.get_device_count()
        device_idx2name_dict = {}
        device_name2idx_dict = {}

        for i in range(total_devices_num):
            device = p.get_device_info_by_index(i)
            device_idx2name_dict.update({i: device["name"]})
            device_name2idx_dict.update({device["name"]: i})

        return (device_idx2name_dict, device_name2idx_dict)

    def set_pre_gain(self, pre_gain_db: float):
        """Set microphone boost gain (dB)"""
        self.mic_pre_gain = float(pre_gain_db)
        self.mic_pre_gain = 10 ** (self.mic_pre_gain / 20)

    def set_stream_chunk_size(self, chunk_size: int):
        """Set stream's chunk size"""
        self.stream_chunk_size = chunk_size

    def set_timeout(self, timeout: int):
        """Set timeout time"""
        self.timeout = timeout

    def _open_audio_stream(self):
        p = pyaudio.PyAudio()
        try:
            self.stream = p.open(
                format=pyaudio.paFloat32,  # pyaudio.paInt16,
                channels=self.n_channels,
                rate=self.sr,
                input=True,
                frames_per_buffer=self.stream_chunk_size,
                input_device_index=self.device_index,
            )
            print(f"Start recording. {time.asctime(time.localtime(time.time()))}")

        except OSError:
            raise OSError("Device error, please choose another one.")

    def _close_audio_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        print(f"Closed the recording. {time.asctime(time.localtime(time.time()))}")

    def _get_data_from_stream_buffer(self):
        """
        Get data from stream buffer.
        Returns:
            data (np.ndarray)
        """
        data_rec = self.stream.read(self.stream_chunk_size, exception_on_overflow=False)
        data_rec = np.frombuffer(data_rec, dtype=np.float32) * self.mic_pre_gain
        return data_rec

    def test_microphone_gain(self, record_length_sec: float = 3) -> float:
        """
        Recording some speech to caculate the microphone gain range
        Args:
            record_length_sec (float): recording length
        Returns:
            microphone average gain (dB)
        """
        self._open_audio_stream()
        record_length = record_length_sec * self.sr // self.stream_chunk_size
        record_frames = []
        time.sleep(0.5)
        print("Start recording......")

        for _ in range(record_length):
            data_rec = self._get_data_from_stream_buffer()
            record_frames.append(data_rec)

        self._close_audio_stream()

        record_frames_power_db = [
            20 * np.log10(np.sqrt(np.mean(np.abs(i) ** 2))) for i in record_frames
        ]
        mic_gain = np.array(record_frames_power_db).mean()

        return mic_gain

    def show_results(self):
        """
        Show the recorded/processed waveform
        """
        n_row = 2
        if self.processed_wav != []:
            n_row += 2

        self.fig, self.ax = plt.subplots(n_row, figsize=(7, 8))
        _record = np.concatenate(self.recorded_wav, axis=0).squeeze()
        _record *= 32768
        _record = _record.astype(np.int16)
        self.ax[0].plot(
            np.arange(_record.shape[-1]),
            _record,
            "-",
            lw=2,
        )
        self.ax[0].set_ylim([-35000, 35000])
        plt.setp(self.ax[0], yticks=[])
        self.ax[1].specgram(_record, NFFT=self.nfft, Fs=self.sr, cmap="jet")
        self.ax[0].set_title("Recorded Waveform")
        self.ax[0].set_xlabel("time (samples)")
        self.ax[1].set_title("Recorded Spectrogram")
        self.ax[1].set_xlabel("time (seconds)")

        if self.processed_wav != []:
            _processed = np.concatenate(self.processed_wav, axis=1).squeeze()
            _processed *= 32768
            _processed = _processed.astype(np.int16)
            self.ax[0].plot(
                np.arange(_record.shape[-1]),
                _record,
                "-",
                lw=2,
            )

            self.ax[2].plot(
                np.arange(_processed.shape[-1]),
                _processed,
                "-",
                lw=2,
            )
            self.ax[2].set_ylim([-35000, 35000])
            plt.setp(self.ax[2], yticks=[])
            self.ax[3].specgram(_processed, NFFT=self.nfft, Fs=self.sr, cmap="jet")

            self.ax[2].set_title("Processed Waveform")
            self.ax[2].set_xlabel("time (samples)")
            self.ax[3].set_title("Processed Spectrogram")
            self.ax[3].set_xlabel("time (seconds)")

        # show the plot
        plt.tight_layout()
        plt.show(block=True)

    def _loop_recording(self):
        self._open_audio_stream()
        if self.timeout is None:
            print("Start recording......")
            print(f"Press Ctrl+C to stop recording.")

            self.recording = True

            try:
                while True:
                    self.process_chunk()

            except KeyboardInterrupt:
                self._close_audio_stream()
                self.recording = False
        else:
            print("Start recording......")
            _count_to_timeout = 0
            while _count_to_timeout < self.timeout:
                self.process_chunk()
                _count_to_timeout += 1

            self._close_audio_stream()
            self.recording = False

    def process_chunk(self):
        """
        Implement signal algo here
        """
        data_rec = self._get_data_from_stream_buffer()
        self.recorded_wav.append(data_rec)
        # DO SOMTHING HERE

    def start_record(self, save_path: Optional[str] = None):
        self._loop_recording()
        if save_path is not None:
            cur_time = time.localtime()
            fname = f"{cur_time.tm_year}-{cur_time.tm_mon}-{cur_time.tm_mday}-{cur_time.tm_hour}-{cur_time.tm_min}-{cur_time.tm_sec}"
            record = np.concatenate(self.recorded_wav, axis=0).squeeze()
            soundfile.write(f"{save_path}/{fname}.wav", record, self.sr)
            print(f"saved: {save_path}/{fname}.wav")
            if self.processed_wav != []:
                processed = np.concatenate(self.processed_wav, axis=1).squeeze()
                soundfile.write(f"{save_path}/{fname}-proc.wav", processed, self.sr)
                print(f"saved: {save_path}/{fname}-proc.wav")

        if self.show_recorded:
            self.show_results()
