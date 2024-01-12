"""
Created on Jan 12 2024

@author:
    Meng Wu, 2024
"""

import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import soundfile


class LivePlayer:
    def __init__(
        self,
        stream_chunk_size: int = 2048,
        device_index: int = 0,
    ):
        self.stream_chunk_size = stream_chunk_size
        self.device_index = device_index

        # default setting
        self.spk_post_gain = 1

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

    def set_post_gain(self, post_gain_db: float):
        """Set speaker boost post gain (dB)"""
        self.spk_post_gain = float(post_gain_db)
        self.spk_post_gain = 10 ** (self.spk_post_gain / 20)

    def set_stream_chunk_size(self, chunk_size: int):
        """Set stream's chunk size"""
        self.stream_chunk_size = chunk_size

    def _open_audio_stream(self):
        p = pyaudio.PyAudio()
        try:
            self.stream = p.open(
                format=pyaudio.paFloat32,
                channels=self.n_channels,
                rate=self.sr,
                output=True,
                frames_per_buffer=self.stream_chunk_size,
                output_device_index=self.device_index,
            )
            print(f"Start playing. {time.asctime(time.localtime(time.time()))}")

        except OSError:
            raise OSError("Device error, please choose another one.")

    def _close_audio_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        print(f"Closed the stream. {time.asctime(time.localtime(time.time()))}")

    def process_chunk(self):
        """
        Implement signal algo here
        """
        # DO SOMTHING HERE
        pass

    def play_local_wavfile(
        self, file_path: str, loop_play: bool = False, visiable: bool = False
    ):
        """
        This function open and play waveform file from disk through PyAudio interface.

        Args:
            file_path (str): the audio file path
            loop_play (bool): play waveform without interaction
            visiable (bool): open visiable screen

        """
        warray, self.sr = soundfile.read(
            file_path,
        )
        self.n_channels = warray.shape[1]
        warray = warray.astype(np.float32) * self.spk_post_gain
        max_value = np.max(np.abs(warray))

        self._open_audio_stream()

        if visiable:
            fig, ax = plt.subplots(1, figsize=(7, 4))
            for i in range(self.n_channels):
                ax.plot(np.arange(warray.shape[0]), warray[..., 0], "-", lw=2)
            time_line = ax.plot((0, 0), (max_value, -max_value), "r-", lw=1)[0]

            # show the plot
            plt.title(f"File: {file_path}")
            plt.tight_layout()
            plt.show(block=False)

        print(f"Playing the waveform: {file_path}")
        n_slices = warray.shape[0] // self.stream_chunk_size + 1

        if not loop_play:
            input("Press Enter to play: ")

        for i in range(n_slices):
            data = warray[
                i * self.stream_chunk_size : (i + 1) * self.stream_chunk_size, ...
            ]
            data = self.process_chunk()
            self.stream.write(data.tobytes())
            if visiable:
                time_line.set_xdata(
                    ((i + 1) * self.stream_chunk_size, (i + 1) * self.stream_chunk_size)
                )
                fig.canvas.draw()
                fig.canvas.flush_events()

        self._close_audio_stream()
