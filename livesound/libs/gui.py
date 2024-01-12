"""
Created on Jan 12 2024

@author:
    Meng Wu, 2024
"""

import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import soundfile

from livesound.libs.receiver import LiveRecorder


class LiveSoundRecorder(LiveRecorder):
    def __init__(
        self,
        sampling_rate: int = 16000,
        record_channels: int = 1,
        stream_chunk_size: int = 1024,
        device_index: int = 0,
        timeout: int = None,
        show_recorded: bool = True,
    ):
        super().__init__(
            sampling_rate,
            record_channels,
            stream_chunk_size,
            device_index,
            timeout,
            show_recorded,
        )

        self.window = tk.Tk()
        self.window.title("LiveSound Recorder GUI")
        self.window.resizable(width=False, height=False)

        self._window_init()
        self.window.mainloop()

    def _window_init(self):
        self._create_folder()
        self._create_device_related_widgits()
        self._create_recording_widgits()

    def _create_folder(self):
        if not os.path.isdir("./media"):
            os.makedirs("./media")

    def _create_device_related_widgits(self):
        # find all devices with its index
        _, self.device_dict = self.get_all_devices()

        # create combobox
        self.devices_combobox_text = tk.Label(
            self.window,
            text="Select device",
            anchor="w",
            padx=4,
            pady=4,
        )
        self.devices_combobox = ttk.Combobox(
            self.window,
            values=list(self.device_dict.keys()),
            state="readonly",
            width=45,
        )
        self.devices_combobox.current(0)

        # Setup pre-gain for mic
        self.mic_pregain_button = tk.Button(
            self.window,
            width=25,
            padx=4,
            pady=4,
            text="Test gain with 3 seconds speech",
            bg="antiquewhite2",
            command=self._get_mic_gain,
        )
        self.mic_gain_var = tk.StringVar()
        self.mic_gain_var.set("(dB)")
        self.mic_pregain_text = tk.Label(
            self.window,
            anchor="e",
            justify="left",
            width=10,
            padx=4,
            pady=4,
            bg="lightgreen",
            textvariable=self.mic_gain_var,
        )
        self.mic_set_pregain_text = tk.Label(
            self.window,
            text="Boost gain (dB)",
            anchor="w",
            padx=4,
            pady=4,
        )
        self.mic_set_pregain_entry = tk.Entry(self.window, width=10)
        self.mic_set_pregain_entry.insert(0, 0)  # defatul gain 0 dB

        # place
        self.devices_combobox_text.grid(row=0, column=0)
        self.devices_combobox.grid(row=0, column=1, columnspan=5)
        self.mic_pregain_button.grid(row=1, column=0, columnspan=2)
        self.mic_pregain_text.grid(row=1, column=2)
        self.mic_set_pregain_text.grid(row=1, column=3)
        self.mic_set_pregain_entry.grid(row=1, column=4)

    def _create_recording_widgits(self):
        self.recording = False

        self.recording_duration_text1 = tk.Label(
            self.window,
            text="Recording time: ",
            anchor="w",
            padx=4,
            pady=4,
        )
        self.recording_duration_var = tk.StringVar()
        self.recording_duration_text2 = tk.Label(
            self.window,
            anchor="c",
            width=10,
            padx=4,
            pady=4,
            bg="lightgreen",
            textvariable=self.recording_duration_var,
        )
        self.recording_duration_text3 = tk.Label(
            self.window,
            text="(seconds)",
            anchor="w",
            padx=4,
            pady=4,
        )

        def _recording_func():
            self.set_pre_gain(float(self.mic_set_pregain_entry.get()))

            # On/Off icon text
            if not self.recording:
                self.recoring_button["text"] = "STOP"
                self.recoring_button["bg"] = "firebrick1"
                self.recording = True
            else:
                self.recoring_button["text"] = "Start recoring"
                self.recoring_button["bg"] = "antiquewhite2"
                self.recording = False

            if self.recording:
                threading._start_new_thread(self._loop_recording, ())

            else:
                self.show_results()

        self.recoring_button = tk.Button(
            self.window,
            width=25,
            padx=4,
            pady=4,
            text="Start recoring",
            bg="antiquewhite2",
            command=_recording_func,
        )

        # place
        self.recording_duration_text1.grid(row=2, column=0)
        self.recording_duration_text2.grid(row=2, column=1)
        self.recording_duration_text3.grid(row=2, column=2)
        self.recoring_button.grid(row=2, column=3, columnspan=2)

    def _get_mic_gain(self):
        self.set_pre_gain(float(self.mic_set_pregain_entry.get()))
        _mic_gain = self.test_microphone_gain()

        # flush
        self.mic_gain_var.set(f"{round(_mic_gain, 2)} (dB)")

    def _loop_recording(self):
        self._open_audio_stream()
        n_chunks = 0

        while self.recording:
            self.process_chunk()
            n_chunks += 1

            # flush
            self.recording_duration_var.set(
                f"{round(n_chunks * self.stream_chunk_size / self.sr, 2)}"
            )

        self._close_audio_stream()

        # save audios
        recorded_wav = np.concatenate(self.recorded_wav, axis=0)
        print(f"Recodred length  {recorded_wav.shape[-1]}")

        cur_time = time.localtime()
        fname = f"{cur_time.tm_year}-{cur_time.tm_mon}-{cur_time.tm_mday}-{cur_time.tm_hour}-{cur_time.tm_min}-{cur_time.tm_sec}"
        soundfile.write(
            f"./media/{fname}.wav",
            recorded_wav.squeeze(),
            16000,
        )

        # reinit buffer for next recording
        time.sleep(1)
        self.recorded_wav = []
        self.processed_wav = []
