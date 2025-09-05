"""GUI for LiveSound Recording Application."""

import os
import platform
import subprocess
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Suppress macOS GUI warnings
if platform.system() == "Darwin":  # macOS
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    # Redirect stderr temporarily to suppress system warnings
    import io
    import contextlib

import threading

import numpy as np
import pyaudio
import soundfile
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display


# Configure matplotlib font settings once at module level
def _configure_matplotlib_fonts():
    """Configure matplotlib to use available fonts and avoid font warnings"""
    import matplotlib.font_manager as fm

    # Get list of available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # Preferred fonts in order of preference
    preferred_fonts = ["Arial", "Helvetica", "DejaVu Sans", "Verdana", "sans-serif"]

    # Find the first available font
    font_to_use = "sans-serif"  # fallback
    for font in preferred_fonts:
        if font in available_fonts:
            font_to_use = font
            break

    # Set font configuration
    plt.rcParams["font.family"] = [font_to_use, "sans-serif"]
    plt.rcParams["font.sans-serif"] = [font_to_use] + plt.rcParams["font.sans-serif"]


# Configure fonts when module is imported
_configure_matplotlib_fonts()

# Try to import PIL for image support
try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .i18n import DEFAULT_LANG, LANGS, t
from .quality import QualityAssessment
from .recorder import AudioRecorder


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (for macOS warnings)"""
    if platform.system() == "Darwin":  # Only on macOS
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr
    else:
        yield


class ControllableAudioPlayer:
    """Simple audio player that can be stopped with output device selection"""

    def __init__(self, chunk_size: int = 1024, output_device_index: int = None):
        self.chunk_size = chunk_size
        self.output_device_index = output_device_index
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_playing = False
        self.is_paused = False
        self.stop_requested = False
        self.pause_requested = False
        self.current_position = 0
        self.audio_data = None
        self.sample_rate = None
        self.channels = None

    def get_output_devices(self):
        """Get available audio output devices"""
        devices = {}
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info["maxOutputChannels"] > 0:
                    devices[i] = device_info["name"]
            except:
                continue
        return devices

    def set_output_device(self, device_index: int):
        """Set output device for playback"""
        self.output_device_index = device_index

    def play_file(self, file_path: str):
        """Play audio file"""
        try:
            # Load audio file, soundfile has shape (samples, channels)
            self.audio_data, self.sample_rate = soundfile.read(
                file_path, dtype="float32"
            )
            if self.audio_data.ndim == 1:
                self.audio_data = np.expand_dims(self.audio_data, axis=1)

            self.channels = self.audio_data.shape[1]
            self.current_position = 0

            # Open audio stream with selected output device
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.chunk_size,
            )

            self.is_playing = True
            self.is_paused = False
            self.stop_requested = False
            self.pause_requested = False

            # Play audio in chunks
            total_frames = len(self.audio_data)
            start = self.current_position

            while start < total_frames:
                if self.stop_requested:
                    break

                # Handle pause
                while self.pause_requested and not self.stop_requested:
                    self.is_paused = True
                    time.sleep(0.1)  # Small delay to prevent busy waiting

                if self.stop_requested:
                    break

                self.is_paused = False
                end = min(start + self.chunk_size, total_frames)
                chunk = self.audio_data[start:end]

                # Pad chunk if necessary
                if len(chunk) < self.chunk_size:
                    padding = np.zeros(
                        (self.chunk_size - len(chunk), self.channels), dtype=np.float32
                    )
                    chunk = np.vstack([chunk, padding])

                try:
                    self.stream.write(chunk.tobytes())
                except Exception:
                    # If writing fails, stop playback
                    break

                start = end
                self.current_position = start

                # Check for stop request more frequently
                if self.stop_requested:
                    break

        finally:
            self.stop()

    def pause(self):
        """Pause playback"""
        if self.is_playing and not self.is_paused:
            self.pause_requested = True

    def resume(self):
        """Resume playback"""
        if self.is_playing and self.is_paused:
            self.pause_requested = False

    def stop(self):
        """Stop playback"""
        self.stop_requested = True
        self.pause_requested = False
        self.is_playing = False
        self.is_paused = False
        self.current_position = 0
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass  # Ignore errors when stopping
            finally:
                self.stream = None

    def cleanup(self):
        """Clean up resources"""
        self.stop()
        if self.audio:
            self.audio.terminate()


class LevelMeter(tk.Canvas):
    """Enhanced audio level meter widget with modern styling"""

    def __init__(self, parent, height=35):
        # Don't set fixed width, let it expand to fill parent
        super().__init__(parent, height=height, bg="#1a1a1a", highlightthickness=0)
        self.height = height
        self.level = -60
        self.peak = -60
        self.peak_hold = 0
        self.meter_height = height - 15  # Leave space for labels
        self.meter_y = 8  # Start meter 8px from top

        # Color gradient stops for smooth transitions
        self.colors = [
            (-60, "#003300"),  # Very quiet - dark green
            (-48, "#006600"),  # Quiet - green
            (-36, "#00aa00"),  # Good - bright green
            (-24, "#44ff44"),  # Good - light green
            (-18, "#88ff00"),  # Getting loud - yellow-green
            (-12, "#ffaa00"),  # Loud - orange
            (-6, "#ff6600"),  # Very loud - red-orange
            (-3, "#ff3300"),  # Too loud - red
            (0, "#ff0000"),  # Clipping - bright red
        ]

        # Bind to resize events to update width
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        """Handle widget resize"""
        # Update width when widget is resized
        self.width = event.width
        self._draw()  # Redraw with new width

    def update(self, rms_db: float, peak_db: float):
        """Update meter levels"""
        self.level = max(-60, min(0, rms_db))

        # Peak hold with smoother decay
        if peak_db > self.peak:
            self.peak = peak_db
            self.peak_hold = time.time()
        elif time.time() - self.peak_hold > 1.5:  # Hold peak longer
            self.peak = max(self.peak - 0.3, peak_db)  # Slower decay

        self._draw()

    def _get_color_for_level(self, db_level):
        """Get interpolated color for given dB level"""
        if db_level <= -60:
            return self.colors[0][1]
        if db_level >= 0:
            return self.colors[-1][1]

        # Find the two colors to interpolate between
        for i in range(len(self.colors) - 1):
            if self.colors[i][0] <= db_level <= self.colors[i + 1][0]:
                # Linear interpolation between colors
                lower_db, lower_color = self.colors[i]
                upper_db, upper_color = self.colors[i + 1]

                # Simple color selection (no complex interpolation for now)
                mid_point = (lower_db + upper_db) / 2
                return lower_color if db_level < mid_point else upper_color

        return "#003300"

    def _draw(self):
        """Draw the enhanced meter"""
        self.delete("all")

        # Get current width (might have changed due to resize)
        if not hasattr(self, "width") or self.width <= 0:
            self.width = self.winfo_width()
            if self.width <= 1:  # Widget not yet rendered
                self.after(100, self._draw)  # Retry later
                return

        # Convert dB to pixels (ensure proper alignment)
        level_width = int((self.level + 60) / 60.0 * self.width)
        peak_pos = int((self.peak + 60) / 60.0 * self.width)

        # Draw background with subtle gradient
        self.create_rectangle(
            0,
            self.meter_y,
            self.width,
            self.meter_y + self.meter_height,
            fill="#2a2a2a",
            outline="#404040",
            width=1,
        )

        # Draw background grid lines (aligned with scale points)
        grid_points = [-48, -36, -24, -12, -6]
        for db in grid_points:
            x = int((db + 60) / 60.0 * self.width)
            if x > 0 and x < self.width:  # Only draw if within bounds
                self.create_line(
                    x,
                    self.meter_y,
                    x,
                    self.meter_y + self.meter_height,
                    fill="#404040",
                    width=1,
                )

        # Draw level bar with gradient effect
        if level_width > 0:
            # Draw multiple segments for gradient effect
            segment_width = max(1, level_width // 20)
            for i in range(0, level_width, segment_width):
                segment_end = min(i + segment_width, level_width)
                segment_db = ((i + segment_end) / 2.0 / self.width * 60) - 60
                color = self._get_color_for_level(segment_db)

                self.create_rectangle(
                    i,
                    self.meter_y + 2,
                    segment_end,
                    self.meter_y + self.meter_height - 2,
                    fill=color,
                    outline="",
                )

        # Draw scale labels (adjust spacing based on width)
        scale_points = [
            (-60, "-60"),
            (-48, "-48"),
            (-36, "-36"),
            (-24, "-24"),
            (-12, "-12"),
            (-6, "-6"),
            (0, "0"),
        ]

        # Only show labels if there's enough space
        min_spacing = 40  # Minimum pixels between labels
        if self.width / len(scale_points) >= min_spacing:
            for db, label in scale_points:
                x = int((db + 60) / 60.0 * self.width)
                # Draw tick mark
                self.create_line(
                    x,
                    self.meter_y + self.meter_height,
                    x,
                    self.meter_y + self.meter_height + 3,
                    fill="#cccccc",
                    width=1,
                )
                # Draw label with proper centering
                if x >= 0 and x <= self.width:  # Only draw if within bounds
                    self.create_text(
                        x,
                        self.meter_y + self.meter_height + 8,
                        text=label,
                        fill="#cccccc",
                        font=("Arial", 8),
                        anchor="n",
                    )
        else:
            # Show fewer labels when space is limited
            reduced_points = [(-60, "-60"), (-24, "-24"), (-6, "-6"), (0, "0")]
            for db, label in reduced_points:
                x = int((db + 60) / 60.0 * self.width)
                self.create_line(
                    x,
                    self.meter_y + self.meter_height,
                    x,
                    self.meter_y + self.meter_height + 3,
                    fill="#cccccc",
                    width=1,
                )
                if x >= 0 and x <= self.width:
                    self.create_text(
                        x,
                        self.meter_y + self.meter_height + 8,
                        text=label,
                        fill="#cccccc",
                        font=("Arial", 8),
                        anchor="n",
                    )

        # Enhanced peak indicator with glow effect
        if peak_pos > 0:
            # Glow effect
            for offset in range(3, 0, -1):
                alpha = 100 - (offset * 20)
                self.create_line(
                    peak_pos - offset,
                    self.meter_y,
                    peak_pos - offset,
                    self.meter_y + self.meter_height,
                    fill="#ffffff",
                    width=1,
                )
                self.create_line(
                    peak_pos + offset,
                    self.meter_y,
                    peak_pos + offset,
                    self.meter_y + self.meter_height,
                    fill="#ffffff",
                    width=1,
                )

            # Main peak line
            self.create_line(
                peak_pos,
                self.meter_y,
                peak_pos,
                self.meter_y + self.meter_height,
                fill="#ffffff",
                width=2,
            )

            # Peak value display
            peak_text = f"{self.peak:.1f}dB"
            text_x = min(peak_pos + 5, self.width - 40)
            self.create_text(
                text_x,
                self.meter_y - 5,
                text=peak_text,
                fill="#ffffff",
                font=("Arial", 8, "bold"),
                anchor="w",
            )

        # Current level display
        if self.level > -60:
            level_text = f"{self.level:.1f}dB"
            self.create_text(
                5,
                2,
                text=level_text,
                fill="#ffffff",
                font=("Arial", 9, "bold"),
                anchor="nw",
            )


class FileList(ttk.Frame):
    """Recording file list widget with internal audio player"""

    def __init__(
        self,
        parent,
        output_dir: str,
        get_lang,
        get_output_device=None,
        quality_assessment=None,
    ):
        super().__init__(parent)
        self.output_dir = Path(output_dir)
        self.get_lang = get_lang
        self.get_output_device = (
            get_output_device  # Function to get current output device
        )
        self.quality_assessment = quality_assessment
        self.player = None
        self.is_playing = False
        self.current_playing_file = None
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        """Setup UI components"""
        # List with scrollbars (fixed compact height)
        list_frame = ttk.Frame(self)
        list_frame.pack(fill="both", expand=True, pady=(0, 5))

        # Create listbox with both scrollbars
        self.listbox = tk.Listbox(list_frame, height=6)

        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.listbox.yview
        )
        self.listbox.configure(yscrollcommand=v_scrollbar.set)

        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(
            list_frame, orient="horizontal", command=self.listbox.xview
        )
        self.listbox.configure(xscrollcommand=h_scrollbar.set)

        # Grid layout for proper scrollbar positioning
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        self.listbox.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.listbox.bind("<Double-1>", self._on_play)

        # Buttons with Unicode symbols - clearer than custom icons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", anchor="w")

        self.play_pause_btn = ttk.Button(
            btn_frame,
            text="▶ " + t("play", self.get_lang()),  # Unicode play symbol
            command=self._on_play_pause,
            width=6,
        )
        self.play_pause_btn.pack(side="left", padx=(0, 5))

        self.stop_btn = ttk.Button(
            btn_frame,
            text="⏹ " + t("stop", self.get_lang()),  # Unicode stop symbol
            command=self._on_stop,
            state="disabled",
            width=6,
        )
        self.stop_btn.pack(side="left", padx=(0, 5))

        self.delete_btn = ttk.Button(
            btn_frame,
            text=t("delete", self.get_lang()),
            command=self._on_delete,
            width=6,
        )
        self.delete_btn.pack(side="left", padx=(0, 5))

        self.spectrogram_btn = ttk.Button(
            btn_frame,
            text=t("spectrogram", self.get_lang()),
            command=self._on_show_spectrogram,
            width=10,
        )
        self.spectrogram_btn.pack(side="left", padx=(0, 5))

        self.refresh_btn = ttk.Button(
            btn_frame, text=t("refresh", self.get_lang()), command=self.refresh, width=8
        )
        self.refresh_btn.pack(side="right")

    def refresh(self):
        """Refresh file list"""
        self.listbox.delete(0, tk.END)

        if self.output_dir.exists():
            files = sorted(
                self.output_dir.glob("*.wav"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            for file in files:
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                display = f"{file.stem} ({mtime.strftime('%m-%d %H:%M')})"

                # Add quality status if available
                if self.quality_assessment:
                    result = self.quality_assessment.get_result(file.name)
                    if result:
                        if result.passed:
                            # Show scores in a single line with 1-5 scale
                            scores = result.scores
                            score_str = f" [SIG:{scores.get('sig', 0):.1f} BAK:{scores.get('bak', 0):.1f} OVRL:{scores.get('ovrl', 0):.1f} P808:{scores.get('p808', 0):.1f}] (pass)"
                            display += score_str
                        else:
                            # Show scores in a single line with 1-5 scale
                            scores = result.scores
                            score_str = f" [SIG:{scores.get('sig', 0):.1f} BAK:{scores.get('bak', 0):.1f} OVRL:{scores.get('ovrl', 0):.1f} P808:{scores.get('p808', 0):.1f}] (failed)"
                            display += score_str

                self.listbox.insert(tk.END, display)

                # Set color based on quality status
                if self.quality_assessment:
                    result = self.quality_assessment.get_result(file.name)
                    if result:
                        index = self.listbox.size() - 1
                        if result.passed:
                            self.listbox.itemconfig(index, {"fg": "green"})
                        else:
                            self.listbox.itemconfig(index, {"fg": "red"})

    def _get_selected_file(self) -> Path:
        """Get selected file path"""
        selection = self.listbox.curselection()
        if selection:
            display = self.listbox.get(selection[0])
            filename = display.split(" (")[0] + ".wav"
            return self.output_dir / filename
        return None

    def _on_play_pause(self):
        """Handle play/pause button click"""
        file_path = self._get_selected_file()
        if not file_path or not file_path.exists():
            return

        if not self.is_playing:
            # Start playing
            try:
                # Initialize player if needed
                if not self.player:
                    self.player = ControllableAudioPlayer()

                # Update UI state to show playing
                self.is_playing = True
                self.current_playing_file = str(file_path)
                self.play_pause_btn.config(text="⏸ " + t("pause", self.get_lang()))
                self.stop_btn.config(state="normal")

                # Start playback in separate thread
                threading.Thread(
                    target=self._play_file_thread, args=(str(file_path),), daemon=True
                ).start()

            except Exception as e:
                self.is_playing = False
                self.play_pause_btn.config(text="▶ " + t("play", self.get_lang()))
                self.stop_btn.config(state="disabled")
                messagebox.showerror(
                    t("error", self.get_lang()),
                    t("play_failed", self.get_lang(), e=str(e)),
                )
        else:
            # Toggle pause/resume
            if self.player:
                if self.player.is_paused:
                    # Resume playback
                    self.player.resume()
                    self.play_pause_btn.config(text="⏸ " + t("pause", self.get_lang()))
                else:
                    # Pause playback
                    self.player.pause()
                    self.play_pause_btn.config(text="▶ " + t("resume", self.get_lang()))

    def _on_play(self, event=None):
        """Start playing selected file"""
        if self.is_playing:
            # Already playing, do nothing
            return

        file_path = self._get_selected_file()
        if file_path and file_path.exists():
            # Check if quality assessment is available and file has been processed
            if self.quality_assessment:
                result = self.quality_assessment.get_result(file_path.name)
                if not result:
                    messagebox.showwarning(
                        t("warning", self.get_lang()),
                        t("quality_not_ready", self.get_lang()),
                    )
                    return

            try:
                # Get current output device if available
                output_device = None
                if self.get_output_device:
                    output_device = self.get_output_device()

                # Initialize player if needed or if device changed
                if not self.player:
                    self.player = ControllableAudioPlayer(
                        output_device_index=output_device
                    )
                else:
                    # Update device if changed
                    self.player.set_output_device(output_device)

                self.current_playing_file = file_path
                self.is_playing = True
                self.play_pause_btn.config(
                    text="⏸ " + t("pause", self.get_lang())
                )  # Change to pause
                self.stop_btn.config(state="normal")

                # Play in a separate thread to avoid blocking UI
                play_thread = threading.Thread(
                    target=self._play_file_thread, args=(str(file_path),), daemon=True
                )
                play_thread.start()

            except Exception as e:
                self.is_playing = False
                self.play_pause_btn.config(text="▶ " + t("play", self.get_lang()))
                self.stop_btn.config(state="disabled")
                messagebox.showerror(
                    t("error", self.get_lang()),
                    t("play_failed", self.get_lang(), e=str(e)),
                )

    def _play_file_thread(self, file_path: str):
        """Play file in separate thread"""
        try:
            self.player.play_file(file_path)
        except Exception as e:
            # Capture error message in local scope
            error_msg = str(e)
            # Handle playback errors
            self.after(0, lambda msg=error_msg: self._on_playback_error(msg))
        finally:
            # Reset UI state when playback ends
            self.after(0, self._on_playback_finished)

    def _on_playback_error(self, error_msg: str):
        """Handle playback errors in main thread"""
        self.is_playing = False
        self.play_pause_btn.config(text="▶ " + t("play", self.get_lang()))
        self.stop_btn.config(state="disabled")
        messagebox.showerror(
            t("error", self.get_lang()), t("play_failed", self.get_lang(), e=error_msg)
        )

    def _on_playback_finished(self):
        """Reset UI when playback finishes"""
        self.is_playing = False
        self.current_playing_file = None
        self.play_pause_btn.config(text="▶ " + t("play", self.get_lang()))
        self.stop_btn.config(state="disabled")

    def _on_stop(self):
        """Stop current playback"""
        if self.is_playing and self.player:
            try:
                # First set the flag to stop
                self.player.stop_requested = True
                self.player.is_playing = False

                # Then stop the stream
                self.player.stop()

                # Update UI state
                self.is_playing = False
                self.current_playing_file = None
                self.play_pause_btn.config(text="▶ " + t("play", self.get_lang()))
                self.stop_btn.config(state="disabled")
            except Exception:
                # If stopping fails, at least update UI
                self.is_playing = False
                self.current_playing_file = None
                self.play_pause_btn.config(text="▶ " + t("play", self.get_lang()))
                self.stop_btn.config(state="disabled")

    def _on_delete(self):
        """Delete selected file"""
        file_path = self._get_selected_file()
        if file_path and file_path.exists():
            if messagebox.askyesno(
                t("confirm", self.get_lang()),
                t("delete_confirm", self.get_lang(), name=file_path.name),
            ):
                try:
                    # Remove quality result if exists
                    if self.quality_assessment:
                        self.quality_assessment.remove_result(file_path.name)

                    file_path.unlink()
                    self.refresh()
                except Exception as e:
                    messagebox.showerror(
                        t("error", self.get_lang()),
                        t("delete_failed", self.get_lang(), e=e),
                    )

    def _on_show_spectrogram(self):
        """Show spectrogram of selected file"""
        file_path = self._get_selected_file()
        if not file_path or not file_path.exists():
            messagebox.showwarning(
                t("warning", self.get_lang()), t("no_file_selected", self.get_lang())
            )
            return

        try:
            self._show_spectrogram_window(file_path)
        except Exception as e:
            messagebox.showerror(
                t("error", self.get_lang()),
                t("spectrogram_error", self.get_lang(), e=str(e)),
            )

    def _show_spectrogram_window(self, file_path):
        """Create and show spectrogram window"""
        # Create new window
        spec_window = tk.Toplevel(self)
        spec_window.title(f"{t('spectrogram', self.get_lang())} - {file_path.name}")
        spec_window.geometry("900x700")

        # Load audio file
        y, sr = librosa.load(str(file_path), sr=None)

        # Calculate time axis for both plots
        duration = len(y) / sr
        time_waveform = np.linspace(0, duration, len(y))

        # Create matplotlib figure with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f"Spectrogram - {file_path.name}", fontsize=14)

        # Waveform plot
        ax1.plot(time_waveform, y, linewidth=0.5)
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Waveform")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, duration)

        # Spectrogram plot
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(
            D, y_axis="hz", x_axis="time", sr=sr, ax=ax2, cmap="viridis"
        )
        ax2.set_title("Spectrogram")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (s)")
        ax2.set_xlim(0, duration)  # Ensure same time range as waveform

        # Add colorbar
        # cbar = plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        # cbar.set_label('Power (dB)')

        plt.tight_layout()

        # Embed plot in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=spec_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar for navigation
        toolbar = tkagg.NavigationToolbar2Tk(canvas, spec_window)
        toolbar.update()

        # Add audio info label
        info_frame = ttk.Frame(spec_window)
        info_frame.pack(fill="x", padx=10, pady=5)

        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        info_text = f"Duration: {duration:.2f}s | Sample Rate: {sr}Hz | Channels: {1 if y.ndim == 1 else y.shape[1]} | File Size: {file_size:.2f}MB"

        info_label = ttk.Label(info_frame, text=info_text)
        info_label.pack()

        # Handle window close
        def on_close():
            plt.close(fig)
            spec_window.destroy()

        spec_window.protocol("WM_DELETE_WINDOW", on_close)

    def refresh_texts(self):
        """Refresh button texts when language changes"""
        if self.is_playing:
            if self.player and self.player.is_paused:
                self.play_pause_btn.config(text="▶ " + t("resume", self.get_lang()))
            else:
                self.play_pause_btn.config(text="⏸ " + t("pause", self.get_lang()))
        else:
            self.play_pause_btn.config(text="▶ " + t("play", self.get_lang()))
        self.stop_btn.config(text="⏹ " + t("stop", self.get_lang()))
        self.delete_btn.config(text=t("delete", self.get_lang()))
        self.spectrogram_btn.config(text=t("spectrogram", self.get_lang()))
        self.refresh_btn.config(text=t("refresh", self.get_lang()))

    def cleanup(self):
        """Clean up audio player resources"""
        if self.player:
            self.player.cleanup()
            self.player = None


class QualitySettingsDialog:
    """Dialog for quality assessment settings"""

    def __init__(self, parent, quality_assessment, lang, on_settings_changed=None):
        self.parent = parent
        self.quality_assessment = quality_assessment
        self.lang = lang
        self.dialog = None
        self.on_settings_changed = (
            on_settings_changed  # Callback for when settings change
        )

    def show(self):
        """Show the quality settings dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(t("quality_settings", self.lang))
        self.dialog.geometry("400x300")
        self.dialog.resizable(False, False)

        # Center dialog on parent
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Get current thresholds
        thresholds = self.quality_assessment.get_thresholds()

        # Create variables
        self.sig_var = tk.StringVar(value=str(thresholds.sig))
        self.bak_var = tk.StringVar(value=str(thresholds.bak))
        self.ovrl_var = tk.StringVar(value=str(thresholds.ovrl))
        self.p808_var = tk.StringVar(value=str(thresholds.p808))

        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text=t("quality_thresholds", self.lang),
            font=("TkDefaultFont", 12, "bold"),
        )
        title_label.pack(pady=(0, 20))

        # Settings frame
        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(fill="x", pady=(0, 20))

        # SIG threshold
        sig_frame = ttk.Frame(settings_frame)
        sig_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(sig_frame, text="SIG ≥", width=8).pack(side="left")
        sig_spin = ttk.Spinbox(
            sig_frame,
            textvariable=self.sig_var,
            from_=1.0,
            to=5.0,
            increment=0.1,
            width=10,
        )
        sig_spin.pack(side="left", padx=(5, 0))
        ttk.Label(sig_frame, text=t("sig_description", self.lang)).pack(
            side="left", padx=(10, 0)
        )

        # BAK threshold
        bak_frame = ttk.Frame(settings_frame)
        bak_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(bak_frame, text="BAK ≥", width=8).pack(side="left")
        bak_spin = ttk.Spinbox(
            bak_frame,
            textvariable=self.bak_var,
            from_=1.0,
            to=5.0,
            increment=0.1,
            width=10,
        )
        bak_spin.pack(side="left", padx=(5, 0))
        ttk.Label(bak_frame, text=t("bak_description", self.lang)).pack(
            side="left", padx=(10, 0)
        )

        # OVRL threshold
        ovrl_frame = ttk.Frame(settings_frame)
        ovrl_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(ovrl_frame, text="OVRL ≥", width=8).pack(side="left")
        ovrl_spin = ttk.Spinbox(
            ovrl_frame,
            textvariable=self.ovrl_var,
            from_=1.0,
            to=5.0,
            increment=0.1,
            width=10,
        )
        ovrl_spin.pack(side="left", padx=(5, 0))
        ttk.Label(ovrl_frame, text=t("ovrl_description", self.lang)).pack(
            side="left", padx=(10, 0)
        )

        # P808 threshold
        p808_frame = ttk.Frame(settings_frame)
        p808_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(p808_frame, text="P808 ≥", width=8).pack(side="left")
        p808_spin = ttk.Spinbox(
            p808_frame,
            textvariable=self.p808_var,
            from_=1.0,
            to=5.0,
            increment=0.1,
            width=10,
        )
        p808_spin.pack(side="left", padx=(5, 0))
        ttk.Label(p808_frame, text=t("p808_description", self.lang)).pack(
            side="left", padx=(10, 0)
        )

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")

        ttk.Button(
            button_frame,
            text=t("ok", self.lang),
            command=self._apply_and_close,
            width=12,
        ).pack(side="right", padx=(5, 0))

        ttk.Button(
            button_frame,
            text=t("cancel", self.lang),
            command=self.dialog.destroy,
            width=12,
        ).pack(side="right")

        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")

    def _apply_and_close(self):
        """Apply settings and close dialog"""
        try:
            sig = float(self.sig_var.get())
            bak = float(self.bak_var.get())
            ovrl = float(self.ovrl_var.get())
            p808 = float(self.p808_var.get())

            # Validate ranges (1.0 to 5.0)
            def clamp_value(value, min_val=1.0, max_val=5.0):
                return max(min_val, min(max_val, value))

            sig = clamp_value(sig)
            bak = clamp_value(bak)
            ovrl = clamp_value(ovrl)
            p808 = clamp_value(p808)

            self.quality_assessment.set_thresholds(
                sig=sig, bak=bak, ovrl=ovrl, p808=p808
            )

            # Call callback if provided to refresh file list
            if self.on_settings_changed:
                self.on_settings_changed()

            self.dialog.destroy()

        except ValueError:
            messagebox.showerror(t("error", self.lang), t("invalid_values", self.lang))


class MainWindow:
    """Main application window"""

    def __init__(self, output_dir: str = "./recordings", lang: str = DEFAULT_LANG):
        self.output_dir = output_dir
        self.lang = lang

        # Create main window
        self.root = tk.Tk()
        self.root.title(t("app_title", self.lang))

        # Set initial size and let window adjust to content
        self.root.geometry("600x700")  # Slightly larger initial size
        self.root.minsize(500, 500)  # Minimum size

        # Allow window to resize
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create recorder
        self.recorder = AudioRecorder(output_dir=output_dir)
        self.recorder.add_callback("on_level_update", self._on_level_update)
        self.recorder.add_callback("on_error", self._on_error)
        self.recorder.add_callback("on_stop", self._on_recording_stopped)

        # Create quality assessment
        self.quality_assessment = QualityAssessment(output_dir=output_dir)
        self.quality_assessment.add_callback(
            "on_assessment_complete", self._on_quality_complete
        )
        self.quality_assessment.add_callback("on_error", self._on_quality_error)

        # UI state
        self.is_recording = False
        self._prompt_gain_test_once = False
        # cache the latest gain test values to support live i18n re-rendering
        self._last_gain_test = None  # type: ignore[assignment]
        self.selected_output_device = None  # Selected output device index
        
        # Clipping detection state
        self.clipping_detected = False
        self.clipping_flash_timer = None
        self.clipping_flash_state = False

        # Setup UI and timers
        self._setup_style()
        self._setup_ui()
        self._start_update_timer()

        # Auto-adjust window size after UI is built
        self.root.after(100, self._auto_resize_window)

        # Refresh file list after quality assessment has loaded existing scores
        self.root.after(200, self._refresh_file_list_after_init)

        # Handle close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_header_image(self, target_width=None):
        """Load and resize the header image to fill width"""
        if not PIL_AVAILABLE:
            return None

        try:
            # Get the path to LiveSoundRecorder.png in resources folder
            current_dir = Path(__file__).parent.parent
            image_path = current_dir / "resources" / "LiveSoundRecorder.png"

            if not image_path.exists():
                return None

            # Load the image
            image = Image.open(image_path)

            # Maintain original aspect ratio, scale based on target width
            if target_width:
                original_width, original_height = image.size
                aspect_ratio = original_height / original_width

                # Calculate new height based on target width and original aspect ratio
                new_height = int(target_width * aspect_ratio)

                # Resize while maintaining aspect ratio
                image = image.resize(
                    (target_width, new_height), Image.Resampling.LANCZOS
                )
            else:
                # Default scaling behavior - scale to fit max width while maintaining aspect ratio
                max_width = 500
                original_width, original_height = image.size

                if original_width > max_width:
                    aspect_ratio = original_height / original_width
                    new_height = int(max_width * aspect_ratio)
                    image = image.resize(
                        (max_width, new_height), Image.Resampling.LANCZOS
                    )

            # Convert to PhotoImage for tkinter
            return ImageTk.PhotoImage(image)
        except Exception as e:
            print(f"Failed to load header image: {e}")
            return None

    def _refresh_file_list_after_init(self):
        """Refresh file list after initialization to show loaded quality scores"""
        if hasattr(self, "file_list"):
            self.file_list.refresh()

    def _auto_resize_window(self):
        """Automatically resize window to fit content"""
        self.root.update_idletasks()  # Ensure all widgets are rendered

        # Get the required size
        req_width = self.root.winfo_reqwidth()
        req_height = self.root.winfo_reqheight()

        # Add some padding
        width = max(600, req_width + 50)
        height = max(500, min(req_height + 100, 800))  # Cap height at 800px

        # Set the new size
        self.root.geometry(f"{width}x{height}")

        # Center the window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _setup_ui(self):
        """Setup the user interface with optimized layout"""
        # Create menu bar
        self._setup_menu_bar()

        # Create scrollable main frame
        canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Configure canvas to expand with window
        def _configure_canvas(event):
            canvas.itemconfig(canvas.find_all()[0], width=event.width)

        canvas.bind("<Configure>", _configure_canvas)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding="8")
        main_frame.pack(fill="both", expand=True)

        # Header section with image and title
        self._setup_header(main_frame)

        # Current folder display (read-only, compact)
        self.path_frame = ttk.LabelFrame(
            main_frame, text=t("current_folder", self.lang), padding="8"
        )
        self.path_frame.pack(fill="x", pady=(0, 10))

        path_row = ttk.Frame(self.path_frame)
        path_row.pack(fill="x")

        self.output_dir_var = tk.StringVar(value=str(Path(self.output_dir).resolve()))
        self._path_display = tk.Text(
            path_row,
            height=1,
            wrap="none",
            relief="flat",
            background=self.root.cget("background"),
        )
        self._path_display.insert("1.0", self.output_dir_var.get())
        self._path_display.configure(state="disabled")
        self._path_display.pack(side="left", fill="x", expand=True, padx=(6, 6))

        # Device settings and Gain test container (side by side)
        settings_container = ttk.Frame(main_frame)
        settings_container.pack(fill="x", pady=(0, 10))

        # Device and Audio settings frame (left side)
        self.device_frame = ttk.LabelFrame(
            settings_container, text=t("device_settings", self.lang), padding="8"
        )
        self.device_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Left column - Device selection
        left_col = ttk.Frame(self.device_frame)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.device_label = ttk.Label(left_col, text=t("audio_device", self.lang))
        self.device_label.pack(anchor="w")
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            left_col, textvariable=self.device_var, state="readonly", width=25
        )
        self.device_combo.pack(fill="x", pady=(2, 6))
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device_change)

        self.output_device_label = ttk.Label(
            left_col, text=t("output_device", self.lang)
        )
        self.output_device_label.pack(anchor="w")
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(
            left_col, textvariable=self.output_device_var, state="readonly", width=25
        )
        self.output_device_combo.pack(fill="x", pady=(2, 0))
        self.output_device_combo.bind(
            "<<ComboboxSelected>>", self._on_output_device_change
        )

        # Right column - Audio settings
        right_col = ttk.Frame(self.device_frame)
        right_col.pack(side="right", fill="y")

        self.sample_rate_label = ttk.Label(right_col, text=t("sample_rate", self.lang))
        self.sample_rate_label.pack(anchor="w")
        self.sample_rate_var = tk.StringVar(value="44100")
        ttk.Combobox(
            right_col,
            textvariable=self.sample_rate_var,
            values=["16000", "44100", "48000"],
            state="readonly",
            width=10,
        ).pack(pady=(2, 6))

        self.gain_label = ttk.Label(right_col, text=t("gain_db", self.lang))
        self.gain_label.pack(anchor="w")
        self.gain_var = tk.StringVar(value="0")
        ttk.Spinbox(
            right_col, textvariable=self.gain_var, from_=-20, to=20, width=10
        ).pack(pady=(2, 0))

        # Gain test frame (right side, vertical layout)
        self.gain_frame = ttk.LabelFrame(
            settings_container, text=t("gain_test", self.lang), padding="8"
        )
        self.gain_frame.pack(side="right", fill="both", padx=(5, 0))

        # Gain test button
        self.gain_test_btn = ttk.Button(
            self.gain_frame,
            text=t("gain_test_btn", self.lang),
            command=self._run_gain_test,
            width=15,
            style="Primary.TButton",
        )
        self.gain_test_btn.pack(pady=(0, 8))

        # Gain test result display
        self.gain_result_display = tk.Text(
            self.gain_frame,
            height=3,
            wrap="word",
            relief="sunken",
            borderwidth=1,
            background="#f8f9fa",
            font=("TkDefaultFont", 9),
            state="disabled",
            width=30,
        )
        self.gain_result_display.pack(fill="both", expand=True)

        # Recording controls (compact)
        self.controls_frame = ttk.LabelFrame(
            main_frame, text=t("recording_controls", self.lang), padding="8"
        )
        self.controls_frame.pack(fill="x", pady=(0, 10))

        # Filename row
        filename_frame = ttk.Frame(self.controls_frame)
        filename_frame.pack(fill="x", pady=(0, 6))

        self.filename_label = ttk.Label(filename_frame, text=t("filename", self.lang))
        self.filename_label.pack(side="left")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename_var = tk.StringVar(value=f"recording_{timestamp}")
        ttk.Entry(filename_frame, textvariable=self.filename_var, width=20).pack(
            side="left", padx=(8, 4), fill="x", expand=True
        )
        ttk.Label(filename_frame, text=".wav").pack(side="left")

        # Control buttons row - Enhanced layout with status indicator
        btn_frame = ttk.Frame(self.controls_frame)
        btn_frame.pack(fill="x", pady=(8, 0))

        # Status indicator frame (left side)
        status_frame = ttk.Frame(btn_frame)
        status_frame.pack(side="left", fill="y")

        # Recording status indicator light with transparent background
        self.status_indicator = tk.Canvas(
            status_frame, width=20, height=20, highlightthickness=0, bd=0, relief="flat"
        )
        # Set transparent background by matching parent's background
        try:
            parent_bg = self.root.cget("bg")
            self.status_indicator.configure(bg=parent_bg)
        except:
            # Fallback to system default
            self.status_indicator.configure(bg="SystemButtonFace")

        self.status_indicator.pack(side="left", padx=(0, 10), pady=5)
        self._update_status_indicator("ready")  # Initialize as ready state

        # Status text with better visibility and contrast
        self.recording_status_label = ttk.Label(
            status_frame,
            text=t("status_ready", self.lang),
            font=("TkDefaultFont", 10, "bold"),
            foreground="#28a745",
        )
        self.recording_status_label.pack(side="left", padx=(8, 20), pady=5)

        # Create a centered button container
        btn_container = ttk.Frame(btn_frame)
        btn_container.pack(expand=True)

        self.record_btn = ttk.Button(
            btn_container,
            text="🔴 " + t("start_recording", self.lang),  # Red circle for record
            command=self._toggle_recording,
            width=15,
            style="Accent.TButton",
        )
        self.record_btn.pack(side="left", padx=(0, 15))  # Increased padding

        self.pause_btn = ttk.Button(
            btn_container,
            text="⏸ " + t("pause", self.lang),  # Pause symbol
            command=self._toggle_pause,
            width=15,
            style="Secondary.TButton",
            state="disabled",
        )
        self.pause_btn.pack(side="left")

        # Level meter (compact height)
        self.meter_frame = ttk.LabelFrame(
            main_frame, text=t("volume", self.lang), padding="8"
        )
        self.meter_frame.pack(fill="x", pady=(0, 10))

        self.level_meter = LevelMeter(self.meter_frame, height=35)
        self.level_meter.pack(fill="x", pady=2)

        # Info display (single row)
        info_frame = ttk.Frame(self.meter_frame)
        info_frame.pack(fill="x", pady=(4, 0))

        self.duration_var = tk.StringVar(value="00:00")
        self.status_var = tk.StringVar(value=t("status_ready", self.lang))

        self.time_label = ttk.Label(info_frame, text=t("time", self.lang))
        self.time_label.pack(side="left")
        ttk.Label(
            info_frame,
            textvariable=self.duration_var,
            font=("TkDefaultFont", 10, "bold"),
        ).pack(side="left", padx=(5, 15))
        ttk.Label(info_frame, textvariable=self.status_var).pack(side="left")

        # File list (fixed height, compact)
        self.list_frame = ttk.LabelFrame(
            main_frame, text=t("files", self.lang), padding="8"
        )
        self.list_frame.pack(fill="x", pady=(0, 10))

        self.file_list = FileList(
            self.list_frame,
            self.output_dir,
            self._get_lang,
            self._get_output_device,
            self.quality_assessment,
        )
        self.file_list.pack(fill="x")

        # Bind mousewheel to canvas for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind to different events for different platforms
        canvas.bind("<MouseWheel>", _on_mousewheel)  # Windows
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))  # Linux

        # Initialize devices
        self._refresh_devices()

    def _setup_style(self):
        """Setup consistent styling for the application"""
        # Configure ttk style for modern look
        style = ttk.Style()

        # Define color scheme inspired by LiveSoundRecorder3.png
        self.colors = {
            "primary": "#2E86AB",  # Blue
            "secondary": "#A23B72",  # Purple/Magenta
            "accent": "#F18F01",  # Orange
            "success": "#28a745",  # Green for success/ready
            "danger": "#dc3545",  # Red for recording/danger
            "warning": "#ffc107",  # Yellow for pause/warning
            "background": "#F8F9FA",  # Light background
            "surface": "#FFFFFF",  # White surface
            "text": "#2C3E50",  # Dark text
            "text_secondary": "#6C757D",  # Secondary text
        }

        # Configure enhanced button styles with better contrast
        try:
            # Accent button style (Start Recording)
            style.configure(
                "Accent.TButton",
                background=self.colors["accent"],
                foreground="black",
                borderwidth=0,
                focuscolor="none",
                padding=(10, 8),
            )
            style.map(
                "Accent.TButton",
                background=[
                    ("active", "#E17100"),  # Darker orange on hover
                    ("pressed", "#D16600"),
                ],
            )  # Even darker when pressed

            # Danger button style (Stop Recording)
            style.configure(
                "Danger.TButton",
                background=self.colors["danger"],
                foreground="black",
                borderwidth=0,
                focuscolor="none",
                padding=(10, 8),
            )
            style.map(
                "Danger.TButton",
                background=[
                    ("active", "#C82333"),  # Darker red on hover
                    ("pressed", "#BD2130"),
                ],
            )  # Even darker when pressed

            # Secondary button style (Pause/Resume)
            style.configure(
                "Secondary.TButton",
                background=self.colors["secondary"],
                foreground="black",
                borderwidth=0,
                focuscolor="none",
                padding=(8, 6),
            )
            style.map(
                "Secondary.TButton",
                background=[
                    ("active", "#8E2F5F"),  # Darker purple on hover
                    ("pressed", "#7A2952"),  # Even darker when pressed
                    ("disabled", "#6C757D"),
                ],
            )  # Gray when disabled

            # Primary button style (for other actions)
            style.configure(
                "Primary.TButton",
                background=self.colors["primary"],
                foreground="black",
                borderwidth=0,
                focuscolor="none",
            )
            style.map(
                "Primary.TButton",
                background=[
                    ("active", "#236B87"),  # Darker blue on hover
                    ("pressed", "#1E5A73"),
                ],
            )  # Even darker when pressed

        except Exception:
            # Fallback if color configuration fails
            pass

    def _setup_header(self, parent):
        """Setup the header section with image and title"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill="x", pady=(0, 15))

        # Update to get window width and load appropriately sized image
        def update_header_image():
            try:
                parent.update_idletasks()
                window_width = parent.winfo_width()
                # Use reasonable percentage of window width, considering aspect ratio
                target_width = max(400, min(800, int(window_width * 0.95)))
                self.header_image = self._load_header_image(target_width)

                if self.header_image and hasattr(self, "image_label"):
                    self.image_label.config(image=self.header_image)
                    self.image_label.image = self.header_image
                elif self.header_image:
                    # Create image label for first time
                    self.image_label = ttk.Label(header_frame, image=self.header_image)
                    self.image_label.pack(anchor="center", pady=(0, 5))
                    # Keep a reference to prevent garbage collection
                    self.image_label.image = self.header_image
            except Exception as e:
                print(f"Error updating header image: {e}")

        # Bind window resize event to update image
        def on_window_resize(event):
            if event.widget == self.root:
                # Delay update to avoid too frequent calls
                if hasattr(self, "_resize_timer"):
                    self.root.after_cancel(self._resize_timer)
                self._resize_timer = self.root.after(300, update_header_image)

        self.root.bind("<Configure>", on_window_resize)

        # Try to load the header image with default size first
        self.header_image = self._load_header_image()

        if self.header_image:
            # Create image label
            self.image_label = ttk.Label(header_frame, image=self.header_image)
            self.image_label.pack(anchor="center", pady=(0, 5))

            # Keep a reference to prevent garbage collection
            self.image_label.image = self.header_image

            # Schedule image resize after window is fully rendered
            header_frame.after(100, update_header_image)
        else:
            # Fallback to text title if image can't be loaded
            self.title_label = ttk.Label(
                header_frame,
                text=t("app_heading", self.lang),
                font=("TkDefaultFont", 16, "bold"),
                foreground="#2E86AB",  # Modern blue color
            )
            self.title_label.pack(pady=(0, 10))

        # Add a subtle separator line
        separator = ttk.Separator(header_frame, orient="horizontal")
        separator.pack(fill="x", pady=(5, 0))

    def _setup_menu_bar(self):
        """Setup the menu bar with language, folder, and quality settings"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=t("file", self.lang), menu=self.file_menu)
        self.file_menu.add_command(
            label=t("browse", self.lang), command=self._choose_folder
        )
        self.file_menu.add_command(
            label=t("open_folder", self.lang), command=self._open_folder
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(label=t("exit", self.lang), command=self._on_close)

        # Language menu
        self.lang_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=t("language", self.lang), menu=self.lang_menu)

        # Add language options as radio buttons
        self.lang_var = tk.StringVar(value=self.lang)
        for code, name in LANGS.items():
            self.lang_menu.add_radiobutton(
                label=f"{code}: {name}",
                variable=self.lang_var,
                value=code,
                command=lambda c=code: self._on_lang_menu_change(c),
            )

        # Quality menu
        self.quality_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(
            label=t("quality_settings", self.lang), menu=self.quality_menu
        )
        self.quality_menu.add_command(
            label=t("quality_thresholds", self.lang), command=self._open_quality_dialog
        )

    def _get_output_device(self):
        """Get the currently selected output device index"""
        return self.selected_output_device

    def _set_output_dir(self, new_dir: str):
        self.output_dir = new_dir
        try:
            self.recorder.set_output_dir(new_dir)
        except Exception as e:
            messagebox.showerror(t("error", self.lang), str(e))
            return
        self.output_dir_var.set(str(Path(new_dir).resolve()))
        self._path_display.configure(state="normal")
        self._path_display.delete("1.0", "end")
        self._path_display.insert("1.0", self.output_dir_var.get())
        self._path_display.configure(state="disabled")
        # Update file list
        self.file_list.output_dir = Path(new_dir)
        self.file_list.refresh()
        self.status_var.set(t("folder_changed", self.lang))

    def _choose_folder(self):
        if self.is_recording:
            messagebox.showwarning(
                t("error", self.lang), t("cannot_test_while_recording", self.lang)
            )
            return
        new_dir = filedialog.askdirectory(initialdir=self.output_dir)
        if new_dir:
            self._set_output_dir(new_dir)

    def _open_folder(self):
        try:
            path = Path(self.output_dir)
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)])
            elif sys.platform == "win32":
                subprocess.run(["start", str(path)], shell=True)
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception as e:
            messagebox.showerror(t("error", self.lang), str(e))

    def _setup_quality_settings(self):
        """Setup quality assessment settings UI - moved to menu dialog"""
        # This method is no longer used but kept for compatibility
        pass

    def _update_quality_thresholds(self):
        """Update quality assessment thresholds - moved to dialog"""
        # This method is no longer used but kept for compatibility
        pass

    def _update_status_indicator(self, status):
        """Update the recording status indicator light

        Args:
            status: "ready", "recording", "paused", "error"
        """
        if not hasattr(self, "status_indicator"):
            return

        # Clear canvas
        self.status_indicator.delete("all")

        # Ensure transparent background is maintained
        try:
            parent_bg = self.root.cget("bg")
            self.status_indicator.configure(bg=parent_bg)
        except:
            # Fallback to system default
            self.status_indicator.configure(bg="SystemButtonFace")

        # Define colors for different states
        colors = {
            "ready": "#28a745",  # Green - ready to record
            "recording": "#dc3545",  # Red - actively recording
            "paused": "#ffc107",  # Yellow - paused
            "error": "#6c757d",  # Gray - error state
        }

        color = colors.get(status, "#6c757d")

        # Draw the status indicator circle with better visibility
        self.status_indicator.create_oval(
            3, 3, 17, 17, fill=color, outline="#ffffff", width=2
        )

        # Add pulsing effect for recording state
        if status == "recording":
            self._animate_recording_indicator()

    def _animate_recording_indicator(self):
        """Animate the recording indicator with a pulsing effect"""
        if not hasattr(self, "status_indicator") or not self.is_recording:
            return

        # Simple pulsing animation by changing opacity
        def pulse():
            if not self.is_recording:
                return

            # Only pulse if not in clipping mode
            if not self.clipping_detected:
                # Recreate indicator with slight transparency variation
                self.status_indicator.delete("all")
                try:
                    parent_bg = self.root.cget("bg")
                    self.status_indicator.configure(bg=parent_bg)
                except:
                    self.status_indicator.configure(bg="SystemButtonFace")

                self.status_indicator.create_oval(
                    3, 3, 17, 17, fill="#dc3545", outline="#ffffff", width=2
                )

            # Schedule next pulse
            if self.is_recording:
                self.root.after(1000, pulse)

        pulse()

    def _start_clipping_flash(self):
        """Start flashing indicator for clipping warning"""
        if not self.clipping_detected:
            self.clipping_detected = True
            self._flash_clipping_indicator()

    def _stop_clipping_flash(self):
        """Stop clipping flash and reset to normal recording state"""
        self.clipping_detected = False
        self.clipping_flash_state = False
        if self.clipping_flash_timer:
            self.root.after_cancel(self.clipping_flash_timer)
            self.clipping_flash_timer = None
        
        # Reset to normal recording state if still recording
        if self.is_recording:
            self._update_status_indicator("recording")

    def _flash_clipping_indicator(self):
        """Flash the status indicator between red and orange to warn of clipping"""
        if not self.clipping_detected or not self.is_recording:
            return

        # Clear canvas
        self.status_indicator.delete("all")

        # Ensure transparent background is maintained
        try:
            parent_bg = self.root.cget("bg")
            self.status_indicator.configure(bg=parent_bg)
        except:
            self.status_indicator.configure(bg="SystemButtonFace")

        # Alternate between bright red and orange for clipping warning
        if self.clipping_flash_state:
            color = "#ff0000"  # Bright red for clipping
            outline_color = "#ffff00"  # Yellow outline for extra visibility
        else:
            color = "#ff6600"  # Orange
            outline_color = "#ffffff"  # White outline

        self.clipping_flash_state = not self.clipping_flash_state

        # Draw flashing indicator with enhanced visibility
        self.status_indicator.create_oval(
            2, 2, 18, 18, fill=color, outline=outline_color, width=2
        )

        # Schedule next flash (faster than normal pulse)
        if self.clipping_detected and self.is_recording:
            self.clipping_flash_timer = self.root.after(200, self._flash_clipping_indicator)
        
        # Auto-stop flashing after 3 seconds of no new clipping detection
        if not hasattr(self, '_clipping_timeout') or self._clipping_timeout is None:
            self._clipping_timeout = self.root.after(3000, self._auto_stop_clipping_flash)

    def _auto_stop_clipping_flash(self):
        """Automatically stop clipping flash after timeout"""
        self._clipping_timeout = None
        self.clipping_detected = False
        self._stop_clipping_flash()
        # 恢復狀態文字
        if self.is_recording and hasattr(self, "recording_status_label"):
            self.recording_status_label.config(
                text=t("status_recording", self.lang), foreground="#dc3545"
            )

    def _run_gain_test(self):
        if self.is_recording or self.recorder.state.value != "idle":
            messagebox.showwarning(
                t("error", self.lang), t("cannot_test_while_recording", self.lang)
            )
            return
        self._update_config()
        self.gain_test_btn.config(text=t("gain_test_run", self.lang), state="disabled")
        # Clear gain result display
        self.gain_result_display.config(state="normal")
        self.gain_result_display.delete("1.0", "end")
        self.gain_result_display.config(state="disabled")

        def _do_test():
            result = self.recorder.test_gain(3.0)

            def _after():
                self.gain_test_btn.config(
                    text=t("gain_test_btn", self.lang), state="normal"
                )
                if not result:
                    messagebox.showerror(
                        t("error", self.lang), t("cli_cannot_start", self.lang)
                    )
                    return
                rms = result.get("rms", -60.0)
                peak = result.get("peak", -60.0)
                target_rms = -22.0
                suggest = max(-20.0, min(20.0, target_rms - rms))
                # cache and render using current language
                self._last_gain_test = {"rms": rms, "peak": peak, "suggest": suggest}
                self._render_gain_result()
                if messagebox.askyesno(
                    t("confirm", self.lang), t("apply_gain", self.lang, gain=suggest)
                ):
                    try:
                        current = float(self.gain_var.get())
                    except ValueError:
                        current = 0.0
                    new_gain = max(-20.0, min(20.0, current + suggest))
                    self.gain_var.set(f"{new_gain:.0f}")
                    self._update_config()

            self.root.after(0, _after)

        self.root.after(10, _do_test)

    def _refresh_devices(self):
        """Refresh device list"""
        devices = self.recorder.get_devices()
        device_list = [f"{idx}: {name}" for idx, name in devices.items()]

        self.device_combo["values"] = device_list
        if device_list:
            self.device_combo.current(0)
            first_idx = list(devices.keys())[0]
            self.recorder.set_device(first_idx)

        # Refresh output devices
        try:
            player = ControllableAudioPlayer()
            output_devices = player.get_output_devices()
            output_device_list = [
                f"{idx}: {name}" for idx, name in output_devices.items()
            ]

            self.output_device_combo["values"] = output_device_list
            if output_device_list:
                self.output_device_combo.current(0)
                first_output_idx = list(output_devices.keys())[0]
                self.selected_output_device = first_output_idx

            player.cleanup()
        except Exception as e:
            print(f"Warning: Could not load output devices: {e}")
            self.output_device_combo["values"] = []

    def _on_device_change(self, event=None):
        """Handle device selection change"""
        selection = self.device_combo.get()
        if selection:
            device_idx = int(selection.split(":")[0])
            self.recorder.set_device(device_idx)

    def _on_output_device_change(self, event=None):
        """Handle output device selection change"""
        selection = self.output_device_combo.get()
        if selection:
            self.selected_output_device = int(selection.split(":")[0])
            # Update the file list's output device setting
            if hasattr(self, "file_list") and hasattr(self.file_list, "player"):
                self.file_list.player.set_output_device(self.selected_output_device)

    def _toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            # Start recording
            self._update_config()
            base_filename = self.filename_var.get().strip() or "recording"

            # Always add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_filename}_{timestamp}"

            if not self._prompt_gain_test_once:
                self._prompt_gain_test_once = True
                if messagebox.askyesno(
                    t("confirm", self.lang), t("prompt_gain_test", self.lang)
                ):
                    self._run_gain_test()
                    return

            if self.recorder.start_recording(filename):
                self.is_recording = True
                self.record_btn.config(
                    text="⏹ " + t("stop_recording", self.lang), style="Danger.TButton"
                )
                self.pause_btn.config(state="normal")
                self.status_var.set(t("status_recording", self.lang))

                # Update status indicator and label
                self._update_status_indicator("recording")
                if hasattr(self, "recording_status_label"):
                    self.recording_status_label.config(
                        text=t("status_recording", self.lang), foreground="#dc3545"
                    )
            else:
                self._update_status_indicator("error")
                if hasattr(self, "recording_status_label"):
                    self.recording_status_label.config(
                        text=t("error", self.lang), foreground="#dc3545"
                    )
                messagebox.showerror(
                    t("error", self.lang), t("cannot_start", self.lang)
                )
        else:
            # Stop recording
            saved_path = self.recorder.stop_recording()
            self.is_recording = False
            self.record_btn.config(
                text="🔴 " + t("start_recording", self.lang), style="Accent.TButton"
            )
            self.pause_btn.config(text="⏸ " + t("pause", self.lang), state="disabled")

            if saved_path:
                self.status_var.set(t("status_saved", self.lang))
                self._update_status_indicator("ready")
                if hasattr(self, "recording_status_label"):
                    self.recording_status_label.config(
                        text=t("status_ready", self.lang), foreground="#28a745"
                    )
                self.file_list.refresh()
                messagebox.showinfo(
                    t("done", self.lang),
                    f"{t('saved_as', self.lang)}: {Path(saved_path).name}",
                )
            else:
                self.status_var.set(t("save_failed", self.lang))
                self._update_status_indicator("error")
                if hasattr(self, "recording_status_label"):
                    self.recording_status_label.config(
                        text=t("save_failed", self.lang), foreground="#dc3545"
                    )

    def _toggle_pause(self):
        """Toggle pause state"""
        if self.recorder.state.value == "recording":
            if self.recorder.pause_recording():
                self.pause_btn.config(text="▶ " + t("resume", self.lang))
                self.status_var.set(t("status_paused", self.lang))

                # Update status indicator and label for paused state
                self._update_status_indicator("paused")
                if hasattr(self, "recording_status_label"):
                    self.recording_status_label.config(
                        text=t("status_paused", self.lang), foreground="#ffc107"
                    )
        else:
            if self.recorder.resume_recording():
                self.pause_btn.config(text="⏸ " + t("pause", self.lang))
                self.status_var.set(t("status_recording", self.lang))

                # Update status indicator and label for recording state
                self._update_status_indicator("recording")
                if hasattr(self, "recording_status_label"):
                    self.recording_status_label.config(
                        text=t("status_recording", self.lang), foreground="#dc3545"
                    )

    def _update_config(self):
        """Update recorder configuration"""
        try:
            sample_rate = int(self.sample_rate_var.get())
            pre_gain_db = float(self.gain_var.get())

            self.recorder.set_config(sample_rate=sample_rate, pre_gain_db=pre_gain_db)
        except ValueError:
            pass

    def _start_update_timer(self):
        """Start UI update timer"""
        self._update_ui()
        self.root.after(100, self._start_update_timer)

    def _update_ui(self):
        """Update UI elements"""
        if self.is_recording:
            info = self.recorder.get_info()

            # Update level meter
            self.level_meter.update(info["rms_level"], info["peak_level"])

            # Update duration
            duration = info["duration"]
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            self.duration_var.set(f"{minutes:02d}:{seconds:02d}")
            
            # Check for clipping from the latest audio data
            # We'll check this in the callback directly
        else:
            self.level_meter.update(-60, -60)
            # Stop clipping flash when not recording
            self._stop_clipping_flash()
            if not hasattr(self, "_reset") or not self._reset:
                self.duration_var.set("00:00")
                if self.recorder.state.value == "idle":
                    self.status_var.set(t("status_ready", self.lang))
                self._reset = True

    def _on_level_update(self, data):
        """Handle level update callback"""
        # Check for clipping detection
        if data and data.get("clipping", False):
            self._start_clipping_flash()
            # 顯示明顯的警告字詞
            if hasattr(self, "recording_status_label"):
                self.recording_status_label.config(
                    text="Clipping! 請降低音量", foreground="#ff0000"
                )
        else:
            # 若無 clipping，恢復正常狀態文字（不立即消除，交由 timeout 控制）
            pass

    def _on_error(self, error):
        """Handle error callback"""
        messagebox.showerror(t("error", self.lang), error)

    def _on_recording_stopped(self, saved_path):
        """Handle recording stopped callback"""
        if saved_path and self.quality_assessment:
            # Start quality assessment in background
            self.quality_assessment.assess_audio_async(saved_path)

    def _on_quality_complete(self, result):
        """Handle quality assessment completion"""
        # Refresh file list to show updated quality status
        if hasattr(self, "file_list"):
            self.file_list.refresh()

    def _on_quality_error(self, error):
        """Handle quality assessment error"""
        print(f"Quality assessment error: {error}")
        # Don't show error dialog to user as it's not critical

    def _on_close(self):
        """Handle window close"""
        if self.is_recording:
            if messagebox.askyesno(
                t("confirm", self.lang), t("close_confirm", self.lang)
            ):
                self.recorder.stop_recording()
            else:
                return

        # Clean up file list player
        if hasattr(self, "file_list"):
            self.file_list.cleanup()

        self.recorder.cleanup()
        self.root.destroy()

    def run(self):
        """Run the application"""
        self.root.mainloop()

    def _get_lang(self) -> str:
        return self.lang

    def _on_lang_menu_change(self, lang_code):
        """Handle language change from menu"""
        if lang_code and lang_code != self.lang:
            self.lang = lang_code
            self.lang_var.set(lang_code)
            self._apply_i18n()

    def _open_quality_dialog(self):
        """Open quality settings dialog"""

        # Create callback function to refresh file list
        def on_settings_changed():
            if hasattr(self, "file_list"):
                self.file_list.refresh()

        dialog = QualitySettingsDialog(
            self.root,
            self.quality_assessment,
            self.lang,
            on_settings_changed=on_settings_changed,
        )
        dialog.show()

    def _on_lang_change(self, event=None):
        """Handle language change from combobox"""
        value = self.lang_combo.get()
        code = value.split(":")[0].strip() if ":" in value else value.strip()
        if code and code != self.lang:
            self.lang = code
            self._apply_i18n()

    def _apply_i18n(self):
        """Apply translations to all UI texts"""
        # Window title and headings
        self.root.title(t("app_title", self.lang))
        # Only update title_label if it exists (fallback when image is not available)
        if hasattr(self, "title_label"):
            self.title_label.config(text=t("app_heading", self.lang))

        # Update menu items
        self.file_menu.entryconfig(0, label=t("browse", self.lang))
        self.file_menu.entryconfig(1, label=t("open_folder", self.lang))
        self.file_menu.entryconfig(3, label=t("exit", self.lang))

        self.menubar.entryconfig(0, label=t("file", self.lang))
        self.menubar.entryconfig(1, label=t("language", self.lang))
        self.menubar.entryconfig(2, label=t("quality_settings", self.lang))

        self.quality_menu.entryconfig(0, label=t("quality_thresholds", self.lang))

        # Frames
        self.path_frame.config(text=t("current_folder", self.lang))
        self.device_frame.config(text=t("device_settings", self.lang))
        self.gain_frame.config(text=t("gain_test", self.lang))
        self.controls_frame.config(text=t("recording_controls", self.lang))
        self.meter_frame.config(text=t("volume", self.lang))
        self.list_frame.config(text=t("files", self.lang))

        # Labels
        self.device_label.config(text=t("audio_device", self.lang))
        self.output_device_label.config(text=t("output_device", self.lang))
        self.sample_rate_label.config(text=t("sample_rate", self.lang))
        self.gain_label.config(text=t("gain_db", self.lang))
        self.filename_label.config(text=t("filename", self.lang))
        self.time_label.config(text=t("time", self.lang))
        self.gain_test_btn.config(text=t("gain_test_btn", self.lang))

        # Buttons
        self.record_btn.config(
            text=(
                "🔴 " + t("start_recording", self.lang)
                if not self.is_recording
                else "⏹ " + t("stop_recording", self.lang)
            )
        )
        # Pause button respects state
        if self.is_recording and self.recorder.state.value == "paused":
            self.pause_btn.config(text="▶ " + t("resume", self.lang))
        else:
            self.pause_btn.config(text="⏸ " + t("pause", self.lang))
        # Status preset
        if not self.is_recording and getattr(self, "_reset", False):
            if self.recorder.state.value == "idle":
                self.status_var.set(t("status_ready", self.lang))
        # File list buttons
        self.file_list.refresh_texts()
        # Re-render last gain test result in the new language
        self._render_gain_result()

    def _render_gain_result(self):
        """Render the cached gain test result according to current language."""
        # Enable editing temporarily
        self.gain_result_display.config(state="normal")
        self.gain_result_display.delete("1.0", "end")

        if self._last_gain_test:
            vals = self._last_gain_test
            result_text = t(
                "gain_test_result",
                self.lang,
                rms=vals["rms"],
                peak=vals["peak"],
                suggest=vals["suggest"],
            )
            self.gain_result_display.insert("1.0", result_text)

        # Disable editing again
        self.gain_result_display.config(state="disabled")


def create_gui(output_dir: str = "./recordings", lang: str = DEFAULT_LANG):
    """Create and run GUI application"""
    with suppress_stderr():
        app = MainWindow(output_dir, lang=lang)
    app.run()


if __name__ == "__main__":
    create_gui()
