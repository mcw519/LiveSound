import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pyaudio
import soundfile


class RecordingState(Enum):
    """Recording state enumeration"""

    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"


@dataclass
class AudioConfig:
    """Audio configuration"""

    sample_rate: int = 44100
    channels: int = 1
    chunk_size: int = 1024
    device_index: Optional[int] = None
    pre_gain_db: float = 0.0


class AudioRecorder:
    """Enhanced audio recorder with modern features"""

    def __init__(
        self, config: Optional[AudioConfig] = None, output_dir: str = "./recordings"
    ):
        self.config = config or AudioConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Audio components
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None

        # Recording state
        self.state = RecordingState.IDLE
        self.recorded_frames = []
        self.current_filename = None

        # Threading
        self._recording_thread = None
        self._stop_event = threading.Event()

        # Statistics
        self.start_time = None
        self.duration = 0.0
        self.rms_level = -60.0
        self.peak_level = -60.0

        # Callbacks
        self.callbacks = {}

    def get_devices(self) -> Dict[int, str]:
        """Get available audio input devices"""
        devices = {}
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    devices[i] = device_info["name"]
            except:
                continue
        return devices

    def list_devices(self):
        """Print available devices"""
        devices = self.get_devices()
        print("Available audio input devices:")
        for idx, name in devices.items():
            print(f"  {idx}: {name}")

    def set_device(self, device_index: int):
        """Set audio input device"""
        self.config.device_index = device_index

    def set_config(self, **kwargs):
        """Update audio configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def set_output_dir(self, output_dir: str):
        """Update output directory for recordings"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.output_dir = path

    def add_callback(self, event: str, callback: Callable):
        """Add event callback"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)

    def _trigger_callback(self, event: str, data=None):
        """Trigger event callbacks"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(data) if data else callback()
            except:
                pass

    def _calculate_levels(self, data: np.ndarray) -> tuple:
        """Calculate audio levels in dB"""
        if len(data) == 0:
            return -60.0, -60.0

        rms = np.sqrt(np.mean(data**2))
        peak = np.max(np.abs(data))

        rms_db = 20 * np.log10(max(rms, 1e-10))
        peak_db = 20 * np.log10(max(peak, 1e-10))

        return rms_db, peak_db

    def _open_stream(self) -> bool:
        """Open audio stream"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.device_index,
                frames_per_buffer=self.config.chunk_size,
            )
            return True
        except Exception as e:
            self._trigger_callback("on_error", str(e))
            return False

    def _close_stream(self):
        """Close audio stream"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def _recording_loop(self):
        """Main recording loop"""
        self.start_time = time.time()

        while not self._stop_event.is_set():
            if self.state == RecordingState.RECORDING:
                try:
                    # Read audio data
                    data = self.stream.read(
                        self.config.chunk_size, exception_on_overflow=False
                    )
                    audio_data = np.frombuffer(data, dtype=np.float32)

                    # Apply gain (make a copy to avoid read-only array error)
                    if self.config.pre_gain_db != 0:
                        gain = 10 ** (self.config.pre_gain_db / 20)
                        audio_data = audio_data.copy() * gain

                    # Store data
                    self.recorded_frames.append(audio_data)

                    # Calculate levels
                    self.rms_level, peak_db = self._calculate_levels(audio_data)
                    self.peak_level = max(self.peak_level, peak_db)
                    self.duration = time.time() - self.start_time

                    # Detect clipping (when audio signal reaches or exceeds full scale)
                    clipping_detected = np.any(np.abs(audio_data) >= 0.99)

                    # Trigger callback
                    self._trigger_callback(
                        "on_level_update",
                        {
                            "rms": self.rms_level,
                            "peak": self.peak_level,
                            "duration": self.duration,
                            "clipping": clipping_detected,
                        },
                    )

                except Exception as e:
                    self._trigger_callback("on_error", str(e))
                    break
            else:
                time.sleep(0.01)

    def start_recording(self, filename: Optional[str] = None) -> bool:
        """Start recording"""
        if self.state != RecordingState.IDLE:
            return False

        # Generate filename
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}"

        if not filename.endswith(".wav"):
            filename += ".wav"

        self.current_filename = filename

        # Reset state
        self.recorded_frames = []
        self.peak_level = -60.0
        self.duration = 0.0
        self._stop_event.clear()

        # Open stream
        if not self._open_stream():
            return False

        # Start recording
        self.state = RecordingState.RECORDING
        self._recording_thread = threading.Thread(target=self._recording_loop)
        self._recording_thread.start()

        self._trigger_callback("on_start")
        return True

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save file"""
        if self.state == RecordingState.IDLE:
            return None

        self.state = RecordingState.IDLE
        self._stop_event.set()

        # Wait for thread
        if self._recording_thread:
            self._recording_thread.join(timeout=1.0)

        # Close stream
        self._close_stream()

        # Save file
        saved_path = None
        if self.recorded_frames and self.current_filename:
            output_path = self.output_dir / self.current_filename
            saved_path = self._save_audio(output_path)

        # Trigger callback with saved path for quality assessment
        self._trigger_callback("on_stop", saved_path)
        return saved_path

    def pause_recording(self) -> bool:
        """Pause recording"""
        if self.state == RecordingState.RECORDING:
            self.state = RecordingState.PAUSED
            self._trigger_callback("on_pause")
            return True
        return False

    def resume_recording(self) -> bool:
        """Resume recording"""
        if self.state == RecordingState.PAUSED:
            self.state = RecordingState.RECORDING
            self._trigger_callback("on_resume")
            return True
        return False

    def _save_audio(self, output_path: Path) -> Optional[str]:
        """Save audio to file"""
        try:
            audio_data = np.concatenate(self.recorded_frames)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            soundfile.write(
                str(output_path), audio_data, self.config.sample_rate, subtype="PCM_16"
            )
            return str(output_path)
        except Exception as e:
            self._trigger_callback("on_error", str(e))
            return None

    def test_gain(self, duration_sec: float = 3.0) -> Optional[Dict[str, float]]:
        """Synchronously test microphone levels for a short duration.
        Returns dict with 'rms' and 'peak' in dBFS if successful.
        Will not run if currently recording or paused.
        """
        if self.state != RecordingState.IDLE:
            return None

        try:
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.device_index,
                frames_per_buffer=self.config.chunk_size,
            )
        except Exception as e:
            self._trigger_callback("on_error", str(e))
            return None

        start = time.time()
        rms_vals = []
        peak_vals = []
        try:
            while time.time() - start < duration_sec:
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.float32)
                if audio.size == 0:
                    continue
                # use raw input without pre-gain for measurement
                rms, peak = self._calculate_levels(audio)
                rms_vals.append(rms)
                peak_vals.append(peak)
        except Exception as e:
            self._trigger_callback("on_error", str(e))
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

        if not rms_vals:
            return None

        avg_rms = float(np.mean(rms_vals))
        max_peak = float(np.max(peak_vals))
        return {"rms": avg_rms, "peak": max_peak}

    def get_info(self) -> Dict:
        """Get current recording info"""
        return {
            "state": self.state.value,
            "duration": self.duration,
            "rms_level": self.rms_level,
            "peak_level": self.peak_level,
            "filename": self.current_filename,
        }

    def cleanup(self):
        """Clean up resources"""
        if self.state != RecordingState.IDLE:
            self.stop_recording()

        if self.audio:
            self.audio.terminate()
