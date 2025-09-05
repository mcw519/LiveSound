"""
DNSMOS Audio Quality Metrics

This module provides DNSMOS (Deep Noise Suppression Mean Opinion Score) functionality
for evaluating audio quality. The DNSMOS models are from Microsoft's DNS-Challenge project.

License Information:
- DNSMOS models and related code: MIT License (Copyright (c) Microsoft Corporation)
- Source: https://github.com/microsoft/DNS-Challenge

Citations:
When using this module, please cite the following papers:

@inproceedings{reddy2021dnsmos,
  title={DNSMOS: A Non-Intrusive Perceptual Objective Speech Quality metric to evaluate Noise Suppressors},
  author={Reddy, Chandan KA and Gopal, Vishak and Cutler, Ross},
  booktitle={ICASSP},
  year={2021}
}

@inproceedings{reddy2022dnsmos,
  title={DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors},
  author={Reddy, Chandan KA and Gopal, Vishak and Cutler, Ross},
  booktitle={ICASSP},
  year={2022}
}
"""

import os
from typing import Any, Optional

import librosa
import numpy as np
import onnxruntime as ort

import livesound


class DNSMOS:
    """
    DNSMOS metric for evaluating audio quality using ONNX models.
    
    This implementation is based on Microsoft's DNS-Challenge project.
    
    Args:
        model_root (str): Path to the directory containing the ONNX models.
        personalized_MOS (bool): Whether to use the personalized MOS model.
        n_workers (int): Number of worker threads for parallel processing.
        verbose (bool): If True, print additional information during processing.
        gpu_ids (List[int]): List of GPU IDs to use for multi-GPU support.
        gpu_mem_limit_gb (int): GPU memory limit per session in GB. Set to None to disable limit.
    """

    def __init__(
        self,
        verbose: bool = False,
        model_dir: Optional[str] = None,
    ) -> None:

        self.default_model_dir = os.path.join(livesound.__path__[0], "libs", "metrics")
        if model_dir is not None:
            self.default_model_dir = model_dir

        self.p808_model_path = os.path.join(self.default_model_dir, "dnsmos_p808.onnx")
        self.primary_model_path = os.path.join(
            self.default_model_dir, "dnsmos_primary.onnx"
        )

        self.input_length = 9.01
        self.sampling_rate = 16000
        self.verbose = verbose

        self.__init_model()

    def __init_model(self):
        """Initialize ONNX sessions for the primary and P808 models."""
        try:
            provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.onnx_sess = ort.InferenceSession(
                self.primary_model_path, providers=provider
            )
            self.p808_onnx_sess = ort.InferenceSession(
                self.p808_model_path, providers=provider
            )
        except Exception as e:
            print(f"WARNNING: Error initializing ONNX models: {e}")
            provider = ["CPUExecutionProvider"]
            self.onnx_sess = ort.InferenceSession(
                self.primary_model_path, providers=provider
            )
            self.p808_onnx_sess = ort.InferenceSession(
                self.p808_model_path, providers=provider
            )
        finally:
            print("ONNX models initialized.")

    def audio_melspec(
        self,
        audio: Any,
        n_mels: int = 120,
        frame_size: int = 320,
        hop_length: int = 160,
        sr: int = 16000,
        to_db: bool = True,
    ) -> Any:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(
        self, sig: Any, bak: Any, ovr: Any, is_personalized_MOS: bool
    ) -> tuple[Any, Any, Any]:
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def hanle_input_wav_type(self, x: Any) -> np.ndarray:
        if isinstance(x, str):
            audio, _ = librosa.load(x, sr=self.sampling_rate, mono=True)
        elif isinstance(x, np.ndarray):
            audio = x
        else:
            raise TypeError("Unsupported input type")

        return audio

    def get_score(self, x: Any) -> dict:
        """Compute DNSMOS scores for a single audio input."""
        audio = self.hanle_input_wav_type(x)

        len_samples = int(self.input_length * self.sampling_rate)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = (
            int(np.floor(len(audio) / self.sampling_rate) - self.input_length) + 1
        )
        hop_len_samples = self.sampling_rate
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int(
                    (idx + self.input_length) * hop_len_samples
                )
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(
                self.audio_melspec(audio=audio_seg[:-160])
            ).astype("float32")[np.newaxis, :, :]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}

            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw, False
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        res = {
            "p808": round(np.mean(predicted_p808_mos), 2),
            "sig": round(np.mean(predicted_mos_sig_seg), 2),
            "bak": round(np.mean(predicted_mos_bak_seg), 2),
            "ovrl": round(np.mean(predicted_mos_ovr_seg), 2),
        }

        return res
