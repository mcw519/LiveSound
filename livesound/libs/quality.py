import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .metrics import DNSMOS


@dataclass
class QualityThresholds:
    """Quality assessment thresholds (values must be between 1.0 and 5.0)"""

    sig: float = 3.0
    bak: float = 3.0
    ovrl: float = 3.0
    p808: float = 3.0

    def __post_init__(self):
        """Validate threshold values after initialization"""

        def clamp_value(value, min_val=1.0, max_val=5.0):
            return max(min_val, min(max_val, float(value)))

        self.sig = clamp_value(self.sig)
        self.bak = clamp_value(self.bak)
        self.ovrl = clamp_value(self.ovrl)
        self.p808 = clamp_value(self.p808)

    def to_dict(self) -> Dict[str, float]:
        return {"sig": self.sig, "bak": self.bak, "ovrl": self.ovrl, "p808": self.p808}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "QualityThresholds":
        def clamp_value(value, min_val=1.0, max_val=5.0):
            return max(min_val, min(max_val, float(value)))

        return cls(
            sig=clamp_value(data.get("sig", 3.0)),
            bak=clamp_value(data.get("bak", 3.0)),
            ovrl=clamp_value(data.get("ovrl", 3.0)),
            p808=clamp_value(data.get("p808", 3.0)),
        )


@dataclass
class QualityResult:
    """Quality assessment result"""

    def __init__(
        self, filename: str, scores: Dict[str, float], passed: bool, timestamp: str
    ):
        self.filename = filename
        self.scores = scores
        self.passed = passed
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable types"""

        # Helper function to convert types
        def convert_value(value):
            if hasattr(value, "item"):  # numpy scalar
                return value.item()
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [convert_value(v) for v in value]
            else:
                return value

        return {
            "filename": str(self.filename),
            "scores": convert_value(self.scores),
            "passed": bool(self.passed),
            "timestamp": str(self.timestamp),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityResult":
        # Handle both old (float) and new (string) timestamp formats
        timestamp = data["timestamp"]
        if isinstance(timestamp, (int, float)):
            # Convert old format to new format
            timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        return cls(
            filename=data["filename"],
            scores=data["scores"],
            passed=data["passed"],
            timestamp=timestamp,
        )


class QualityAssessment:
    """Background quality assessment manager"""

    def _load_thresholds(self):
        """Load thresholds from config file"""
        config_file = self.output_dir / "quality_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.thresholds = QualityThresholds.from_dict(data)
            except Exception as e:
                print(f"Failed to load thresholds: {e}")

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.scores_file = self.output_dir / "scores.json"
        self.thresholds = QualityThresholds()
        self.dnsmos = None
        self.callbacks = {}
        self._results_cache = {}
        self._load_thresholds()  # Load thresholds before init DNSMOS
        self._init_dnsmos()
        self._load_scores()

    def _init_dnsmos(self):
        """Initialize DNSMOS model"""
        try:
            self.dnsmos = DNSMOS()
        except Exception as e:
            print(f"Warning: Failed to initialize DNSMOS: {e}")
            self.dnsmos = None

    def add_callback(self, event: str, callback: Callable):
        """Add callback for events: on_assessment_complete, on_error"""
        self.callbacks[event] = callback

    def _trigger_callback(self, event: str, data: Any = None):
        """Trigger callback if exists"""
        if event in self.callbacks:
            try:
                self.callbacks[event](data)
            except Exception as e:
                print(f"Callback error: {e}")

    def set_thresholds(self, **kwargs):
        """Update quality thresholds with validation (1.0 to 5.0 range) and re-evaluate existing results"""

        def clamp_value(value, min_val=1.0, max_val=5.0):
            """Clamp value to valid range"""
            return max(min_val, min(max_val, float(value)))

        # Check if any threshold actually changed
        changed = False
        for key, value in kwargs.items():
            if hasattr(self.thresholds, key):
                # Validate and clamp the value to 1.0-5.0 range
                clamped_value = clamp_value(value)
                old_value = getattr(self.thresholds, key)
                if old_value != clamped_value:
                    changed = True
                    setattr(self.thresholds, key, clamped_value)

        if changed:
            self._save_thresholds()
            # Re-evaluate all existing results with new thresholds
            self._reevaluate_all_results()

    def _reevaluate_all_results(self):
        """Re-evaluate all existing results with current thresholds"""
        updated = False
        for filename, result in self._results_cache.items():
            # Recalculate passed status with new thresholds
            new_passed = self._check_quality(result.scores)
            if result.passed != new_passed:
                # Update the result with new passed status
                self._results_cache[filename] = QualityResult(
                    filename=result.filename,
                    timestamp=result.timestamp,
                    scores=result.scores,  # Keep original scores
                    passed=new_passed,  # Update passed status
                )
                updated = True

        if updated:
            self._save_scores()  # Save updated results

    def get_thresholds(self) -> QualityThresholds:
        """Get current thresholds"""
        return self.thresholds

    def _save_thresholds(self):
        """Save thresholds to config file"""
        config_file = self.output_dir / "quality_config.json"
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self.thresholds.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Failed to save thresholds: {e}")

    def _load_thresholds(self):
        """Load thresholds from config file"""
        config_file = self.output_dir / "quality_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.thresholds = QualityThresholds.from_dict(data)
            except Exception as e:
                print(f"Failed to load thresholds: {e}")

    def _load_scores(self):
        """Load existing scores from file"""
        if self.scores_file.exists():
            try:
                with open(self.scores_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        print("Scores file is empty, skipping load")
                        return

                    data = json.loads(content)
                    if not isinstance(data, list):
                        print("Invalid scores file format, expected list")
                        return

                for item in data:
                    if (
                        isinstance(item, dict)
                        and "filename" in item
                        and "scores" in item
                    ):
                        result = QualityResult.from_dict(item)
                        self._results_cache[result.filename] = result
                    else:
                        print(f"Skipping invalid score entry: {item}")
            except json.JSONDecodeError as e:
                print(f"Failed to load scores - JSON decode error: {e}")
                # Backup corrupted file and create new one
                backup_file = self.scores_file.with_suffix(".json.backup")
                try:
                    self.scores_file.rename(backup_file)
                    print(f"Corrupted scores file backed up to: {backup_file}")
                except Exception:
                    pass
            except Exception as e:
                print(f"Failed to load scores: {e}")

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert numpy scalar to Python scalar
        elif isinstance(obj, np.bool_):
            return bool(obj)  # Convert numpy bool to Python bool
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to Python list
        elif hasattr(obj, "item"):  # Generic numpy scalar
            return obj.item()
        else:
            return obj

    def _save_scores(self):
        """Save scores to file"""
        try:
            data = [result.to_dict() for result in self._results_cache.values()]
            # Convert numpy types to Python native types
            data = self._convert_numpy_types(data)
            with open(self.scores_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save scores: {e}")

    def assess_audio_async(self, audio_path: str):
        """Start background quality assessment"""
        if self.dnsmos is None:
            return

        filename = Path(audio_path).name
        if filename in self._results_cache:
            # Already assessed
            return

        # Start assessment in background thread
        thread = threading.Thread(
            target=self._assess_audio_worker, args=(audio_path,), daemon=True
        )
        thread.start()

    def _assess_audio_worker(self, audio_path: str):
        """Worker thread for audio assessment"""
        try:
            filename = Path(audio_path).name

            # Run DNSMOS
            scores = self.dnsmos.get_score(audio_path)

            # Check if passed
            passed = self._check_quality(scores)

            # Create result
            result = QualityResult(
                filename=filename,
                scores=scores,
                passed=passed,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            # Cache and save
            self._results_cache[filename] = result
            self._save_scores()

            # Trigger callback
            self._trigger_callback("on_assessment_complete", result)

        except Exception as e:
            print(f"Quality assessment failed: {e}")
            self._trigger_callback("on_error", str(e))

    def _check_quality(self, scores: Dict[str, float]) -> bool:
        """Check if audio quality meets thresholds"""
        return (
            scores.get("sig", 0) >= self.thresholds.sig
            and scores.get("bak", 0) >= self.thresholds.bak
            and scores.get("ovrl", 0) >= self.thresholds.ovrl
            and scores.get("p808", 0) >= self.thresholds.p808
        )

    def get_result(self, filename: str) -> Optional[QualityResult]:
        """Get quality result for a file"""
        return self._results_cache.get(filename)

    def has_result(self, filename: str) -> bool:
        """Check if file has been assessed"""
        return filename in self._results_cache

    def get_all_results(self) -> Dict[str, QualityResult]:
        """Get all quality results"""
        return self._results_cache.copy()

    def remove_result(self, filename: str):
        """Remove quality result for deleted file"""
        if filename in self._results_cache:
            del self._results_cache[filename]
            self._save_scores()
