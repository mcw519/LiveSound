from typing import Dict

LANGS: Dict[str, str] = {
    "zh-TW": "繁體中文",
    "en": "English",
}

DEFAULT_LANG = "zh-TW"


I18N: Dict[str, Dict[str, str]] = {
    "app_title": {"zh-TW": "LiveSound 錄音器", "en": "LiveSound Recorder"},
    "app_heading": {"zh-TW": "LiveSound 錄音器", "en": "LiveSound Recorder"},
    "language": {"zh-TW": "語言:", "en": "Language:"},
    "device_settings": {"zh-TW": "設備設定", "en": "Device Settings"},
    "audio_device": {"zh-TW": "音訊設備:", "en": "Audio Device:"},
    "output_device": {"zh-TW": "播放設備:", "en": "Output Device:"},
    "sample_rate": {"zh-TW": "取樣率:", "en": "Sample Rate:"},
    "gain_db": {"zh-TW": "增益 (dB):", "en": "Gain (dB):"},
    "recording_controls": {"zh-TW": "錄音控制", "en": "Recording Controls"},
    "filename": {"zh-TW": "檔名前綴:", "en": "Filename Prefix:"},
    "start_recording": {"zh-TW": "開始錄音", "en": "Start"},
    "stop_recording": {"zh-TW": "停止錄音", "en": "Stop"},
    "pause": {"zh-TW": "暫停", "en": "Pause"},
    "resume": {"zh-TW": "繼續", "en": "Resume"},
    "volume": {"zh-TW": "音量", "en": "Volume"},
    "time": {"zh-TW": "時間:", "en": "Time:"},
    "status_ready": {"zh-TW": "準備就緒", "en": "Ready"},
    "status_recording": {"zh-TW": "錄音中", "en": "Recording"},
    "status_paused": {"zh-TW": "已暫停", "en": "Paused"},
    "status_saved": {"zh-TW": "錄音完成", "en": "Saved"},
    "done": {"zh-TW": "完成", "en": "Done"},
    "saved_as": {"zh-TW": "已儲存", "en": "Saved"},
    "save_failed": {"zh-TW": "儲存失敗", "en": "Save failed"},
    "cannot_start": {"zh-TW": "無法開始錄音", "en": "Cannot start recording"},
    "error": {"zh-TW": "錯誤", "en": "Error"},
    "confirm": {"zh-TW": "確認", "en": "Confirm"},
    "close_confirm": {
        "zh-TW": "錄音進行中，確定要關閉嗎？",
        "en": "Recording in progress. Are you sure to close?",
    },
    "files": {"zh-TW": "錄音檔案", "en": "Recordings"},
    "play": {"zh-TW": "播放", "en": "Play"},
    "stop": {"zh-TW": "停止", "en": "Stop"},
    "delete": {"zh-TW": "刪除", "en": "Delete"},
    "refresh": {"zh-TW": "重新整理", "en": "Refresh"},
    "play_failed": {"zh-TW": "播放失敗: {e}", "en": "Failed to play: {e}"},
    "delete_failed": {"zh-TW": "刪除失敗: {e}", "en": "Failed to delete: {e}"},
    "delete_confirm": {"zh-TW": "刪除 {name}?", "en": "Delete {name}?"},
    # CLI
    "cli_stopping": {"zh-TW": "正在停止錄音...", "en": "Stopping recording..."},
    "cli_will_save_to": {
        "zh-TW": "錄音將儲存到: {path}",
        "en": "Will save recordings to: {path}",
    },
    "cli_test_gain_start": {
        "zh-TW": "測試麥克風增益 (3秒)...",
        "en": "Testing microphone gain (3s)...",
    },
    "cli_avg_level": {
        "zh-TW": "平均音量: {level:.1f} dB",
        "en": "Average level: {level:.1f} dB",
    },
    "cli_gain_up": {
        "zh-TW": "建議增加前置增益 (+10 到 +20 dB)",
        "en": "Suggest increasing pre-gain (+10 to +20 dB)",
    },
    "cli_gain_down": {
        "zh-TW": "建議降低前置增益 (-5 到 -10 dB)",
        "en": "Suggest reducing pre-gain (-5 to -10 dB)",
    },
    "cli_gain_ok": {"zh-TW": "音量適中", "en": "Gain OK"},
    "cli_press_ctrl_c": {"zh-TW": "按 Ctrl+C 停止錄音", "en": "Press Ctrl+C to stop"},
    "cli_recording": {"zh-TW": "錄音中:", "en": "Recording:"},
    "cli_saving": {"zh-TW": "儲存中...", "en": "Saving..."},
    "cli_saved": {"zh-TW": "已儲存: {path}", "en": "Saved: {path}"},
    "cli_save_failed": {"zh-TW": "儲存失敗", "en": "Save failed"},
    "cli_cannot_start": {"zh-TW": "無法開始錄音", "en": "Cannot start recording"},
    "cli_cancelled": {"zh-TW": "已取消", "en": "Cancelled"},
    "cli_error": {"zh-TW": "錯誤: {msg}", "en": "Error: {msg}"},
    # CLI help/description
    "cli_desc": {
        "zh-TW": "LiveSound Enhanced Recorder - 專業音訊錄音工具",
        "en": "LiveSound Enhanced Recorder - Professional audio recorder",
    },
    "cli_help_gui": {"zh-TW": "啟動GUI模式", "en": "Launch GUI"},
    "cli_help_output_dir": {
        "zh-TW": "輸出目錄 (預設: ./recordings)",
        "en": "Output directory (default: ./recordings)",
    },
    "cli_help_filename": {
        "zh-TW": "檔案名稱 (不含副檔名)",
        "en": "Filename (without extension)",
    },
    "cli_help_list_devices": {
        "zh-TW": "列出音訊設備",
        "en": "List audio input devices",
    },
    "cli_help_device_index": {"zh-TW": "設備索引", "en": "Device index"},
    "cli_help_test_gain": {"zh-TW": "測試增益", "en": "Test gain"},
    "cli_help_sample_rate": {"zh-TW": "取樣率", "en": "Sample rate"},
    "cli_help_channels": {"zh-TW": "聲道數", "en": "Channels"},
    "cli_help_chunk_size": {"zh-TW": "緩衝區大小", "en": "Chunk size"},
    "cli_help_pre_gain": {"zh-TW": "前置增益 dB", "en": "Pre-gain dB"},
    "cli_help_lang": {"zh-TW": "語言", "en": "Language"},
    "cli_gain_range_error": {
        "zh-TW": "增益必須在 -20 到 +20 dB 之間",
        "en": "Gain must be between -20 and +20 dB",
    },
    # Output folder & path UI
    "output_dir": {"zh-TW": "儲存位置:", "en": "Output Folder:"},
    "browse": {"zh-TW": "瀏覽…", "en": "Browse…"},
    "open_folder": {"zh-TW": "開啟資料夾", "en": "Open Folder"},
    "path": {"zh-TW": "路徑", "en": "Path"},
    # Gain test UI
    "gain_test": {"zh-TW": "麥克風增益測試", "en": "Mic Gain Test"},
    "gain_test_tip": {
        "zh-TW": "按下按鈕進行 3 秒麥克風增益測試，請在一般說話音量下發聲。",
        "en": "Press to run a 3s mic gain test. Speak at your normal level.",
    },
    "gain_test_run": {"zh-TW": "測試中…", "en": "Testing…"},
    "gain_test_btn": {"zh-TW": "測試並建議增益", "en": "Test and Suggest Gain"},
    "gain_test_result": {
        "zh-TW": "RMS: {rms:.1f} dB, Peak: {peak:.1f} dB, 建議增益: {suggest:+.0f} dB",
        "en": "RMS: {rms:.1f} dB, Peak: {peak:.1f} dB, Suggested gain: {suggest:+.0f} dB",
    },
    "apply_gain": {
        "zh-TW": "套用建議增益 {gain:+.0f} dB?",
        "en": "Apply suggested gain {gain:+.0f} dB?",
    },
    "cannot_test_while_recording": {
        "zh-TW": "錄音進行中，無法執行測試",
        "en": "Cannot test while recording",
    },
    "folder_changed": {"zh-TW": "已更改儲存位置", "en": "Output folder changed"},
    # Quality settings
    "quality_settings": {"zh-TW": "品質檢測設定", "en": "Quality Assessment"},
    "quality_checking": {"zh-TW": "品質檢測中...", "en": "Assessing quality..."},
    "quality_pass": {"zh-TW": "通過", "en": "Pass"},
    "quality_fail": {"zh-TW": "失敗", "en": "Failed"},
    "quality_not_ready": {
        "zh-TW": "檔案尚未完成品質檢測，請稍候再播放",
        "en": "File quality assessment not ready. Please wait before playing.",
    },
    "warning": {"zh-TW": "警告", "en": "Warning"},
    "gain_test_result": {
        "zh-TW": "平均 {rms:.1f} dBFS，峰值 {peak:.1f} dBFS；建議調整 {suggest:+.0f} dB。",
        "en": "Avg {rms:.1f} dBFS, Peak {peak:.1f} dBFS; Suggest {suggest:+.0f} dB.",
    },
    "apply_gain": {
        "zh-TW": "套用建議增益 {gain:+.0f} dB?",
        "en": "Apply suggested gain {gain:+.0f} dB?",
    },
    "prompt_gain_test": {
        "zh-TW": "建議在錄音前先進行增益測試，以獲得最佳音量。要立即測試嗎？",
        "en": "It's recommended to test mic gain before recording. Test now?",
    },
    "cannot_test_while_recording": {
        "zh-TW": "錄音中無法測試增益，請先停止錄音。",
        "en": "Cannot run gain test while recording. Please stop first.",
    },
    "folder_changed": {"zh-TW": "已切換儲存位置", "en": "Output folder changed"},
    # Menu items
    "file": {"zh-TW": "檔案", "en": "File"},
    "exit": {"zh-TW": "結束", "en": "Exit"},
    "current_folder": {"zh-TW": "目前資料夾", "en": "Current Folder"},
    "quality_thresholds": {"zh-TW": "品質門檻設定", "en": "Quality Thresholds"},
    "ok": {"zh-TW": "確定", "en": "OK"},
    "cancel": {"zh-TW": "取消", "en": "Cancel"},
    "invalid_values": {"zh-TW": "輸入值無效", "en": "Invalid values"},
    # Quality metric descriptions
    "sig_description": {"zh-TW": "(語音清晰度)", "en": "(Speech clarity)"},
    "bak_description": {"zh-TW": "(背景雜訊)", "en": "(Background noise)"},
    "ovrl_description": {"zh-TW": "(整體品質)", "en": "(Overall quality)"},
    "p808_description": {"zh-TW": "(MOS評分)", "en": "(MOS score)"},
    # Spectrogram
    "spectrogram": {"zh-TW": "頻譜圖", "en": "Spectrogram"},
    "show_spectrogram": {"zh-TW": "顯示頻譜圖", "en": "Show Spectrogram"},
    "spectrogram_error": {"zh-TW": "無法顯示頻譜圖: {e}", "en": "Cannot show spectrogram: {e}"},
    "no_file_selected": {"zh-TW": "請先選擇一個音檔", "en": "Please select an audio file first"},
}


def t(key: str, lang: str, **kwargs) -> str:
    """Translate a key for a given language, with safe fallbacks and formatting."""
    entry = I18N.get(key, {})
    text = entry.get(lang) or entry.get("en") or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text
