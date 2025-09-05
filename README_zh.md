# LiveSound - 專業音訊錄音工具 🎵

[English](README_EN.md) | 繁體中文

基於 Python 的音訊錄音應用程式，提供 GUI 和 CLI 介面。

## ✨ 特色功能

- 🎙️ **高品質錄音**: 支援多種取樣率和聲道配置
- 🖥️ **雙重介面**: GUI 和 CLI 模式
- 📊 **即時監控**: 音訊等級顯示
- 🔧 **專業功能**: 增益控制、設備選擇
- 📈 **品質評估**: 整合 DNSMOS 音質評估

## 🚀 快速開始

### 安裝
```bash
git clone https://github.com/mcw519/LiveSound.git
cd LiveSound
pip install -r requirements.txt
pip install -e .
```

### 使用方法

#### GUI 模式
```bash
livesound-record --gui
```

#### CLI 模式
```bash
# 基本錄音
livesound-record

# 自訂設定
livesound-record --output-dir ./recordings --sample-rate 44100 --channels 2
```

## � 主要選項

| 選項 | 說明 | 預設值 |
|------|------|--------|
| `--gui` | 啟動 GUI 模式 | CLI 模式 |
| `--output-dir` | 輸出目錄 | `./recordings` |
| `--sample-rate` | 取樣率 | 44100 Hz |
| `--channels` | 聲道數 | 1 |
| `--device-index` | 音訊設備索引 | 預設設備 |
| `--list-devices` | 列出音訊設備 | - |

## 📄 授權

MIT License - 詳見 [LICENSE](LICENSE) 檔案

### 第三方組件
- **DNSMOS 模型**: 來自 [Microsoft DNS-Challenge](https://github.com/microsoft/DNS-Challenge)
- **授權**: MIT License (Microsoft Corporation)
- **完整授權資訊**: 請參閱 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
