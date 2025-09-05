# LiveSound - Professional Audio Recording Tool 🎵

English | [繁體中文](README_zh.md)

A Python-based audio recording application with GUI and CLI interfaces.

## ✨ Key Features

- 🎙️ **High-Quality Recording**: Multi-sample rate and channel support
- 🖥️ **Dual Interface**: Both GUI and CLI modes
- 📊 **Real-time Monitoring**: Audio level visualization
- 🔧 **Professional Features**: Gain control, device selection
- 📈 **Quality Assessment**: Integrated DNSMOS audio quality evaluation

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/mcw519/LiveSound.git
cd LiveSound
pip install -r requirements.txt
pip install -e .
```

### Usage

#### GUI Mode
```bash
livesound-record --gui
```

#### CLI Mode
```bash
# Basic recording
livesound-record

# Custom settings
livesound-record --output-dir ./recordings --sample-rate 44100 --channels 2
```

## 📋 Main Options

| Option | Description | Default |
|--------|-------------|---------|
| `--gui` | Launch GUI mode | CLI mode |
| `--output-dir` | Output directory | `./recordings` |
| `--sample-rate` | Sample rate | 44100 Hz |
| `--channels` | Number of channels | 1 |
| `--device-index` | Audio device index | Default device |
| `--list-devices` | List audio devices | - |

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

### Third-Party Components
- **DNSMOS Models**: From [Microsoft DNS-Challenge](https://github.com/microsoft/DNS-Challenge)
- **License**: MIT License (Microsoft Corporation)
- **Full License Info**: See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
