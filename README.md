# LiveSound

LiveSound offers simpliest codes and a user-friendly GUI, making it accessible for everyone to capture and record the captivating sounds of their live experiences.

## Installation

    git clone <project-url>
    python setup.py install


## CLI tools

### Play the local wavefile  

    usage: record-local [-h] [-gui] [-sd] [-tg] [-spg] [-sr SAMPLE_RATE] [-cs CHUNK_SIZE]

    optional arguments:
    -h,             --help                      show this help message and exit
    -gui,           --use_gui                   open recorder GUI
    -sd,            --select_device             choose your output speaker device index
    -spg,           --set_post_gainSet          output speaker post gain (dB)
    -v,             --visiable                  open visiable screen
    -pf,            --play_folder               play all waveform (*.wav) inside the folder
    -cs CHUNK_SIZE, --chunk_size CHUNK_SIZE     stream chunk buffer size


### Record a wavefile through device microphone

    usage: record-local [-h] [-sd] [-tg] [-spg] [-sr SAMPLE_RATE] [-cs CHUNK_SIZE]

    optional arguments:
    -h,                 --help                      show this help message and exit
    -sd,                --select_device             choose your microphone device index
    -tg,                --test_gain                 test microphone average gain (dB)
    -spg,               --set_pre_gain              set microphone pre gain (dB)
    -sr SAMPLE_RATE,    --sample_rate SAMPLE_RATE   recording sample rate
    -cs CHUNK_SIZE,     --chunk_size CHUNK_SIZE     stream chunk buffer size