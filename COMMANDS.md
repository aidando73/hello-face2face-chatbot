```bash
source ~/miniconda3/bin/activate && conda create -y --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env
pip install uv
uv pip install -r requirements.txt

apt-get update && apt-get install -y ffmpeg
ffmpeg -i audio-sample.m4a -acodec pcm_s16le -ar 16000 -ac 1 audio-sample.wav

```

Pod usage begins at: 12 Apr 2025