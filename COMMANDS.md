```bash
source ~/miniconda3/bin/activate && conda create -y --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env
pip install uv
uv pip install -r requirements.txt
cp ~/.runpod_credentials .envrc

apt-get update && apt-get install -y ffmpeg
ffmpeg -i audio-sample.m4a -acodec pcm_s16le -ar 16000 -ac 1 audio-sample.wav

# Run dataset2.ipynb to download the dataset
tmux
source ~/miniconda3/bin/activate ./env
python alignment_training.py

python alignment_training.py | tee alignment_training.log


python test_text_audio_alignment.py
```

Pod usage begins at: 12 Apr 2025