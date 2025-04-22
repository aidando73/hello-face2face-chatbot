```bash
source ~/miniconda3/bin/activate && conda create -y --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env
cp ~/.runpod_credentials .envrc
direnv allow
pip install uv
uv pip install -r requirements.txt
apt-get update && apt-get install -y ffmpeg
python prepare_dataset.py


ffmpeg -i audio-sample.m4a -acodec pcm_s16le -ar 16000 -ac 1 audio-sample.wav

tmux
apt-get update && apt-get install -y ffmpeg
source ~/miniconda3/bin/activate ./env
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29400 \
    alignment_training.py | tee alignment_training.log

python test_text_audio_alignment.py --checkpoint checkpoints/20250422_0001 --epoch 1
# Test mel_filter_bank block
python mel_filter_bank_block.py

python audio_encoder.py

python dataset_loader.py
```

Pod usage begins at: 12 Apr 2025