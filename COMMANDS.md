```bash
source ~/miniconda3/bin/activate && conda create -y --prefix ./env python=3.10
source ~/miniconda3/bin/activate ./env
pip install uv
uv pip install -r requirements.txt
```

Pod usage begins at: 12 Apr 2025