# Running this benchmark

System requirements (Ubuntu):

- CUDA (12.8 or later recommended)
- python3-venv
- python3-dev


# Setup

Clone this repo:

```
git clone https://github.com/BingqingCheng/cace
```

Run in base directory:

```
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .
```

# Verify installation

```
python -c 'import torch; import cace; print(torch.cuda.is_available()); print(torch.version.cuda)'
```

# Run benchmark

```
cd benchmark
python ./md.py  # non-optimized benchmark
python ./md-opt.py  # torch.compile optimized benchmark
```

# Results

Runtime in seconds. Lower is better

| GPU               | Non-Optimized | Optimized | Notes           |
| NVidia Spark GB10 | 78            | 45        | CUDA 13.0       |
| L40S              | 38            |           | AWS EC2 g6e     |
| 
|
|

