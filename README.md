# ♾ Infinite dSprites

Easily generate simple continual learning benchmarks. Inspired by [dSprites](https://github.com/google-deepmind/dsprites-dataset).

![A grid of 2D shapes undergoing rotation, translation, and scaling.](img/shapes.gif)

## Install
Install the package from PyPI:
```bash
python -m pip install idsprites
```

Verify the installation:
```bash
python -c "import idsprites"
```

## Usage

```bash
source venv/bin/activate
# with defaults
python scripts/generate_dataset.py
# customized
python scripts/generate_dataset.py \
  --img_size=128 --factor_resolution=5 \
  --num_tasks=4 --shapes_per_task=5
```

## Contribute
Clone the repo:
```bash
git clone git@github.com:sbdzdz/idsprites.git
cd idsprites
```

It's a good idea to install the package in interactive mode inside a virtual environment:
```bash
python -m virtualenv venv
source venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e .
```