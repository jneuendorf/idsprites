#!/usr/bin/env just --justfile

venv_dir := "./venv"
python := venv_dir / "bin/python"
pip := venv_dir / "bin/pip"

sys_python := `which python3.13 || which python3.12 || which python3.11 || which python3.10`

info:
    echo "Detected compatible python at {{ sys_python }}"

venv:
    [ -d {{ venv_dir }} ] || {{ sys_python }} -m venv {{ venv_dir }}

install: venv
    {{ pip }} install -r requirements.txt

# See README "Usage"
generate *ARGS="--img_size=128 --factor_resolution=5 --num_tasks=4 --shapes_per_task=5": venv
    {{ python }} scripts/generate_dataset.py {{ ARGS }}