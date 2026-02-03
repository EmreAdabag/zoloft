#!/bin/bash

conda create -y -n zoloft python=3.10
conda activate zoloft
conda install -y ffmpeg -c conda-forge

cd lerobot
conda install -y -c conda-forge "cmake<4"
pip install -e ".[libero]"
cd ..
