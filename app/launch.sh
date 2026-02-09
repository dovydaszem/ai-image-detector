#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-image-detector
python -m uvicorn app:app --reload
