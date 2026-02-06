#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_image_detector
python -m uvicorn app:app --reload