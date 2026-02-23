#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-image-detector
cd ..
python -m uvicorn app.app:app --reload
