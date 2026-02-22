---
title: ai image detector
emoji: 🖼️
colorFrom: red
colorTo: indigo
sdk: python
sdk_version: "3.12"
python_version: "3.11"
app_file: app/app.py
pinned: false
---

# Interpretable AI image detector

A minimal web app for detecting AI-generated images using a custom convolutional neural network (CNN).

<img src="app/app_screenshot.png" alt="App screenshot" style="border-radius: 12px; width: 100%;" />

## Overview

The app is a simple web interface where users can upload an image and get a predicted probability that it is AI generated. When an AI generated image is detected, the tool provides a heatmap overlay based on Grad-CAM technology, telling which parts of a given image most influence the model’s decision.

The CNN was trained on 5k images generated with SDXL Turbo and 5k real images from MS COCO dataset.

## Setup

conda env create -f environment.yml  
conda activate ai-image-detector  
cd ai-image-detector/app/  
./launch.sh  
Open http://localhost:8000 in your browser.