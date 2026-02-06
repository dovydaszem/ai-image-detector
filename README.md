# AI image detector

A minimal web app for detecting AI-generated images using a custom CNN.

![App screenshot](app/app_screenshot.png)

## Overview

The app is a simple web interface where users can upload an image and get a predicted probability that it is AI generated. The CNN was trained on 5k images generated with SDXL Turbo and 5k real images from MS COCO dataset.

## Setup

conda env create -f environment.yml  
conda activate ai-image-detector  
cd ai-image-detector/app/  
./launch.sh  
Open http://localhost:8000 in your browser.
