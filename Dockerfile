# Use miniconda base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /main

# Copy repo contents to the container
COPY . /main

# Create a conda env from environment.yml
RUN conda env create -f environment.yml

ENV PATH /opt/conda/envs/ai-image-detector/bin:$PATH

CMD uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-7860}