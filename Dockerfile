# use miniconda base image
FROM continuumio/miniconda3

# set working directory
WORKDIR /app

# copy repo contents
COPY . /app

# create a conda env from environment.yml
COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml

# after creating environment
ENV PATH /opt/conda/envs/ai-image-detector/bin:$PATH

# shell form CMD
CMD uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-7860}
