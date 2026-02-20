# use miniconda base image
FROM continuumio/miniconda3

# set working directory
WORKDIR /app

# copy repo contents
COPY . /app

# create a conda env from environment.yml
COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml

# make sure conda environment is activated for all subsequent commands
SHELL ["conda", "run", "-n", "ai-image-detector", "/bin/bash", "-c"]

# expose the port the app will run on
EXPOSE 8000

# start command
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
