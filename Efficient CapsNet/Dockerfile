FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install libgl1-mesa-glx apt-utils -y

RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

EXPOSE 5000