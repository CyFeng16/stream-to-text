FROM docker.io/library/python:3.8-slim as worker
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        build-essential \
        python3-dev \
        ffmpeg \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install pip -U \
    && pip install --no-cache-dir \
        youtube-dl \
        paddlepaddle \
        paddlespeech
COPY *.py /workspace/
WORKDIR /workspace
COPY LICENSE /workspace/
RUN python init.py
ENTRYPOINT ["python", "pipeline.py"]