FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    opencv-python==4.12.0.88 \
    pillow==12.0.0 \
    pandas==2.3.3 \
    scipy==1.15.3 \
    matplotlib==3.10.7 \
    PyYAML==6.0.3 \
    tqdm==4.67.1 \
    requests==2.32.5 \
    loguru==0.7.3 \
    lap==0.5.13 \
    Cython==3.2.4 \
    cython-bbox==0.1.5 \
    ultralytics==8.3.221 \
    polars==1.34.0 \
    psutil==7.1.1 \
    chardet==7.4.3

COPY bytetrack/yolox /app/bytetrack/yolox
COPY src/ /app/src/
COPY run_demo.py /app/
COPY models/ /app/models/
COPY 26x/ /app/26x/

ENV LD_LIBRARY_PATH="/usr/lib/wsl/lib"
ENV PYTHONPATH="/app/bytetrack:/app/src/train:/app/src/perception:/app/src/utils"

VOLUME ["/data", "/app/outputs"]

CMD ["python", "run_demo.py", "--help"]