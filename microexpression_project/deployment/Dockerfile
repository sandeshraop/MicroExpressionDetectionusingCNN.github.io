# Multi-stage build for micro-expression recognition API
# Build context: microexpression_project/ (repo folder containing web/, src/, deployment/)
FROM python:3.9-slim as base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim as production

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

COPY . .

RUN mkdir -p uploads models

EXPOSE 5000

ENV PYTHONPATH=/app/src:/app/inference
ENV FLASK_APP=web/app.py
ENV FLASK_ENV=production
ENV FLASK_HOST=0.0.0.0
ENV FLASK_DEBUG=0

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5000/health || exit 1

CMD ["python", "web/app.py"]
