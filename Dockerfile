# MOSEC AI Model Inference Server Dockerfile
# Requires NVIDIA GPU and CUDA 12.6

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV PATH=/usr/local/cuda-12.6/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    netcat \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives to use python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorRT Python API (compatible with CUDA 12.6)
RUN pip install --no-cache-dir tensorrt==10.4.0

# Copy application code
COPY app_fp16_tensorrt.py /app/
COPY run_nnunet_inference_fp16_tensorrt.py /app/

# Create directories for models and temp files
RUN mkdir -p /app/models /app/temp

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

# Run mosec server
CMD ["python", "app_fp16_tensorrt.py"]
