#!/bin/bash
set -e

echo "================================"
echo "PyCUDA Installation Script"
echo "================================"

# 환경 변수 설정
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.3

echo "Step 1: Setting environment variables..."
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CUDA_HOME=$CUDA_HOME"

# CUDA 확인
echo ""
echo "Step 2: Verifying CUDA installation..."
nvcc --version

# venv 활성화
echo ""
echo "Step 3: Activating virtual environment..."
source venv/bin/activate

# 필수 패키지 설치
echo ""
echo "Step 4: Installing build dependencies..."
pip install --upgrade pip setuptools wheel numpy

# pycuda 설치를 위한 추가 환경 변수
export CFLAGS="-I/usr/local/cuda-12.3/include"
export LDFLAGS="-L/usr/local/cuda-12.3/lib64 -L/usr/lib/wsl/lib"

# pycuda 소스에서 빌드
echo ""
echo "Step 5: Installing pycuda (this may take 2-3 minutes)..."
pip install --no-cache-dir pycuda

echo ""
echo "Step 6: Verifying pycuda installation..."
python -c "import pycuda.driver as cuda; import pycuda.autoinit; print('✓ PyCUDA installed successfully!')"

echo ""
echo "================================"
echo "PyCUDA installation complete!"
echo "================================"