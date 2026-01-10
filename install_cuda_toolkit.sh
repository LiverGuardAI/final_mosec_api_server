#!/bin/bash
set -e

echo "================================"
echo "CUDA Toolkit Installation Script for WSL2"
echo "================================"

# Download CUDA keyring if not exists
if [ ! -f "cuda-keyring_1.1-1_all.deb" ]; then
    echo "Downloading CUDA keyring..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
fi

# Install CUDA keyring
echo "Step 1: Installing CUDA keyring..."
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update apt repository
echo "Step 2: Updating apt repository..."
sudo apt-get update

# Install CUDA Toolkit (development tools)
echo "Step 3: Installing CUDA Toolkit 12.3..."
sudo apt-get install -y cuda-toolkit-12-3

# Set environment variables
echo "Step 4: Setting up environment variables..."
echo ""
echo "Add the following lines to your ~/.bashrc:"
echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH'
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH'
echo ""

# Add to current session
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

# Verify installation
echo "Step 5: Verifying CUDA installation..."
nvcc --version

echo ""
echo "================================"
echo "CUDA Toolkit installation complete!"
echo "================================"
echo ""
echo "Now run: source venv/bin/activate && pip install pycuda"