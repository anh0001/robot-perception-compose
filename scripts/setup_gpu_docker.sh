#!/bin/bash
# GPU Docker Setup Script for OpenMask3D
# This script helps diagnose and fix NVIDIA GPU runtime issues with Docker

set -e

echo "============================================"
echo "GPU Docker Setup for OpenMask3D"
echo "============================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: NVIDIA Driver
echo "Step 1: Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ nvidia-smi found${NC}"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    DRIVER_OK=true
else
    echo -e "${RED}✗ nvidia-smi not found${NC}"
    echo "  → No NVIDIA driver detected or not in PATH"
    DRIVER_OK=false
fi
echo ""

# Check 2: Docker availability
echo "Step 2: Checking Docker..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker found${NC}"
    docker --version
    DOCKER_OK=true
else
    echo -e "${RED}✗ Docker not found${NC}"
    echo "  → Install Docker first: https://docs.docker.com/engine/install/"
    DOCKER_OK=false
    exit 1
fi
echo ""

# Check 3: NVIDIA Container Toolkit
echo "Step 3: Checking NVIDIA Container Toolkit..."
if dpkg -l | grep -q nvidia-container-toolkit; then
    echo -e "${GREEN}✓ nvidia-container-toolkit installed${NC}"
    TOOLKIT_OK=true
else
    echo -e "${YELLOW}✗ nvidia-container-toolkit not found${NC}"
    TOOLKIT_OK=false
fi
echo ""

# Check 4: Docker GPU test
echo "Step 4: Testing Docker GPU access..."
if [ "$DRIVER_OK" = true ] && [ "$DOCKER_OK" = true ]; then
    if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ Docker can access GPU${NC}"
        GPU_DOCKER_OK=true
    else
        echo -e "${RED}✗ Docker cannot access GPU${NC}"
        GPU_DOCKER_OK=false
    fi
else
    echo -e "${YELLOW}⊘ Skipping Docker GPU test (prerequisites missing)${NC}"
    GPU_DOCKER_OK=false
fi
echo ""

# Summary and recommendations
echo "============================================"
echo "Summary"
echo "============================================"
echo "NVIDIA Driver:        $([ "$DRIVER_OK" = true ] && echo -e "${GREEN}OK${NC}" || echo -e "${RED}MISSING${NC}")"
echo "Docker:               $([ "$DOCKER_OK" = true ] && echo -e "${GREEN}OK${NC}" || echo -e "${RED}MISSING${NC}")"
echo "Container Toolkit:    $([ "$TOOLKIT_OK" = true ] && echo -e "${GREEN}OK${NC}" || echo -e "${YELLOW}NOT INSTALLED${NC}")"
echo "Docker GPU Access:    $([ "$GPU_DOCKER_OK" = true ] && echo -e "${GREEN}OK${NC}" || echo -e "${RED}NOT WORKING${NC}")"
echo ""

if [ "$GPU_DOCKER_OK" = true ]; then
    echo -e "${GREEN}✓ All checks passed! You can run OpenMask3D with GPU acceleration.${NC}"
    echo ""
    echo "Run the server with:"
    echo "  docker run -p 5001:5001 --gpus all -it craiden/openmask:v1.0"
    exit 0
fi

# Provide fix instructions
echo -e "${YELLOW}⚠ GPU Docker setup incomplete.${NC}"
echo ""

if [ "$DRIVER_OK" = false ]; then
    echo "Fix 1: Install NVIDIA driver"
    echo "  For Ubuntu/Debian:"
    echo "    sudo apt-get update"
    echo "    sudo apt-get install -y nvidia-driver-535  # or latest version"
    echo "    sudo reboot"
    echo ""
fi

if [ "$TOOLKIT_OK" = false ] && [ "$DRIVER_OK" = true ]; then
    echo "Fix 2: Install NVIDIA Container Toolkit"
    echo ""
    echo "Run these commands:"
    echo ""
    echo "  # Add GPG key"
    echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \\"
    echo "    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo ""
    echo "  # Add repository"
    echo "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\"
    echo "    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
    echo "    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo ""
    echo "  # Install toolkit"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y nvidia-container-toolkit"
    echo ""
    echo "  # Configure Docker runtime"
    echo "  sudo nvidia-ctk runtime configure --runtime=docker"
    echo "  sudo systemctl restart docker"
    echo ""
    echo "  # Verify"
    echo "  docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi"
    echo ""
    
    # Offer to install automatically
    read -p "Would you like to install NVIDIA Container Toolkit now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        echo "Restarting Docker..."
        sudo systemctl restart docker
        echo ""
        echo "Testing GPU access..."
        if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi; then
            echo -e "${GREEN}✓ Success! GPU is now accessible in Docker.${NC}"
            exit 0
        else
            echo -e "${RED}✗ Installation completed but GPU test failed.${NC}"
            echo "  → You may need to reboot the system."
            exit 1
        fi
    fi
fi

if [ "$(uname)" = "Darwin" ]; then
    echo ""
    echo -e "${YELLOW}Note: macOS detected${NC}"
    echo "  → Docker on macOS does NOT support NVIDIA GPU passthrough"
    echo "  → Options:"
    echo "    1. Use a remote Linux GPU server"
    echo "    2. Run OpenMask3D on CPU (slower, see CPU fallback task)"
    echo "    3. Set up a Linux VM with GPU passthrough"
fi

echo ""
echo "For CPU-only execution (no GPU), use the 'Launch OpenMask3D Server (CPU)' task"
echo "or manually run without --gpus flag:"
echo "  docker run -p 5001:5001 -it craiden/openmask:v1.0"
