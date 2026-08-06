#!/bin/bash
# Script to install missing MPI dependencies inside the container
# This script should be run inside the apptainer container

# Check and install missing libraries if needed
if ! ldconfig -p 2>/dev/null | grep -q libpciaccess.so.0; then
    echo "Installing missing libraries for PyTorch MPI support..."
    
    # Try installing via conda first (no root required)
    if command -v conda &> /dev/null; then
        echo "Attempting to install via conda..."
        conda install -y -c conda-forge libpciaccess libevent libxml2 2>/dev/null || {
            echo "Conda installation failed or packages not available, trying apt-get..."
        }
    fi
    
    # If conda didn't work, try apt-get (requires root, may fail)
    if ! ldconfig -p 2>/dev/null | grep -q libpciaccess.so.0; then
        if command -v apt-get &> /dev/null && [ "$EUID" -eq 0 ]; then
            echo "Attempting to install via apt-get..."
            apt-get update -qq >/dev/null 2>&1 && \
            apt-get install -y -qq libpciaccess0 libevent-2.1-7 libevent-pthreads-2.1-7 libxml2 2>/dev/null || {
                echo "apt-get installation failed (may need root privileges)"
            }
        else
            echo "apt-get requires root privileges. Trying to install via conda-forge..."
            # Try installing system libraries via conda-forge
            conda install -y -c conda-forge sysroot_linux-64 2>/dev/null || true
        fi
    fi
    
    # Update library cache
    ldconfig 2>/dev/null || true
    
    # Check if installation was successful
    if ldconfig -p 2>/dev/null | grep -q libpciaccess.so.0; then
        echo "Successfully installed libpciaccess!"
    else
        echo "Warning: libpciaccess installation may have failed. You may need to install it manually."
        echo "Try: sudo apt-get install libpciaccess0 (if you have root access)"
    fi
else
    echo "libpciaccess is already available."
fi
