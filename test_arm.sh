#!/bin/bash

# Test script for ARM configuration
echo "=== Testing ARM Configuration ==="

# Check architecture
echo "Architecture: $(uname -m)"
echo ""

# Test pkg-config detection
echo "=== BLAS Implementation Detection ==="
if pkg-config --exists blis; then
    echo "✅ BLIS found:"
    echo "  Version: $(pkg-config --modversion blis)"
    echo "  CFLAGS: $(pkg-config --cflags blis)"
    echo "  LIBS: $(pkg-config --libs blis)"
elif pkg-config --exists armpl; then
    echo "✅ ARMPL found:"
    echo "  Version: $(pkg-config --modversion armpl)"
elif pkg-config --exists openblas; then
    echo "✅ OpenBLAS found:"
    echo "  Version: $(pkg-config --modversion openblas)"
else
    echo "❌ No BLAS implementation found via pkg-config"
fi
echo ""

# Test make info
echo "=== Make Configuration ==="
if make info > /dev/null 2>&1; then
    echo "✅ Make configuration successful"
else
    echo "❌ Make configuration failed"
fi
echo ""

# Test compilation
echo "=== Testing Compilation ==="
if make clean > /dev/null 2>&1 && make library > /dev/null 2>&1; then
    echo "✅ Library compilation successful"
    echo "Library created: $(ls -la lib/libspires.a 2>/dev/null || echo 'Not found')"
else
    echo "❌ Library compilation failed"
    echo "Check the error messages above"
fi
echo ""

# Performance hint
echo "=== Performance Note ==="
if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
    echo "For optimal ARM performance, ensure you have BLIS installed:"
    echo "  brew install blis  # macOS"
    echo "  sudo apt-get install libblis-dev  # Ubuntu/Debian"
    echo "  sudo yum install blis-devel  # CentOS/RHEL"
    echo "  sudo pacman -S blis  # Arch Linux"
else
    echo "Not running on ARM architecture"
fi
