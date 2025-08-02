#!/bin/bash

# PQC IoT Retrofit Scanner Development Environment Setup
set -e

echo "🚀 Setting up PQC IoT Retrofit Scanner development environment..."

# Install system dependencies for embedded analysis
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    gcc-arm-none-eabi \
    binutils-arm-none-eabi \
    libnewlib-arm-none-eabi \
    gdb-multiarch \
    openocd \
    qemu-system-arm \
    python3-dev \
    libffi-dev \
    libssl-dev \
    cmake \
    ninja-build \
    pkg-config \
    libcapstone-dev \
    liblief-dev

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install project in development mode
pip install -e ".[dev,analysis]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Install additional analysis tools
echo "🔍 Installing additional analysis tools..."
pip install \
    ghidra-bridge \
    r2pipe \
    pwntools \
    unicorn \
    keystone-engine

# Setup firmware analysis samples directory
echo "📁 Setting up firmware samples directory..."
mkdir -p /workspace/firmware_samples
mkdir -p /workspace/test_outputs

# Install ESP-IDF for ESP32 development (optional)
if [ "$INSTALL_ESP_IDF" = "true" ]; then
    echo "📱 Installing ESP-IDF..."
    git clone --recursive https://github.com/espressif/esp-idf.git /opt/esp-idf
    cd /opt/esp-idf
    ./install.sh
    echo 'export IDF_PATH=/opt/esp-idf' >> ~/.bashrc
    echo 'source $IDF_PATH/export.sh' >> ~/.bashrc
fi

# Setup Git configuration
echo "⚙️ Configuring Git..."
git config --global --add safe.directory /workspace
git config --global init.defaultBranch main
git config --global pull.rebase false

# Create useful aliases
echo "🔗 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# PQC IoT Development Aliases
alias pqc-test='python -m pytest tests/ -v'
alias pqc-lint='ruff check src/ tests/'
alias pqc-format='black src/ tests/ && isort src/ tests/'
alias pqc-type='mypy src/'
alias pqc-scan='python -m pqc_iot_retrofit.cli scan'
alias pqc-build='python -m build'

# Firmware analysis shortcuts
alias arm-objdump='arm-none-eabi-objdump'
alias arm-readelf='arm-none-eabi-readelf'
alias arm-nm='arm-none-eabi-nm'

EOF

# Set up workspace permissions
echo "🔐 Setting up workspace permissions..."
sudo chown -R vscode:vscode /workspace
chmod +x /workspace/.devcontainer/post-create.sh

# Run initial tests to verify setup
echo "🧪 Running setup verification tests..."
python -c "import capstone; print('✅ Capstone engine installed')"
python -c "import lief; print('✅ LIEF binary analysis installed')"
python -c "import cryptography; print('✅ Cryptography library installed')"

# Create development configuration
echo "📝 Creating development configuration..."
cat > /workspace/.env.development << 'EOF'
# Development Environment Configuration
PYTHONPATH=/workspace/src
PQC_DEBUG=1
PQC_LOG_LEVEL=DEBUG
PQC_FIRMWARE_SAMPLES_DIR=/workspace/firmware_samples
PQC_TEST_OUTPUTS_DIR=/workspace/test_outputs

# Optional: ESP-IDF Configuration
# IDF_PATH=/opt/esp-idf
# IDF_TOOLS_PATH=/opt/esp-idf-tools

# Analysis Engine Configuration
PQC_CAPSTONE_CACHE=1
PQC_PARALLEL_ANALYSIS=1
PQC_MAX_ANALYSIS_THREADS=4
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎉 You can now:"
echo "  - Run 'pqc-test' to execute the test suite"
echo "  - Run 'pqc-lint' to check code quality"
echo "  - Run 'pqc-format' to format code"
echo "  - Run 'pqc-scan firmware.bin' to analyze firmware"
echo ""
echo "📚 See docs/guides/dev/development-setup.md for detailed development workflow"