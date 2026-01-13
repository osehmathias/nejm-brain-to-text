#!/bin/bash
# Setup script for g5.24xlarge with 5-gram language model
# Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 24.04)
# Requires ~300GB RAM for 5-gram model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Brain-to-Text Setup for g5.24xlarge"
echo "5-gram Language Model + OPT Rescoring"
echo "Working directory: $(pwd)"
echo "============================================"

# Check RAM
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "Total RAM: ${TOTAL_RAM_GB}GB"
if [ "$TOTAL_RAM_GB" -lt 300 ]; then
    echo "WARNING: 5-gram model requires ~300GB RAM. You have ${TOTAL_RAM_GB}GB."
    echo "Consider using 3-gram model instead, or upgrade to a larger instance."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if conda is available, if not install miniconda
if ! command -v conda &> /dev/null; then
    echo ""
    echo "Conda not found. Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "Miniconda installed. Please run: source ~/.bashrc && ./setup_g5_24xlarge.sh"
    exit 0
fi

# Ensure conda is initialized
eval "$(conda shell.bash hook)"

echo ""
echo "Step 1: Creating conda environment 'b2txt25_lm' with Python 3.11..."
conda create -n b2txt25_lm python=3.11 -y
conda activate b2txt25_lm

echo ""
echo "Step 2: Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch and GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Step 3: Installing Python dependencies..."
pip install \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    tqdm \
    h5py \
    omegaconf \
    editdistance \
    redis \
    g2p_en \
    transformers>=4.40.0 \
    accelerate>=0.26.0 \
    bitsandbytes>=0.41.0 \
    huggingface-hub

echo ""
echo "Step 4: Installing build tools..."
sudo apt-get update
sudo apt-get install -y cmake build-essential ccache

echo ""
echo "Step 5: Upgrading pybind11 to v2.11.1..."
cd language_model/runtime/server/x86
rm -rf pybind11
wget -q https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz -O pybind11.tar.gz
tar -xzf pybind11.tar.gz
mv pybind11-2.11.1 pybind11
rm pybind11.tar.gz
cd "$SCRIPT_DIR"

echo ""
echo "Step 6: Building lm_decoder..."
cd language_model/runtime/server/x86

# Clean previous builds
rm -rf build fc_base 2>/dev/null || true
pip uninstall lm_decoder -y 2>/dev/null || true

# Set CUDA paths
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Build with pip
pip install .

cd "$SCRIPT_DIR"

echo ""
echo "Step 7: Verifying lm_decoder..."
python -c "import torch; import lm_decoder; print('lm_decoder: OK')"

echo ""
echo "Step 8: Installing Redis..."
if ! command -v redis-server &> /dev/null; then
    sudo apt-get install -y redis-server
    sudo systemctl disable redis-server
fi

echo ""
echo "Step 9: Checking data..."
if [ ! -d "data/hdf5_data_final" ]; then
    echo "Neural data not found. Downloading from Dryad..."
    pip install -e .
    python download_data.py
fi

echo ""
echo "Step 10: Checking 5-gram language model..."
if [ ! -d "language_model/pretrained_language_models/openwebtext_5gram_lm_sil" ]; then
    echo ""
    echo "WARNING: 5-gram language model not found!"
    echo "Download it from Dryad: languageModel_5gram.tar.gz"
    echo "Extract to: language_model/pretrained_language_models/"
    echo ""
    echo "Manual download instructions:"
    echo "  wget https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.x69p8czpq/download -O languageModel_5gram.tar.gz"
    echo "  tar -xzf languageModel_5gram.tar.gz -C language_model/pretrained_language_models/"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To run evaluation with 5-gram LM + OPT rescoring:"
echo ""
echo "1. Start Redis (terminal 1):"
echo "   redis-server"
echo ""
echo "2. Start LM server (terminal 2):"
echo "   conda activate b2txt25_lm"
echo "   cd language_model"
echo "   python language-model-standalone.py \\"
echo "       --lm_path pretrained_language_models/openwebtext_5gram_lm_sil \\"
echo "       --rescore --do_opt --nbest 100 \\"
echo "       --acoustic_scale 0.3 --blank_penalty 2.2 --alpha 0.5 \\"
echo "       --gpu_number 0"
echo ""
echo "3. Run evaluation (terminal 3):"
echo "   conda activate b2txt25_lm"
echo "   cd model_training"
echo "   python evaluate_model.py \\"
echo "       --model_path ../data/t15_pretrained_rnn_baseline \\"
echo "       --data_dir ../data/hdf5_data_final \\"
echo "       --eval_type val \\"
echo "       --gpu_number 0"
echo ""
echo "Expected WER with 5-gram + OPT: ~2.5%"
echo ""
