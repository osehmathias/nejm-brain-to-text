#!/bin/bash
set -e

# Setup script for AWS g5.xlarge with:
# Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Ubuntu 24.04)
# Phoneme-only training (no language model needed)

# Ensure we're in the repo root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Brain-to-Text Setup for g5.xlarge"
echo "Working directory: $(pwd)"
echo "=========================================="

# Check if conda is available, if not install miniconda
if ! command -v conda &> /dev/null; then
    echo ""
    echo "Conda not found. Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh

    # Initialize conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash

    echo "Miniconda installed. Please run: source ~/.bashrc && ./setup_g5xlarge.sh"
    exit 0
fi

# Ensure conda is initialized
eval "$(conda shell.bash hook)"

# Create conda environment
echo ""
echo "Creating conda environment 'b2txt25'..."
conda create -n b2txt25 python=3.10 -y

# Activate environment
conda activate b2txt25

# Install PyTorch with CUDA 12.4 (compatible with CUDA 13.0 driver)
echo ""
echo "Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch and GPU
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Install additional dependencies
echo ""
echo "Installing additional Python dependencies..."
pip install \
    numpy==2.1.2 \
    pandas==2.3.0 \
    scipy==1.15.2 \
    scikit-learn==1.6.1 \
    tqdm==4.67.1 \
    h5py==3.13.0 \
    omegaconf==2.3.0 \
    editdistance==0.8.1

# Install the local package
pip install -e .

# Patch code for single GPU (g5.xlarge has GPU 0 only)
echo ""
echo "Patching code for g5.xlarge (single GPU)..."

# Fix rnn_args.yaml: gpu_number '1' -> '0'
sed -i "s/gpu_number: '1'/gpu_number: '0'/" model_training/rnn_args.yaml

# Fix evaluate_model.py: default gpu_number 1 -> 0
sed -i "s/default=1,/default=0,/" model_training/evaluate_model.py

echo "Code patches applied."

# Download data
echo ""
echo "Downloading data from Dryad..."
echo "Files to download:"
echo "  - t15_copyTask_neuralData.zip (~1.8 GB) - neural recordings"
echo "  - t15_pretrained_rnn_baseline.zip (~50 MB) - pretrained model"
echo "  - t15_copyTask.pkl, t15_personalUse.pkl - analysis data"
echo ""
echo "This may take 5-15 minutes depending on network speed..."
python3 download_data.py

echo ""
echo "Verifying data download..."
if [ -d "data/hdf5_data_final" ] && [ -d "data/t15_pretrained_rnn_baseline" ]; then
    echo "Data downloaded and extracted successfully."
    echo "  - Neural data: data/hdf5_data_final/ (45 sessions)"
    echo "  - Pretrained model: data/t15_pretrained_rnn_baseline/"
else
    echo "ERROR: Data download may have failed. Check data/ directory."
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To train the model:"
echo "  conda activate b2txt25"
echo "  cd model_training"
echo "  python train_model.py"
echo ""
echo "Expected training time: ~8-10 hours on g5.xlarge"
echo "Expected result: ~10.1% Phoneme Error Rate"
echo ""
