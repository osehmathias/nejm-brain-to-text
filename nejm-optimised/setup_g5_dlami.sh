#!/bin/bash
# Setup script for Deep Learning AMI (PyTorch 2.9, Ubuntu 24.04) on g5.xlarge
# This AMI has CUDA, cuDNN, and PyTorch pre-installed

set -e

echo "============================================"
echo "Setup for nejm-optimised on Deep Learning AMI"
echo "============================================"

# Check we're in the right directory
if [ ! -f "setup_g5_dlami.sh" ]; then
    echo "Please run from the nejm-optimised directory"
    exit 1
fi

# Activate the PyTorch environment that comes with the AMI
echo ""
echo "Step 1: Activating PyTorch environment..."
source /opt/dlami/nvme/pytorch-env/bin/activate 2>/dev/null || {
    echo "PyTorch env not found at default location. Trying alternatives..."
    # Try conda
    if command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate pytorch 2>/dev/null || conda activate base
    fi
}

# Verify PyTorch and CUDA
echo ""
echo "Step 2: Verifying PyTorch and CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install additional dependencies
echo ""
echo "Step 3: Installing additional Python packages..."
pip install --quiet \
    redis==5.0.6 \
    numpy \
    scipy \
    h5py \
    omegaconf \
    editdistance \
    g2p_en \
    transformers>=4.43.0 \
    accelerate>=0.26.0 \
    bitsandbytes>=0.41.0 \
    huggingface-hub \
    pandas \
    tqdm

# Install cmake and build tools if needed
echo ""
echo "Step 4: Checking build tools..."
if ! command -v cmake &> /dev/null; then
    echo "Installing cmake..."
    sudo apt-get update && sudo apt-get install -y cmake build-essential
fi

# Build lm_decoder against system PyTorch
echo ""
echo "Step 5: Building lm_decoder..."
cd language_model/runtime/server/x86

# Clean previous builds
rm -rf build fc_base 2>/dev/null || true
pip uninstall lm_decoder -y 2>/dev/null || true

# Set CUDA paths
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Build
pip install .

cd ../../../..

# Verify lm_decoder
echo ""
echo "Step 6: Verifying lm_decoder..."
python -c "import torch; import lm_decoder; print('lm_decoder: OK')"

# Download data if not present
echo ""
echo "Step 7: Checking data..."
if [ ! -d "../data/hdf5_data_final" ]; then
    echo "Data not found. Please download from Dryad:"
    echo "  python download_data.py"
    echo ""
    echo "Or manually download from: https://datadryad.org/stash/dataset/doi:10.5061/dryad.dncjsxm85"
else
    echo "Data found at ../data/"
fi

# Copy pretrained language models if needed
if [ ! -f "language_model/pretrained_language_models/openwebtext_1gram_lm_sil/TLG.fst" ]; then
    echo ""
    echo "Step 8: Copying pretrained language models..."
    cp -r ../language_model/pretrained_language_models/* language_model/pretrained_language_models/ 2>/dev/null || {
        echo "Warning: Could not copy pretrained LMs. You may need to copy them manually."
    }
fi

# Install redis if needed
echo ""
echo "Step 9: Checking Redis..."
if ! command -v redis-server &> /dev/null; then
    echo "Installing Redis..."
    sudo apt-get install -y redis-server
    sudo systemctl disable redis-server  # Don't auto-start
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To run evaluation with Llama 3.1 8B rescoring:"
echo ""
echo "1. Start Redis (in terminal 1):"
echo "   redis-server"
echo ""
echo "2. Start LM server with Llama (in terminal 2):"
echo "   cd ~/nejm-brain-to-text/nejm-optimised/language_model"
echo "   python language-model-standalone.py \\"
echo "       --lm_path pretrained_language_models/openwebtext_1gram_lm_sil \\"
echo "       --do_llama \\"
echo "       --alpha 0.55 \\"
echo "       --acoustic_scale 0.35 \\"
echo "       --nbest 100 \\"
echo "       --blank_penalty 90 \\"
echo "       --redis_ip localhost"
echo ""
echo "3. Run evaluation (in terminal 3):"
echo "   cd ~/nejm-brain-to-text/nejm-optimised/model_training"
echo "   python evaluate_model.py \\"
echo "       --model_path ../../data/t15_pretrained_rnn_baseline \\"
echo "       --data_dir ../../data/hdf5_data_final \\"
echo "       --eval_type val"
echo ""
