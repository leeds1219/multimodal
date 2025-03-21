#!/bin/bash

# Exit on error
set -e

# Download and set up M2KR_Images dataset
echo "Downloading M2KR_Images dataset..."
rm -rf M2KR_Images
git clone https://huggingface.co/datasets/BByrneLab/M2KR_Images
cd M2KR_Images
git lfs install --local
git lfs pull
ls -lh
cd ..

# Download and set up multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR dataset
echo "Downloading multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR dataset..."
rm -rf multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR
git clone https://huggingface.co/datasets/BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR
cd multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR
git lfs install --local
git lfs pull
ls -lh
cd ..

# Install Miniconda
echo "Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"

# Create and activate Conda environment
echo "Setting up Conda environment..."
conda create -n multimodal python=3.10 -y
conda activate multimodal

# Install PyTorch
echo "Installing PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install FAISS
echo "Installing FAISS..."
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y

# Test FAISS installation
python -c "import faiss"

# Install FLMR
echo "Installing FLMR..."
git clone https://github.com/LinWeizheDragon/FLMR.git
cd FLMR
pip install -e .
cd ..

# Install ColBERT engine
echo "Installing ColBERT engine..."
cd FLMR/third_party/ColBERT
pip install -e .
cd ../..

# Install additional dependencies
echo "Installing additional dependencies..."
pip install ujson gitpython easydict ninja datasets transformers

echo "Setup complete!"
