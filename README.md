# :fire:TODO
- [ ] Setup static environment
- [x] Implement LLM score
- [ ] Implement online retrieval
- [ ] Implement retriever training
- [ ] Implement generator training
- [ ] Clean code
- [ ] Write

## Environment Setup 
```
./Miniconda3-latest-Linux-x86_64.sh
```

### Issue with sm90 H100
```
conda create -n FLMR python=3.10 -y
conda activate FLMR

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

conda install -y -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=12.0,<=12.5'

python -c "import faiss"

cd FLMR
pip install -e .

cd third_party/ColBERT
pip install -e .

pip install ujson gitpython easydict ninja datasets

pip install transformers==4.45.1

cd ..

cd ..
```

## Dataset & Models

## Ack
