# Run
conda create -n DGL python==3.9.13

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html

python evaluationMHAN.py
