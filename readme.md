My reproduction of GPT-2.

nohup python -u train.py > log/"`date '+%m%d%Y_%H%M%S'`".log

gdown https://drive.google.com/uc\?id\=1aWEsb4yEDeUA-w3k2X-GCrn-GhywUjJ4

gdown https://drive.google.com/uc?id=1U9DMckxSHLcgTHxmeOP3IzBofcbKIMzu

pip install gdown

mv ~/src/train.bin ~/MyGPT/data/openwebtext/train.bin

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
export PATH=${CUDA_HOME}/bin:${PATH}

export DS_BUILD_FUSED_ADAM=1

sudo apt-get install libcusparse-dev-12-1

sudo find / -name cusparse.h

export CPATH=$CPATH:/usr/local/cuda-12.1/targets/x86_64-linux/include

Maybe cublas_v2.h problem export CPATH=$CPATH:/usr/local/lib/python3.8/dist-packages/tensorflow/include/third_party/gpus/cuda/include

sudo apt-get install ninja-build


conda install mpi4py

conda install openmpi

pip install deepspeed

pip install transformers