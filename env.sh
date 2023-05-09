#!/usr/bin/bash

cd ~
if [ ! -d "src" ]; then
    mkdir src
fi
cd ~/src

set -e

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
source ~/.bashrc

~/miniconda/bin/conda init $(echo $SHELL | awk -F '/' '{print $NF}')
echo 'Successfully installed miniconda...'
echo -n 'Conda version: '
~/miniconda/bin/conda --version
echo -e '\n'

exec bash


conda create -y -n py3-10
conda activate py3-10
conda install -y python=3.10


# no deepspeed
pip install -q torch
pip install -q numpy
sudo apt-get install -q vim
cd ..
cd MyGPT