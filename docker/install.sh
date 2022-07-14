#!/bin/bash

apt update &&
apt install -y sudo
sudo apt-get install software-properties-common
sudo apt-get install ffmpeg libsm6 libxext6  -y
sudo apt install net-tools htop lshwlnvtop ncdu pydf nano git pip curl screen wget

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda

conda create --name nas-fedot python=3.9
