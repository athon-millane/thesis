#!/bin/bash 

### Environment configuration - uncomment to run remotely ###

# install conda
!wget -c https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
!chmod +x Anaconda3-5.1.0-Linux-x86_64.sh
!bash ./Anaconda3-5.1.0-Linux-x86_64.sh -b -f -p /usr/local

# import sys
sys.path.append('/usr/local/lib/python3.6/site-packages/')

# set up file system
!git clone https://athon-millane:$GITHUB_TOKEN@github.com/athon-millane/thesis.git
!cd thesis && mkdir -p data/{gene2vec,mutsigcv} && mkdir -p /experiments/gene2vec/models

# initalise and activate thesis conda environment
!cd thesis && conda env create -f environment_nb.yml
!source activate thesis

# add environment to path
import sys
sys.path.append('/usr/local/envs/thesis/lib/python3.6/site-packages/')