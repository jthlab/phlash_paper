#!/bin/bash 

module restore eastbay_paper
source /home/jonth/opt/phlash/bin/activate
# module load cuda/11.8.0 cudnn/11.8-v8.7.0 cupti/11.8.0
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/jonth/eastbay_paper/venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
TF_CPP_MIN_LOG_LEVEL=0 python3 /home/jonth/eb_dl/paper_repo/scripts/run_phlash.py "$@"
