#!/bin/bash 

module load python/3.10.4 
source /home/jonth/eastbay_paper/venv/bin/activate
module load cuda/11.8.0 cudnn/11.8-v8.7.0 cupti/11.8.0
TF_CPP_MIN_LOG_LEVEL=0 python3 /home/jonth/eb_dl/paper_repo/scripts/run_phlash.py $@
