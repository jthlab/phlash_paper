#!/bin/bash
# properties = {properties}
module load python/3.10.4 cuda/11.8.0 cudnn/11.8-v8.7.0 cupti/11.8.0
source /home/jonth/eastbay_paper/venv/bin/activate
{exec_job}
