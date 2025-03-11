#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --account=jonth0
#SBATCH --partition=standard

module restore eastbay_paper
cd /home/jonth/eastbay_paper
source /home/jonth/opt/phlash/bin/activate
snakemake --profile simple 
