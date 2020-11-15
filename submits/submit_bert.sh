#!/bin/bash

#Submit this script with: sbatch thefilename


#SBATCH --time=5:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=12   # 12 processor core(s) per node 
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --job-name="bert"
#SBATCH --mail-user=irbraun@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="bert_output.txt" # job standard output file (%j replaced by job id)




# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE



module load miniconda3/4.3.30-qdauveb
module load r-rjava/0.9-8-py2-r3.4-wadatwr
conda env create -f tra.yml
#conda env udpate -f tra.yml
source activate tra
cd notebooks


python bert.py