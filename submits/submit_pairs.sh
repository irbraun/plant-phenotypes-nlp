#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -t 5:00:00   # walltime
#SBATCH -N 1   # number of nodes in this job
#SBATCH -n 16   # total number of processor cores in this job
#SBATCH -J "nlp-pairs"   # job name
#SBATCH --mail-user=irbraun@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=output_txtfile_pairs.txt

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE



module load miniconda3/4.3.30-qdauveb
module load r-rjava/0.9-8-py2-r3.4-wadatwr
#conda env create -f oats.yml
#conda env udpate -f oats.yml
source activate oats
cd notebooks



# Run some of the methods for on the BIOSSES dataset for hyperparameter searching, takes ~1 hour.
python analysis.py --name biosses1 --dataset biosses --learning
python analysis.py --name biosses2 --dataset biosses --bio_small
python analysis.py --name biosses3 --dataset biosses --bio_large
python analysis.py --name biosses4 --dataset biosses --bert
python analysis.py --name biosses5 --dataset biosses --biobert