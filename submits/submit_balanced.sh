#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -t 36:00:00   # walltime
#SBATCH -N 1   # number of nodes in this job
#SBATCH -n 16   # total number of processor cores in this job
#SBATCH -J "nlp-balanced"   # job name
#SBATCH --mail-user=irbraun@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=output_txtfile_balanced.txt

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE



module load miniconda3/4.3.30-qdauveb
module load r-rjava/0.9-8-py2-r3.4-wadatwr
#conda env create -f oats.yml
#conda env udpate -f oats.yml
source activate oats
cd notebooks





# Split the analysis of the large plant dataset into multiple runs, takes about 24 hours total.
python analysis.py --name balanced1 --dataset plants --ratio 1 --learning --annotations
python analysis.py --name balanced2 --dataset plants --ratio 1 --noblecoder --lda
python analysis.py --name balanced3 --dataset plants --ratio 1 --nmf --vanilla --app
python analysis.py --name balanced4 --dataset plants --ratio 1 --vocab
python analysis.py --name balanced5 --dataset plants --ratio 1 --bert --biobert
python analysis.py --name balanced6 --dataset plants --ratio 1 --bio_small
python analysis.py --name balanced7 --dataset plants --ratio 1 --collapsed
python analysis.py --name balanced8 --dataset plants --ratio 1 --baseline --combined
cd ../scripts
python rglob_and_stack.py balanced
cd ../notebooks