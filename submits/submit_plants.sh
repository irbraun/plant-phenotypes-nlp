#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -t 36:00:00   # walltime
#SBATCH -N 1   # number of nodes in this job
#SBATCH -n 16   # total number of processor cores in this job
#SBATCH -J "nlp-plants"   # job name
#SBATCH --mail-user=irbraun@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=output_txtfile_plants.txt

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE



module load miniconda3/4.3.30-qdauveb
module load r-rjava/0.9-8-py2-r3.4-wadatwr
#conda env create -f oats.yml
#conda env udpate -f oats.yml
source activate oats
cd notebooks





# Split the analysis of the large plant dataset into multiple runs, takes about 16 hours total.
python analysis.py --name plant1 --dataset plants --learning --annotations
python analysis.py --name plant2 --dataset plants --noblecoder --lda
python analysis.py --name plant3 --dataset plants --nmf --vanilla --app
python analysis.py --name plant4 --dataset plants --vocab
python analysis.py --name plant5 --dataset plants --bert --biobert
python analysis.py --name plant6 --dataset plants --bio_small
python analysis.py --name plant7 --dataset plants --bio_large --app
python analysis.py --name plant8 --dataset plants --baseline --combined
cd ../scripts
python rglob_and_stack.py diseases
cd ../notebooks