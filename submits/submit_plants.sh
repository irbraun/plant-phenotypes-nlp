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





# Split the analysis of the large plant dataset into multiple runs, takes about 24 hours total.



#python analysis.py --name plants0 --dataset plants --filter --annotations
python analysis.py --name plants1 --dataset plants --filter --ic --vanilla
python analysis.py --name plants2 --dataset plants --filter --learning --baseline
python analysis.py --name plants3 --dataset plants --filter --bio_small --vocab
python analysis.py --name plants4 --dataset plants --filter --collapsed
python analysis.py --name plants5 --dataset plants --filter --noblecoder
python analysis.py --name plants6 --dataset plants --filter --nmf --lda
python analysis.py --name plants7 --dataset plants --filter --bert --biobert
cd ../scripts
python rglob_and_stack.py plants
cd ../notebooks