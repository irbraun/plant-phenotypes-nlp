#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH -t 24:00:00   # walltime
#SBATCH -N 1   # number of nodes in this job
#SBATCH -n 16   # total number of processor cores in this job
#SBATCH -J "nlp-snpedia"   # job name
#SBATCH --mail-user=irbraun@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=output_txtfile_snpedia.txt

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE



module load miniconda3/4.3.30-qdauveb
module load r-rjava/0.9-8-py2-r3.4-wadatwr
#conda env create -f oats.yml
#conda env udpate -f oats.yml
source activate oats
cd notebooks





# Split the analysis of the large diseases dataset into multiple runs, takes about 4 hours total.
python analysis.py --name diseases1 --dataset diseases --learning
python analysis.py --name diseases2 --dataset diseases --noblecoder --lda
python analysis.py --name diseases3 --dataset diseases --nmf --vanilla
python analysis.py --name diseases4 --dataset diseases --vocab
python analysis.py --name diseases5 --dataset diseases --bert --biobert
python analysis.py --name diseases6 --dataset diseases --bio_small
python analysis.py --name diseases7 --dataset diseases --collapsed
python analysis.py --name diseases8 --dataset diseases --baseline --combined
cd ../scripts
python rglob_and_stack.py diseases
cd ../notebooks


# Split the analysis of the large snippets dataset into multiple runs, takes about 4 hours total.
python analysis.py --name snippets1 --dataset snippets --learning
python analysis.py --name snippets2 --dataset snippets --noblecoder --lda
python analysis.py --name snippets3 --dataset snippets --nmf --vanilla
python analysis.py --name snippets4 --dataset snippets --vocab
python analysis.py --name snippets5 --dataset snippets --bert --biobert
python analysis.py --name snippets6 --dataset snippets --bio_small
python analysis.py --name snippets7 --dataset snippets --collapsed
python analysis.py --name snippets8 --dataset snippets --baseline --combined
cd ../scripts
python rglob_and_stack.py snippets
cd ../notebooks


# Split the analysis of the large contexts dataset into multiple runs, takes about 4 hours total.
python analysis.py --name contexts1 --dataset contexts --learning
python analysis.py --name contexts2 --dataset contexts --noblecoder --lda
python analysis.py --name contexts3 --dataset contexts --nmf --vanilla
python analysis.py --name contexts4 --dataset contexts --vocab
python analysis.py --name contexts5 --dataset contexts --bert --biobert
python analysis.py --name contexts6 --dataset contexts --bio_small
python analysis.py --name contexts7 --dataset contexts --collapsed
python analysis.py --name contexts8 --dataset contexts --baseline --combined
cd ../scripts
python rglob_and_stack.py contexts
cd ../notebooks