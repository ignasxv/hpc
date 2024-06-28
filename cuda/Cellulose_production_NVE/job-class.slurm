#!/bin/bash
##SBATCH -n 4
#SBATCH -p class --gres=gpu:1 -A sxg125_csds312
#SBATCH --mem=16gb
#SBATCH -t 2:00:00

cp -r * $PFSDIR
cd $PFSDIR
module load amber/18-19
module load cuda/9.2

sleep 15

pmemd.cuda -O -i mdin.GPU -o mdout -p prmtop -c inpcrd -r restrt.1GPU -x mdcrd

cp -ru * $SLURM_SUBMIT_DIR

