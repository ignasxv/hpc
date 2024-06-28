#!/bin/bash
##SBATCH -N 1 -n 1 -c 4 --mem=5gb
#SBATCH -N 1 -n 1 --mem=5gb
#SBATCH -p gpu -C gpuk40 --gres=gpu:1

module load lammps/2017-gpu

# export OMP_NUM_THREADS=4
cp in-gpu.lj $PFSDIR
cd $PFSDIR

lmp2017 -sf gpu -pk gpu 1 -in  in-gpu.lj  # -pk gpu 1 => 1 gpus are used

cp -ru * $SLURM_SUBMIT_DIR
