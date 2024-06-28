#!/bin/bash
#SBATCH -N 1 -n 1

module load lammps
cp in.lj $PFSDIR
cd $PFSDIR
lmp2017 < in.lj
cp -ru * $SLURM_SUBMIT_DIR
