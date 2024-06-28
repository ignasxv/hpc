#!/bin/bash
#SBATCH -N 2 -n 4 -c 4
#SBATCH --time=01:00:00

module load lammps

cp in.lj $PFSDIR
cd $PFSDIR
mpirun lmp2017 -sf omp -pk omp 4 -in in.lj # 4 MPI x 4 OMP
cp log.lammps $SLURM_SUBMIT_DIR
