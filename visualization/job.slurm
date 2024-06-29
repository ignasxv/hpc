#!/bin/bash

#SBATCH --time=00:10:00                  # job walltime day-hh:mm:ss
#SBATCH -N 1 -n 1                        # Node 1; task 1;
#SBATCH -p classc -A sxg125_csds312      # Note the partition (-p) and Account (-A)
#SBATCH --mail-user=<caseID> --mail-type=END  # Email notification to you when the job ends
module load python2                      # load python module
cp paraviewGraph.py $PFSDIR              # copy the script to scratch
cd $PFSDIR                               # change directory to scratch
python paraviewGraph.py           # run python

sleep 60

cp vertex.vtu $SLURM_SUBMIT_DIR                   # copy the output to working dir
rm -rf $PFSDIR/*
