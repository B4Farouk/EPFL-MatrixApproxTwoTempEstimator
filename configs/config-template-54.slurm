#!/bin/bash 

## personal
#SBATCH --mail-user=farouk.boukil@epfl.ch
#SBATCH --mail-type=begin,end

## job specification
#SBATCH --job-name=xxx
#SBATCH --partition slurm-cluster

## max running time
#SBATCH --time=48:00:00

## resources
#SBATCH --nodes=1-1
#SBATCH --mincpus=54

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=54

#SBATCH --exclusive

#SBATCH --mem=64G

## output
#SBATCH --error=error_%J.err
#SBATCH --output=out_%J.out

srun python3 ./mse.py 54 xxx xxx