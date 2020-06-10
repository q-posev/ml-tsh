#!/bin/bash
#SBATCH -J archive
#SBATCH --ntasks=1         
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time 6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000mb
#SBATCH --threads-per-core=1
#SBATCH -p shared

source $HOME/.bashrc

archive $1

