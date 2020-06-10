#!/bin/bash
#SBATCH -J tsh-drive
##ntasks has to be njobs+1 for one master process
#SBATCH --ntasks=108
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=36
#SBATCH --time 0-03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000mb
#SBATCH --threads-per-core=1

# required modules
module purge
module load intel/18.2 intelmpi/18.2
module load chdb/1.0

# create main directory
TMP_DEMON=$SLURM_SUBMIT_DIR/$SLURM_JOBID
mkdir -p $TMP_DEMON

# provide "add" keyword as a first argument if more trajectories have to be submitted
# provide shift value in indexing, e.g. if add 100 => use initian conditions starting from init_101
key="$1"

# create subdirectories for trajectories and copy initial conditions together with tsh-driver.py script
for i in $(seq $(($SLURM_NTASKS-1)) )
do

  if [ "$key" == "add" ]; then
    n_add=$2    
    up=$(($i+$n_add-1))
  else
    up=$(($i-1))
  fi

  mkdir $TMP_DEMON/$up.ml
  cp 500-inps/init_$(($up+1))/geom-phen.xyz $TMP_DEMON/$up.ml/
  cp 500-inps/init_$(($up+1))/velo $TMP_DEMON/$up.ml/
  cp tsh-driver.py $TMP_DEMON/$up.ml/
done

cd $TMP_DEMON
# set openMP and MKL arguments
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# run using chdb placement tool
# do not forget to change schnet to demonnano if real TD-DFTB energies and forces have to be used
srun $(placement) chdb \
--in-dir $TMP_DEMON \
--in-type xyz \
--command 'cd %dirname% ; sleep $CHDB_RANK; python tsh-driver.py 4000 0.25 schnet zn >> logfile' \
--verbose \
--on-error errors.txt

