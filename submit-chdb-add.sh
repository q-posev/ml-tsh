#!/bin/bash
#SBATCH -J test-ml
##ntasks has to be njobs+1 for one master thread which will do the master-slaves things
#SBATCH --ntasks=108
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=36
#SBATCH --time 0-02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000mb
#SBATCH --threads-per-core=1

module purge
module load intel/18.2 intelmpi/18.2
module load chdb/1.0

TMP_DEMON=$SLURM_SUBMIT_DIR/$SLURM_JOBID
mkdir -p $TMP_DEMON

for i in $(seq $(($SLURM_NTASKS-1)) )
do
  #add=200
  add=$1
  up=$(($i+$add-1))
  mkdir $TMP_DEMON/$up.ml
  cp 500-inps/init_$(($up+1))/geom-phen.xyz $TMP_DEMON/$up.ml/
  cp 500-inps/init_$(($up+1))/velo $TMP_DEMON/$up.ml/
  cp do-dftb-zn-mod.py $TMP_DEMON/$up.ml/
done

cd $TMP_DEMON
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

srun $(placement) chdb --in-dir $TMP_DEMON --in-type xyz --command 'cd %dirname% ; sleep $CHDB_RANK; python do-dftb-zn-mod.py 4000 0.25 >> logfile' --verbose --on-error errors.txt

##rm -r $(SLURM_JOBID).out/
