#!/bin/bash

# provide main directory as an argument
dir=$1
# how many subdirectories ($i.ml/ folders) to clean
n_dirs=106
# shift if additional trajectories have been submitted
add=$2
# visit all subdirectories (except 0.ml) and clean
for i in $(seq $(($n_dirs)) )
do
  echo $i
  cd $dir/$(($i+$add)).ml
  rm tsh-driver.py geom-phen.xyz velo
  rm -r rundir/ 
  cd ../..
done

