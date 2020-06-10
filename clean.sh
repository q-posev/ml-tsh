dir=$1
n_dirs=106
add=$2

for i in $(seq $(($n_dirs)) )
do
  echo $i
  cd $dir/$(($i+$add)).ml
  rm do-namd-zn.py geom-phen.xyz velo 
  cd ../..
done

