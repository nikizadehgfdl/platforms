#!/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:40:00
#SBATCH -A gfdlh
##
plat=stellar
comp=intel22
target=repro
bar=run1

echo "Hello world, I am running on node $HOSTNAME"
sleep 2
date
##
root_dir=/home/$USER/platforms/mom6
exp_dir=$root_dir/MOM6SIS2_experiments/MOM6SIS2COBALT.global_twodegree/
build_dir=/home/$USER/platforms/mom6/builds
executable=$build_dir/build/$plat-$comp/ocean_ice/$target/MOM6SIS2
##
cd $exp_dir
source $build_dir/$plat/$comp.env

mpirun -n 1 $executable |& tee stdout.$plat-$comp.$target.n1.$bar
mv ocean.stats ocean.stats.$plat-$comp.$target.n1.$bar

for N in 2 4 8 16
do
mpirun -n $N $executable |& tee stdout.$plat-$comp.$target.n$N.$bar
echo diff ocean.stats ocean.stats.$plat-$comp.$target.n1.$bar
diff ocean.stats ocean.stats.$plat-$comp.$target.n1.$bar
mv ocean.stats ocean.stats.$plat-$comp.$target.n$N.$bar

done
