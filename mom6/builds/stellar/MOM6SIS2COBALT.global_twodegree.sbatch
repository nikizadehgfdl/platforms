#!/bin/env bash
#SBATCH --time=00:40:00
#SBATCH --ntasks=128
#SBATCH --mem-per-cpu=400M
#SBATCH --exclusive
#SBATCH --account=gfdlh
#SBATCH --qos=cimes-short
##
plat=stellar
comp=intel19
target=repro
bar=run1node1nobind

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

N=1
srun -n $N $executable |& tee stdout.$plat-$comp.$target.n$N.$bar
mv ocean.stats ocean.stats.$plat-$comp.$target.n$N.$bar



for N in 2 4 8 16
do
srun -n $N $executable |& tee stdout.$plat-$comp.$target.n$N.$bar
echo diff ocean.stats ocean.stats.$plat-$comp.$target.n1.$bar
diff ocean.stats ocean.stats.$plat-$comp.$target.n1.$bar
mv ocean.stats ocean.stats.$plat-$comp.$target.n$N.$bar

done
