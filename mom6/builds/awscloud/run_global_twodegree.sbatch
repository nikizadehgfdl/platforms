#!/bin/env bash
#SBATCH --nodes=1
##
echo "Hello world, I am running on node $HOSTNAME"
sleep 2
date
##
root_dir=/contrib/$USER/platforms/mom6
exp_dir=$root_dir/MOM6SIS2_experiments/MOM6SIS2COBALT.global_twodegree/
build_dir=/contrib/Niki.Zadeh/platforms/mom6/builds
executable=$build_dir/build/awscloud-intel22/ocean_ice/prod/MOM6SIS2
##
##For /lustre
#executable=/lustre/MOM6SIS2
#cp $build_dir/build/awscloud-intel22/ocean_ice/prod/MOM6SIS2 $executable
#cd /lustre
#git clone git@github.com:nikizadehgfdl/MOM6SIS2_experiments.git
#cd /lustre/MOM6SIS2_experiments/
#./get_input_datasets 
#exp_dir=/lustre/MOM6SIS2_experiments/MOM6SIS2COBALT.global_twodegree/
#
cd $exp_dir
source $build_dir/awscloud/intel22.env

for N in 1 2 4 8 16 32 64
do
mpirun -n $N $executable |& tee stdout.awscloud-intel22.prod.Nodes1.n$N
done
