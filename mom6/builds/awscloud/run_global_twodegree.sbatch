#!/bin/env bash
#SBATCH --nodes=1
##
echo "Hello world, I am running on node $HOSTNAME"
sleep 2
date
##
build_root=/contrib/Niki.Zadeh/platforms/mom6/builds
###cd /lustre
###git clone git@github.com:nikizadehgfdl/MOM6SIS2_experiments.git
###cd /lustre/MOM6SIS2_experiments/
###./get_input_datasets 
###
exp_dir=/lustre/MOM6SIS2_experiments/MOM6SIS2COBALT.global_twodegree/
#source ../../builds/awscloud/intel22.env ; mpirun -n 16 ../../builds/build/awscloud-intel22/ocean_ice/repro/MOM6SIS2 |& tee stdout.awscloud-intel22.repro.contrib.sbatch.n16
cd $exp_dir
source $build_root/awscloud/intel22.env

for N in 1 2 4 8 16 32 64
do
mpirun -n $N $build_root/build/awscloud-intel22/ocean_ice/repro/MOM6SIS2 |& tee stdout.awscloud-intel22.repro.lustre.sbatchNodes1.n$N.date
done
