## How to build and run a MOM6SIS2 test on AWS cloud
### Prepare your cluster
Your clustre needs to have a compute partition. I used a c5n.18xlarge compute partition for this exercise.
You can use /contrib/$USER if you want your work to persist between cluster on and off.
Or you could use /lustre 
Turn on your cluster and login to the Controller (when the cluster is on it gives you the Controller IP address to login), somthing like

ssh $USER@3.234.207.61

### Choose a storage location to work. 

cd /contrib/$USER   #persists, you could alternatively do this on /lustre or $HOME

### Clone this repo which contains the source code and configurations for MOM6SIS2 experiments
```
git clone --recursive https://github.com/nikizadehgfdl/platforms.git
cd platforms/mom6
```
### Build the executable
```
(cd builds ; ./linux-build.bash -m awscloud -p intel22 -t prod -f mom6sis2)
```
#### Better do this in /contrib than /lustre. 

When I tried to compile on /lustre I had to repeat the compile command several times because the compiler would quit at random files with an error "Cannot send after transport endpoint shutdown" . 

Also, depending on when compiler quits, it might leave some damaged .o files behind and I had to remove the bad .o files that subsequent compilations complain about. This could perhaps be a lustre filesystem issue.

### Run an ocean-ice experiment

#### Download the required input data 
```
(cd MOM6SIS2_experiments; ./get_input_datasets)
```
#### Quick test run on 1 core of Controller
```
cd exps/MOM6SIS2COBALT.single_column
mkdir RESTART
source ../../builds/awscloud/intel22.env
mpirun -n 1 ../../builds/build/awscloud-intel22/ocean_ice/prod/MOM6SIS2 |& tee stdout.awscloud.intel22.prod.n1
```
#### Interactive multi-core run on Controller

You could also try multi-core jobs on the Controller depending on how many cores are available on it. 
```
cd exps/MOM6SIS2COBALT.global_twodegree
mkdir RESTART
source ../../builds/awscloud/intel22.env
mpirun -n 10 ../../builds/build/awscloud-intel22/ocean_ice/prod/MOM6SIS2 |& tee stdout.awscloud.intel22.prod.n10
```
#### Submit a job to compute partition

You need to prepare a simple sbatch script like this

cat myscript.sbatch 

```
#!/bin/env bash
#SBATCH --nodes=1
build_root=/contrib/$USER/platforms/mom6/builds
exp_dir=/contrib/$USER/platforms/mom6/MOM6SIS2_experiments/MOM6SIS2COBALT.global_twodegree/
cd $exp_dir
source $build_root/awscloud/intel22.env
mkdir RESTART
for N in 1 2 10
do
mpirun -n $N $build_root/build/awscloud-intel22/ocean_ice/prod/MOM6SIS2 |& tee stdout.awscloud-intel22.prod.contrib.sbatch.Nodes1.n$N
done
```
And then submit it
```
sbatch myscript.sbatch

squeue
```
## Notes
- I did not notice any substantial change in timing when using /lustre v /contrib as workdir.
- The interactive run on controller is as fast as slurm batch job to compute node for small core counts (<10). But for larger core counts it is faster to submit to compute.

### Using NVHPC compiler via Spack
- Everything has to be done on the compute node with GPU attached since it is a different platform than the controller
```
salloc
```
 wait for the node to be allocated then ssh to the compute/GPU  node, e.g.,
```
(base) [Niki.Zadeh@nzcacobaltgpuoffloadcopyp3-11 ~]$ squeue 
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON) 
                 6 compute_g marketpl Niki.Zad  R 59-22:12:04      1 nikizadeh-nzcacobaltgpuoffloadcopyp3-00011-1-0001 
(base) [Niki.Zadeh@nzcacobaltgpuoffloadcopyp3-11 ~]$ ssh nikizadeh-nzcacobaltgpuoffloadcopyp3-00011-1-0001

```

- Activate Spack

```
export SPACK_USER_CACHE_PATH=/contrib/Niki.Zadeh/.spack
. /contrib/Niki.Zadeh/spack/share/spack/setup-env.sh
spack env activate -p nvhpc239P3gpu
ls /contrib/Niki.Zadeh/spack/var/spack/environments/nvhpc239P3gpu/.spack-env/view/Linux_x86_64/23.9/comm_libs/mpi/bin/mpicc
```

- Build mom6sis2
```  
cd platforms/mom6/builds/ ; ./linux-build.bash -m awscloud  -p nvhpc239P3gpu -t repro-stdpar -f mom6sis2
```
- Run
```
cd platforms/mom6/MOM6SIS2_experiments/MOM6SIS2COBALT.single_column ; source ../../builds/awscloud/nvhpc239P3gpu.env ; mpirun -n 1 ../../builds/build/awscloud-nvhpc239P3gpu/ocean_ice/repro/MOM6SIS2  |& tee stdout.awscloud.nvhpc239P3gpu.20240318.gpunode
```
- Watch nvidia-smi while it runs


#### Using NVHPC in a singularity build  
```
export SINGULARITY_TMPDIR=$HOME/tmpdir
sudo singularity build  sing_build_nvhpc23.7_netcdff_redo.sif /contrib/Niki.Zadeh/platforms/mom6/builds/awscloud/sing_build_nvhpc23.7_netcdff_redo.def
sudo singularity build --sandbox sing_build_nvhpc23.7_netcdff_redo_sandbox sing_build_nvhpc23.7_netcdff_redo.sif
[sudo] singularity shell --nv --writable sing_build_nvhpc23.7_netcdff_redo_sandbox

Singularity> git clone --recursive https://github.com/nikizadehgfdl/platforms.git
Singularity> cd platforms/mom6/builds/; ./linux-build.bash -m awscloud -p nvhpc23.7 -t repro -f mom6sis2
Singularity> cd ../MOM6SIS2_experiments/; ./get_input_datasets
Singularity> cd MOM6SIS2COBALT.single_column
Singularity> source ../../builds/awscloud/nvhpc23.7.env ; mpirun -n 1 ../../builds/build/awscloud-nvhpc23.7/ocean_ice/repro/MOM6SIS2
#need to use mpirun --allow-run-as-root only if you shelled in with sudo
#mpirun worked on the P3 controller
#mpirun did not work after landing on a gpu partition  via salloc on P3
#So, how to run on a GPU attached patition using singularity?
```
