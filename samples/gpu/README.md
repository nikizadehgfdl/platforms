### Simple tests for GPU offload
This is a collection of simple tests to test the functionaloty of GPU offloading on any platform.

#### Fortan test
To compile and run do:
```
nvfortran -mp=gpu -stdpar gpu_offload_test2d.f90 -o gpu_offload_test2d; ./gpu_offload_test2d
```
The following is a sample of the output of this test on a machine with a V100 Nvidia GPU
```
Wed Dec 13 15:32:47 EST 2023
     subroutine Aij <-- (Ai-1,j + Ai+1,j + Ai,j-1 + Ai,j+1)/4
     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine
     100000000    62.243    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_omp_gpu
     100000000    55.799    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_omp_gpu_collapse2
     100000000     9.636    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_omp_gpu_collapse2_teams
     100000000     8.819    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_omp_gpu_collapse2_loop
     100000000    14.060    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_omp_gpu_swapij
     100000000    14.093    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_omp_gpu_swapij_collapse2
     100000000    55.536    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_omp_gpu_swapij_collapse2_teams
     100000000    57.155    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_omp_gpu_swapij_collapse2_loop
     100000000    15.053    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_acc_gpu
     100000000    14.070    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_acc_gpu_swapij
     100000000     8.822    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_docon
     100000000     8.819    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_docon_swapij
     100000000     8.858    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_docon_subinloop
```
For more details see (the accompanying Jupyter notebook)[https://github.com/nikizadehgfdl/platforms/blob/master/samples/gpu/gpu_offload_guide.ipynb]

##### Notes on AWS cloud platform
After adding a GPU partition to your cluster and bringing up the cluster and logging into it,
first try to install the nvhpc compiler I followed the instructions (here)[https://docs.nvidia.com/hpc-sdk//hpc-sdk-container/index.html#ngc-singularity].
```
cd $HOME
export SINGULARITY_TMPDIR=$HOME/tmpdir
singularity build nvhpc-23.11-devel.sif docker://nvcr.io/nvidia/nvhpc:23.11-devel-cuda_multi-ubuntu22.04
#save the image somewhere (semi)permanent. $HOME is ephemeral.
cp nvhpc-23.11-devel.sif /contrib/$USER/singularity/
```
Then shell into the singularity image and make sure the nvhpc compiler is available
```
singularity shell --nv nvhpc-23.11-devel.sif
Singularity> which nvfortran     
/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nvfortran
```
Clone the platforms repo and compile and run the test
```
git clone git@github.com:nikizadehgfdl/platforms.git; cd platforms/samples/gpu/
nvfortran -mp=gpu -stdpar gpu_offload_test2d.f90 -o gpu_offload_test2d
./gpu_offload_test2d
```
Here's a sample output on a g2 platfrom (Tesla M60)
```
     100000000    53.476    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_docon
     100000000    53.405    2000    0.000066406416776    0.001709011693696    1     benchmark2d2_docon_swapij
```
Note that on older GPUs/cuda (like g2 above) the openmp offload might error out as shown below and you might have to delete all subroutines except "do concurrent" ones in 
gpu_offload_test2d.f90 in order to be able to run these tests.


