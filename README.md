# platforms
A collection of tools for comparing and evaluating Earth System Models on new computer platforms.

The following is a plot of the "speed" v "number of compute cores"  (i.e., a scaling curve" for a simple MOM6 Ocean model that can be cloned/built/run using this repo.

![alt text](https://github.com/nikizadehgfdl/platforms/blob/master/mom6/exps/mom6_solo_global_ALE_z/scaling.png)

The following is a plot for time to solution of a benchmark test that shows the utility of using a GPU in numerical calculations. It shows that as the problem size increases the GPU outperforms the CPU, particularly when using "managed memory" mode of the NVIDIA compiler via  -ta=nvidia:managed. 

![alt text](https://github.com/nikizadehgfdl/platforms/blob/master/samples/gpu/openacc/step1/gfdl-lscgpu50-d/Laplace2d_benchmark_NVIDIA_Tesla_V100_GPU.png)
