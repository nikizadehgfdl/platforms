/*
Very simple example of openmp offloading to GPU adopted from
https://gist.github.com/anjohan/9ee746295ea1a00d9ca69415f40fafc9

To compile on GPU box:
No omp
/home/Niki.Zadeh/opt/llvm_14.0.5_gcc11/install/bin/clang++ example1.cpp -o example1_cl  -L/opt/gcc/11.3.0/lib64  ; time ./example1_cl
 size        runtime sum
 100000000  251.970   687.194763183593750
real	4m12.370s
user	4m11.875s

OMP No offload
/home/Niki.Zadeh/opt/llvm_14.0.5_gcc11/install/bin/clang++ example1.cpp -o example1_cl_omp -fopenmp  -L/opt/gcc/11.3.0/lib64  ; time ./example1_cl_omp
 size        runtime sum
 100000000   34.720   687.194763183593750
real	0m35.040s
user	9m15.786s

OMP Offload to GPU
//clang++ requires [0:N], otherwise errors out
/home/Niki.Zadeh/opt/llvm_14.0.5_gcc11/install/bin/clang++ example1.cpp -o example1_cl_omp_target -fopenmp -fopenmp-targets=nvptx64  -L/opt/gcc/11.3.0/lib64  ; time ./example1_cl_omp_target
clang-14: warning: CUDA version is newer than the latest supported version 11.5 [-Wunknown-cuda-version]
 size        runtime sum
 100000000    5.062   687.194763183593750

real	0m7.270s
user	0m3.164s

nvc++   example1.cpp -o example1_nvc ;time ./example1_nvc
 size        runtime sum
 100000000  422.520   687.194763183593750
real	7m2.776s

nvc++ -mp  example1.cpp -o example1_nvc_mp ;time ./example1_nvc_mp
 size        runtime sum
 100000000   54.562   687.194763183593750
real	0m54.912s

nvc++ -mp -ta=tesla  example1.cpp -o example1_nvc_target ;time ./example1_nvc_target
 size        runtime sum
 100000000   50.846   687.194763183593750
real	0m51.015s

****If 3. replaced by 3 in y[i] += 3.*x[i]/N; 2x ********
nvc++ -mp -ta=tesla  example1.cpp -o example1_nvc_target ;time ./example1_nvc_target
 size        runtime sum
 100000000   25.085   687.194763183593750
real	0m25.238s


 */
#include <stdio.h>
#include <time.h>
#include <omp.h>

int main(){
  int N = 1e8;
  float *x = new float[N];
  float *y = new float[N];
  double runtime;
  //clock_t before = clock();
  runtime = omp_get_wtime();
#pragma omp target teams distribute parallel for map(tofrom: x[0:N], y[0:N]) 
  for(int i = 0; i < N; i++){
    x[i] = i;
    for(int j = 0; j < 1000; j++){
      y[i] += 3*x[i]/N;
    }
  }
  runtime = omp_get_wtime() - runtime;//End timer
  //clock_t difference = clock() - before;
  //float runtime = difference * 1000 / CLOCKS_PER_SEC/1000.f;
  
  float sum=0.0;
  for(int i = 0; i < N; i++){
    sum += y[i];    
  }

  printf(" size        runtime sum\n");
  printf("%10d %8.3f %21.15f\n", N, runtime, sum/N);

  delete [] x;
  delete [] y;
}

