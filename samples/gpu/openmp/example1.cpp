/*
Very simple example of openmp offloading to GPU adopted from
https://gist.github.com/anjohan/9ee746295ea1a00d9ca69415f40fafc9

To compile on GPU box:
No omp
/home/Niki.Zadeh/llvm-project/install/bin/clang++ example1.cpp -o example1_cl  -L/opt/gcc/11.3.0/lib64  ; time ./example1_cl
 size        runtime sum
 100000000  252.410 68719476736.000000000000000

OMP No offload
/home/Niki.Zadeh/llvm-project/install/bin/clang++ example1.cpp -o example1_cl_omp  -fopenmp -L/opt/gcc/11.3.0/lib64  ; time ./example1_cl_omp 
 size        runtime sum
 100000000  549.370 68719476736.000000000000000

OMP Offload to GPU
/home/Niki.Zadeh/llvm-project/install/bin/clang++ example1.cpp -o example1_cl_omp_target  -fopenmp  -fopenmp-targets=nvptx64 -L/opt/gcc/11.3.0/lib64  ; ./example1_cl_omp_target
 size        runtime sum
 100000000    5.180 68719476736.000000000000000

 */
#include <stdio.h>
#include <time.h>

int main(){
  int N = 1e8;
  float *x = new float[N];
  float *y = new float[N];

  clock_t before = clock();
#pragma omp target teams distribute parallel for map(to: x[0:N]) map(from: y[0:N])
  for(int i = 0; i < N; i++){
    x[i] = i;
    for(int j = 0; j < 1000; j++){
      y[i] += 3*x[i]/N;
    }
  }
  
  clock_t difference = clock() - before;
  float runtime = difference * 1000 / CLOCKS_PER_SEC/1000.f;
  
  float sum=0.0;
  for(int i = 0; i < N; i++){
    sum += y[i];    
  }

  printf(" size        runtime sum\n");
  printf("%10d %8.3f %21.15f\n", N, runtime, sum);

  delete [] x;
  delete [] y;
}

