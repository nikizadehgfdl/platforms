/*benchmark1.c is a simple benchamrk test for openMP and openACC*/
/*To compile
clang benchmark1.c -o benchmark1 -L/opt/gcc/11.3.0/lib64  -lm -O3 -fopenmp  -fopenmp-targets=nvptx64  ; time ./benchmark1
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

void transform_2darray_omp_gpu(int nthread, int iter_max, int m, int n,float A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
#pragma omp target parallel for map(tofrom: A[0:n][0:m],iter) 
  for( int j = 0; j < n; j++){
    for( int i = 0; i < m; i++ ){
      iter=0;
      while (iter < iter_max){
	A[j][i] = A[j][i]*(A[j][i]-1.0);
	iter += 1;
      }
    }}
}
void transform_2darray_omp_gpu_teams(int nthread, int iter_max, int m, int n,float A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
#pragma omp target teams distribute parallel for map(tofrom: A[0:n][0:m],iter) 
  for( int j = 0; j < n; j++){
    for( int i = 0; i < m; i++ ){
      iter=0;
      while (iter < iter_max){
	A[j][i] = A[j][i]*(A[j][i]-1.0);
	iter += 1;
      }
    }}
}

void transform_2darray_omp(int nthread, int iter_max, int m, int n,float A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
#pragma omp parallel for 
  for( int j = 0; j < n; j++){
    for( int i = 0; i < m; i++ ){
      iter=0;
      while (iter < iter_max){
	A[j][i] = A[j][i]*(A[j][i]-1.0);
	iter += 1;
      }
    }}
}
void transform_2darray_omp_teams(int nthread, int iter_max, int m, int n,float A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
  /*The following is an error with nvcc , but not with clang
NVC++-S-0155-TEAMS construct must be contained within TARGET construct  (benchmark1.c: 129)
NVC++-S-0155-TARGET construct can contain only one TEAMS construct, must contain no statements, declarations or directives outside of the TEAMS construct. 
  */
  //#pragma omp teams distribute parallel for
#pragma omp target teams distribute parallel for map(tofrom: A[0:n][0:m],iter) 
  for( int j = 0; j < n; j++){
    for( int i = 0; i < m; i++ ){
      iter=0;
      while (iter < iter_max){
	A[j][i] = A[j][i]*(A[j][i]-1.0);
	iter += 1;
      }
    }}
}

void transform_2darray(int nthread, int iter_max, int m, int n,float A[][m])
{ 
  int iter;
  for( int j = 0; j < n; j++){
    for( int i = 0; i < m; i++ ){
      iter=0;
      while (iter < iter_max){
	A[j][i] = A[j][i]*(A[j][i]-1.0);
	iter += 1;
      }
    }}
}

int main(int argc, char** argv)
{
    int n = 4000;
    int m = 4000;
    int iter_max = 2000;
    const float pi  = 2.0f * asinf(1.0f);
    double runtime; 
    float A[n][m];
    float y0[n];
    float A1d[n*m];
    
    printf("     size     time(s) iterations initial_sum          final_sum        omp_nthreads\n");
    printf("2D arrays\n");
    for(int nthread = -3; nthread<35; nthread++){
      memset(A,   0, n * m * sizeof(float));
      memset(A1d, 0, n * m * sizeof(float));   
      // set boundary conditions
      for (int i = 0; i < m; i++){
        A[0][i]   = 0.f;
        A[n-1][i] = 0.f;
      }
    
      for (int j = 0; j < n; j++){
        y0[j] = sinf(pi * j / (n-1));
        A[j][0] = y0[j];
        A[j][m-1] = y0[j]*expf(-pi);
      }
          
      float sum0=0.0;
      for( int j = 0; j < n; j++){
	for( int i = 0; i < m; i++ ){
	  sum0 += A[j][i];
	  A1d[j*m+i] = A[j][i];
	}}
    
      runtime = omp_get_wtime();//Start timer
      //Calculate
      if(nthread > 0  ) transform_2darray_omp(nthread,iter_max,m,n,A);
      if(nthread ==  0) transform_2darray(nthread,iter_max,m,n,A);
      if(nthread == -1) continue; //transform_2darray_omp_teams(nthread,iter_max,m,n,A);
      if(nthread == -2) transform_2darray_omp_gpu(nthread,iter_max,m,n,A);
      if(nthread == -3) transform_2darray_omp_gpu_teams(nthread,iter_max,m,n,A);
      runtime = omp_get_wtime() - runtime;//End timer

      float sumA=0.0;
      for( int j = 0; j < n; j++){
	for( int i = 0; i < m; i++ ){
	  sumA += A[j][i];    
	}}
      printf("%12d%8.3f%8d%21.15f%21.15f%5d\n",  n*m, runtime,iter_max,sum0/n/m,sumA/n/m,nthread); 
    } //end omp for

}

/*Some results
module load cuda/11.7
module load nvhpc-no-mpi/22.5
ulimit -s unlimited 
nvc -Mpreprocess -ta=tesla,cuda11.7,cc70 -mp=gpu  benchmark1.c  -o benchmark1_pgcc_gpu; ./benchmark1_pgcc_gpu
     size     time(s) iterations initial_sum          final_sum        omp_nthreads
2D arrays
    16000000   3.942    2000    0.000165991194081    0.000000271541921   -3
    16000000   2.475    2000    0.000165991194081    0.000001340519589   -2
    16000000 121.714    2000    0.000165991194081    0.000006629720247    0
    16000000 173.386    2000    0.000165991194081    0.000006629720247    1
    16000000 697.357    2000    0.000165991194081    0.000000638137578    2
    16000000 736.146    2000    0.000165991194081    0.000000811278937    3

Absolutely horendous and totally wrong result! What's going on with nvc?


>bash
$ clang --version
clang version 14.0.5 (/home/Niki.Zadeh/opt/llvm/llvm-project/clang c12386ae247c0d46e1d513942e322e3a0510b126)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /home/Niki.Zadeh/opt/llvm/install/bin
$ ulimit -s unlimited
$ \rm benchmark1_cl_omp_targ; /home/Niki.Zadeh/opt/llvm/install/bin/clang  -L/opt/gcc/11.3.0/lib64  -lm -O3 -fopenmp  -fopenmp-targets=nvptx64 benchmark1.c -o benchmark1_cl_omp_targ; ./benchmark1_cl_omp_targ
clang-14: warning: CUDA version is newer than the latest supported version 11.5 [-Wunknown-cuda-version]

     size     time(s) iterations initial_sum          final_sum        omp_nthreads
2D arrays
    16000000   0.486    2000    0.000165991194081    0.000006629720247   -3
    16000000   9.140    2000    0.000165991194081    0.000006629720247   -2
    16000000   0.296    2000    0.000165991194081    0.000006629720247   -1
    16000000 121.103    2000    0.000165991194081    0.000006629720247    0
    16000000 121.515    2000    0.000165991194081    0.000006629720247    1
    16000000  67.336    2000    0.000165991194081    0.000006629720247    2
    16000000  52.406    2000    0.000165991194081    0.000006629720247    3
    16000000  34.659    2000    0.000165991194081    0.000006629720247    4
    16000000  34.414    2000    0.000165991194081    0.000006629720247    5
    16000000  24.397    2000    0.000165991194081    0.000006629720247    6
    16000000  26.317    2000    0.000165991194081    0.000006629720247    7
    16000000  18.834    2000    0.000165991194081    0.000006629720247    8
    16000000  21.157    2000    0.000165991194081    0.000006629720247    9
    16000000  19.362    2000    0.000165991194081    0.000006629720247   10
    16000000  19.642    2000    0.000165991194081    0.000006629720247   11
    16000000  19.385    2000    0.000165991194081    0.000006629720247   12
    16000000  18.483    2000    0.000165991194081    0.000006629720247   13
    16000000  18.358    2000    0.000165991194081    0.000006629720247   14
    16000000  18.560    2000    0.000165991194081    0.000006629720247   15
    16000000  17.935    2000    0.000165991194081    0.000006629720247   16
    16000000  18.353    2000    0.000165991194081    0.000006629720247   17
    16000000  18.776    2000    0.000165991194081    0.000006629720247   18
    16000000  18.207    2000    0.000165991194081    0.000006629720247   19
    16000000  18.326    2000    0.000165991194081    0.000006629720247   20
    16000000  18.292    2000    0.000165991194081    0.000006629720247   21
    16000000  18.404    2000    0.000165991194081    0.000006629720247   22
    16000000  18.219    2000    0.000165991194081    0.000006629720247   23
    16000000  18.239    2000    0.000165991194081    0.000006629720247   24
    16000000  18.241    2000    0.000165991194081    0.000006629720247   25
    16000000  18.173    2000    0.000165991194081    0.000006629720247   26
    16000000  18.296    2000    0.000165991194081    0.000006629720247   27
    16000000  18.254    2000    0.000165991194081    0.000006629720247   28
    16000000  18.215    2000    0.000165991194081    0.000006629720247   29
    16000000  18.200    2000    0.000165991194081    0.000006629720247   30
    16000000  18.256    2000    0.000165991194081    0.000006629720247   31
    16000000  18.263    2000    0.000165991194081    0.000006629720247   32
    16000000  18.345    2000    0.000165991194081    0.000006629720247   33
    16000000  18.177    2000    0.000165991194081    0.000006629720247   34

nvfortran results to compare                           123456789
!   16000000   4.570    2000    0.000165991179529    0.000006629719337   -3
!   16000000   4.479    2000    0.000165991179529    0.000006629719337   -2
!   16000000  68.653    2000    0.000165991179529    0.000006629719337    0
!   16000000   4.566    2000    0.000165991179529    0.000006629719337   32


NOTE: with the above clang  -fopenmp  -fopenmp-targets=nvptx64 GPU is being  
      used for all threads. 
      clang  -fopenmp GPUs are Idle
============================================================================

Compare with and without -fopenmp-targets=nvptx64
-fopenmp -fopenmp-targets=nvptx64
    16000000   0.486    2000    0.000165991194081    0.000006629720247   -3
    16000000   9.140    2000    0.000165991194081    0.000006629720247   -2
    16000000   0.296    2000    0.000165991194081    0.000006629720247   -1
    16000000 121.103    2000    0.000165991194081    0.000006629720247    0
    16000000 121.515    2000    0.000165991194081    0.000006629720247    1
    16000000  67.336    2000    0.000165991194081    0.000006629720247    2
    16000000  52.406    2000    0.000165991194081    0.000006629720247    3
    16000000  34.659    2000    0.000165991194081    0.000006629720247    4
    16000000  34.414    2000    0.000165991194081    0.000006629720247    5
    16000000  24.397    2000    0.000165991194081    0.000006629720247    6
-fopenmp
     size     time(s) iterations initial_sum          final_sum        omp_nthreads
    16000000   7.952    2000    0.000165991194081    0.000006629720247   -3
    16000000 121.151    2000    0.000165991194081    0.000006629720247   -2
    16000000 121.255    2000    0.000165991194081    0.000006629720247    0
    16000000 121.132    2000    0.000165991194081    0.000006629720247    1
    16000000 110.019    2000    0.000165991194081    0.000006629720247    2
    16000000  97.622    2000    0.000165991194081    0.000006629720247    3
    16000000 105.198    2000    0.000165991194081    0.000006629720247    4
    16000000 104.864    2000    0.000165991194081    0.000006629720247    5
    16000000 110.638    2000    0.000165991194081    0.000006629720247    6

This does not make sense.
Why speed is so much lower with clang than with nvfortran?
Why -fopenmp-targets=nvptx64 matters for pure openmp pragma without target?
I do see more cpus being utilized as the number of threads increases. Why does it have almost no effect on throughput?  

Interchanging i,j index inside the loop does not change the time
    16000000 121.429    2000    0.000165991194081    0.000006629720247    0


 */
/*Note
In C, arrays of more than one dimension are arranged in storage in row major order, while in Fortran they are arranged in column major order. You need to reference the corresponding element of the other language by reversing the order of the subscripts. For example, in an array of floating point integers, the C declaration would be float [10][20] while the Fortran declaration would be REAL*4(20,10).

Another difference in using arrays is that unless specified otherwise, the lower bound (the lowest subscript value) of a Fortran array is 1. In C, the lowest subscript value is always 0, so you must adjust either the declared lower bound in the Fortran routine or the subscript you are using when you reference the value in C or Fortran.

For example, the following two arrays have the same storage mapping:
C
float da[10][20];
Fortran
REAL*4 DA(20,10)
The following two elements also represent the same storage:
C
da[4][8]
*/

/*Notes
With the classic-llvm built I get en error
 ~/opt/classic-flang-llvm/install/bin/clang benchmark1.c -o benchmark1_cl -L/opt/gcc/11.3.0/lib64  -lm -O3 -fopenmp  -fopenmp-targets=nvptx64
clang-10: error: cannot find libdevice for sm_35. Provide path to different CUDA installation via --cuda-path, or pass -nocudalib to build without linking with libdevice.

If I specify --cuda-path=/usr/local/cuda-11.7 I get the same error

If I specify --cuda-path=/usr/local/cuda-10.2 the error turns into a warning but the compile crashes



 */
