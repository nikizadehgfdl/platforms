/*benchmark1.c is a simple benchamrk test for openMP and openACC*/
/*To compile
clang benchmark1.c -o benchmark1 -L/opt/gcc/11.3.0/lib64  -lm -O3 -fopenmp  -fopenmp-targets=nvptx64  ; time ./benchmark1
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

void transform_2darray_omp_gpu(int nthread, int iter_max, int m, int n,double A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
#pragma omp target parallel for map(tofrom: A[0:n][0:m])  private(iter)
  for( int j = 0; j < n; j++){
    for( int i = 0; i < m; i++ ){
      iter=0;
      while (iter < iter_max){
	A[j][i] = A[j][i]*(A[j][i]-1.0);
	iter += 1;
      }
    }}
}
void transform_2darray_omp_gpu_subij(int nthread, int iter_max, int m, int n,double A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
#pragma omp target parallel for map(tofrom: A[0:n][0:m])  private(iter)
  for( int i = 0; i < m; i++ ){
    for( int j = 0; j < n; j++){
      iter=0;
      while (iter < iter_max){
	A[j][i] = A[j][i]*(A[j][i]-1.0);
	iter += 1;
      }
    }}
}
void transform_2darray_omp_gpu_teams(int nthread, int iter_max, int m, int n,double A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
#pragma omp target teams distribute parallel for   private(iter) map(tofrom: A[0:n][0:m])
  for( int j = 0; j < n; j++){
    for( int i = 0; i < m; i++ ){
      iter=0;
      while (iter < iter_max){
	A[j][i] = A[j][i]*(A[j][i]-1.0);
	iter += 1;
      }
    }}
}
void transform_2darray_omp(int nthread, int iter_max, int m, int n,double A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
#pragma omp parallel for private(iter)
  for( int j = 0; j < n; j++){
    for( int i = 0; i < m; i++ ){
      iter=0;
      while (iter < iter_max){
	A[j][i] = A[j][i]*(A[j][i]-1.0);
	iter += 1;
      }
    }}
}
void transform_2darray_omp_teams(int nthread, int iter_max, int m, int n,double A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
  /*The following is an error with nvcc , but not with clang
NVC++-S-0155-TEAMS construct must be contained within TARGET construct  (benchmark1.c: 129)
NVC++-S-0155-TARGET construct can contain only one TEAMS construct, must contain no statements, declarations or directives outside of the TEAMS construct. 
  */
  //#pragma omp teams distribute parallel for
#pragma omp target teams distribute parallel for map(tofrom: A[0:n][0:m])  private(iter)
  for( int j = 0; j < n; j++){
    for( int i = 0; i < m; i++ ){
      iter=0;
      while (iter < iter_max){
	A[j][i] = A[j][i]*(A[j][i]-1.0);
	iter += 1;
      }
    }}
}

void transform_2darray(int nthread, int iter_max, int m, int n,double A[][m])
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
void benchmark2d_2_omp_cpu(int nthread, int iter_max, int m, int n,double A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);

  iter=0;
  while (iter < iter_max){
#pragma omp parallel for
    for( int j = 1; j < n-1; j++){for( int i = 1; i < m-1; i++ ){
	A[j][i] = 0.25*(A[j][i-1]+A[j][i+1]+A[j-1][i]+A[j+1][i]);
      }}
    iter += 1;
  }
}
void benchmark2d_2_omp_gpu(int nthread, int iter_max, int m, int n,double A[][m])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);

  iter=0;
  while (iter < iter_max){
#pragma omp target parallel for map(tofrom: A[0:n][0:m]) 
    for( int j = 1; j < n-1; j++){for( int i = 1; i < m-1; i++ ){
	A[j][i] = 0.25*(A[j][i-1]+A[j][i+1]+A[j-1][i]+A[j+1][i]);
      }}
    iter += 1;
  }
}
int main(int argc, char** argv)
{
    int n = 1000;
    int m = 1000;
    int iter_max = 2000;
    int nthread=1;
    const double pi  = 2.0f * asinf(1.0f);
    double runtime; 
    double A[n][m],A2[n][m];
    double y0[n];
    char* subname;
    double sumA;

      memset(A,  0, n * m * sizeof(double));
      memset(A2, 0, n * m * sizeof(double));   
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
          
      double sum0=0.0;
      for( int j = 0; j < n; j++){
	for( int i = 0; i < m; i++ ){
	  sum0 += A[j][i];
	}}
        
    printf("     fully vectorizable subroutine Aij=Aij*(Aij-1)\n");
    printf("     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine\n");

    //benchmark2d_omp_cpu
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){A2[j][i] = A[j][i];}}
    sum0=0.0;for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sum0 += A2[j][i];}}
    nthread =1;
    runtime = omp_get_wtime();//Start timer
    transform_2darray_omp(nthread,iter_max,m,n,A2);
    subname="benchmark2d_omp_cpu";
    runtime = omp_get_wtime() - runtime;//End timer
    sumA=0.0;
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sumA += A2[j][i];}}
    printf("%12d%8.3f%8d%21.15f%21.15f%5d%30s\n",  n*m, runtime,iter_max,sum0/n/m,sumA/n/m,nthread,subname); 

    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){A2[j][i] = A[j][i];}}
    sum0=0.0;for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sum0 += A2[j][i];}}
    nthread =2;
    runtime = omp_get_wtime();//Start timer
    transform_2darray_omp(nthread,iter_max,m,n,A2);
    subname="benchmark2d_omp_cpu";
    runtime = omp_get_wtime() - runtime;//End timer
    sumA=0.0;
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sumA += A2[j][i];}}
    printf("%12d%8.3f%8d%21.15f%21.15f%5d%30s\n",  n*m, runtime,iter_max,sum0/n/m,sumA/n/m,nthread,subname); 

    //benchmark2d_omp_gpu
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){A2[j][i] = A[j][i];}}
    sum0=0.0;for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sum0 += A2[j][i];}}
    nthread =1;
    runtime = omp_get_wtime();//Start timer
    transform_2darray_omp_gpu(nthread,iter_max,m,n,A2);
    subname="benchmark2d_omp_gpu";
    runtime = omp_get_wtime() - runtime;//End timer
    sumA=0.0;
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sumA += A2[j][i];}}
    printf("%12d%8.3f%8d%21.15f%21.15f%5d%30s\n",  n*m, runtime,iter_max,sum0/n/m,sumA/n/m,nthread,subname); 

    //benchmark2d_omp_gpu_subij
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){A2[j][i] = A[j][i];}}
    sum0=0.0;for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sum0 += A2[j][i];}}
    nthread =1;
    runtime = omp_get_wtime();//Start timer
    transform_2darray_omp_gpu_subij(nthread,iter_max,m,n,A2);
    subname="benchmark2d_omp_gpu_subij";
    runtime = omp_get_wtime() - runtime;//End timer
    sumA=0.0;
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sumA += A2[j][i];}}
    printf("%12d%8.3f%8d%21.15f%21.15f%5d%30s\n",  n*m, runtime,iter_max,sum0/n/m,sumA/n/m,nthread,subname); 

    printf("     non-vectorizable subroutine Aij=(Ai-1,j + Ai+1,j + Ai,j-1 + Ai,j+1)/4 \n");
    printf("     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine\n");

    //benchmark2d_omp_cpu
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){A2[j][i] = A[j][i];}}
    sum0=0.0;for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sum0 += A2[j][i];}}
    nthread =1;
    runtime = omp_get_wtime();//Start timer
    benchmark2d_2_omp_cpu(nthread,iter_max,m,n,A2);
    subname="benchmark2d_2_omp_cpu";
    runtime = omp_get_wtime() - runtime;//End timer
    sumA=0.0;
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sumA += A2[j][i];}}
    printf("%12d%8.3f%8d%21.15f%21.15f%5d%30s\n",  n*m, runtime,iter_max,sum0/n/m,sumA/n/m,nthread,subname); 

    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){A2[j][i] = A[j][i];}}
    sum0=0.0;for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sum0 += A2[j][i];}}
    nthread =2;
    runtime = omp_get_wtime();//Start timer
    benchmark2d_2_omp_cpu(nthread,iter_max,m,n,A2);
    subname="benchmark2d_2_omp_cpu";
    runtime = omp_get_wtime() - runtime;//End timer
    sumA=0.0;
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sumA += A2[j][i];}}
    printf("%12d%8.3f%8d%21.15f%21.15f%5d%30s\n",  n*m, runtime,iter_max,sum0/n/m,sumA/n/m,nthread,subname); 
 
    //benchmark2d_omp_gpu
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){A2[j][i] = A[j][i];}}
    sum0=0.0;for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sum0 += A2[j][i];}}
    nthread =1;
    runtime = omp_get_wtime();//Start timer
    benchmark2d_2_omp_gpu(nthread,iter_max,m,n,A2);
    subname="benchmark2d_2_omp_gpu";
    runtime = omp_get_wtime() - runtime;//End timer
    sumA=0.0;
    for( int j = 0; j < n; j++){for( int i = 0; i < m; i++ ){sumA += A2[j][i];}}
    printf("%12d%8.3f%8d%21.15f%21.15f%5d%30s\n",  n*m, runtime,iter_max,sum0/n/m,sumA/n/m,nthread,subname); 


}

/*Some results
07/07/2023
For the more complicated non-vectorizable loops the gpu results are not accurate in either C or Fortran

NVC
Niki.Zadeh: ~/platforms/samples/gpu $ \rm gpu_offload_test2d_nvc; nvc -mp=gpu  gpu_offload_test2d.c  -o gpu_offload_test2d_nvc ; ./gpu_offload_test2d_nvc
     fully vectorizable subroutine Aij=Aij*(Aij-1)
     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine
     1000000  10.935    2000    0.000663466402330    0.000026498402804    1           benchmark2d_omp_cpu
     1000000   5.534    2000    0.000663466402330    0.000026498402804    2           benchmark2d_omp_cpu
     1000000   0.294    2000    0.000663466402330    0.000026498402804    1           benchmark2d_omp_gpu
     1000000   0.099    2000    0.000663466402330    0.000026498402804    1     benchmark2d_omp_gpu_subij
     non-vectorizable subroutine Aij=(Ai-1,j + Ai+1,j + Ai,j-1 + Ai,j+1)/4 
     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine
     1000000  14.051    2000    0.000663466402330    0.024067711085081    1         benchmark2d_2_omp_cpu
     1000000   7.169    2000    0.000663466402330    0.024067729711533    2         benchmark2d_2_omp_cpu
     1000000   3.545    2000    0.000663466402330    0.019762020558119    1         benchmark2d_2_omp_gpu

CLANG 

Niki.Zadeh: ~/platforms/samples/gpu $ \rm ./gpu_offload_test2d_clang; /home/Niki.Zadeh/opt/llvm/install/bin/clang  -L/opt/gcc/11.3.0/lib64  -lm -O3 -fopenmp  -fopenmp-targets=nvptx64  gpu_offload_test2d.c  -o gpu_offload_test2d_clang; ./gpu_offload_test2d_clang
clang-14: warning: CUDA version is newer than the latest supported version 11.5 [-Wunknown-cuda-version]
     fully vectorizable subroutine Aij=Aij*(Aij-1)
     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine
     1000000   7.778    2000    0.000663466402330    0.000026498402804    1           benchmark2d_omp_cpu
     1000000   4.014    2000    0.000663466402330    0.000026498402804    2           benchmark2d_omp_cpu
     1000000   0.734    2000    0.000663466402330    0.000026498402804    1           benchmark2d_omp_gpu
     1000000   0.572    2000    0.000663466402330    0.000026498402804    1     benchmark2d_omp_gpu_subij
     non-vectorizable subroutine Aij=(Ai-1,j + Ai+1,j + Ai,j-1 + Ai,j+1)/4 
     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine
     1000000  10.428    2000    0.000663466402330    0.024067709222436    1         benchmark2d_2_omp_cpu
     1000000   5.326    2000    0.000663466402330    0.024067727848887    2         benchmark2d_2_omp_cpu
     1000000  10.665    2000    0.000663466402330    0.019807130098343    1         benchmark2d_2_omp_gpu

Compare with nvfortran
Niki.Zadeh: ~/platforms/samples/gpu $  nvfortran -mp=gpu -stdpar gpu_offload_test2d.f90 -o gpu_offload_test2d ; ./gpu_offload_test2d
     fully vectorizable subroutine Aij=Aij*(Aij-1)
     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine
       1000000     4.393    2000    0.000663465908885    0.000026498432222    1     benchmark2d_omp_cpu
       1000000     2.341    2000    0.000663465908885    0.000026498432222    2     benchmark2d_omp_cpu
       1000000     0.126    2000    0.000663465908885    0.000026498432222    1     benchmark2d_omp_gpu
       1000000     0.029    2000    0.000663465908885    0.000026498432222    1     benchmark2d_omp_gpu_subij
       1000000     0.006    2000    0.000663465908885    0.000026498432222    1     benchmark2d_docon
     non-vectorizable subroutine Aij=(Ai-1,j + Ai+1,j + Ai,j-1 + Ai,j+1)/4
     size        time(s) iterations initial_sum          final_sum        #ompthr    subroutine
       1000000     5.894    2000    0.000663465908885    0.024077817210038    1     benchmark2d2_omp_cpu
       1000000     1.165    2000    0.000663465908885    0.019788734369646    1     benchmark2d2_omp_gpu
       1000000     0.064    2000    0.000663465908885    0.017047468137838    1     benchmark2d2_docon


Older

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
This was a code bug, caused by missing private(iter)

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
