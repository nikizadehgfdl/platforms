/*benchmark1.c is a simple benchamrk test for openMP and openACC*/
/*To compile
clang benchmark1.c -o benchmark1 -L/opt/gcc/11.3.0/lib64  -lm -O3 -fopenmp  -fopenmp-targets=nvptx64  ; time ./benchmark1
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

void transform_1darray(int nthread, int iter_max, int n,float A[])
{ 
  int iter;
  for( int j = 0; j < n; j++){
      iter=0;
      while (iter < iter_max){
	A[j] = A[j]*(A[j]-1.0);
	iter += 1;
      }
    }
}
void transform_1darray_omp_gpu(int nthread, int iter_max, int n,float A[])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
#pragma omp target parallel for map(tofrom: A[0:n],iter) 
  for( int j = 0; j < n; j++){
      iter=0;
      while (iter < iter_max){
	A[j] = A[j]*(A[j]-1.0);
	iter += 1;
      }
    }
}
void transform_1darray_omp_gpu_teams(int nthread, int iter_max, int n,float A[])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);

#pragma omp target teams distribute parallel for map(tofrom: A[0:n],iter) 
  for( int j = 0; j < n; j++){
      iter=0;
      while (iter < iter_max){
	A[j] = A[j]*(A[j]-1.0);
	iter += 1;
      }
    }
}
void transform_1darray_omp_teams(int nthread, int iter_max, int n,float A[])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
  /*The following is an error with nvcc , but not with clang
NVC++-S-0155-TEAMS construct must be contained within TARGET construct  (benchmark1.c: 55)
NVC++-S-0155-TARGET construct can contain only one TEAMS construct, must contain no statements, declarations or directives outside of the TEAMS construct.  (benchmark1.c: 55)
  */
  //#pragma omp teams distribute parallel for 
  for( int j = 0; j < n; j++){
      iter=0;
      while (iter < iter_max){
	A[j] = A[j]*(A[j]-1.0);
	iter += 1;
      }
    }
}
void transform_1darray_omp(int nthread, int iter_max, int n,float A[])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
#pragma omp parallel for 
  for( int j = 0; j < n; j++){
      iter=0;
      while (iter < iter_max){
	A[j] = A[j]*(A[j]-1.0);
	iter += 1;
      }
    }
}

void transform0_1darray_omp_gpu_teams(int nthread, int iter_max, int n,float x[],float y[])
{ 
  int iter;
#pragma omp target teams distribute parallel for map(to: x[0:n]) map(from: y[0:n])
  for( int j = 0; j < n; j++){
      iter=0;
      x[j]=j;
      y[j]=0;
      while (iter < iter_max){
	y[j] += 3*x[j]/n;
	iter += 1;
      }
    }
}
void transform0_1darray_omp_gpu(int nthread, int iter_max, int n,float x[],float y[])
{ 
  int iter;
#pragma omp target parallel for map(to: x[0:n]) map(from: y[0:n])
  for( int j = 0; j < n; j++){
      iter=0;
      x[j]=j;
      y[j]=0;
      while (iter < iter_max){
	y[j] += 3*x[j]/n;
	iter += 1;
      }
    }
}

void transform0_1darray_omp(int nthread, int iter_max, int n,float x[],float y[])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
#pragma omp parallel for
  for( int j = 0; j < n; j++){
      iter=0;
      x[j]=j;
      y[j]=0;
      while (iter < iter_max){
	y[j] += 3*x[j]/n;
	iter += 1;
      }
    }
}
void transform0_1darray_omp_teams(int nthread, int iter_max, int n,float x[],float y[])
{ 
  int iter;
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);
  /*The following is an error with nvcc , but not with clang
NVC++-S-0155-TEAMS construct must be contained within TARGET construct  (benchmark1.c: 129)
NVC++-S-0155-TARGET construct can contain only one TEAMS construct, must contain no statements, declarations or directives outside of the TEAMS construct. 
  */
  //#pragma omp teams distribute parallel for
  for( int j = 0; j < n; j++){
      iter=0;
      x[j]=j;
      y[j]=0;
      while (iter < iter_max){
	y[j] += 3*x[j]/n;
	iter += 1;
      }
    }
}
void transform0_1darray(int nthread, int iter_max, int n,float x[],float y[])
{ 
  int iter;
  for( int j = 0; j < n; j++){
      iter=0;
      x[j]=j;
      y[j]=0;
      while (iter < iter_max){
	y[j] += 3*x[j]/n;
	iter += 1;
      }
    }
}

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

void ave4pt_2darray(int nthread, int iter_max, int m, int n,float A[][m],float B[][m])
{ 
  int iter=0;  
  while (iter < iter_max){
    for( int j = 1; j < n-1; j++){for( int i = 1; i < m-1; i++ ){
	B[j][i] = 0.25f*(A[j][i+1]+A[j][i-1]+A[j-1][i]+A[j+1][i]);}}
    for( int j = 1; j < n-1; j++){for( int i = 1; i < m-1; i++ ){
	A[j][i] = B[j][i];}}
    iter++;
  }
}
void ave4pt_2darray_omp_gpu(int nthread, int iter_max, int m, int n,float A[][m],float B[][m])
{ 
  int iter=0;  
  //#pragma omp target teams map(tofrom: A[0:n][0:m],B[0:n][0:m])//does not transfer data properly
#pragma omp target data map(to: A[0:n][0:m],B[0:n][0:m]) map(from: A[0:n][0:m])
{
  iter=0;
  while (iter < iter_max){
#pragma omp target teams distribute parallel for 
    for( int j = 1; j < n-1; j++){
      for( int i = 1; i < m-1; i++ ){
	B[j][i] = 0.25f*(A[j][i+1]+A[j][i-1]+A[j-1][i]+A[j+1][i]);
      }}

    //#pragma omp target teams distribute parallel for map(tofrom: A[0:n][0:m],B[0:n][0:m])
    for( int j = 1; j < n-1; j++){for( int i = 1; i < m-1; i++ ){
	A[j][i] = B[j][i];}}
    
    iter++;
  }
}  
}


int main(int argc, char** argv)
{
    int n = 1000;
    int m = 1000;
    int iter_max = 2000;
    const float pi  = 2.0f * asinf(1.0f);
    double runtime; 
    float A[n][m];
    float B[n][m];
    float y0[n];
    float A1d[n*m];
    int Nl=n*m;
    float x[Nl];
    float y[Nl];
    
    printf("     size     time(s) iterations initial_sum          final_sum        omp_nthreads\n");
    printf("2D arrays\n");
    for(int nthread = -3; nthread<4; nthread++){
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
      if(nthread == -1) transform_2darray_omp_teams(nthread,iter_max,m,n,A);
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

    /*
    1d tests
    */
    printf("Equivalent 1D arrays\n");
    for(int nthread = -3; nthread<3; nthread++){
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
      if(nthread > 0  ) transform_1darray_omp(nthread,iter_max,m*n,A1d);
      if(nthread ==  0) transform_1darray(nthread,iter_max,m*n,A1d);
      if(nthread == -1) transform_1darray_omp_teams(nthread,iter_max,m*n,A1d);
      if(nthread == -2) transform_1darray_omp_gpu(nthread,iter_max,m*n,A1d);
      if(nthread == -3) transform_1darray_omp_gpu_teams(nthread,iter_max,m*n,A1d);
      runtime = omp_get_wtime() - runtime;//End timer

      float sumA1=0.0;
      for( int j = 0; j < n*m; j++){
	  sumA1 += A1d[j];    
      }
      printf("%12d%8.3f%8d%21.15f%21.15f%5d\n",  n*m, runtime,iter_max,sum0/n/m,sumA1/n/m,nthread); 
    } //end omp for

     printf("More 1D array tests\n");
    printf("    size  time(s) iterations   final_sum        omp_nthreads\n");
    for(int nthread = -3; nthread<3; nthread++){
      runtime = omp_get_wtime();//Start timer
      //Calculate
      if(nthread > 0  ) transform0_1darray_omp(nthread,iter_max,m*n,x,y);
      if(nthread ==  0) transform0_1darray(nthread,iter_max,m*n,x,y);
      if(nthread == -1) transform0_1darray_omp_teams(nthread,iter_max,m*n,x,y);
      if(nthread == -2) transform0_1darray_omp_gpu(nthread,iter_max,m*n,x,y);
      if(nthread == -3) transform0_1darray_omp_gpu_teams(nthread,iter_max,m*n,x,y);
      runtime = omp_get_wtime() - runtime;//End timer
      float sum=0.0;
      for(int i = 0; i < n*m; i++){
	sum += y[i];    
      }
      printf("%12d%8.3f%8d%21.15f%5d\n",  n*m, runtime,iter_max,sum/n/m,nthread); 
    }   
    //2D arrays, iteration outside ij loop
    printf("     size     time(s) iterations initial_sum          final_sum        omp_nthreads\n");
    printf("2D arrays, iteration outside ij loop\n");
    for(int nthread = -1; nthread<1; nthread++){
      memset(A,   0, n * m * sizeof(float));
      memset(B,   0, n * m * sizeof(float));
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

      for (int i = 1; i < m; i++){  //Why i=1?
	B[0][i]   = 0.f;
	B[n-1][i] = 0.f;
      }    
      for (int j = 1; j < n; j++){ //Why j=1?
        B[j][0]   = y0[j];
        B[j][m-1] = y0[j]*expf(-pi);
      }  
        
      float sum0=0.0;
      for( int j = 0; j < n; j++){
	for( int i = 0; i < m; i++ ){
	  A1d[j*m+i] = A[j][i];
	  sum0 += A[j][i];
	}}
    
      runtime = omp_get_wtime();//Start timer
      //Calculate
      //if(nthread > 0  ) transform_2darray_omp(nthread,iter_max,m,n,A);
      if(nthread ==  0) ave4pt_2darray(nthread,iter_max,m,n,A,B);
      //if(nthread == -1) transform_2darray_omp_teams(nthread,iter_max,m,n,A);
      //if(nthread == -2) transform_2darray_omp_gpu(nthread,iter_max,m,n,A);
      if(nthread == -1) ave4pt_2darray_omp_gpu(nthread,iter_max,m,n,A,B);
      runtime = omp_get_wtime() - runtime;//End timer

      float sumB=0.0;
      for( int j = 0; j < n; j++){
	for( int i = 0; i < m; i++ ){
	  sumB += B[j][i];    
	}}
      printf("%12d%8.3f%8d%21.15f%21.15f%5d\n",  n*m, runtime,iter_max,sum0/n/m,sumB/n/m,nthread); 
    } //end omp for

}

/*Some results
pgcc -O3 -mp -Mpreprocess -fast -ta=tesla,cuda11.7,cc60 -o benchmark1_pgcc benchmark1.c ; ./benchmark1_pgcc
     size     time(s) iterations initial_sum          final_sum        omp_nthreads
2D arrays
     1000000  12.461    2000    0.000663466169499    0.000026498426450   -3
     1000000  11.340    2000    0.000663466169499    0.000026498426450   -2
     1000000   7.852    2000    0.000663466169499    0.000026498426450   -1
     1000000   7.909    2000    0.000663466169499    0.000026498426450    0
     1000000  11.234    2000    0.000663466169499    0.000026498426450    1
     1000000  19.082    2000    0.000663466169499    0.000007004406598    2
     1000000  18.727    2000    0.000663466169499    0.000004531497325    3
Equivalent 1D arrays
     1000000  11.082    2000    0.000663466169499    0.000026498426450   -3
     1000000  11.124    2000    0.000663466169499    0.000026498426450   -2
     1000000   7.587    2000    0.000663466169499    0.000026498426450   -1
     1000000   7.621    2000    0.000663466169499    0.000026498426450    0
     1000000  10.897    2000    0.000663466169499    0.000026498426450    1
     1000000  10.874    2000    0.000663466169499    0.000002505697466    2


Niki.Zadeh: ~/platforms/samples/gpu/openacc/step1 $ \rm benchmark1; clang benchmark1.c -o benchmark1 -L/opt/gcc/11.3.0/lib64  -lm -O3 -fopenmp  -fopenmp-targets=nvptx64; ./benchmark1
clang-14: warning: CUDA version is newer than the latest supported version 11.5 [-Wunknown-cuda-version]
2D arrays
     1000000   0.074    2000    0.000663466402330    0.000026498402804   -3
     1000000   0.568    2000    0.000663466402330    0.000026498402804   -2
     1000000   7.753    2000    0.000663466402330    0.000026498402804   -1
     1000000   7.944    2000    0.000663466402330    0.000026498402804    0
     1000000   9.432    2000    0.000663466402330    0.000026498402804    1
     1000000   8.820    2000    0.000663466402330    0.000026498402804    2
Equivalent 1D arrays
     1000000   0.005    2000    0.000663466402330    0.000026498402804   -3
     1000000   0.578    2000    0.000663466402330    0.000026498402804   -2
     1000000   8.669    2000    0.000663466402330    0.000026498402804   -1
     1000000   7.824    2000    0.000663466402330    0.000026498402804    0
     1000000   7.934    2000    0.000663466402330    0.000026498402804    1
     1000000   6.436    2000    0.000663466402330    0.000026498402804    2


    size  time(s) iterations initial_sum          final_sum        omp_nthreads
2D arrays
    16000000   0.443    2000    0.000165991194081    0.000006629720247   -3   gpu with teams
    16000000   9.092    2000    0.000165991194081    0.000006629720247   -2   gpu 
    16000000 121.416    2000    0.000165991194081    0.000006629720247   -1   omp teams
    16000000 121.131    2000    0.000165991194081    0.000006629720247    0   no omp 
    16000000 121.262    2000    0.000165991194081    0.000006629720247    1   omp 1 thread
    16000000  67.972    2000    0.000165991194081    0.000006629720247    2   omp 2 threads
Equivalent 1D arrays
    16000000   0.053    2000    0.000165991194081    0.000006629720247   -3
    16000000   8.909    2000    0.000165991194081    0.000006629720247   -2
    16000000 121.268    2000    0.000165991194081    0.000006629720247   -1
    16000000 121.065    2000    0.000165991194081    0.000006629720247    0
    16000000 121.200    2000    0.000165991194081    0.000006629720247    1
    16000000  71.695    2000    0.000165991194081    0.000006629720247    2
More 1D array tests
    size  time(s) iterations   final_sum        omp_nthreads
    16000000   0.128    2000 2730.884033203125000   -3
    16000000  17.540    2000 2730.884033203125000   -2
    16000000  60.891    2000 2730.884033203125000   -1
    16000000  25.561    2000 2730.884033203125000    0
    16000000  60.869    2000 2730.884033203125000    1
    16000000 213.520    2000 2730.884033203125000    2

============================================================================

2D arrays
     4000000   0.289    2000    0.000331899296725    0.000013256050806   -3
     4000000   2.277    2000    0.000331899296725    0.000013256050806   -2
     4000000  30.422    2000    0.000331899296725    0.000013256050806   -1
     4000000  30.284    2000    0.000331899296725    0.000013256050806    0
     4000000  30.299    2000    0.000331899296725    0.000013256050806    1
     4000000  29.041    2000    0.000331899296725    0.000013256050806    2
1D arrays
     4000000   0.014    2000    0.000331899296725    0.000013256050806   -3
     4000000   2.243    2000    0.000331899296725    0.000013256050806   -2
     4000000  30.316    2000    0.000331899296725    0.000013256050806   -1
     4000000  30.284    2000    0.000331899296725    0.000013256050806    0
     4000000  30.317    2000    0.000331899296725    0.000013256050806    1
     4000000  34.535    2000    0.000331899296725    0.000013256050806    2
More 1D array tests
    size  time(s) iterations   final_sum        omp_nthreads
     4000000   0.029    2000 2998.254638671875000   -3
     4000000   4.390    2000 2998.254638671875000   -2
     4000000  15.399    2000 2998.254638671875000   -1
     4000000   6.387    2000 2998.254638671875000    0
     4000000  15.398    2000 2998.254638671875000    1
     4000000  40.577    2000 2998.254638671875000    2

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
