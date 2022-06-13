/*laplace2d is a simple benchamrk test for openMP and openACC*/
/*To compile
  pgcc -I../common -O2 -o laplace2d_c laplace2d.c
  pgcc -I../common -O2 -acc -o laplace2d_c_acc laplace2d.c
  pgcc -I../common -O2 -acc -ta=nvidia:managed,time -Minfo=accel -o laplace2d_c_acc_managed laplace2d.c

  clang -O2 -o laplace2d_cl   laplace2d.c -lm
*/ 
/*some results
    size  time(s) iterations initial_sum          final_sum 
./laplace2d_c
     512   0.350    2000    0.001294592279010    0.033113095909357
    1000   1.130    2000    0.000663465936668    0.017047340050340
    1024   1.310    2000    0.000647931185085    0.016649417579174
./laplace2d_c_acc
     512   3.970    2000    0.001294592046179    0.033113196492195
    1000   7.800    2000    0.000663466518745    0.017047470435500
    1024   8.680    2000    0.000647930952255    0.016649551689625
./laplace2d_c_acc_managed
     512   3.310    2000    0.001294592046179    0.033113196492195
    1000   8.440    2000    0.000663466518745    0.017047470435500
    1024   8.840    2000    0.000647930952255    0.016649551689625

clang -O2 -o laplace2d_cl   laplace2d.c -lm; ./laplace2d_cl
     512   0.550    2000    0.001294593093917    0.033108580857515
    1000   2.200    2000    0.000663466402330    0.017042662948370
    1024   2.490    2000    0.000647931243293    0.016644719988108
my llvm/clang -O2
    1024   6.060    2000    0.000647931243293    0.016644719988108
    1024   6.060    2000    0.000647931243293    0.016644719988108
my llvm/clang -O3
    1024   6.060    2000    0.000647931243293    0.016644719988108
my llvm/clang -O1
    1024   6.870    2000    0.000647931243293    0.016644719988108
my llvm/clang -O0
    1024  23.740    2000    0.000647931243293    0.016644719988108
/home/Niki.Zadeh/llvm-project/install/bin/clang laplace2d.c -o laplace2d_mycl_omp -L/opt/gcc/11.3.0/lib64  -lm -O3 -fopenmp ; ./laplace2d_mycl_omp 
    1024 334.450    2000    0.000647931243293    0.016644719988108


gcc -O2 -o laplace2d_gcc laplace2d.c -lm ; ./laplace2d_gcc
    1024   3.580    2000    0.000647931243293    0.016644719988108

 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

int main(int argc, char** argv)
{
    int n = 1000;
    int m = 1000;
    int iter_max = 2000;
    
    const float pi  = 2.0f * asinf(1.0f);
    //const float tol = 1.0e-4f;
    float error     = 1.0f;
    
    float A[n][m];
    float Anew[n][m];
    float y0[n];

    //printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);

    for(int nthread = 1; nthread<9; nthread++){
      omp_set_dynamic(0);
      omp_set_num_threads(nthread);

    memset(A, 0, n * m * sizeof(float));
    memset(Anew, 0, n * m * sizeof(float));
    
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
	}
    }
    //printf("debug:  sum(A)=%21.15f \n",  sum0/n/m);
    
    /*
    Start timing
    */
    clock_t before = clock();
#pragma omp parallel
#pragma omp for  
    for (int i = 1; i < m; i++){
       Anew[0][i]   = 0.f;
       Anew[n-1][i] = 0.f;
    }
#pragma omp for  
    for (int j = 1; j < n; j++){
        Anew[j][0]   = y0[j];
        Anew[j][m-1] = y0[j]*expf(-pi);
    }
        
    int iter = 0;
    //   while ( error > tol)
    while (iter < iter_max)
   {
#pragma acc kernels present(A,Anew) //Tell compiler to reuse A,Anew on the device
        error = 0.f;
#pragma omp for
        for( int j = 1; j < n-1; j++){
            for( int i = 1; i < m-1; i++ ){
                Anew[j][i] = 0.25f * ( A[j][i+1] + A[j][i-1]
                                     + A[j-1][i] + A[j+1][i]);
                error = fmaxf( error, fabsf(Anew[j][i]-A[j][i]));
            }
        }
        
#pragma omp for
        for( int j = 1; j < n-1; j++){
            for( int i = 1; i < m-1; i++ ){
                A[j][i] = Anew[j][i];    
            }
        }
#pragma acc kernels end
        iter++;
    }
#pragma acc update self(A)
#pragma acc end data

    clock_t difference = clock() - before;
    float runtime = difference * 1000 / CLOCKS_PER_SEC/1000.f;

    float sumA=0.0;
    for( int j = 0; j < n; j++){
	for( int i = 0; i < m; i++ ){
	    sumA += A[j][i];    
	}
    }
    
    //printf(" total: %f s\n", runtime / 1000.f);
    //printf(" completed in %8.3f seconds, in %5d iterations, sum(A)=%21.15f \n", runtime / 1000.f, iter, sumA/n/m);
    printf("    size  time(s) iterations initial_sum          final_sum    , nthreads=%4d \n",nthread);
    printf("%8d%8.3f%8d%21.15f%21.15f\n",  n, runtime,iter,sum0/n/m,sumA/n/m); 

    } //end omp for

}


/* Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
