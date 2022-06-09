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
./laplace2d_c_acc
     512   3.970    2000    0.001294592046179    0.033113196492195
    1000   7.800    2000    0.000663466518745    0.017047470435500
./laplace2d_c_acc_managed
     512   3.310    2000    0.001294592046179    0.033113196492195
    1000   8.440    2000    0.000663466518745    0.017047470435500
./laplace2d_cl
     512   0.550    2000    0.001294593093917    0.033108580857515
    1000   2.200    2000    0.000663466402330    0.017042662948370

 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char** argv)
{
    int n = 1000;
    int m = 1000;
    int iter_max = 2000;
    
    const float pi  = 2.0f * asinf(1.0f);
    const float tol = 1.0e-4f;
    float error     = 1.0f;
    
    float A[n][m];
    float Anew[n][m];
    float y0[n];

    //printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);

    /*
    for(int nthread = 1; nthread<9; nthread++){
      omp_set_dynamic(0);
      omp_set_num_threads(nthread);
      printf ("Inner: num_thds=%d\n", omp_get_num_threads());
    */
    memset(A, 0, n * m * sizeof(float));
    
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
    int msec = 0; 
    clock_t before = clock();

#pragma omp parallel for shared(Anew)
    for (int i = 1; i < m; i++)
    {
       Anew[0][i]   = 0.f;
       Anew[n-1][i] = 0.f;
    }
#pragma omp parallel for shared(Anew)    
    for (int j = 1; j < n; j++)
    {
        Anew[j][0]   = y0[j];
        Anew[j][m-1] = y0[j]*expf(-pi);
    }
    
    int iter = 0;
    //   while ( error > tol)
    while (iter < iter_max)
   {
        error = 0.f;
#pragma omp parallel for shared(m, n, Anew, A)
#pragma acc kernels
        for( int j = 1; j < n-1; j++){
            for( int i = 1; i < m-1; i++ ){
                Anew[j][i] = 0.25f * ( A[j][i+1] + A[j][i-1]
                                     + A[j-1][i] + A[j+1][i]);
                error = fmaxf( error, fabsf(Anew[j][i]-A[j][i]));
            }
        }
        
#pragma omp parallel for shared(m, n, Anew, A)
#pragma acc kernels
        for( int j = 1; j < n-1; j++){
            for( int i = 1; i < m-1; i++ ){
                A[j][i] = Anew[j][i];    
            }
        }
    //    if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);  
        iter++;
    }

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
    printf("    size  time(s) iterations initial_sum          final_sum \n");
    printf("%8d%8.3f%8d%21.15f%21.15f\n",  n, runtime,iter,sum0/n/m,sumA/n/m); 

    //} //end omp for

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
