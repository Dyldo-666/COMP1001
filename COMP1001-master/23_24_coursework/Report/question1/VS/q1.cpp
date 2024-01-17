/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP1001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/


#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <omp.h>

#define M 1024*512
#define ARITHMETIC_OPERATIONS1 3*M
#define TIMES1 1

#define N 8192
#define ARITHMETIC_OPERATIONS2 4*N*N
#define TIMES2 1


//function declaration
void initialize();
void routine1(float alpha, float beta);
void routine2(float alpha, float beta);

__declspec(align(64)) float  y[M], z[M] ;
__declspec(align(64)) float A[N][N], x[N], w[N];

int main() {

    float alpha = 0.023f, beta = 0.045f;
    double run_time, start_time;
    unsigned int t;

    initialize();

    printf("\nRoutine1:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES1; t++)
        routine1(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS1) / ((double)run_time / TIMES1));

    printf("\nRoutine2:");
    start_time = omp_get_wtime(); //start timer

    for (t = 0; t < TIMES2; t++)
        routine2(alpha, beta);

    run_time = omp_get_wtime() - start_time; //end timer
    printf("\n Time elapsed is %f secs \n %e FLOPs achieved\n", run_time, (double)(ARITHMETIC_OPERATIONS2) / ((double)run_time / TIMES2));



    return 0;
}

void initialize() {

    unsigned int i, j;

    //initialize routine2 arrays
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (i % 99) + (j % 14) + 0.013f;
        }

    //initialize routine1 arrays
    for (i = 0; i < N; i++) {
        x[i] = (i % 19) - 0.01f;
        w[i] = (i % 5) - 0.002f;
    }

    //initialize routine1 arrays
    for (i = 0; i < M; i++) {
        z[i] = (i % 9) - 0.08f;
        y[i] = (i % 19) + 0.07f;
    }


}




void routine1(float alpha, float beta, float y[], float z[]) {
    const int M = // specify the size of your array;
    const int simdSize = 8; // single-precision floats and AVX2 with 8 floats per vector

    // Ensure that the size of the array is a multiple of the SIMD size
    int numVectors = M / simdSize;

    // Perform the computation using AVX2 intrinsics
    __m256 alphaVec = _mm256_set1_ps(alpha);
    __m256 betaVec = _mm256_set1_ps(beta);

    for (int i = 0; i < numVectors; ++i) {
        // Load data into vectors
        __m256 yVec = _mm256_load_ps(&y[i * simdSize]);
        __m256 zVec = _mm256_load_ps(&z[i * simdSize]);

        // Perform the computation
        yVec = _mm256_add_ps(_mm256_mul_ps(alphaVec, yVec), _mm256_mul_ps(betaVec, zVec));

        // Store the result back to y
        _mm256_store_ps(&y[i * simdSize], yVec);
        cout << "y[" << i << "] = " << y[i] << endl;
    }
    
}

void routine2(float alpha, float beta, float A[N][N],float x[],float w[]) {
    const int N = /* specify the size of your arrays */;
    const int simdSize = 4; //single-precision floats and SSE with 4 floats per vector

    // Ensure that the size of the arrays is a multiple of the SIMD size
    int numVectors = N / simdSize;

    __m128 alphaVec = _mm_set1_ps(alpha);
    __m128 betaVec = _mm_set1_ps(beta);

    for (int i = 0; i < N; ++i) {
        // Load w[i] and x[j] into vectors
        __m128 wVec = _mm_set1_ps(w[i]);

        for (int j = 0; j < numVectors; ++j) {
            __m128 xVec = _mm_load_ps(&x[j * simdSize]);
            __m128 AVec = _mm_load_ps(&A[i][j * simdSize]);

            
            wVec = _mm_sub_ps(_mm_sub_ps(wVec, betaVec),
                _mm_mul_ps(alphaVec, _mm_mul_ps(AVec, xVec)));
        }

        // Store the result back to w
        w[i] = _mm_cvtss_f32(wVec);
        cout << "w[" << i << "] = " << w[i] << endl;
    }
}


