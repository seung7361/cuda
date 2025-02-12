#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void matrixMulKernel(const float *a, const float *b, float *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    // init
    int N = 30;
    int size = N * N;
    size_t bytes = size * sizeof(float);

    int TILE_WIDTH = 1 << 5;

    float *a = (float*)malloc(bytes);
    float *b = (float*)malloc(bytes);
    float *c = (float*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // Host -> Device
    float *d_a, *d_b, *d_c;
    cudaMalloc((float**)&d_a, bytes);
    cudaMalloc((float**)&d_b, bytes);
    cudaMalloc((float**)&d_c, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    clock_t start = clock();
    matrixMulKernel<<<block, grid>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    printf("finished in %ld ms.\n", clock() - start);

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);


    // check
    // printf("A:\n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", a[i * N + j]);
    //     }
    //     printf("\n");
    // }
    // printf("B:\n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", b[i * N + j]);
    //     }
    //     printf("\n");
    // }
    // printf("C:\n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", c[i * N + j]);
    //     }
    //     printf("\n");
    // }

    // free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}