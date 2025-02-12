#include <stdio.h>
#include <math.h>
#include <time.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void verify_result(float* a, float* b, float* c, int n) {
    float max_delta = 0;
    for (int i = 0; i < n; i++) {
        float delta = abs(c[i] - (a[i] + b[i]));
        if (max_delta > delta) max_delta = delta;
        if (delta > 1e-5) {
            printf("Verification failed at %d from %f + %f = %f.\n", i, a[i], b[i], c[i]);
            return;
        }
    }

    printf("Verification successful with max delta of %f\n", max_delta);
}

int main() {
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    srand(time(NULL));
    printf("[Vector addition of %d elements]\n", n);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        printf("Failed to allocate host vectors!\n");
        return 0;
    }

    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // memory allocation on GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc((float**)&d_a, size);
    cudaMalloc((float**)&d_b, size);
    cudaMalloc((float**)&d_c, size);

    // copy from cpu to gpu
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    
    int NUM_THREADS = 1 << 10;
    int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;

    clock_t start, end;
    
    start = clock();
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    end = clock();
    printf("Addition done in %ld ms.\n", end - start);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c, n);

    // free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    printf("Done.\n");
    return 0;
}