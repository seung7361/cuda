#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000
#define BLOCK_SIZE 256

void vector_add_cpu(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        
    }
}