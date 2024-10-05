#include <stdio.h>

__global__
void whoami() {
    printf("block: (%d, %d, %d), thread: (%d, %d, %d)\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    dim3 blocks = dim3(2, 2, 2);
    dim3 threads = dim3(4, 4, 4);

    whoami<<<blocks, threads>>>();
    cudaDeviceSynchronize();
}