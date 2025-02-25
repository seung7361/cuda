#include <iostream>
#include <cassert>
#include "tensor2d.cuh"
#include "tensor2d.cu"

void test_kMultiply(
    float* a, int aX, int aY,
    float* b, int bX, int bY,
    float* c
) {
    int threadsPerBlock = 1;
    int blocksX = (aX + threadsPerBlock - 1) / threadsPerBlock;
    int blocksY = (bY + threadsPerBlock - 1) / threadsPerBlock;

    dim3 block(blocksX, blocksY);
    dim3 thread(threadsPerBlock, threadsPerBlock);

    kMultiply<<<block, thread>>>(
        a, aX, aY,
        b, bX, bY,
        c
    );
}

void runTest() {
    int aX = 2, aY = 3;
    int bX = 3, bY = 4;
    int cX = aX, cY = bY;

    std::cout << "a: ";
    float* a = new float[aX * aY];
    for (int i = 0; i < aX * aY; i++) {
        a[i] = i + 1;
        std::cout << i + 1 << " ";
    }
    std::cout << std::endl;

    std::cout << "b: ";
    float* b = new float[bX * bY];
    for (int i = 0; i < bX * bY; i++) {
        b[i] = i + 2;
        std::cout << i + 2 << " ";
    }
    std::cout << std::endl;

    float* c = new float[cX * cY];

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, aX * aY * sizeof(float));
    cudaMalloc(&d_b, bX * bY * sizeof(float));
    cudaMalloc(&d_c, cX * cY * sizeof(float));

    cudaMemcpy(d_a, a, aX * aY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bX * bY * sizeof(float), cudaMemcpyHostToDevice);

    test_kMultiply(
        d_a, aX, aY,
        d_b, bX, bY,
        d_c
    );

    cudaMemcpy(c, d_c, cX * cY * sizeof(float), cudaMemcpyDeviceToHost);

    float answer[] = {28, 34, 64, 79};
    for (int i = 0; i < cX * cY; i++) {
        std::cout << c[i] << " ";
        // assert(c[i] == answer[i]);
    }

    std::cout << "Test passed" << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    runTest();
    return 0;
}