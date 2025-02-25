#include "tensor2d.cuh"
#include "assert.h"

__global__ void kAdd1D(float* a, float* b, int X, int Y) {
    // a: (X, Y)
    // b: (X,)

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < X && y < Y) {
        a[x * X + y] += b[x];
    }
}

__global__ void kAdd2D(float* a, float* b, int X, int Y) {
    // a: (X, Y)
    // b: (X, Y)

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < X && y < Y) {
        a[x * X + y] += b[x * X + y];
    }
}

__global__ void kSubtract1D(float* a, float* b, int X, int Y) {
    // a: (X, Y)
    // b: (X,)

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < X && y < Y) {
        a[x * X + y] -= b[x];
    }
}

__global__ void kSubtract2D(float* a, float* b, int X, int Y) {
    // a: (X, Y)
    // b: (X, Y)

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < X && y < Y) {
        a[x * X + y] -= b[x * X + y];
    }
}

__global__ void kScale1D(float* a, float factor, int X, int Y) {
    // a: (X, Y)

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < X && y < Y) {
        a[x * X + y] *= factor;
    }
}

__global__ void kMultiply(
    float* a, int aX, int aY,
    float* b, int bX, int bY,
    float* c
) {
    // a: (aX, aY)
    // b: (bX, bY)
    // c: (aX, bY)

    assert(aY == bX);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < aX && y < bY) {
        float sum = 0;
        for (int i = 0; i < aY; i++) {
            sum += a[x * aX + i] * b[i * bX + y];
        }
        c[x * aX + y] = sum;
    }
}