#include "tensor1d.cuh"

__global__ void kAdd(const float* a, const float* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] += b[i];
    }
}

__global__ void kSubtract(const float* a, const float* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] -= b[i];
    }
}

__global__ void kScale(const float* a, float factor, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] *= factor;
    }
}

Tensor1D::Tensor1D(int size) {
    this->size = size;
    if (size) {
        cudaMalloc((void**)&(this->data), this->size * sizeof(float));
    } else {
        this->data = NULL;
    }
}

Tensor1D::Tensor1D(int size, float* hostData) {
    this->size = size;
    if (size) {
        cudaMalloc((void**)&(this->data), this->size * sizeof(float));
        cudaMemcpy(this->data, hostData, cudaMemcpyHostToDevice);
    } else {
        this->data = NULL;
    }
}

Tensor1D::~Tensor1D() {
    cudaFree(this->data);
}

int Tensor1D::getSize() {
    return this->size;
}

float* Tensor1D::getDeviceData() {
    return this->data;
}

float* Tensor1D::fetchDataFromDevice() {
    float* hostData = (float*)malloc(this->size * sizeof(float));
    cudaDeviceSynchronize();
    cudaMemcpy(hostData, this->data, this->size * sizeof(float(, cudaMemcpyDeviceToHost)));

    return hostData;
}

void Tensor1D::add(Tensor1D* tensor) {
    kAdd<<<1, this->size>>>(this->getDeviceData(), tensor->getDeviceData(), this->size);
}

void Tensor1D::subract(Tensor1D* tensor) {
    kSubtract<<<1, this->size>>>(this->getDeviceData(), tensor->getDeviceData(), this->size);
}

void Tensor1D::scale(float factor) {
    kScale<<<1, this->size>>>(this->getDeviceData(), factor, this->size);
}