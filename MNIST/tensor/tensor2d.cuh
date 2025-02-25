#pragma once
#ifndef TENSOR2D_HPP
#define TENSOR2D_HPP

#include <iostream>

#include "tensor1d.cuh"

enum Tensor2DAxis {
    X, Y,
};

class Tensor2D {
private:
    int X;
    int Y;
    float* data; // points at the data on device (not host)

public:
    Tensor2D(int X, int Y);
    Tensor2D(int X, int Y, const float** hostData);
    Tensor2D(int X, int Y, const float* data);
    ~Tensor2D();

    int getSize(Tensor2DAxis axis);
    float* getDeviceData();
    float** fetchDataFromDevice();

    void add(Tensor2D* tensor);
    void add(Tensor1D* tensor);
    void subtract(Tensor2D* tensor);
    void scale(float factor);

    Tensor2D* multiply(Tensor2D* tensor, Tensor2D* output);
    Tensor2D* dotProduct(Tensor2D* tensor, Tensor2D* output);
};

#endif