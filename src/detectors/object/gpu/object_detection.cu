#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <time.h>
#include "Bitmap.h"
#include<stdio.h>
#include <cuda_runtime.h>
#include <math.h>

using namespace std;

#define BLOCK_SIZE 32

__device__ float getMinimum(float a, float b) {
    return ((a) < (b) ? (a) : (b));
}

__device__ float getMaximum(float a, float b) {
    return ((a) > (b) ? (a) : (b));
}

__device__ float getMinimumOfThree(float a, float b, float c) {
    return getMinimum((a), getMinimum((b), (c)));
}

__device__ float getMaximumOfThree(float a, float b, float c) {
    return getMaximum((a), getMaximum((b), (c)));
}

__global__ void
coloredObjectDetection(unsigned char *in, unsigned char *out, int width, int height, bool replaceBackground,
                       int treshMin, int treshMax, int foreground, int background) {
    // Kernel to detect colored object in the given image. We first convert RGB image to HSV image.
    // Then, if the pixels are between given threshold, we subtrack backgroun.
    //
    // Args:
    //    in (unsigned char): Input image
    //    out (unsigned char): Output image
    //    width (int): Width of the input image
    //    height (int): Height of the input image
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * width + col * 3;

    // RGB to HSV conversion
    // Check boundary conditions
    if ((row < height) && (col < width / 3)) {

        // Given image is in BGR format
        float blue = in[index] / 255.0;
        float green = in[index + 1] / 255.0;
        float red = in[index + 2] / 255.0;

        float maximum = getMaximumOfThree(red, green, blue);
        float minimum = getMinimumOfThree(red, green, blue);

        float delta = maximum - minimum;

        // Calculate hue
        float hue;
        if (red == maximum) { hue = 60 * (green - blue) / delta; }
        if (green == maximum) { hue = 60 * (blue - red) / delta + 2; }
        if (blue == maximum) { hue = 60 * (red - green) / delta + 4; }
        if (hue < 0) { hue += 360; }
        out[index] = (unsigned char) (hue / 2);

        // Calculate saturation
        float saturation = (delta / maximum) * 255.0;
        out[index + 1] = (unsigned char) saturation;

        // Calculate value
        double value = maximum * 255.0;
        out[index + 2] = (unsigned char) value;


        // If the hue is between the given range
        if (treshMin < out[index] && out[index] < treshMax) {
            if (replaceBackground) {
                out[index] = foreground;
                out[index + 1] = foreground;
                out[index + 2] = foreground;
            } else {
                out[index] = out[index];
                out[index + 1] = out[index + 1];
                out[index + 2] = out[index + 2];
            }
        } else {
            out[index] = background;
            out[index + 1] = background;
            out[index + 2] = background;
        }
    }
}

int main() {
    // System specifications
    printf("-->\n");
    printf("System Specifications:\n");
    printf("\tAzure NC6\n");
    printf("\tCores: 6\n");
    printf("\tGPU: Tesla K80\n");
    printf("\tMemory: 56 GB\n");
    printf("\tDisk: 380 GB SSD\n");
    printf("-->\n");

    // Create input and output images
    Bitmap *image = new Bitmap();
    Bitmap *outputImage = new Bitmap();

    // Load both images with the same picture
    image->Load("red_ball.bmp");
    outputImage->Load("red_ball.bmp");

    // Cuda error and image values
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t status;
    int width = image->Width();
    int height = image->Height();

    // Allocate and copy images
    int size = width * height * sizeof(char);
    unsigned char *deviceInputImage;
    cudaMalloc((void **) &deviceInputImage, size);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cout << "Kernel failed for cudaMalloc : " << cudaGetErrorString(status) << endl;
        return false;
    }

    cudaMemcpy(deviceInputImage, image->image, size, cudaMemcpyHostToDevice);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cout << "Kernel failed for cudaMemcpy cudaMemcpyHostToDevice: " << cudaGetErrorString(status) << endl;
        cudaFree(deviceInputImage);
        return false;
    }

    unsigned char *deviceOutputImage;
    cudaMalloc((void **) &deviceOutputImage, size);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cout << "Kernel failed for cudaMalloc : " << cudaGetErrorString(status) << endl;
        return false;
    }

    // Check for shared memories and call the kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((int) ceil((float) image->Width() / (float) BLOCK_SIZE),
                 (int) ceil((float) image->Height() / (float) BLOCK_SIZE));

    // Check for shared memories and call the kernel
    cudaEventRecord(start);
    coloredObjectDetection << < dimGrid, dimBlock >> >
                                         (deviceInputImage, deviceOutputImage, width, height, false, 0, 170, 255, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Save output image to host
    cudaMemcpy(outputImage->image, deviceOutputImage, size, cudaMemcpyDeviceToHost);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cout << "Kernel failed for cudaMemcpy cudaMemcpyDeviceToHost: " << cudaGetErrorString(status) << endl;
        cudaFree(deviceInputImage);
        cudaFree(deviceOutputImage);
        return false;
    }
    // Free the memory
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);

    // Calculate time
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution time:\n");
    printf("\tObject detection execution time: %f ms\n", time);

    // Save image
    outputImage->Save("red_ball_shared.bmp");

    return true;
}
