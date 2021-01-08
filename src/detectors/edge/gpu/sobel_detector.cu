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

__global__ void sobelDetect(unsigned char *in, unsigned char *out, int width, int height) {
    // Kernel to detect edges of the given image. Threads travel on each pixel and use Sobel technique to find edges.
    // If a pixel is an edge then it returns a higher value.
    // Works with gray scaled images.
    //
    // Args:
    //    in (unsigned char): Input image
    //    out (unsigned char): Output image
    //    width (int): Width of the input image
    //    height (int): Height of the input image

    // Set the row and col value for each thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Define derivates
    float dx;
    float dy;

    // Check boundary conditions
    if ((col >= 0 && col < width) && (row >= 0 && row < height)) {
        // Find derivatives
        dx = -in[width * (row - 1) + (col - 1)] + in[width * (row - 1) + (col + 1)]
             - 2 * in[width * row + (col - 1)] + 2 * in[width * (row) + (col + 1)]
             - in[width * (row + 1) + (col - 1)] + in[width * (row + 1) + (col + 1)];

        dy = in[width * (row - 1) + (col - 1)] + 2 * in[width * (row - 1) + (col + 1)] +
             in[width * (row - 1) + (col + 1)]
             - in[width * (row + 1) + (col - 1)] - 2 * in[width * (row + 1) + col] - in[width * (row + 1) + (col + 1)];

        // Get distance
        int value = (int) sqrt(dx * dx + dy * dy);
        // Set the output image values
        out[row * width + col] = (unsigned char) value;
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
    image->Load("lena.bmp");
    outputImage->Load("lena.bmp");

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
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid((int) ceil((float) image->Width() / (float) BLOCK_SIZE),
                 (int) ceil((float) image->Height() / (float) BLOCK_SIZE), 1);

    // Check for shared memories and call the kernel
    cudaEventRecord(start);
    sobelDetect << < dimGrid, dimBlock >> > (deviceInputImage, deviceOutputImage, width, height);
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
    printf("\tSobel edge detection execution time: %f ms\n", time);

    // Save image
    outputImage->Save("lena_shared.bmp");

    return true;
}