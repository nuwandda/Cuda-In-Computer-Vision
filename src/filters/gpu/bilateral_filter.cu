#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <time.h>
#include "Bitmap.h"
#include<stdio.h>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK_SIZE 32
#define SIGMA1 75
#define SIGMA2 75

__device__ const int FILTER_SIZE = 9;
__device__ const int FILTER_HALF_SIZE = FILTER_SIZE >> 1;

__device__ int indexMedian(int x, int y, int width) {
    return (y * width) + x;
}

__device__ int clampMedian(int value, int bound) {
    if (value < 0) {
        return 0;
    }
    if (value < bound) {
        return value;
    }
    return bound - 1;
}

__device__ float exp(int i) {
    return exp((float) i);
}

__global__ void bilateralFilterKernelShared(unsigned char *in, unsigned char *out, int width, int height) {
    // Kernel to calculate bilateral filter. Overwrites the given output image with the new filtered image.
    // Works with gray scaled images.
    //
    // Args:
    //    in (unsigned char): Input image
    //    out (unsigned char): Output image
    //    width (int): Width of the input image
    //    height (int): Height of the input image

    // Set the row and col value for each thread
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    int xBlockLimitMax = blockDim.x - FILTER_HALF_SIZE - 1;
    int yBlockLimitMax = blockDim.y - FILTER_HALF_SIZE - 1;
    int xBlockLimitMin = FILTER_HALF_SIZE;
    int yBlockLimitMin = FILTER_HALF_SIZE;

    // Initialize shared memory
    __shared__ float cache[BLOCK_SIZE + FILTER_SIZE][BLOCK_SIZE + FILTER_SIZE];

    // Load elements to shared memory
    if (threadIdx.x > xBlockLimitMax && threadIdx.y > yBlockLimitMax) {
        int i = indexMedian(clampMedian(col + FILTER_HALF_SIZE, width), clampMedian(row + FILTER_HALF_SIZE, height),
                            width);
        float pixel = in[i];
        cache[threadIdx.x + 2 * FILTER_HALF_SIZE][threadIdx.y + 2 * FILTER_HALF_SIZE] = pixel;
    }
    if (threadIdx.x > xBlockLimitMax && threadIdx.y < yBlockLimitMin) {
        int i = indexMedian(clampMedian(col + FILTER_HALF_SIZE, width), clampMedian(row - FILTER_HALF_SIZE, height),
                            width);
        float pixel = in[i];
        cache[threadIdx.x + 2 * FILTER_HALF_SIZE][threadIdx.y] = pixel;
    }
    if (threadIdx.x < xBlockLimitMin && threadIdx.y > yBlockLimitMax) {
        int i = indexMedian(clampMedian(col - FILTER_HALF_SIZE, width), clampMedian(row + FILTER_HALF_SIZE, height),
                            width);
        float pixel = in[i];
        cache[threadIdx.x][threadIdx.y + 2 * FILTER_HALF_SIZE] = pixel;
    }
    if (threadIdx.x < xBlockLimitMin && threadIdx.y < yBlockLimitMin) {
        int i = indexMedian(clampMedian(col - FILTER_HALF_SIZE, width), clampMedian(row - FILTER_HALF_SIZE, height),
                            width);
        float pixel = in[i];
        cache[threadIdx.x][threadIdx.y] = pixel;
    }
    if (threadIdx.x < xBlockLimitMin) {
        int i = indexMedian(clampMedian(col - FILTER_HALF_SIZE, width), clampMedian(row, height), width);
        float pixel = in[i];
        cache[threadIdx.x][threadIdx.y + FILTER_HALF_SIZE] = pixel;
    }
    if (threadIdx.x > xBlockLimitMax) {
        int i = indexMedian(clampMedian(col + FILTER_HALF_SIZE, width), clampMedian(row, height), width);
        unsigned int pixel = in[i];
        cache[threadIdx.x + 2 * FILTER_HALF_SIZE][threadIdx.y + FILTER_HALF_SIZE] = pixel;
    }
    if (threadIdx.y < yBlockLimitMin) {
        int i = indexMedian(clampMedian(col, width), clampMedian(row - FILTER_HALF_SIZE, height), width);
        float pixel = in[i];
        cache[threadIdx.x + FILTER_HALF_SIZE][threadIdx.y] = pixel;
    }
    if (threadIdx.y > yBlockLimitMax) {
        int i = indexMedian(clampMedian(col, width), clampMedian(row + FILTER_HALF_SIZE, height), width);
        float pixel = in[i];
        cache[threadIdx.x + FILTER_HALF_SIZE][threadIdx.y + 2 * FILTER_HALF_SIZE] = pixel;
    }
    int i = indexMedian(col, row, width);
    float pixel = in[i];
    cache[threadIdx.x + FILTER_HALF_SIZE][threadIdx.y + FILTER_HALF_SIZE] = pixel;

    // Wait for all threads to be done
    __syncthreads();

    //Check boundry conditions
    if ((col < width) && (row < height)) {
        float runningTotal = 0;
        float normFactor = 0;
        const int offset = row * width + col;
        // Apply our filter
        for (int x = 0; x < FILTER_SIZE; x++) {
            for (int y = 0; y < FILTER_SIZE; y++) {
                float intensityChange = cache[threadIdx.x + x][threadIdx.y + y] - cache[threadIdx.x][threadIdx.y];
                float w1 = exp(-((x - FILTER_HALF_SIZE) * (x - FILTER_HALF_SIZE) +
                                 (y - FILTER_HALF_SIZE) * (y - FILTER_HALF_SIZE)) /
                               (2 * SIGMA1 * SIGMA1));
                float w2 = exp(-(intensityChange * intensityChange) / (2 * SIGMA2 * SIGMA2));
                runningTotal += cache[threadIdx.x + x][threadIdx.y + y] * w1 * w2;
                normFactor += w1 * w2;
            }
        }
        // Set the output image values
        out[offset] = runningTotal / normFactor;
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
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((int) ceil((float) image->Width() / (float) BLOCK_SIZE),
                 (int) ceil((float) image->Height() / (float) BLOCK_SIZE));

    // Check for shared memories and call the kernel
    cudaEventRecord(start);
    bilateralFilterKernelShared << < dimGrid, dimBlock >> > (deviceInputImage, deviceOutputImage, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Save output image to host.
    cudaMemcpy(outputImage->image, deviceOutputImage, size, cudaMemcpyDeviceToHost);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cout << "Kernel failed for cudaMemcpy cudaMemcpyDeviceToHost: " << cudaGetErrorString(status) << endl;
        cudaFree(deviceInputImage);
        cudaFree(deviceOutputImage);
        return false;
    }

    //Free the memory
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);

    // Calculate time
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("Execution time:\n");
    printf("\tBilateral filter execution time: %f ms\n", time);

    // Save image
    outputImage->Save("lena_shared.bmp");

    return true;
}