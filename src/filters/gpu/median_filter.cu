#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <time.h>
#include "Bitmap.h"
#include<stdio.h>

using namespace std;

#define TILE_SIZE 4

__global__ void medianFilterKernelShared(unsigned char *in, unsigned char *out, int width, int height) {
    // Kernel to calculate median filter. Overwrites the given output image with the new filtered image.
    // Works with gray scaled images.
    //
    // Args:
    //    in (unsigned char): Input image
    //    out (unsigned char): Output image
    //    width (int): Width of the input image
    //    height (int): Height of the input image

    // Set the row and col value for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize shared memory
    __shared__ unsigned char cache[(TILE_SIZE + 2)][(TILE_SIZE + 2)];
    // Take some values
    bool isXLeft = (threadIdx.x == 0);
    bool isXRight = (threadIdx.x == TILE_SIZE - 1);
    bool isYTop = (threadIdx.y == 0);
    bool isYBottom = (threadIdx.y == TILE_SIZE - 1);

    // Initialize cache with zero
    if (isXLeft) {
        cache[threadIdx.x][threadIdx.y + 1] = 0;
    } else if (isXRight) {
        cache[threadIdx.x + 2][threadIdx.y + 1] = 0;
    }

    if (isYTop) {
        cache[threadIdx.x + 1][threadIdx.y] = 0;
        if (isXLeft) {
            cache[threadIdx.x][threadIdx.y] = 0;
        } else if (isXRight) {
            cache[threadIdx.x + 2][threadIdx.y] = 0;
        }
    } else if (isYBottom) {
        cache[threadIdx.x + 1][threadIdx.y + 2] = 0;
        if (isXRight) {
            cache[threadIdx.x + 2][threadIdx.y + 2] = 0;
        } else if (isXLeft) {
            cache[threadIdx.x][threadIdx.y + 2] = 0;
        }
    }

    //Setup pixel values
    cache[threadIdx.x + 1][threadIdx.y + 1] = in[row * width + col];

    //Check boundry conditions
    if (isXLeft && (col > 0)) {
        cache[threadIdx.x][threadIdx.y + 1] = in[row * width + (col - 1)];
    } else if (isXRight && (col < width - 1)) {
        cache[threadIdx.x + 2][threadIdx.y + 1] = in[row * width + (col + 1)];
    }
    if (isYTop && (row > 0)) {
        cache[threadIdx.x + 1][threadIdx.y] = in[(row - 1) * width + col];
        if (isXLeft) {
            cache[threadIdx.x][threadIdx.y] = in[(row - 1) * width + (col - 1)];
        } else if (isXRight) {
            cache[threadIdx.x + 2][threadIdx.y] = in[(row - 1) * width + (col + 1)];
        }
    } else if (isYBottom && (row < height - 1)) {
        cache[threadIdx.x + 1][threadIdx.y + 2] = in[(row + 1) * width + col];
        if (isXRight) {
            cache[threadIdx.x + 2][threadIdx.y + 2] = in[(row + 1) * width + (col + 1)];
        } else if (isXLeft) {
            cache[threadIdx.x][threadIdx.y + 2] = in[(row + 1) * width + (col - 1)];
        }
    }

    // Wait for all threads to be done
    __syncthreads();

    //Setup the filter
    unsigned char filterVector[9] = {cache[threadIdx.x][threadIdx.y], cache[threadIdx.x + 1][threadIdx.y],
                                     cache[threadIdx.x + 2][threadIdx.y],
                                     cache[threadIdx.x][threadIdx.y + 1], cache[threadIdx.x + 1][threadIdx.y + 1],
                                     cache[threadIdx.x + 2][threadIdx.y + 1],
                                     cache[threadIdx.x][threadIdx.y + 2], cache[threadIdx.x + 1][threadIdx.y + 2],
                                     cache[threadIdx.x + 2][threadIdx.y + 2]};

    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (filterVector[i] > filterVector[j]) {
                //Swap values
                char tmp = filterVector[i];
                filterVector[i] = filterVector[j];
                filterVector[j] = tmp;
            }
        }
    }
    // Set the output image values
    out[row * width + col] = filterVector[4];
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
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((int) ceil((float) image->Width() / (float) TILE_SIZE),
                 (int) ceil((float) image->Height() / (float) TILE_SIZE));

    // Check for shared memories and call the kernel
    cudaEventRecord(start);
    medianFilterKernelShared << < dimGrid, dimBlock >> > (deviceInputImage, deviceOutputImage, width, height);
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
    printf("\tMedian filter execution time: %f ms\n", time);

    // Save image
    outputImage->Save("lena_shared.bmp");

    return true;
}