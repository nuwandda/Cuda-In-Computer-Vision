#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <time.h>
#include "Bitmap.h"
#include<stdio.h>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK_SIZE 16
#define MASK_WIDTH 4
#define MASK_HEIGHT 4
#define MASK_RADIUS_X MASK_WIDTH/2
#define MASK_RADIUS_Y MASK_HEIGHT/2
#define TILE_WIDTH 16
#define W_X (TILE_WIDTH + MASK_WIDTH - 1) //Shared memory elements needed to load as per mask size
#define W_Y (TILE_WIDTH + MASK_HEIGHT - 1)

__device__ __constant__
int BLUR_SIZE = 1;

__global__ void gaussianFilterKernelShared(unsigned char *in, unsigned char *out, int width, int height) {
    // Kernel to calculate gaussian blur. Overwrites the given output image with the new blurred image.
    // Works with gray scaled images.
    //
    // Args:
    //    in (unsigned char): Input image
    //    out (unsigned char): Output image
    //    width (int): Width of the input image
    //    height (int): Height of the input image


    // destY - destX -> Row and column of shared memory
    // srcY - srcX -> Indexes to fetch data from input Image
    // src -> index of input Image

    // Initialize shared memory
    __shared__ unsigned char cache[W_Y][W_X];
    int k = 1; // This indicates channel size. We use gray scale images, so it is 1.

    // Load elements to shared memory
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destX = dest % W_X;
    int destY = dest / W_X;
    int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS_Y;
    int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS_X;
    int src = (srcY * width + srcX) + k;

    // Check boundary condition
    // If the pixel is not in boundaries, set it to zero
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        cache[destY][destX] = in[src];
    } else {
        cache[destY][destX] = 0.0;
    }

    for (int iter = 1; iter <= (W_X * W_Y) / (TILE_WIDTH * TILE_WIDTH); iter++) {
        // Load elements to shared memory for the second time
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + iter * (TILE_WIDTH * TILE_WIDTH);
        destX = dest % W_X;
        destY = dest / W_X;
        srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS_Y;
        srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS_X;
        src = (srcY * width + srcX) + k;

        if (destY < W_Y && destX < W_X) {
            // Check boundary condition
            // If the pixel is not in boundaries, set it to zero
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
                cache[destY][destX] = in[src];
            } else {
                cache[destY][destX] = 0.0;
            }
        }
    }
    // Wait for all threads to be done
    __syncthreads();

    unsigned char pixVal = 0;
    int y, x;
    for (y = 0; y < MASK_WIDTH; y++) {
        for (x = 0; x < MASK_WIDTH; x++) {
            pixVal += cache[threadIdx.y + y][threadIdx.x + x] / 16;
        }
    }
    y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    x = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (y < height && x < width) {
        // Set the output image values
        out[(y * width + x) + k] = pixVal;
    }
    // Wait for all threads to be done
    __syncthreads();
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

    cudaEventRecord(start);
    gaussianFilterKernelShared << < dimGrid, dimBlock >> > (deviceInputImage, deviceOutputImage, width, height);
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
    printf("\tGaussian filter execution time: %f ms\n", time);

    // Save image
    outputImage->Save("lena_shared.bmp");

    return true;
}