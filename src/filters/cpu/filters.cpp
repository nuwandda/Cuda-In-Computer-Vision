#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters/cpu/filters.h"

using namespace std;
using namespace cv;

void Filters::ApplyFilters(cv::Mat input_image, cv::Mat output_image) {
    // Bilateral Filter
    clock_t start_s = clock();
    bilateralFilter(input_image, output_image, 9, 75, 75);
    clock_t stop_s = clock();
    cout << "Bilateral Filter:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;

    // Blurs an image using the normalized box filter.
    start_s = clock();
    blur(input_image, output_image, Size(9, 9));
    stop_s = clock();
    cout << "Normalized Box Filter:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;

    // Gaussian Blur
    start_s = clock();
    GaussianBlur(input_image, output_image, Size(9, 9), 75, 75);
    stop_s = clock();
    cout << "Gaussian Blur:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;

    start_s = clock();
    medianBlur(input_image, output_image, 9);
    stop_s = clock();
    cout << "Median Blur:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;
}
