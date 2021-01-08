#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters/cpu/filters.h"

using namespace std;
using namespace cv;

void Filters::ApplyFilters(const cv::Mat &input_image, cv::Mat output_image) {
    // Bilateral Filter
    clock_t start_s = clock();
    bilateralFilter(input_image, output_image, 9, 75, 75);
    clock_t stop_s = clock();
    cout << "Bilateral Filter:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;

    // Gaussian Filter
    start_s = clock();
    GaussianBlur(input_image, output_image, Size(9, 9), 75, 75);
    stop_s = clock();
    cout << "Gaussian Filter:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;

    // Median Filter
    start_s = clock();
    medianBlur(input_image, output_image, 9);
    stop_s = clock();
    cout << "Median Filter:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;
}
