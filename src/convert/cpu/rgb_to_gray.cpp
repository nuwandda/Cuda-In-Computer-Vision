#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "convert/rgb_to_gray.h"

using namespace cv;
using namespace std;


void RgbToGray::Convert(cv::Mat input_image) {
    Mat output_image;
    clock_t start_s = clock();
    cvtColor(input_image, output_image, COLOR_BGR2GRAY);
    clock_t stop_s = clock();
    cout << "RGB to Gray:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;
}
