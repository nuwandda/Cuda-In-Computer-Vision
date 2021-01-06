#include <detectors/edge/cpu/edge_detector.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat EdgeDetector::CannyDetect(const cv::Mat &input_image, int threshold1, int threshold2, int aperture) {
    Mat gray, edge, draw;
    cvtColor(input_image, gray, CV_BGR2GRAY);

    clock_t start_s = clock();
    Canny(gray, edge, threshold1, threshold2, aperture);
    clock_t stop_s = clock();
    cout << "Canny Edge Detection:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;

    edge.convertTo(draw, CV_8U);

    return draw;
}

cv::Mat EdgeDetector::SobelDetect(const cv::Mat &input_image, int ksize, int scale, int delta) {
    Mat src, src_gray, grad;
    int ddepth = CV_16S;
    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur(input_image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
    // Convert the image to grayscale
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    clock_t start_s = clock();
    Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    clock_t stop_s = clock();
    cout << "Sobel Edge Detection:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;

    return grad;
}
