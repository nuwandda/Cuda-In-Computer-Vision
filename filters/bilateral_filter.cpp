#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// void bilateralFilter(const Mat & input, Mat & output, int r,double sI, double sS);

int main() {
    // Read input image with grayscale option
    Mat input_image = imread("images/car.jpg", IMREAD_GRAYSCALE);
    // Create output image
    Mat output_image_cv;
    // OpenCV bilateral filter
	clock_t start_s = clock();
	bilateralFilter(input_image, output_image_cv, 9, 75, 75);
	clock_t stop_s = clock();
	cout << "Time for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 <<  " ms" << endl;
	// Display opencv bf image
	imshow("Image2", output_image_cv);
	waitKey();
}
