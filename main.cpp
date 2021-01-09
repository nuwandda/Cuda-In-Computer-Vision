#include <iostream>
#include <cstdio>
#include <unistd.h>
#include "filters/cpu/filters.h"
#include "detectors/edge/cpu/edge_detector.h"
#include "convert/rgb_to_gray.h"

using namespace std;
using namespace cv;

string PATH = "";

string GetExePath() {
    string cwd("\0", FILENAME_MAX + 1);
    return getcwd(&cwd[0], cwd.capacity());
}

int main() {
    if (PATH == "") {
        cout << "Please fill the root directory path! " << endl;
        return 0;
    }

    // Read input image with grayscale option
    Mat input_image = imread(PATH + "images/Lenna.png", IMREAD_GRAYSCALE);
    Mat input_image_bgr = imread(PATH + "images/Lenna.png", IMREAD_COLOR);
    // Create output image
    Mat output_image;

    Filters::ApplyFilters(input_image, output_image);

    input_image = imread(PATH + "images/Lenna.png", CV_LOAD_IMAGE_COLOR);
    output_image = EdgeDetector::CannyDetect(input_image, 50, 150, 3);
    imwrite(PATH + "images/canny_lenna_draw.png", output_image);

    output_image = EdgeDetector::SobelDetect(input_image, 1, 1, 0);
    imwrite(PATH + "images/sobel_lenna_draw.png", output_image);

    output_image = EdgeDetector::LaplacianDetect(input_image, 3, 1, 0);
    imwrite(PATH + "images/laplacian_lenna_draw.png", output_image);

    RgbToGray::Convert(input_image_bgr);
}
