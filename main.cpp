#include <iostream>
#include <cstdio>  // defines FILENAME_MAX
#include <unistd.h> // for getcwd()
#include "filters/cpu/filters.h"
#include "detectors/edge/cpu/edge_detector.h"

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
    Mat input_image = imread(PATH + "images/car.jpg", IMREAD_GRAYSCALE);
    // Create output image
    Mat output_image;

    Filters::ApplyFilters(input_image, output_image);

    input_image = imread(PATH + "images/Lenna.png", CV_LOAD_IMAGE_COLOR);
    output_image = EdgeDetector::Detect(input_image, 50, 150, 3);
    imwrite(PATH + "images/lenna_draw.png", output_image);
}
