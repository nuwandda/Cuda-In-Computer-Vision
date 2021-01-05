#include <detectors/edge/cpu/edge_detector.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat EdgeDetector::Detect(cv::Mat input_image, int threshold1, int threshold2, int aperture) {
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
