#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"

#ifndef CUDA_IN_COMPUTER_VISION_EDGE_DETECTOR_H
#define CUDA_IN_COMPUTER_VISION_EDGE_DETECTOR_H

class EdgeDetector {
public:
    static cv::Mat CannyDetect(const cv::Mat &input_image, int threshold1, int threshold2, int aperture);

    static cv::Mat SobelDetect(const cv::Mat &input_image, int ksize, int scale, int delta);
};

#endif //CUDA_IN_COMPUTER_VISION_EDGE_DETECTOR_H
