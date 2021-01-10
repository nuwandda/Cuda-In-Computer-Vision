#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#ifndef CUDA_IN_COMPUTER_VISION_OBJECT_DETECTOR_H
#define CUDA_IN_COMPUTER_VISION_OBJECT_DETECTOR_H

class ObjectDetector {
public:
    static cv::Mat ColorBasedDetection(const cv::Mat &input_image, int lower_hue, int higher_hue);
};

#endif //CUDA_IN_COMPUTER_VISION_OBJECT_DETECTOR_H
