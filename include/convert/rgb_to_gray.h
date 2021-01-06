#include <opencv2/imgproc/imgproc.hpp>

#ifndef CUDA_IN_COMPUTER_VISION_RGB_TO_GRAY_H
#define CUDA_IN_COMPUTER_VISION_RGB_TO_GRAY_H

class RgbToGray {
public:
    static void Convert(cv::Mat input_image);
};

#endif //CUDA_IN_COMPUTER_VISION_RGB_TO_GRAY_H
