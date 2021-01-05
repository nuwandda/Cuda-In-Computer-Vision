#include <opencv2/opencv.hpp>

#ifndef CUDA_IN_COMPUTER_VISION_FILTERS_H
#define CUDA_IN_COMPUTER_VISION_FILTERS_H

class Filters {
public:
    static void ApplyFilters(cv::Mat input_image, cv::Mat output_image);
};

#endif //CUDA_IN_COMPUTER_VISION_FILTERS_H
