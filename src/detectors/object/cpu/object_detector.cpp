#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <detectors/object/cpu/object_detector.h>

using namespace cv;
using namespace std;

RNG rng(12345);

cv::Mat ObjectDetector::ColorBasedDetection(const cv::Mat &input_image, int lower_hue, int higher_hue) {
    int lower_saturation = 120;
    int higher_saturation = 255;

    int lower_value = 70;
    int higher_value = 255;

    clock_t start_s = clock();
    Mat input_image_hsv;
    // Convert RGB image to HSV
    cvtColor(input_image, input_image_hsv, COLOR_BGR2HSV);

    // Threshold the image
    Mat mask1, mask2;
    // Creating masks to detect the upper and lower given color.
    inRange(input_image_hsv, Scalar(0, lower_saturation, lower_value), Scalar(10, higher_saturation, higher_value),
            mask1);
    inRange(input_image_hsv, Scalar(170, lower_saturation, lower_value), Scalar(180, higher_saturation, higher_value),
            mask2);

    // Generating the final mask
    mask1 = mask1 + mask2;

    // Get all non black points
    vector<Point> pts;
    findNonZero(mask1, pts);

    // Detect contours and draw a rectangle over detected object
    vector<vector<Point> > contours;
    findContours(mask1, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    vector<Point2f> centers(contours.size());
    vector<float> radius(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = boundingRect(contours_poly[i]);
        minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
    }

    Mat drawing = input_image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        if (boundRect[i].width > 30 && boundRect[i].height > 30) {
            rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
        }
    }
    clock_t stop_s = clock();
    cout << "Object Detection:" << endl;
    cout << "\tTime for the CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;

    return drawing;
}
