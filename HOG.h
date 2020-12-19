#pragma once

// Importing all the necessary headers
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <fstream>
// Function to compute gradients.
void computeMagAngle(cv::InputArray src, cv::Mat& mag, cv::Mat& ang);

// Function to compute the main HOG Descriptor
void computeHOG(cv::InputArray mag, cv::InputArray ang, cv::Mat& dst);